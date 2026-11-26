"""Sesión de análisis por cámara: verificación de alarmas + seguimiento corporal.

Una sesión empieza con una alarma de la cámara y dura mientras la alarma siga
activa o haya personas seguidas. Durante la sesión:

- cada alarma recibe su veredicto (YOLO a ritmo de confirmación);
- confirmada la persona, YOLO baja a ~1 fps y el flujo óptico sigue a cada
  persona entre detecciones;
- las salvaguardas de ``tracking.yolo_reason`` despiertan a YOLO ante
  cualquier duda.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.analysis.face_policy import (
    IDENTIFIED,
    FaceState,
    register_attempt,
    should_attempt,
)
from src.analysis.face_worker import FaceWorker
from src.analysis.policy import yolo_reason
from src.analysis.verdict import VerificationSummary, VerificationTracker
from src.detection.base import FrameResult, PersonDetector
from src.outputs.annotate import annotate
from src.settings import CameraSettings
from src.sources.video import open_rtsp
from src.tracking import FLOW_SCALE, MotionGate, PersonTrack, TrackManager
from src.outputs.viewer import draw_overlay




@dataclass
class SessionStats:
    camera: str
    started_wall: datetime
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ended_wall: datetime | None = None
    frames: int = 0               # fotogramas procesados (flujo + puerta)
    yolo_runs: int = 0
    reasons: Counter = field(default_factory=Counter)
    new_by_reason: Counter = field(default_factory=Counter)
    recoveries: Counter = field(default_factory=Counter)    # posicion / apariencia
    face_grabs: int = 0           # ráfagas del vídeo HD
    face_attempts: int = 0        # intentos de rostro (todas las personas)
    face_compared: int = 0        # intentos que llegaron a SFace + galería
    identified: int = 0
    unknown: int = 0
    tracks: int = 0
    max_people: int = 0
    error: str | None = None
    alarms: int = 0               # alarmas de la cámara atendidas en la sesión
    confirmed: int = 0            # de ellas, confirmadas por YOLO
    snapshot: Path | None = None  # foto de la sesión (la de más personas)

    @property
    def savings(self) -> float:
        """Fracción de fotogramas que no necesitaron YOLO."""
        return 1.0 - self.yolo_runs / self.frames if self.frames else 0.0


@dataclass
class SessionCallbacks:
    on_verdict: Callable[[str, VerificationSummary], None] = lambda *_: None
    on_finished: Callable[[VerificationSummary], None] = lambda _: None
    on_track: Callable[[str, PersonTrack, str | None], None] = lambda *_: None
    on_session: Callable[[SessionStats], None] = lambda _: None
    on_face_attempt: Callable[[PersonTrack, object, float], None] = lambda *_: None


class AnalysisSession(threading.Thread):
    def __init__(self, camera: str, url: str, detector: PersonDetector,
                 callbacks: SessionCallbacks, snapshot_dir: Path | None,
                 alarm_id: str | None, settings: CameraSettings | None = None,
                 viewer=None, faces=None, next_label=None) -> None:
        super().__init__(name=f"analisis-{camera}", daemon=True)
        self.camera = camera
        self._url = url
        self._detector = detector
        self._cb = callbacks
        self._snapshot_dir = snapshot_dir
        self.settings = settings or CameraSettings(camera)
        self._max_seconds = self.settings.get("sesion", "max_s")
        self._viewer = viewer
        # Rostro: (FaceEngine, HdFrameGrabber) o None si está desactivado.
        self._faces = faces
        self._next_label = next_label or (lambda: None)
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._pending_alarms: list[str | None] = [alarm_id]
        self._alarm_active = True
        self._closing = False
        self._frame_lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._latest_id = 0
        self.stats = SessionStats(camera, datetime.now())

    # --- control desde el nodo (otros hilos) ------------------------------

    def attach_alarm(self, alarm_id: str | None) -> bool:
        """Una alarma nueva durante la sesión: se verifica y fuerza YOLO.

        Devuelve False si la sesión ya está cerrando (hay que abrir otra).
        """
        with self._lock:
            if self._closing:
                return False
            self._pending_alarms.append(alarm_id)
            self._alarm_active = True
            return True

    def alarm_ended(self) -> None:
        with self._lock:
            self._alarm_active = False

    def request_stop(self) -> None:
        self._halt.set()

    # --- lectura de vídeo ---------------------------------------------------

    def _read_frames(self, capture: cv2.VideoCapture) -> None:
        # Lee continuamente para que el búfer RTSP no entregue fotogramas
        # atrasados; el análisis siempre toma el más reciente.
        while not self._halt.is_set():
            ok, frame = capture.read()
            if not ok or frame is None:
                self._halt.set()
                return
            with self._frame_lock:
                self._latest = frame
                self._latest_id += 1

    # --- bucle principal ----------------------------------------------------

    def run(self) -> None:
        verifications: list[VerificationTracker] = []
        cfg = self.settings
        verdict_cfg, policy_cfg = cfg.section("veredicto"), cfg.section("politica")
        gate_cfg, yolo_cfg = cfg.section("puerta"), cfg.section("yolo")
        detect_options = {"confidence": yolo_cfg["confianza"],
                          "weak_confidence": yolo_cfg["confianza_debil"],
                          "image_size": yolo_cfg["tamano"]}
        process_interval = cfg.get("sesion", "procesar_cada_s")
        tracks = TrackManager.from_settings(self.camera, cfg)
        gate = MotionGate(threshold=gate_cfg["umbral"], min_fraction=gate_cfg["fraccion_min"],
                          margin=gate_cfg["margen"])
        best: tuple[np.ndarray, FrameResult] | None = None
        best_key = (-1, 0.0)
        current_alarm: str | None = None
        alarm_pending = False
        # Una aparición repentina cambia la imagen en un solo fotograma: el
        # movimiento nuevo y la pérdida de trayectoria quedan retenidos hasta
        # que YOLO se ejecute, aunque el límite de frecuencia lo retrase.
        motion_latched = False
        lost_latched = False
        last_yolo = 0.0
        previous_gray: np.ndarray | None = None
        last_id = 0
        started = time.monotonic()
        face_cfg = cfg.section("rostro")
        face_states: dict[str, FaceState] = {}
        boxes_lock = threading.Lock()
        box_snapshot: dict[str, tuple] = {}
        worker = None
        if self._faces is not None:
            engine, grabber = self._faces

            def current_boxes(ids: set[str]) -> dict[str, tuple]:
                with boxes_lock:
                    return {tid: box_snapshot[tid] for tid in ids if tid in box_snapshot}

            worker = FaceWorker(self.camera, engine, grabber, current_boxes,
                                lambda: tracks.frame_size, face_cfg)
            worker.start()

        def apply_face_results(now: float) -> None:
            while worker is not None:
                try:
                    track_id, attempt, hd_ms = worker.results.get_nowait()
                except queue.Empty:
                    return
                state = face_states.get(track_id)
                track = next((t for t in tracks.tracks + tracks.grace if t.id == track_id), None)
                if state is None or track is None:
                    continue
                if attempt is None:                  # no se pudo abrir el vídeo HD
                    state.requested, state.last_attempt = False, now
                    continue
                self.stats.face_attempts += 1
                self.stats.face_compared += int(attempt.counts)
                event = register_attempt(state, attempt, now, face_cfg)
                track.face_attempts = state.attempts
                track.face_status = state.status
                self._cb.on_face_attempt(track, attempt, hd_ms)
                if event is None:
                    continue
                status, name, score = event
                track.face_score = score
                if status == IDENTIFIED:
                    track.identity = name
                    self.stats.identified += 1
                    self._cb.on_track("identified", track, None)
                else:
                    self.stats.unknown += 1
                    self._cb.on_track("unknown", track, None)

        def request_faces(now: float) -> None:
            if worker is None:
                return
            with boxes_lock:
                box_snapshot.clear()
                box_snapshot.update({t.id: t.box for t in tracks.tracks})
            for track in tracks.tracks:
                state = face_states.setdefault(track.id, FaceState())
                if should_attempt(state, now, track.box[3] - track.box[1], face_cfg):
                    state.requested = True
                    worker.request(track.id)

        def take_pending() -> bool:
            nonlocal current_alarm
            with self._lock:
                pending, self._pending_alarms = self._pending_alarms, []
            for alarm_id in pending:
                tracker = VerificationTracker(
                    self.camera, time.monotonic(),
                    confirm_frames=verdict_cfg["confirmar_fotogramas"],
                    decide_seconds=verdict_cfg["decidir_s"],
                    seen_decide_seconds=verdict_cfg["decidir_con_persona_s"])
                tracker.summary.alarm_id = alarm_id
                verifications.append(tracker)
                current_alarm = alarm_id
            return bool(pending)

        def check_timeouts(now: float) -> None:
            for tracker in verifications:
                verdict = tracker.check_timeout(now)
                if verdict is not None:
                    self._cb.on_verdict(verdict, tracker.summary)

        take_pending()
        capture = open_rtsp(self._url)
        if not capture.isOpened():
            capture.release()
            self.stats.error = "no se pudo abrir el vídeo RTSP"
            with self._lock:
                self._closing = True
            take_pending()
            for tracker in verifications:
                tracker.summary.error = self.stats.error
                tracker.check_timeout(float("inf"))
                self._cb.on_verdict("discarded", tracker.summary)
                self._cb.on_finished(tracker.summary)
            self._finish(tracks)
            return
        reader = threading.Thread(target=self._read_frames, args=(capture,),
                                  name=f"rtsp-{self.camera}", daemon=True)
        reader.start()
        try:
            while not self._halt.is_set():
                alarm_pending = take_pending() or alarm_pending
                now = time.monotonic()
                if now - started > self._max_seconds:
                    self.stats.error = f"sesión detenida tras {self._max_seconds:.0f} s"
                    break
                with self._frame_lock:
                    frame, frame_id = self._latest, self._latest_id
                if frame is None or frame_id == last_id:
                    check_timeouts(now)
                    self._halt.wait(0.02)
                    continue
                last_id = frame_id
                self.stats.frames += 1
                apply_face_results(now)
                if tracks.frame_size is None:
                    tracks.frame_size = (frame.shape[1], frame.shape[0])

                gray = cv2.cvtColor(cv2.resize(frame, None, fx=FLOW_SCALE, fy=FLOW_SCALE,
                                               interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_BGR2GRAY)
                lost, moving = False, False
                if previous_gray is not None and tracks.tracks:
                    lost, moving = tracks.flow_step(previous_gray, gray, now)
                motion_outside, motion_inside = gate.update(
                    gray, [track.box for track in tracks.tracks])
                motion_latched = motion_latched or motion_outside
                lost_latched = lost_latched or lost
                with self._lock:
                    alarm_active = self._alarm_active
                verdict_pending = any(v.summary.verdict is None for v in verifications)
                reason = yolo_reason(
                    now=now, last_yolo=last_yolo,
                    verdict_pending=verdict_pending,
                    alarm_pending=alarm_pending, alarm_active=alarm_active,
                    searching=bool(tracks.grace),
                    has_tracks=bool(tracks.tracks), motion_outside=motion_latched,
                    track_lost=lost_latched, tracks_moving=moving or motion_inside,
                    confirm_interval=policy_cfg["confirmacion_s"],
                    trigger_interval=policy_cfg["disparo_min_s"],
                    moving_interval=policy_cfg["movimiento_s"],
                    still_interval=policy_cfg["quieto_s"],
                    watch_interval=policy_cfg["vigilancia_s"])

                if reason is not None:
                    result = self._detector.detect(frame, **detect_options)
                    last_yolo, alarm_pending = time.monotonic(), False
                    motion_latched = lost_latched = False
                    self.stats.yolo_runs += 1
                    self.stats.reasons[reason] += 1
                    self.stats.max_people = max(self.stats.max_people, result.people)
                    for tracker in verifications:
                        verdict = tracker.add(result)
                        if verdict is not None:
                            self._cb.on_verdict(verdict, tracker.summary)
                    key = (result.people, max(result.confidences, default=0.0))
                    if result.people and key > best_key:
                        best, best_key = (frame, result), key
                    created, ended = tracks.apply_detections(
                        gray, [tuple(map(float, box)) for box in result.boxes],
                        list(result.confidences), last_yolo, datetime.now(),
                        reason, current_alarm,
                        [tuple(map(float, box)) for box in result.weak_boxes],
                        list(result.weak_confidences), frame)
                    for track, how, unseen in tracks.last_recovered:
                        self.stats.recoveries[how] += 1
                        if track.id in face_states:      # pudo cambiar de persona
                            face_states[track.id].recheck = True
                        self._cb.on_track("recovered", track, f"{how}:{unseen:.1f}")
                    for track in created:
                        track.session_id = self.stats.id
                        track.label = self._next_label()
                        self.stats.tracks += 1
                        self.stats.new_by_reason[reason] += 1
                        self._cb.on_track("entered", track, None)
                    for track in ended:
                        self._cb.on_track("left", track, None)
                    request_faces(last_yolo)
                else:
                    check_timeouts(now)
                previous_gray = gray
                if self._viewer is not None:
                    self._viewer.publish(self.camera, draw_overlay(
                        frame, tracks.tracks + tracks.grace, reason, motion_outside,
                        self.stats,
                        alarm_active, verdict_pending))

                decided = all(v.summary.verdict is not None for v in verifications)
                if (not alarm_active and decided and not tracks.tracks
                        and not tracks.grace):
                    with self._lock:
                        # Se cierra solo si no llegó otra alarma entretanto.
                        if not self._pending_alarms:
                            self._closing = True
                    if self._closing:
                        break
                self._halt.wait(process_interval)
            check_timeouts(float("inf"))
        finally:
            if worker is not None:
                worker.request_stop()
                worker.join(timeout=8)
                apply_face_results(time.monotonic())
                self.stats.face_grabs = worker.grabs
            with self._lock:
                self._closing = True
            take_pending()      # alarmas llegadas durante el cierre
            check_timeouts(float("inf"))
            self._halt.set()
            reader.join(timeout=6)
            capture.release()

        snapshot = self._save(*best) if best is not None else None
        self.stats.snapshot = snapshot
        self.stats.alarms = len(verifications)
        self.stats.confirmed = sum(v.summary.verdict == "confirmed" for v in verifications)
        for tracker in verifications:
            tracker.summary.snapshot = snapshot
            if self.stats.error:
                tracker.summary.error = self.stats.error
            self._cb.on_finished(tracker.summary)
        self._finish(tracks)

    def _finish(self, tracks: TrackManager) -> None:
        note = self.stats.error or "sesión terminada"
        for track in tracks.close_all():
            self._cb.on_track("left", track, note)
        self.stats.ended_wall = datetime.now()
        if self._viewer is not None:
            self._viewer.idle(self.camera)
        self._cb.on_session(self.stats)

    def _save(self, frame: np.ndarray, result: FrameResult) -> Path | None:
        if self._snapshot_dir is None:
            return None
        moment = self.stats.started_wall
        directory = self._snapshot_dir / moment.strftime("%Y-%m-%d")
        name = (f"{moment:%H%M%S}_{self.camera.replace('.', '-')}"
                f"_{result.people}p.jpg")
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(directory / name), annotate(frame, result),
                           [cv2.IMWRITE_JPEG_QUALITY, 90]):
                return directory / name
        except (OSError, cv2.error):
            pass
        return None


class AnalysisManager:
    """Una sesión de análisis por cámara, iniciada y alimentada por las alarmas."""

    def __init__(self, detector: PersonDetector, urls: dict[str, str],
                 snapshot_dir: Path | None, viewer=None,
                 settings: dict[str, CameraSettings] | None = None,
                 faces: dict | None = None) -> None:
        self._detector = detector
        self._settings = settings or {}
        # cámara -> (FaceEngine, HdFrameGrabber), solo con el rostro activo.
        self._faces = faces or {}
        self._labels: dict[str, tuple[str, int]] = {}    # cámara -> (día, último N)
        self._viewer = viewer
        self._urls = urls
        self._snapshot_dir = snapshot_dir
        self._callbacks = SessionCallbacks()
        self._sessions: dict[str, AnalysisSession] = {}
        self._lock = threading.Lock()

    def set_callbacks(self, on_verdict, on_finished, on_track=None, on_session=None,
                      on_face_attempt=None) -> None:
        self._callbacks = SessionCallbacks(
            on_verdict, on_finished,
            on_track or (lambda *_: None), on_session or (lambda _: None),
            on_face_attempt or (lambda *_: None))

    def alarm_started(self, camera: str, alarm_id: str | None = None) -> bool:
        """Devuelve True si abrió una sesión nueva; False si se unió a la actual."""
        with self._lock:
            session = self._sessions.get(camera)
            if (session is not None and session.is_alive()
                    and session.attach_alarm(alarm_id)):
                return False
            session = AnalysisSession(camera, self._urls[camera], self._detector,
                                      self._callbacks, self._snapshot_dir, alarm_id,
                                      settings=self._settings.get(camera),
                                      viewer=self._viewer,
                                      faces=self._faces.get(camera),
                                      next_label=lambda c=camera: self.next_label(c))
            self._sessions[camera] = session
            session.start()
            return True

    def alarm_ended(self, camera: str) -> None:
        with self._lock:
            session = self._sessions.get(camera)
        if session is not None:
            session.alarm_ended()

    def next_label(self, camera: str) -> str:
        """'Persona N', con numeración que se reinicia cada día por cámara."""
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            day, number = self._labels.get(camera, (today, 0))
            number = number + 1 if day == today else 1
            self._labels[camera] = (today, number)
        return f"Persona {number}"

    @property
    def detector(self) -> PersonDetector:
        return self._detector

    def has_session(self, camera: str) -> bool:
        with self._lock:
            session = self._sessions.get(camera)
        return session is not None and session.is_alive()

    def close(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.request_stop()
        for session in sessions:
            session.join(timeout=8)
