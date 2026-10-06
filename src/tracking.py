"""Seguimiento corporal de bajo consumo: YOLO poco frecuente + flujo óptico.

YOLO crea las trayectorias y corrige su posición; entre dos detecciones cada
persona se sigue con flujo óptico Lucas-Kanade (sin redes neuronales). Una
puerta de movimiento por diferencia de fotogramas despierta a YOLO si algo se
mueve fuera de las personas ya seguidas.

Regla de seguridad: el ahorro solo aplica a seguir a quien ya fue detectado;
ante cualquier duda (movimiento nuevo, trayectoria perdida, alarma nueva o
demasiado tiempo sin YOLO) se ejecuta YOLO.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np


Box = tuple[float, float, float, float]       # x1, y1, x2, y2 en píxeles

# Motivos por los que se ejecuta YOLO (se registran para auditar el ahorro).
REASON_CONFIRM = "confirmacion"       # alarma sin veredicto todavía
REASON_ALARM = "alarma"               # la cámara inició una alarma nueva
REASON_MOTION = "movimiento_nuevo"    # movimiento fuera de las personas seguidas
REASON_LOST = "trayectoria_perdida"   # el flujo óptico perdió a alguien
REASON_INTERVAL = "intervalo"         # tiempo máximo sin YOLO
REASON_WATCH = "vigilancia"           # alarma activa sin personas seguidas

FLOW_SCALE = 0.5            # el flujo y la puerta trabajan a media resolución


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def camera_alias(camera: str) -> str:
    """``192.168.100.109`` → ``cam109``; otros nombres se usan tal cual."""
    last = camera.rsplit(".", 1)[-1]
    return f"cam{last}" if last.isdigit() else camera


class FlowTracker:
    """Sigue una caja moviendo puntos característicos con Lucas-Kanade.

    Trabaja sobre imágenes en gris a escala ``FLOW_SCALE``. Usa verificación
    ida y vuelta: los puntos que no regresan a su origen se descartan. Si
    quedan muy pocos puntos (zona oscura, oclusión) la trayectoria se marca
    perdida y YOLO decide.
    """

    MAX_POINTS = 30
    MIN_POINTS = 5
    MAX_FB_ERROR = 1.5

    def __init__(self, gray: np.ndarray, box: Box) -> None:
        self.box = box
        self.lost = False
        self.moved = 0.0
        self._points = self._seed(gray, box)
        self._initial = len(self._points)
        if self._initial < self.MIN_POINTS:
            self.lost = True

    @staticmethod
    def _seed(gray: np.ndarray, box: Box) -> np.ndarray:
        height, width = gray.shape[:2]
        x1, y1, x2, y2 = (int(round(v * FLOW_SCALE)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return np.empty((0, 1, 2), np.float32)
        points = cv2.goodFeaturesToTrack(gray[y1:y2, x1:x2], FlowTracker.MAX_POINTS,
                                         0.01, 3)
        if points is None:
            return np.empty((0, 1, 2), np.float32)
        return points.astype(np.float32) + np.float32([x1, y1])

    def update(self, previous: np.ndarray, current: np.ndarray) -> bool:
        """Mueve la caja; devuelve False si la trayectoria se perdió."""
        if self.lost or len(self._points) == 0:
            self.lost = True
            return False
        params = dict(winSize=(15, 15), maxLevel=2)
        new, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, self._points,
                                                  None, **params)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, new,
                                                        None, **params)
        error = np.linalg.norm((self._points - back).reshape(-1, 2), axis=1)
        good = ((status.ravel() == 1) & (back_status.ravel() == 1)
                & (error < self.MAX_FB_ERROR))
        if good.sum() < max(self.MIN_POINTS, 0.3 * self._initial):
            self.lost = True
            return False
        shift = np.median((new - self._points).reshape(-1, 2)[good], axis=0)
        dx, dy = float(shift[0]) / FLOW_SCALE, float(shift[1]) / FLOW_SCALE
        x1, y1, x2, y2 = self.box
        self.box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        self.moved = float(np.hypot(dx, dy))
        self._points = new[good].reshape(-1, 1, 2)
        return True


class MotionGate:
    """Diferencia de fotogramas a baja resolución.

    Indica si hay movimiento fuera de las cajas seguidas (posible persona
    nueva) y si las personas seguidas se mueven.
    """

    def __init__(self, threshold: int = 25, min_fraction: float = 0.004,
                 margin: float = 0.15) -> None:
        self.threshold = threshold
        self.min_fraction = min_fraction
        self.margin = margin
        self._previous: np.ndarray | None = None

    def update(self, gray: np.ndarray, boxes: list[Box]) -> tuple[bool, bool]:
        """Devuelve (movimiento fuera de las cajas, movimiento dentro)."""
        small = cv2.GaussianBlur(cv2.resize(gray, (160, 90),
                                            interpolation=cv2.INTER_AREA), (5, 5), 0)
        previous, self._previous = self._previous, small
        if previous is None:
            return False, False
        mask = cv2.absdiff(small, previous) > self.threshold
        inside = np.zeros_like(mask)
        sx = 160 / (gray.shape[1] / FLOW_SCALE)
        sy = 90 / (gray.shape[0] / FLOW_SCALE)
        for x1, y1, x2, y2 in boxes:
            mx, my = (x2 - x1) * self.margin, (y2 - y1) * self.margin
            inside[max(0, int((y1 - my) * sy)):int((y2 + my) * sy) + 1,
                   max(0, int((x1 - mx) * sx)):int((x2 + mx) * sx) + 1] = True
        minimum = self.min_fraction * mask.size
        outside_count = int((mask & ~inside).sum())
        inside_count = int((mask & inside).sum())
        return outside_count >= minimum, inside_count >= minimum / 2


@dataclass
class PersonTrack:
    camera: str
    code: str                     # legible: cam109-20261006-145731-1
    started_wall: datetime
    first_seen: float             # time.monotonic()
    last_seen: float              # última vez que YOLO la vio
    box: Box
    confidence: float
    reason: str                   # motivo de la ejecución de YOLO que la creó
    alarm_id: str | None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    max_confidence: float = 0.0
    last_seen_wall: datetime | None = None
    yolo_hits: int = 1
    misses: int = 0
    flow: FlowTracker | None = None

    @property
    def visible_seconds(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


class TrackManager:
    """Asocia las detecciones de YOLO con las trayectorias existentes."""

    def __init__(self, camera: str, match_iou: float = 0.25, max_misses: int = 2,
                 forget_seconds: float = 2.0) -> None:
        self.camera = camera
        self.alias = camera_alias(camera)
        self.match_iou = match_iou
        self.max_misses = max_misses
        self.forget_seconds = forget_seconds
        self.tracks: list[PersonTrack] = []
        self._codes_in_second: dict[str, int] = {}

    def new_code(self, moment: datetime) -> str:
        stamp = moment.strftime("%Y%m%d-%H%M%S")
        number = self._codes_in_second.get(stamp, 0) + 1
        self._codes_in_second = {stamp: number}   # solo importa el segundo actual
        return f"{self.alias}-{stamp}-{number}"

    def flow_step(self, previous: np.ndarray, current: np.ndarray) -> tuple[bool, bool]:
        """Mueve las cajas entre detecciones. Devuelve (alguna perdida, movimiento)."""
        any_lost, moving = False, False
        for track in self.tracks:
            if track.flow is None:
                continue
            if not track.flow.update(previous, current):
                any_lost = True
            else:
                track.box = track.flow.box
                moving = moving or track.flow.moved > 1.0
        return any_lost, moving

    def apply_detections(self, gray: np.ndarray | None, boxes: list[Box],
                         confidences: list[float], now: float, wall: datetime,
                         reason: str, alarm_id: str | None
                         ) -> tuple[list[PersonTrack], list[PersonTrack]]:
        """Actualiza con una ejecución de YOLO. Devuelve (nuevas, terminadas)."""
        pairs = sorted(((iou(track.box, box), t_index, d_index)
                        for t_index, track in enumerate(self.tracks)
                        for d_index, box in enumerate(boxes)), reverse=True)
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for overlap, t_index, d_index in pairs:
            if overlap < self.match_iou:
                break
            if t_index in used_tracks or d_index in used_detections:
                continue
            used_tracks.add(t_index)
            used_detections.add(d_index)
            track = self.tracks[t_index]
            track.box = boxes[d_index]
            track.confidence = confidences[d_index]
            track.max_confidence = max(track.max_confidence, track.confidence)
            track.last_seen = now
            track.last_seen_wall = wall
            track.yolo_hits += 1
            track.misses = 0
            track.flow = FlowTracker(gray, track.box) if gray is not None else None

        ended: list[PersonTrack] = []
        for t_index, track in enumerate(self.tracks):
            if t_index in used_tracks:
                continue
            track.misses += 1
            if (track.misses >= self.max_misses
                    and now - track.last_seen >= self.forget_seconds):
                ended.append(track)
        self.tracks = [track for track in self.tracks if track not in ended]

        created: list[PersonTrack] = []
        for d_index, box in enumerate(boxes):
            if d_index in used_detections:
                continue
            track = PersonTrack(
                camera=self.camera, code=self.new_code(wall), started_wall=wall,
                first_seen=now, last_seen=now, box=box,
                confidence=confidences[d_index], reason=reason, alarm_id=alarm_id,
                max_confidence=confidences[d_index], last_seen_wall=wall,
                flow=FlowTracker(gray, box) if gray is not None else None)
            self.tracks.append(track)
            created.append(track)
        return created, ended

    def close_all(self) -> list[PersonTrack]:
        ended, self.tracks = self.tracks, []
        return ended


def yolo_reason(*, now: float, last_yolo: float, verdict_pending: bool,
                alarm_pending: bool, alarm_active: bool, has_tracks: bool,
                motion_outside: bool, track_lost: bool, tracks_moving: bool,
                confirm_interval: float = 0.2, trigger_interval: float = 0.5,
                moving_interval: float = 1.0, still_interval: float = 3.0,
                watch_interval: float = 1.0) -> str | None:
    """Decide si este fotograma necesita YOLO y por qué (None: basta el flujo).

    ``trigger_interval`` limita los disparos por movimiento o trayectoria
    perdida: una cortina que se mueve sin parar no debe ejecutar YOLO en cada
    fotograma. Es también la demora máxima para detectar a una persona nueva.
    """
    elapsed = now - last_yolo
    if alarm_pending:
        return REASON_ALARM
    if verdict_pending:
        return REASON_CONFIRM if elapsed >= confirm_interval else None
    if motion_outside and elapsed >= trigger_interval:
        return REASON_MOTION
    if track_lost and elapsed >= trigger_interval:
        return REASON_LOST
    if has_tracks:
        limit = moving_interval if tracks_moving else still_interval
        return REASON_INTERVAL if elapsed >= limit else None
    if alarm_active and elapsed >= watch_interval:
        return REASON_WATCH
    return None
