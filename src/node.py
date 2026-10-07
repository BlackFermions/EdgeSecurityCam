"""Servicio del nodo: alarmas de la cámara + configuración + YOLO + base local."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.camera_config import ConfigWatcher, config_hash, diff_config, summarize
from src.camera_events import (
    AlarmStateTracker,
    CameraAlarm,
    CameraEventListener,
    OnvifEventClient,
)
from src.camera_web import CameraWeb
from src.storage import NodeStore


START_LABELS = {"human": "PERSONA DETECTADA", "motion": "MOVIMIENTO DETECTADO"}
NAMES = {"human": "persona", "motion": "movimiento"}

log = logging.getLogger("alarmas")


def describe_people(summary) -> str:
    noun = "persona" if summary.max_people == 1 else "personas"
    return (f"{summary.max_people} {noun} · confianza {summary.best_confidence:.2f}"
            f" · a los {summary.first_person_after:.1f} s")


class CamDetectorNode:
    def __init__(self, hosts: list[str], username: str, password: str,
                 store: NodeStore | None, motion_mode: str = "auto",
                 verifier=None, ia: dict[str, Any] | None = None,
                 config_interval: float = 60.0, base_dir: Path | None = None,
                 idle_sources: dict | None = None, idle_interval: float = 10.0,
                 idle_min_confidence: float = 0.50) -> None:
        self.hosts = hosts
        self.store = store
        self.verifier = verifier
        self.base_dir = base_dir
        self._lock = threading.Lock()
        self._closing = False
        self._tracker = AlarmStateTracker()
        self._open_alarms: dict[tuple[str, str], str] = {}
        self._configs: dict[str, dict[str, Any]] = {}
        # La cámara reenvía el mismo estado varias veces por segundo; en el
        # registro detallado solo se muestran los cambios de valor.
        self._raw_last: dict[tuple[str, str], tuple[bool | None, int]] = {}
        self.clients = {host: OnvifEventClient(host, username, password,
                                               motion_mode=motion_mode)
                        for host in hosts}
        self.watchers = {host: ConfigWatcher(host, CameraWeb(host, username, password),
                                             {"ia": ia or {}}, self._on_config,
                                             self._on_config_error, config_interval,
                                             self._on_config_recovered)
                         for host in hosts}
        self.listeners = [CameraEventListener(self.clients[host], self._on_alarm,
                                              self._status_handler(host))
                          for host in hosts]
        if verifier is not None:
            verifier.set_callbacks(self._on_verdict, self._on_verification_finished,
                                   self._on_track, self._on_session)
        # Verificación en reposo: solo con YOLO disponible y una fuente de
        # fotogramas por cámara.
        self.idle_checkers = []
        if verifier is not None and idle_sources and idle_interval > 0:
            from src.idle_check import IdleChecker
            self.idle_checkers = [
                IdleChecker(host, idle_sources[host], verifier.detector,
                            verifier.has_session, self._on_idle_person,
                            self._on_idle_status, interval=idle_interval,
                            min_confidence=idle_min_confidence)
                for host in hosts if host in idle_sources]

    # --- ciclo de vida ----------------------------------------------------

    def start(self) -> None:
        self._system_event(None, "inicio", f"nodo iniciado con {len(self.hosts)} cámara(s)")
        for host in self.hosts:
            # La configuración se lee antes de suscribirse: define si el tema
            # de movimiento es una persona o cualquier movimiento.
            self.watchers[host].read_once()
            self.watchers[host].start()
        for listener in self.listeners:
            listener.start()
        for checker in self.idle_checkers:
            checker.start()
        if self.idle_checkers:
            log.info("Verificación en reposo: 1 fotograma cada %.0f s si la cámara no avisa.",
                     self.idle_checkers[0].interval)
        log.info("Escuchando %d cámara(s). Ctrl+C para salir.", len(self.hosts))

    def close(self) -> None:
        # Las alarmas que llegan durante el apagado ya no se pueden verificar:
        # se ignoran para no dejar registros "sin vídeo" falsos.
        self._closing = True
        for watcher in self.watchers.values():
            watcher.request_stop()
        for checker in self.idle_checkers:
            checker.request_stop()
        for listener in self.listeners:
            listener.request_stop()
        for host in self.hosts:
            self._close_open_alarms(host, "nodo detenido")
        for listener in self.listeners:
            listener.close()
        if self.verifier is not None:
            self.verifier.close()
        self._system_event(None, "fin", "nodo detenido")

    # --- configuración ----------------------------------------------------

    def _on_config(self, camera: str, config: dict[str, Any]) -> None:
        with self._lock:
            previous_seen = self._configs.get(camera)
            self._configs[camera] = config
        previous_stored = None
        if self.store is not None:
            _, previous_stored = self.store.record_config(camera, config,
                                                         config_hash(config))
        if previous_seen is None:
            for line in summarize(config):
                log.info("[%s] %s", camera, line)
            if previous_stored is not None:
                for change in diff_config(previous_stored, config):
                    log.info("[%s] cambio desde la ejecución anterior: %s", camera, change)
        elif previous_stored is not None or previous_seen != config:
            changes = diff_config(previous_seen, config)
            for change in changes:
                log.info("[%s] CONFIGURACIÓN CAMBIADA: %s", camera, change)
            self._system_event(camera, "config", "; ".join(changes))

        client = self.clients[camera]
        motion_enabled = bool(config.get("motion", {}).get("enable"))
        changed = client.set_profile(config.get("camara", {}).get("modelo"), motion_enabled)
        if previous_seen is None or changed:
            meaning = "persona" if client.motion_as_human else "movimiento"
            log.info("[%s] las alarmas se registran como %s%s", camera, meaning,
                     "" if client.motion_as_human or client.model is None else
                     " (Motion Detect activo: puede ser cualquier movimiento)")
        if changed and previous_seen is not None:
            self._close_open_alarms(camera, "interpretación cambiada por configuración")

    def _on_config_error(self, camera: str, message: str) -> None:
        log.info("[%s] no se pudo leer la configuración web (%s)", camera, message)
        self._system_event(camera, "error", f"configuración: {message}")

    def _on_config_recovered(self, camera: str, seconds: float) -> None:
        log.info("[%s] configuración web recuperada tras %.0f s", camera, seconds)
        self._system_event(camera, "recuperacion",
                           f"configuración web recuperada tras {seconds:.0f} s")

    # --- alarmas ----------------------------------------------------------

    def _log_raw(self, alarm: CameraAlarm) -> None:
        key = (alarm.camera, alarm.topic)
        previous, repeats = self._raw_last.get(key, (None, -1))
        if repeats >= 0 and previous == alarm.active:
            self._raw_last[key] = (previous, repeats + 1)
            return
        if repeats > 0:
            log.debug("[%s]     (%s=%s se repitió %d veces)", alarm.camera,
                      alarm.topic, previous, repeats)
        self._raw_last[key] = (alarm.active, 0)
        log.debug("[%s] evento %s %s=%s %s", alarm.camera, alarm.operation or "-",
                  alarm.topic, alarm.active, alarm.items)

    def _on_alarm(self, alarm: CameraAlarm) -> None:
        if self._closing:
            return
        with self._lock:
            self._log_raw(alarm)
            transition = self._tracker.update(alarm)
        if transition is None:
            if alarm.kind == "other":
                log.info("[%s] evento no clasificado: %s %s",
                         alarm.camera, alarm.topic, alarm.items)
            return
        camera, name = transition.camera, NAMES[transition.kind]
        key = (camera, transition.kind)
        if transition.initial and not transition.active:
            log.info("[%s] estado inicial: sin %s", camera, name)
            return
        if transition.active:
            prefix = "estado inicial: " if transition.initial else ""
            log.info("[%s] >>> %s%s", camera, prefix, START_LABELS[transition.kind])
            alarm_id = None
            if self.store is not None:
                alarm_id = self.store.alarm_started(camera, transition.kind, transition.at)
                with self._lock:
                    self._open_alarms[key] = alarm_id
            if self.verifier is not None:
                self.verifier.alarm_started(camera, alarm_id)
            return
        with self._lock:
            alarm_id = self._open_alarms.pop(key, None)
        if self.verifier is not None:
            self.verifier.alarm_ended(camera)
        if transition.duration is None:
            log.info("[%s] <<< fin de %s (inicio no observado)", camera, name)
        else:
            log.info("[%s] <<< fin de %s (%.1f s)", camera, name, transition.duration)
        if alarm_id is not None and self.store is not None:
            self.store.alarm_ended(alarm_id, transition.at, transition.duration)

    def _close_open_alarms(self, camera: str, note: str) -> None:
        """Cierra las alarmas abiertas cuyo fin ya no se va a poder observar."""
        with self._lock:
            keys = [key for key in self._open_alarms if key[0] == camera]
            alarm_ids = [self._open_alarms.pop(key) for key in keys]
            self._tracker.forget_camera(camera)
        if self.verifier is not None and alarm_ids:
            self.verifier.alarm_ended(camera)
        if self.store is not None:
            now = datetime.now(timezone.utc)
            for alarm_id in alarm_ids:
                self.store.alarm_ended(alarm_id, now, None, note)

    def _status_handler(self, camera: str):
        def report(message: str, connected: bool) -> None:
            if not connected:
                self._close_open_alarms(camera, "conexión perdida")
            log.info("[%s] %s", camera, message)
            self._system_event(camera, "conexion" if connected else "desconexion", message)
        return report

    # --- verificación en reposo ------------------------------------------------

    def _on_idle_person(self, camera: str, result) -> None:
        """YOLO vio a alguien sin que la cámara avisara: se abre una sesión."""
        if self._closing:
            return
        log.info("[%s] >>> PERSONA DETECTADA EN REPOSO (la cámara no avisó) · "
                 "confianza %.2f", camera, max(result.confidences, default=0.0))
        alarm_id = None
        if self.store is not None:
            now = datetime.now(timezone.utc)
            alarm_id = self.store.alarm_started(camera, "reposo", now)
            self.store.alarm_ended(alarm_id, now, None,
                                   "detectada por verificación en reposo; "
                                   "la cámara no avisó")
        if self.verifier is not None:
            # La sesión dura mientras haya personas o falte el veredicto.
            self.verifier.alarm_started(camera, alarm_id)
            self.verifier.alarm_ended(camera)

    def _on_idle_status(self, camera: str, message: str) -> None:
        log.info("[%s] %s", camera, message)
        self._system_event(camera, "reposo", message)

    # --- verificación YOLO -------------------------------------------------

    def _on_verdict(self, verdict: str, summary) -> None:
        if verdict == "confirmed":
            log.info("[%s]     ✔ PERSONA CONFIRMADA · %s", summary.camera,
                     describe_people(summary))
        elif summary.frames == 0:
            log.info("[%s]     ? sin vídeo para verificar", summary.camera)
        else:
            log.info("[%s]     ✘ descartada: YOLO no vio personas (%d fotogramas)",
                     summary.camera, summary.frames)
        if self.store is not None and summary.alarm_id:
            self.store.alarm_verdict(summary.alarm_id, verdict, summary.frames,
                                     summary.max_people, summary.best_confidence,
                                     summary.first_person_after)

    def _relative(self, path: Path | None) -> str | None:
        if path is None:
            return None
        if self.base_dir and path.is_relative_to(self.base_dir):
            return path.relative_to(self.base_dir).as_posix()
        return path.as_posix()

    def _on_verification_finished(self, summary) -> None:
        # Sin registro por alarma: la sesión escribe un único resumen al cerrar.
        photo = self._relative(summary.snapshot)
        if self.store is not None and summary.alarm_id:
            self.store.alarm_verification_finished(
                summary.alarm_id, summary.frames, summary.max_people,
                summary.best_confidence, photo, summary.error)

    # --- seguimiento corporal --------------------------------------------------

    def _on_track(self, event: str, track, note: str | None) -> None:
        if event == "recovered":
            how, unseen = (note or "posicion:0").split(":")
            log.info("[%s]     persona %s reapareció tras %s s sin verse (por %s)",
                     track.camera, track.code, unseen, how.replace("posicion", "posición"))
            if self.store is not None:
                self.store.track_recovered(track)
            return
        if event == "entered":
            log.info("[%s]     persona %s entró (confianza %.2f · YOLO por %s)",
                     track.camera, track.code, track.confidence, track.reason)
            if self.store is not None:
                self.store.track_entered(track)
            return
        description = {
            "borde": "salió por el borde",
            "interior": "DESAPARECIÓ en el centro de la imagen (oclusión, zona "
                        "oscura o falla de cámara)",
        }.get(track.exit_kind, "dejó de seguirse")
        if note:                      # la sesión terminó con la persona presente
            description = f"dejó de seguirse: {note}"
        else:
            note = description
        log.info("[%s]     persona %s %s (%.1f s visible · %d detecciones YOLO)",
                 track.camera, track.code, description, track.visible_seconds,
                 track.yolo_hits)
        if self.store is not None:
            self.store.track_left(track, note)

    def _on_session(self, stats) -> None:
        reasons = ", ".join(f"{name} {count}" for name, count in stats.reasons.most_common())
        if stats.error:
            log.info("[%s]     análisis: %s", stats.camera, stats.error)
        log.info("[%s]     resumen de sesión: %d alarma(s), %d confirmada(s) · máximo "
                 "%d persona(s) · %d seguida(s) · %d reaparición(es) · foto %s",
                 stats.camera, stats.alarms, stats.confirmed, stats.max_people,
                 stats.tracks, sum(stats.recoveries.values()),
                 self._relative(stats.snapshot) or "sin personas")
        log.info("[%s]     eficiencia: %d fotogramas · %d YOLO (%.0f%% sin YOLO) · "
                 "motivos: %s", stats.camera, stats.frames, stats.yolo_runs,
                 stats.savings * 100, reasons or "-")
        if self.store is not None:
            self.store.session_finished(stats)

    def _system_event(self, camera: str | None, kind: str, message: str) -> None:
        if self.store is not None:
            self.store.system_event(camera, kind, message)
