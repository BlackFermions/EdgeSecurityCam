"""Política de activación de YOLO: cuándo despertarlo y por qué.

Regla de seguridad: el ahorro solo aplica a seguir a quien ya fue detectado;
ante cualquier duda (movimiento nuevo, trayectoria perdida, alarma nueva o
demasiado tiempo sin YOLO) se ejecuta YOLO.
"""

from __future__ import annotations

# Motivos por los que se ejecuta YOLO (se registran para auditar el ahorro).
REASON_CONFIRM = "confirmacion"       # alarma sin veredicto todavía
REASON_ALARM = "alarma"               # la cámara inició una alarma nueva
REASON_MOTION = "movimiento_nuevo"    # movimiento fuera de las personas seguidas
REASON_LOST = "trayectoria_perdida"   # el flujo óptico perdió a alguien
REASON_INTERVAL = "intervalo"         # tiempo máximo sin YOLO
REASON_WATCH = "vigilancia"           # alarma activa sin personas seguidas


def yolo_reason(*, now: float, last_yolo: float, verdict_pending: bool,
                alarm_pending: bool, alarm_active: bool, has_tracks: bool,
                searching: bool = False,
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
    if searching and elapsed >= still_interval:
        return REASON_WATCH       # personas en gracia u ocultas: buscarlas
    return None
