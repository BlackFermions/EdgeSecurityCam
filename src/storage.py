"""Base de datos local del nodo (SQLite).

Guarda cada alarma con su veredicto YOLO y la versión de configuración de la
cámara vigente en ese momento. Es la fuente para la calibración y, más
adelante, la cola de envío al NOC: cada fila nace con un UUID y ``enviado``
queda en NULL hasta que el NOC la confirme; cualquier actualización la vuelve
a marcar como pendiente.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS config_camara (
    id                  INTEGER PRIMARY KEY,
    camara              TEXT NOT NULL,
    hash                TEXT NOT NULL,
    vigente_desde       TEXT NOT NULL,
    vigente_hasta       TEXT,
    modelo              TEXT,
    firmware            TEXT,
    human_activo        INTEGER,
    human_sensibilidad  INTEGER,
    human_duracion      INTEGER,
    motion_activo       INTEGER,
    motion_sensibilidad INTEGER,
    motion_duracion     INTEGER,
    ia_modelo           TEXT,
    ia_confianza        REAL,
    datos               TEXT NOT NULL,          -- configuración completa (JSON)
    enviado             TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_config_vigente
    ON config_camara(camara) WHERE vigente_hasta IS NULL;

CREATE TABLE IF NOT EXISTS alarma (
    id                  TEXT PRIMARY KEY,       -- UUID generado en el nodo
    camara              TEXT NOT NULL,
    config_id           INTEGER REFERENCES config_camara(id),
    tipo                TEXT NOT NULL,          -- human | motion
    inicio              TEXT NOT NULL,          -- UTC ISO 8601
    fin                 TEXT,
    duracion_s          REAL,
    veredicto           TEXT,                   -- confirmada | descartada | sin_video
    max_personas        INTEGER,
    confianza           REAL,
    fotogramas          INTEGER,
    primera_persona_s   REAL,
    foto                TEXT,
    nota                TEXT,
    actualizado         TEXT NOT NULL,
    enviado             TEXT
);
CREATE INDEX IF NOT EXISTS ix_alarma_camara_inicio ON alarma(camara, inicio);
CREATE INDEX IF NOT EXISTS ix_alarma_pendiente ON alarma(enviado) WHERE enviado IS NULL;

-- Versión 2: seguimiento corporal.
CREATE TABLE IF NOT EXISTS persona_track (
    id                  TEXT PRIMARY KEY,       -- UUID generado en el nodo
    codigo              TEXT NOT NULL UNIQUE,   -- legible: cam109-20261006-145731-1
    camara              TEXT NOT NULL,
    alarma_id           TEXT,                   -- alarma vigente al aparecer
    sesion_id           TEXT,
    entrada             TEXT NOT NULL,
    ultima_vista        TEXT,                   -- última detección de YOLO
    salida              TEXT,                   -- cuando se dio por terminada
    segundos_visible    REAL,
    max_confianza       REAL,
    detecciones_yolo    INTEGER,
    motivo_deteccion    TEXT,                   -- qué despertó a YOLO al aparecer
    nota                TEXT,
    actualizado         TEXT NOT NULL,
    enviado             TEXT
);
CREATE INDEX IF NOT EXISTS ix_track_camara_entrada ON persona_track(camara, entrada);

CREATE TABLE IF NOT EXISTS sesion_analisis (
    id                  TEXT PRIMARY KEY,
    camara              TEXT NOT NULL,
    inicio              TEXT NOT NULL,
    fin                 TEXT,
    fotogramas          INTEGER,                -- procesados con flujo/puerta
    inferencias_yolo    INTEGER,
    ahorro              REAL,                   -- fracción sin YOLO
    motivos             TEXT,                   -- JSON: inferencias por motivo
    nuevas_por_motivo   TEXT,                   -- JSON: personas nuevas por motivo
    trayectorias        INTEGER,
    max_personas        INTEGER,
    error               TEXT,
    enviado             TEXT
);

CREATE TABLE IF NOT EXISTS evento_sistema (
    id       INTEGER PRIMARY KEY,
    camara   TEXT,
    momento  TEXT NOT NULL,
    tipo     TEXT NOT NULL,      -- conexion | desconexion | config | inicio | error
    mensaje  TEXT,
    enviado  TEXT
);
"""

VERDICTS = {"confirmed": "confirmada", "discarded": "descartada"}


def utc_iso(moment: datetime | None = None) -> str:
    moment = moment or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds")


class NodeStore:
    """Acceso seguro entre hilos a la base del nodo (una conexión con candado)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False,
                                   isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"la base {path} es de una versión más nueva ({version})")
        self._db.executescript(SCHEMA)
        self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- configuración ---------------------------------------------------

    def current_config(self, camera: str) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute(
                "SELECT * FROM config_camara WHERE camara=? AND vigente_hasta IS NULL",
                (camera,)).fetchone()

    def record_config(self, camera: str, config: dict[str, Any], config_hash: str,
                      at: datetime | None = None) -> tuple[int, dict[str, Any] | None]:
        """Guarda la configuración si cambió.

        Devuelve el id vigente y la configuración anterior si hubo cambio
        (``None`` si es la misma o es la primera).
        """
        moment = utc_iso(at)
        human, motion = config.get("human", {}), config.get("motion", {})
        camera_info, ia = config.get("camara", {}), config.get("ia", {})
        with self._lock:
            current = self._db.execute(
                "SELECT id, hash, datos FROM config_camara "
                "WHERE camara=? AND vigente_hasta IS NULL", (camera,)).fetchone()
            if current is not None and current["hash"] == config_hash:
                return current["id"], None
            self._db.execute("BEGIN")
            try:
                if current is not None:
                    self._db.execute(
                        "UPDATE config_camara SET vigente_hasta=?, enviado=NULL "
                        "WHERE id=?", (moment, current["id"]))
                cursor = self._db.execute(
                    """INSERT INTO config_camara (
                        camara, hash, vigente_desde, modelo, firmware,
                        human_activo, human_sensibilidad, human_duracion,
                        motion_activo, motion_sensibilidad, motion_duracion,
                        ia_modelo, ia_confianza, datos)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (camera, config_hash, moment, camera_info.get("modelo"),
                     camera_info.get("firmware"), human.get("enable"),
                     human.get("sensitivity"), human.get("duration"),
                     motion.get("enable"), motion.get("sensitivity"),
                     motion.get("duration"), ia.get("modelo"), ia.get("confianza"),
                     json.dumps(config, ensure_ascii=False, sort_keys=True)))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        previous = json.loads(current["datos"]) if current is not None else None
        return cursor.lastrowid, previous

    # --- alarmas -----------------------------------------------------------

    def alarm_started(self, camera: str, kind: str, at: datetime) -> str:
        alarm_id = str(uuid.uuid4())
        with self._lock:
            config = self._db.execute(
                "SELECT id FROM config_camara WHERE camara=? AND vigente_hasta IS NULL",
                (camera,)).fetchone()
            self._db.execute(
                "INSERT INTO alarma (id, camara, config_id, tipo, inicio, actualizado) "
                "VALUES (?,?,?,?,?,?)",
                (alarm_id, camera, config["id"] if config else None, kind,
                 utc_iso(at), utc_iso()))
        return alarm_id

    def alarm_ended(self, alarm_id: str, at: datetime, duration: float | None,
                    note: str | None = None) -> None:
        self._update(alarm_id, fin=utc_iso(at), duracion_s=duration, nota=note)

    def alarm_verdict(self, alarm_id: str, verdict: str, frames: int,
                      max_people: int, confidence: float,
                      first_person_after: float | None) -> None:
        label = VERDICTS.get(verdict, verdict)
        if verdict == "discarded" and frames == 0:
            label = "sin_video"
        self._update(alarm_id, veredicto=label, fotogramas=frames,
                     max_personas=max_people, confianza=round(confidence, 4),
                     primera_persona_s=first_person_after)

    def alarm_verification_finished(self, alarm_id: str, frames: int, max_people: int,
                                    confidence: float, photo: str | None,
                                    error: str | None) -> None:
        values: dict[str, Any] = {"fotogramas": frames, "max_personas": max_people,
                                  "confianza": round(confidence, 4), "foto": photo}
        if error:
            values["nota"] = error
        self._update(alarm_id, **values)

    def _update(self, alarm_id: str, **values: Any) -> None:
        values = {key: value for key, value in values.items() if value is not None}
        if not values:
            return
        assignments = ", ".join(f"{key}=?" for key in values)
        with self._lock:
            self._db.execute(
                f"UPDATE alarma SET {assignments}, actualizado=?, enviado=NULL "
                "WHERE id=?", (*values.values(), utc_iso(), alarm_id))

    # --- seguimiento corporal ---------------------------------------------

    def track_entered(self, track) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO persona_track (id, codigo, camara, alarma_id, sesion_id,
                       entrada, ultima_vista, max_confianza, detecciones_yolo,
                       motivo_deteccion, actualizado)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (track.id, track.code, track.camera, track.alarm_id, track.session_id,
                 utc_iso(track.started_wall), utc_iso(track.last_seen_wall),
                 round(track.max_confidence, 4), track.yolo_hits, track.reason,
                 utc_iso()))

    def track_left(self, track, note: str | None = None) -> None:
        with self._lock:
            self._db.execute(
                """UPDATE persona_track SET ultima_vista=?, salida=?,
                       segundos_visible=?, max_confianza=?, detecciones_yolo=?,
                       nota=?, actualizado=?, enviado=NULL WHERE id=?""",
                (utc_iso(track.last_seen_wall), utc_iso(),
                 round(track.visible_seconds, 2), round(track.max_confidence, 4),
                 track.yolo_hits, note, utc_iso(), track.id))

    def session_finished(self, stats) -> None:
        with self._lock:
            self._db.execute(
                """INSERT OR REPLACE INTO sesion_analisis (id, camara, inicio, fin,
                       fotogramas, inferencias_yolo, ahorro, motivos,
                       nuevas_por_motivo, trayectorias, max_personas, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (stats.id, stats.camera, utc_iso(stats.started_wall),
                 utc_iso(stats.ended_wall), stats.frames, stats.yolo_runs,
                 round(stats.savings, 4),
                 json.dumps(dict(stats.reasons), ensure_ascii=False),
                 json.dumps(dict(stats.new_by_reason), ensure_ascii=False),
                 stats.tracks, stats.max_people, stats.error))

    # --- eventos del sistema ----------------------------------------------

    def system_event(self, camera: str | None, kind: str, message: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO evento_sistema (camara, momento, tipo, mensaje) "
                "VALUES (?,?,?,?)", (camera, utc_iso(), kind, message))

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()
