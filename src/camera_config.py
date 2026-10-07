"""Configuración de detección de la cámara: lectura, huella y cambios.

La configuración vigente en el momento de una alarma es necesaria para
interpretarla (una alarma de 2 s significa algo distinto con Motion Detect
activo o apagado). Se lee periódicamente y cada cambio crea una versión.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Callable

from src.camera_web import CameraWeb, CameraWebError


DETECTION_KEYS = ("enable", "sensitivity", "threshold", "duration", "show_human",
                  "rect_num", "rect", "schedule")
SECTIONS = {"human": "pd", "motion": "md"}
SECTION_LABELS = {"camara": "Cámara", "human": "Human Detect",
                  "motion": "Motion Detect", "ia": "IA"}
STRUCTURED_KEYS = {"rect": "región", "schedule": "horario"}


def read_camera_config(web: CameraWeb) -> dict[str, Any]:
    """Lee y normaliza modelo, firmware, Human Detect y Motion Detect."""
    device = web.get("device")
    config: dict[str, Any] = {
        "camara": {"modelo": device.get("devtype"), "firmware": device.get("version")},
    }
    for section, module in SECTIONS.items():
        raw = web.get(module)
        config[section] = {key: raw[key] for key in DETECTION_KEYS if key in raw}
    return config


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _format(key: str, value: Any) -> str:
    if key in ("enable", "show_human"):
        return "ACTIVADO" if value else "desactivado"
    return str(value)


def diff_config(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Cambios legibles entre dos configuraciones."""
    changes: list[str] = []
    present = set(old) | set(new)
    ordered = [name for name in SECTION_LABELS if name in present]
    for section in ordered + sorted(present - set(SECTION_LABELS)):
        before, after = old.get(section, {}), new.get(section, {})
        label = SECTION_LABELS.get(section, section)
        for key in sorted(set(before) | set(after)):
            if before.get(key) == after.get(key):
                continue
            if key in STRUCTURED_KEYS:
                changes.append(f"{label}: {STRUCTURED_KEYS[key]} modificada")
            else:
                changes.append(f"{label}: {key} {_format(key, before.get(key))} → "
                               f"{_format(key, after.get(key))}")
    return changes


def summarize(config: dict[str, Any]) -> list[str]:
    lines = []
    for section in ("human", "motion"):
        data = config.get(section, {})
        lines.append(f"{SECTION_LABELS[section]:<13} {_format('enable', data.get('enable'))}"
                     f" · sensibilidad {data.get('sensitivity')}"
                     f" · duración {data.get('duration')} s"
                     f" · región {'toda la imagen' if not data.get('rect_num') else str(data.get('rect_num')) + ' zona(s)'}")
    return lines


class ConfigWatcher(threading.Thread):
    """Relee la configuración cada ``interval`` segundos y avisa de cada lectura."""

    def __init__(self, camera: str, web: CameraWeb, extra: dict[str, Any],
                 on_config: Callable[[str, dict[str, Any]], None],
                 on_error: Callable[[str, str], None], interval: float = 60.0,
                 on_recovered: Callable[[str, float], None] | None = None) -> None:
        super().__init__(name=f"config-{camera}", daemon=True)
        self.camera = camera
        self._web = web
        self._extra = extra
        self._on_config = on_config
        self._on_error = on_error
        self._on_recovered = on_recovered or (lambda *_: None)
        self._interval = interval
        self._halt = threading.Event()
        self._failing_since: float | None = None   # inicio de la caída actual

    def read_once(self) -> dict[str, Any] | None:
        try:
            config = read_camera_config(self._web)
        except CameraWebError as error:
            if self._failing_since is None:          # se avisa una vez por caída
                self._failing_since = time.monotonic()
                self._on_error(self.camera, str(error))
            return None
        if self._failing_since is not None:
            self._on_recovered(self.camera, time.monotonic() - self._failing_since)
            self._failing_since = None
        config.update(self._extra)
        self._on_config(self.camera, config)
        return config

    def run(self) -> None:
        while not self._halt.wait(self._interval):
            self.read_once()

    def request_stop(self) -> None:
        self._halt.set()
