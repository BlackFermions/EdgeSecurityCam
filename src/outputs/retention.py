"""Retención de los recortes de rostro (datos biométricos).

Los recortes de ``capturas/rostros/AAAA-MM-DD/`` sirven para calibrar, pero
son datos personales: se borran las carpetas de días más antiguos que
``rostro.recortes_dias`` (0 = no borrar). Se ejecuta al arrancar y cada hora.
"""

from __future__ import annotations

import shutil
from datetime import date, datetime, timedelta
from pathlib import Path


def purge_old_days(folder: Path, days: int, today: date | None = None) -> list[str]:
    """Borra las subcarpetas AAAA-MM-DD con más de ``days`` días. Devuelve las borradas."""
    if days <= 0 or not folder.is_dir():
        return []
    limit = (today or date.today()) - timedelta(days=days)
    removed = []
    for child in sorted(folder.iterdir()):
        try:
            day = datetime.strptime(child.name, "%Y-%m-%d").date()
        except ValueError:
            continue                    # solo carpetas con fecha: nada más se toca
        if child.is_dir() and day < limit:
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child.name)
    return removed
