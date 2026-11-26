"""Un resumen de SQLite por día, con fechas en America/Lima.

Uso: python tools/resumen_diario.py --fecha hoy
     python tools/resumen_diario.py --fecha 2026-11-26
     python tools/resumen_diario.py --recientes  # ayer y hoy, para un timer

Solo lee la base. Cada ejecución reemplaza el resumen del día sin duplicar
registros. Las apariciones se agrupan por su fecha de entrada; los intentos
faciales y alarmas, por su propia fecha. No requiere paquetes adicionales.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


ROOT = Path(__file__).resolve().parents[1]


def day_bounds(day: date, zone) -> tuple[str, str]:
    start = datetime.combine(day, time.min, tzinfo=zone)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=zone)
    return tuple(moment.astimezone(timezone.utc).isoformat(timespec="milliseconds")
                 for moment in (start, end))


def cell(value) -> str:
    if value is None:
        return "-"
    return " ".join(str(value).split())


def local_time(value: str | None, zone) -> str:
    if not value:
        return "sin cierre registrado"
    return datetime.fromisoformat(value).astimezone(zone).strftime("%Y-%m-%d %H:%M:%S")


def build_report(db_path: Path, day: date, zone, now: datetime | None = None) -> str:
    """Lee una instantánea consistente, sin modificar el esquema ni los datos."""
    bounds = day_bounds(day, zone)
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        alarms = db.execute(
            "SELECT veredicto, COUNT(*) AS n FROM alarma "
            "WHERE inicio >= ? AND inicio < ? GROUP BY veredicto", bounds).fetchall()
        tracks = db.execute(
            "SELECT codigo, camara, entrada, salida, etiqueta, identidad, estado_rostro, "
            "similitud_rostro, intentos_rostro FROM persona_track "
            "WHERE entrada >= ? AND entrada < ? ORDER BY entrada, codigo", bounds).fetchall()
        attempts = db.execute(
            "SELECT resultado, COUNT(*) AS n FROM rostro_intento "
            "WHERE momento >= ? AND momento < ? GROUP BY resultado", bounds).fetchall()
        events = db.execute(
            "SELECT momento, camara, tipo, mensaje FROM evento_sistema "
            "WHERE momento >= ? AND momento < ? ORDER BY momento, id", bounds).fetchall()

    generated = (now or datetime.now(timezone.utc)).astimezone(zone)
    verdicts = {row["veredicto"]: row["n"] for row in alarms}
    identities = Counter(row["identidad"] for row in tracks if row["identidad"])
    unknown = sum(row["estado_rostro"] == "desconocida" for row in tracks)
    unidentified = sum(not row["identidad"] and row["estado_rostro"] != "desconocida"
                       for row in tracks)
    lines = [
        f"RESUMEN DIARIO - {day.isoformat()}",
        f"Zona horaria: {zone}",
        f"Actualizado: {generated:%Y-%m-%d %H:%M:%S %z}",
        "Fuente: SQLite. Instantánea de los registros guardados hasta esta actualización.",
        "Cada aparición cuenta una entrada; no equivale a una persona distinta.",
        "Las apariciones se asignan al día de entrada; su salida puede ser posterior.",
        "Un registro sin cierre no demuestra que la persona siga presente.",
        "",
        "TOTALES",
        f"Alarmas iniciadas: {sum(verdicts.values())}",
        f"  Confirmadas: {verdicts.get('confirmada', 0)}",
        f"  Descartadas: {verdicts.get('descartada', 0)}",
        f"  Sin vídeo: {verdicts.get('sin_video', 0)}",
        f"  Sin veredicto: {verdicts.get(None, 0)}",
        f"Apariciones iniciadas: {len(tracks)}",
        f"  Con identidad reconocida: {sum(identities.values())}",
        f"  Clasificadas desconocidas: {unknown}",
        f"  Sin identificación concluida: {unidentified}",
        f"Nombres reconocidos distintos: {len(identities)}",
        f"Intentos faciales del día: {sum(row['n'] for row in attempts)}",
        "",
        "APARICIONES POR NOMBRE RECONOCIDO",
    ]
    lines.extend(f"{cell(name)}: {count}" for name, count in sorted(identities.items()))
    if not identities:
        lines.append("Sin identidades reconocidas registradas.")
    lines.extend([
        "", "DETALLE DE APARICIONES",
        "Código | Cámara | Entrada local | Salida local | Etiqueta | Identidad | "
        "Estado facial | Similitud facial | Intentos faciales",
    ])
    for row in tracks:
        score = f"{row['similitud_rostro']:.3f}" if row["similitud_rostro"] is not None else "-"
        lines.append(" | ".join(cell(value) for value in (
            row["codigo"], row["camara"], local_time(row["entrada"], zone),
            local_time(row["salida"], zone), row["etiqueta"], row["identidad"],
            row["estado_rostro"], score, row["intentos_rostro"])))
    if not tracks:
        lines.append("Sin apariciones iniciadas durante este día.")
    lines.extend(["", "RESULTADOS DE INTENTOS FACIALES DEL DÍA"])
    lines.extend(f"{cell(row['resultado'])}: {row['n']}" for row in attempts)
    if not attempts:
        lines.append("Sin intentos faciales registrados.")
    lines.extend(["", "EVENTOS DEL SISTEMA DEL DÍA"])
    lines.extend(" | ".join(cell(value) for value in (
        local_time(row["momento"], zone), row["camara"], row["tipo"], row["mensaje"]))
        for row in events)
    if not events:
        lines.append("Sin eventos del sistema registrados.")
    return "\n".join(lines) + "\n"


def write_report(db_path: Path, folder: Path, day: date, zone) -> Path:
    content = build_report(db_path, day, zone)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = folder / f"{day.isoformat()}.log"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder,
                                         prefix=f".{day}-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "camdetector.db")
    parser.add_argument("--carpeta", type=Path, default=ROOT / "logs" / "resumen_diario")
    parser.add_argument("--zona", default="America/Lima")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fecha", default=None, help="hoy, ayer o AAAA-MM-DD; por defecto hoy")
    group.add_argument("--recientes", action="store_true", help="Actualiza ayer y hoy")
    args = parser.parse_args()
    try:
        zone = ZoneInfo(args.zona)
        today = datetime.now(zone).date()
        if args.recientes:
            days = [today - timedelta(days=1), today]
        else:
            selected = args.fecha or "hoy"
            day = (today if selected == "hoy" else today - timedelta(days=1)
                   if selected == "ayer" else date.fromisoformat(selected))
            days = [day]
        for day in days:
            print(f"Resumen guardado: {write_report(args.db, args.carpeta, day, zone)}")
    except (OSError, ValueError, sqlite3.Error, ZoneInfoNotFoundError) as error:
        print(f"No se pudo generar el resumen: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
