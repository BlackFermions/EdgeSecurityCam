"""Resumen de la base del nodo para calibrar la detección.

Uso:
    python tools\\resumen.py
    python tools\\resumen.py --desde 2026-10-06T20:00 --ultimas 30

Muestra, por cada versión de configuración de la cámara, cuántas alarmas hubo
y cuántas confirmó o descartó YOLO, y las últimas alarmas registradas.
Las horas se muestran en la zona horaria de este equipo.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def local(value: str | None) -> str:
    if not value:
        return "-"
    return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def to_utc(value: str) -> str:
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Resumen de alarmas del nodo")
    parser.add_argument("--db", type=Path, default=BASE_DIR / "data" / "camdetector.db")
    parser.add_argument("--desde", help="Fecha/hora local inicial, p. ej. 2026-10-06T20:00")
    parser.add_argument("--ultimas", type=int, default=15)
    args = parser.parse_args()
    if not args.db.exists():
        print(f"No existe la base {args.db}")
        return 2
    db = sqlite3.connect(f"file:{args.db.as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    since = to_utc(args.desde) if args.desde else "0000"

    print("\nAlarmas por versión de configuración")
    print(f"{'ver':>3}  {'vigente desde':<19}  {'human s/d':<9}  {'motion':<11}  "
          f"{'total':>5}  {'confirm':>7}  {'descart':>7}  {'sin vid':>7}  {'pend':>4}  "
          f"{'dur. media':>10}")
    for row in db.execute(
            """SELECT c.id, c.vigente_desde, c.human_activo, c.human_sensibilidad,
                      c.human_duracion, c.motion_activo, c.motion_sensibilidad,
                      c.motion_duracion,
                      COUNT(a.id) AS total,
                      SUM(a.veredicto = 'confirmada') AS confirmadas,
                      SUM(a.veredicto = 'descartada') AS descartadas,
                      SUM(a.veredicto = 'sin_video') AS sin_video,
                      SUM(a.id IS NOT NULL AND a.veredicto IS NULL) AS pendientes,
                      AVG(a.duracion_s) AS duracion
               FROM config_camara c
               LEFT JOIN alarma a ON a.config_id = c.id AND a.inicio >= ?
               GROUP BY c.id ORDER BY c.id""", (since,)):
        human = (f"{row['human_sensibilidad']}/{row['human_duracion']}s"
                 if row["human_activo"] else "apagado")
        motion = (f"{row['motion_sensibilidad']}/{row['motion_duracion']}s"
                  if row["motion_activo"] else "apagado")
        duration = f"{row['duracion']:.1f} s" if row["duracion"] is not None else "-"
        print(f"{row['id']:>3}  {local(row['vigente_desde']):<19}  {human:<9}  "
              f"{motion:<11}  {row['total']:>5}  {row['confirmadas'] or 0:>7}  "
              f"{row['descartadas'] or 0:>7}  {row['sin_video'] or 0:>7}  "
              f"{row['pendientes'] or 0:>4}  {duration:>10}")

    print(f"\nÚltimas {args.ultimas} alarmas")
    print(f"{'inicio':<19}  {'tipo':<6}  {'dur.':>6}  {'veredicto':<11}  "
          f"{'pers.':>5}  {'conf.':>5}  {'ver':>3}  nota / foto")
    rows = db.execute(
        """SELECT * FROM alarma WHERE inicio >= ? ORDER BY inicio DESC LIMIT ?""",
        (since, args.ultimas)).fetchall()
    for row in reversed(rows):
        duration = f"{row['duracion_s']:.1f}" if row["duracion_s"] is not None else "-"
        confidence = f"{row['confianza']:.2f}" if row["confianza"] else "-"
        print(f"{local(row['inicio']):<19}  {row['tipo']:<6}  {duration:>6}  "
              f"{row['veredicto'] or 'pendiente':<11}  {row['max_personas'] or 0:>5}  "
              f"{confidence:>5}  {row['config_id'] or '-':>3}  "
              f"{row['nota'] or row['foto'] or ''}")
    if not rows:
        print("  (sin alarmas)")

    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
    if "persona_track" in tables:
        print(f"\nÚltimas {args.ultimas} personas seguidas")
        print(f"{'código':<28}  {'entrada':<19}  {'visible':>8}  {'conf.':>5}  "
              f"{'YOLO':>4}  {'detectada por':<20}  nota")
        tracks = db.execute(
            "SELECT * FROM persona_track WHERE entrada >= ? ORDER BY entrada DESC LIMIT ?",
            (since, args.ultimas)).fetchall()
        for row in reversed(tracks):
            visible = (f"{row['segundos_visible']:.1f} s"
                       if row["segundos_visible"] is not None else "en curso")
            print(f"{row['codigo']:<28}  {local(row['entrada']):<19}  {visible:>8}  "
                  f"{row['max_confianza'] or 0:>5.2f}  {row['detecciones_yolo'] or 0:>4}  "
                  f"{row['motivo_deteccion'] or '-':<20}  {row['nota'] or ''}")
        if not tracks:
            print("  (sin personas)")

    if "sesion_analisis" in tables:
        row = db.execute(
            """SELECT COUNT(*) AS sesiones, SUM(fotogramas) AS fotogramas,
                      SUM(inferencias_yolo) AS yolo, SUM(trayectorias) AS personas
               FROM sesion_analisis WHERE inicio >= ?""", (since,)).fetchone()
        if row["sesiones"]:
            saved = 1 - (row["yolo"] or 0) / row["fotogramas"] if row["fotogramas"] else 0
            print(f"\nSesiones de análisis: {row['sesiones']} · fotogramas "
                  f"{row['fotogramas'] or 0} · inferencias YOLO {row['yolo'] or 0} "
                  f"({saved:.0%} de fotogramas sin YOLO) · personas {row['personas'] or 0}")
            counts: dict[str, list[int]] = {}
            for session in db.execute(
                    "SELECT motivos, nuevas_por_motivo FROM sesion_analisis WHERE inicio >= ?",
                    (since,)):
                for column, slot in (("motivos", 0), ("nuevas_por_motivo", 1)):
                    for reason, count in json.loads(session[column] or "{}").items():
                        counts.setdefault(reason, [0, 0])[slot] += count
            print(f"  {'motivo de YOLO':<22} {'inferencias':>11}  {'personas nuevas':>15}")
            for reason, (runs, new) in sorted(counts.items(), key=lambda kv: -kv[1][0]):
                print(f"  {reason:<22} {runs:>11}  {new:>15}")
            print("  (personas nuevas por 'intervalo' = entradas que la puerta de "
                  "movimiento no vio a tiempo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
