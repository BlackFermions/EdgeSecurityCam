"""Muestra los parámetros efectivos de cada cámara y de dónde sale cada valor.

Uso:
    python tools\\parametros.py --camera 192.168.100.109
    python tools\\parametros.py --camera 192.168.100.109 --config config\\otra.toml
    python tools\\parametros.py --markdown > docs\\parametros_tabla.md

Valida el archivo igual que al arrancar: sirve para comprobar un cambio de
calibración antes de reiniciar CamDetector.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.settings import SCHEMA, SettingsError, load_settings, markdown_table  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Parámetros efectivos por cámara")
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "camdetector.toml")
    parser.add_argument("--markdown", action="store_true",
                        help="Imprime la tabla de todos los parámetros (documentación)")
    args = parser.parse_args()
    if args.markdown:
        print(markdown_table())
        return 0
    if not args.camera:
        parser.error("indica al menos una --camera")
    try:
        settings = load_settings(args.config, args.camera)
    except SettingsError as error:
        print(f"ERROR en {args.config}: {error}")
        return 2
    print(f"Archivo: {args.config}{'' if args.config.exists() else ' (no existe: valores por defecto)'}")
    for camera, values in settings.items():
        print(f"\n[{camera}] {values.name}")
        for group, params in SCHEMA.items():
            for name, spec in params.items():
                value = values.get(group, name)
                source = values.sources.get(f"{group}.{name}")
                mark = f"  ← {source}" if source else ""
                unit = f" {spec.unit}" if spec.unit and spec.unit != "fracción" else ""
                print(f"  {group + '.' + name:<36} {str(value) + unit:<20}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
