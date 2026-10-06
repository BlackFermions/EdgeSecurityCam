"""Lectura local de credenciales de la cámara."""

from __future__ import annotations

import getpass
from pathlib import Path


def read_credentials(path: Path | None = None) -> tuple[str, str]:
    """Obtiene credenciales sin registrarlas ni mostrarlas."""
    if path is None:
        username = input("Usuario de la cámara: ").strip()
        password = getpass.getpass("Contraseña (oculta): ")
    else:
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError) as error:
            raise ValueError("no se pudo leer el archivo de credenciales") from error
        values = [line.split(":", 1)[1].strip() for line in lines if line and ":" in line]
        if len(values) != 2:
            raise ValueError("el archivo debe tener dos líneas con 'etiqueta: valor'")
        username, password = values
    if not username or not password:
        raise ValueError("usuario o contraseña vacíos")
    return username, password
