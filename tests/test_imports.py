"""Toda importación ``from src... import ...`` del proyecto debe resolverse.

Incluye las que están dentro de funciones, que solo se ejecutan en ciertas
condiciones (p. ej. con YOLO activado) y que las demás pruebas no recorren.
Evita repetir el fallo del 2026-10-13: tras la reorganización modular quedó
``from src.person_verifier import rtsp_url`` en escuchar_alarmas.py.
"""

import ast
import importlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [path for folder in ("src", "tools", "tests") for path in (ROOT / folder).rglob("*.py")]
FILES.append(ROOT / "escuchar_alarmas.py")


class ImportTests(unittest.TestCase):
    def test_every_src_import_resolves(self):
        problems = []
        for path in FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("src"):
                    continue
                where = f"{path.relative_to(ROOT)}:{node.lineno}"
                try:
                    module = importlib.import_module(node.module)
                except ImportError as error:
                    problems.append(f"{where} → {node.module}: {error}")
                    continue
                for alias in node.names:
                    if not hasattr(module, alias.name):
                        try:
                            importlib.import_module(f"{node.module}.{alias.name}")
                        except ImportError:
                            problems.append(f"{where} → {node.module}.{alias.name} no existe")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
