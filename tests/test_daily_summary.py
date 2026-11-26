import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.storage import NodeStore
from tools.resumen_diario import build_report, day_bounds, write_report


LIMA = timezone(timedelta(hours=-5), "America/Lima")
DAY = date(2026, 11, 26)


class DailySummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "node.db"
        # Esquema real, abierto como WAL mientras se generan los informes.
        self.store = NodeStore(self.path)
        self.addCleanup(self.store.close)

    def insert_track(self, key, entered, identity=None, status=None, left=None):
        with closing(sqlite3.connect(self.path, isolation_level=None)) as db:
            db.execute("""INSERT INTO persona_track
                (id,codigo,camara,entrada,ultima_vista,salida,etiqueta,identidad,
                 estado_rostro,similitud_rostro,intentos_rostro,actualizado)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (key, key, "192.168.100.109", entered, entered, left, "Persona 1",
                 identity, status, 0.72 if identity else None, 2, entered))

    def test_local_day_includes_start_and_excludes_next_midnight(self):
        self.assertEqual(day_bounds(DAY, LIMA),
                         ("2026-11-26T05:00:00.000+00:00", "2026-11-27T05:00:00.000+00:00"))
        self.insert_track("ANTES", "2026-11-26T04:59:59.999+00:00")
        self.insert_track("INICIO", "2026-11-26T05:00:00.000+00:00", "Brayan", "identificada")
        self.insert_track("ULTIMO", "2026-11-27T04:59:59.999+00:00", status="desconocida")
        self.insert_track("DESPUES", "2026-11-27T05:00:00.000+00:00")
        report = build_report(self.path, DAY, LIMA)
        self.assertIn("Apariciones iniciadas: 2", report)
        self.assertIn("INICIO |", report)
        self.assertIn("ULTIMO |", report)
        self.assertNotIn("ANTES |", report)
        self.assertNotIn("DESPUES |", report)
        self.assertIn("2026-11-26 00:00:00", report)

    def test_reentries_are_not_counted_as_different_names(self):
        for key in ("UNO", "DOS"):
            self.insert_track(key, "2026-11-26T18:00:00.000+00:00", "Brayan", "identificada")
        self.insert_track("TRES", "2026-11-26T18:00:00.000+00:00", status="desconocida")
        self.insert_track("CUATRO", "2026-11-26T18:00:00.000+00:00")
        report = build_report(self.path, DAY, LIMA)
        self.assertIn("Apariciones iniciadas: 4", report)
        self.assertIn("Con identidad reconocida: 2", report)
        self.assertIn("Clasificadas desconocidas: 1", report)
        self.assertIn("Sin identificación concluida: 1", report)
        self.assertIn("Nombres reconocidos distintos: 1", report)
        self.assertIn("Brayan: 2", report)

    def test_rewrite_updates_late_identity_and_exit_without_duplicating(self):
        self.insert_track("REPETIDA", "2026-11-27T04:59:00.000+00:00")
        folder = Path(self.tmp.name) / "reports"
        path = write_report(self.path, folder, DAY, LIMA)
        self.assertIn("Sin identidades reconocidas", path.read_text(encoding="utf-8"))
        with closing(sqlite3.connect(self.path, isolation_level=None)) as db:
            db.execute("UPDATE persona_track SET identidad='Brayan', estado_rostro='identificada', "
                       "salida='2026-11-27T05:10:00.000+00:00' WHERE id='REPETIDA'")
        write_report(self.path, folder, DAY, LIMA)
        content = path.read_text(encoding="utf-8")
        self.assertIn("Brayan: 1", content)
        self.assertIn("2026-11-27 00:10:00", content)
        self.assertEqual(content.count("REPETIDA |"), 1)
        self.assertEqual([file.name for file in folder.iterdir()], ["2026-11-26.log"])

    def test_alarm_and_face_counts_include_only_the_requested_day(self):
        self.store.alarm_started("cam", "motion", datetime(2026, 11, 26, 18, tzinfo=timezone.utc))
        self.store.alarm_started("cam", "motion", datetime(2026, 11, 26, 4, tzinfo=timezone.utc))
        with closing(sqlite3.connect(self.path, isolation_level=None)) as db:
            db.execute("INSERT INTO rostro_intento (persona_id,camara,momento,resultado) "
                       "VALUES ('old-track','cam','2026-11-26T18:00:00.000+00:00','coincide')")
        report = build_report(self.path, DAY, LIMA)
        self.assertIn("Alarmas iniciadas: 1", report)
        self.assertIn("Sin veredicto: 1", report)
        self.assertIn("Intentos faciales del día: 1", report)
        self.assertIn("coincide: 1", report)

    def test_missing_database_is_not_created_and_existing_report_is_preserved(self):
        folder = Path(self.tmp.name) / "reports"
        folder.mkdir()
        target = folder / "2026-11-26.log"
        target.write_text("resumen anterior", encoding="utf-8")
        missing = Path(self.tmp.name) / "missing.db"
        with self.assertRaises(sqlite3.OperationalError):
            write_report(missing, folder, DAY, LIMA)
        self.assertFalse(missing.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "resumen anterior")

    def test_empty_day_and_control_characters_are_readable(self):
        self.assertIn("Apariciones iniciadas: 0", build_report(self.path, DAY, LIMA))
        self.insert_track("UNO", "2026-11-26T18:00:00.000+00:00", "Ana\nNombre", "identificada")
        report = build_report(self.path, DAY, LIMA)
        self.assertIn("Ana Nombre: 1", report)
        self.assertNotIn("Ana\nNombre", report)


if __name__ == "__main__":
    unittest.main()
