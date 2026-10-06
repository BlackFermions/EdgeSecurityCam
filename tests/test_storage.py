import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.camera_config import config_hash, diff_config
from src.storage import NodeStore


T0 = datetime(2026, 10, 6, 19, 0, 0, tzinfo=timezone.utc)


def config(motion_enable=0, sensitivity=90, rect=None):
    return {
        "camara": {"modelo": "W51-TY", "firmware": "V1"},
        "human": {"enable": 1, "sensitivity": sensitivity, "duration": 20,
                  "rect": rect or []},
        "motion": {"enable": motion_enable, "sensitivity": 80, "duration": 2},
        "ia": {"modelo": "yolo26n.pt", "confianza": 0.4},
    }


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = NodeStore(Path(self.tmp.name) / "data" / "node.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_config_versions_only_on_change(self):
        first = config()
        id1, prev = self.store.record_config("cam", first, config_hash(first), T0)
        self.assertIsNone(prev)
        same_id, prev = self.store.record_config("cam", first, config_hash(first), T0)
        self.assertEqual((same_id, prev), (id1, None))
        changed = config(motion_enable=1)
        id2, prev = self.store.record_config(
            "cam", changed, config_hash(changed), T0 + timedelta(minutes=5))
        self.assertNotEqual(id1, id2)
        self.assertEqual(prev, first)
        rows = self.store.query(
            "SELECT id, vigente_hasta, motion_activo FROM config_camara ORDER BY id")
        self.assertEqual(rows[0]["vigente_hasta"], "2026-10-06T19:05:00.000+00:00")
        self.assertIsNone(rows[1]["vigente_hasta"])
        self.assertEqual(rows[1]["motion_activo"], 1)

    def test_alarm_links_current_config_and_verdict(self):
        cfg = config()
        config_id, _ = self.store.record_config("cam", cfg, config_hash(cfg), T0)
        alarm_id = self.store.alarm_started("cam", "human", T0)
        self.store.alarm_verdict(alarm_id, "confirmed", 3, 2, 0.91234, 1.2)
        self.store.alarm_ended(alarm_id, T0 + timedelta(seconds=9), 9.0)
        self.store.alarm_verification_finished(alarm_id, 12, 2, 0.95, "capturas/x.jpg", None)
        row = self.store.query("SELECT * FROM alarma WHERE id=?", (alarm_id,))[0]
        self.assertEqual(row["config_id"], config_id)
        self.assertEqual(row["veredicto"], "confirmada")
        self.assertEqual((row["max_personas"], row["fotogramas"]), (2, 12))
        self.assertEqual(row["duracion_s"], 9.0)
        self.assertEqual(row["foto"], "capturas/x.jpg")
        self.assertIsNone(row["enviado"])

    def test_discarded_without_frames_is_no_video(self):
        alarm_id = self.store.alarm_started("cam", "motion", T0)
        self.store.alarm_verdict(alarm_id, "discarded", 0, 0, 0.0, None)
        row = self.store.query("SELECT veredicto, config_id FROM alarma")[0]
        self.assertEqual(row["veredicto"], "sin_video")
        self.assertIsNone(row["config_id"])

    def test_reopen_keeps_data(self):
        alarm_id = self.store.alarm_started("cam", "motion", T0)
        self.store.close()
        self.store = NodeStore(Path(self.tmp.name) / "data" / "node.db")
        self.assertEqual(len(self.store.query("SELECT id FROM alarma WHERE id=?",
                                              (alarm_id,))), 1)


class ConfigDiffTests(unittest.TestCase):
    def test_diff_scalars_and_region(self):
        changes = diff_config(config(), config(motion_enable=1, sensitivity=70,
                                               rect=[{"x": 1}]))
        self.assertIn("Human Detect: sensitivity 90 → 70", changes)
        self.assertIn("Human Detect: región modificada", changes)
        self.assertIn("Motion Detect: enable desactivado → ACTIVADO", changes)

    def test_hash_is_stable(self):
        self.assertEqual(config_hash(config()), config_hash(config()))
        self.assertNotEqual(config_hash(config()), config_hash(config(sensitivity=80)))


if __name__ == "__main__":
    unittest.main()
