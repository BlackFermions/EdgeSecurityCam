import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.camera_events import CameraAlarm
from src.node import CamDetectorNode
from src.storage import NodeStore


T0 = datetime(2026, 10, 6, 19, 0, 0, tzinfo=timezone.utc)
CAM = "192.168.100.109"


def config(motion_enable):
    return {"camara": {"modelo": "W51-TY", "firmware": "V1"},
            "human": {"enable": 1, "sensitivity": 90, "duration": 20},
            "motion": {"enable": motion_enable, "sensitivity": 80, "duration": 2},
            "ia": {}}


def alarm(active, at, kind="human"):
    return CameraAlarm(CAM, kind, active, "RuleEngine/CellMotionDetector/Motion",
                       "Changed", at, None)


class FakeVerifier:
    def __init__(self):
        self.started, self.ended = [], []

    def set_callbacks(self, on_verdict, on_finished, on_track=None, on_session=None):
        self.on_verdict, self.on_finished = on_verdict, on_finished
        self.on_track, self.on_session = on_track, on_session

    def alarm_started(self, camera, alarm_id):
        self.started.append((camera, alarm_id))

    def alarm_ended(self, camera):
        self.ended.append(camera)

    def close(self):
        pass


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = NodeStore(Path(self.tmp.name) / "node.db")
        self.verifier = FakeVerifier()
        self.node = CamDetectorNode([CAM], "u", "p", self.store, verifier=self.verifier)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_config_decides_interpretation(self):
        self.node._on_config(CAM, config(motion_enable=0))
        self.assertTrue(self.node.clients[CAM].motion_as_human)
        self.node._on_config(CAM, config(motion_enable=1))
        self.assertFalse(self.node.clients[CAM].motion_as_human)
        self.assertEqual(len(self.store.query("SELECT id FROM config_camara")), 2)

    def test_alarm_is_stored_and_verified(self):
        self.node._on_config(CAM, config(motion_enable=0))
        self.node._on_alarm(alarm(True, T0))
        self.node._on_alarm(alarm(True, T0 + timedelta(seconds=1)))
        self.node._on_alarm(alarm(False, T0 + timedelta(seconds=6)))
        rows = self.store.query("SELECT * FROM alarma")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["duracion_s"], 6.0)
        self.assertIsNotNone(rows[0]["config_id"])
        self.assertEqual(self.verifier.started, [(CAM, rows[0]["id"])])
        self.assertEqual(self.verifier.ended, [CAM])

    def test_config_change_closes_open_alarm(self):
        self.node._on_config(CAM, config(motion_enable=0))
        self.node._on_alarm(alarm(True, T0))
        self.node._on_config(CAM, config(motion_enable=1))
        row = self.store.query("SELECT fin, nota FROM alarma")[0]
        self.assertIsNotNone(row["fin"])
        self.assertIn("interpretación", row["nota"])

    def test_disconnect_closes_open_alarm(self):
        self.node._on_config(CAM, config(motion_enable=0))
        self.node._on_alarm(alarm(True, T0))
        self.node._status_handler(CAM)("conexión perdida (timeout)", False)
        row = self.store.query("SELECT nota FROM alarma")[0]
        self.assertEqual(row["nota"], "conexión perdida")


if __name__ == "__main__":
    unittest.main()
