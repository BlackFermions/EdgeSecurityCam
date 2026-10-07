import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.analysis.idle import IdleChecker
from src.sources.snapshot import SnapshotSource
from src.node import CamDetectorNode
from src.detection.base import FrameResult
from src.storage import NodeStore


CAM = "192.168.100.109"
FRAME = np.zeros((360, 640, 3), np.uint8)


class FakeSource:
    def __init__(self):
        self.frames = [FRAME]

    def grab(self):
        return self.frames[0] if self.frames else None


class FakeDetector:
    def __init__(self, *confidences):
        self.confidences = confidences

    def detect(self, frame):
        boxes = tuple((0, 0, 10, 20) for _ in self.confidences)
        return FrameResult(0.0, tuple(self.confidences), boxes)


class IdleCheckerTests(unittest.TestCase):
    def make(self, detector, busy=False):
        self.people, self.status = [], []
        self.busy = busy
        return IdleChecker(CAM, FakeSource(), detector, lambda camera: self.busy,
                           lambda camera, result: self.people.append(result),
                           lambda camera, message: self.status.append(message),
                           cooldown=60.0, min_confidence=0.5)

    def test_person_seen_without_camera_alarm_opens_session(self):
        checker = self.make(FakeDetector(0.8))
        self.assertTrue(checker.check_once(now=100.0))
        self.assertEqual(len(self.people), 1)

    def test_empty_room_does_nothing(self):
        checker = self.make(FakeDetector())
        self.assertFalse(checker.check_once(now=100.0))
        self.assertEqual(checker.checks, 1)

    def test_low_confidence_does_not_trigger(self):
        checker = self.make(FakeDetector(0.42))
        self.assertFalse(checker.check_once(now=100.0))

    def test_no_check_while_a_session_is_open(self):
        checker = self.make(FakeDetector(0.9), busy=True)
        self.assertFalse(checker.check_once(now=100.0))
        self.assertEqual(checker.checks, 0)          # ni siquiera toma fotograma

    def test_cooldown_after_trigger(self):
        checker = self.make(FakeDetector(0.9))
        self.assertTrue(checker.check_once(now=100.0))
        self.assertFalse(checker.check_once(now=150.0))
        self.assertTrue(checker.check_once(now=161.0))

    def test_grab_failure_reported_once_and_recovery(self):
        checker = self.make(FakeDetector())
        checker._source.frames = []
        checker.check_once(now=100.0)
        checker.check_once(now=110.0)
        checker._source.frames = [FRAME]
        checker.check_once(now=130.0)
        self.assertEqual(len(self.status), 2)
        self.assertIn("recuperada tras 30 s", self.status[1])


class SnapshotSourceTests(unittest.TestCase):
    def test_uses_onvif_snapshot_and_falls_back_to_rtsp(self):
        source = SnapshotSource(CAM, "u", "p", "rtsp://x")
        source._discover_snapshot_uri = lambda: "http://cam/snap.jpg"
        source._fetch_snapshot = lambda uri: FRAME
        source._grab_rtsp = lambda: None
        self.assertIsNotNone(source.grab())
        self.assertEqual(source.method, "onvif")
        source._fetch_snapshot = lambda uri: None    # la foto falla: respaldo RTSP
        source._grab_rtsp = lambda: FRAME
        self.assertIsNotNone(source.grab())
        self.assertEqual(source.method, "rtsp")

    def test_retries_discovery_when_camera_did_not_answer(self):
        from src.sources.camera_events import CameraEventError
        source = SnapshotSource(CAM, "u", "p", "rtsp://x")
        calls = []

        def discover():
            calls.append(1)
            if len(calls) == 1:
                raise CameraEventError("la cámara no respondió")
            return None                               # respondió, sin foto ONVIF

        source._discover_snapshot_uri = discover
        source._grab_rtsp = lambda: FRAME
        source.grab()
        source.grab()
        source.grab()
        self.assertEqual(len(calls), 2)               # no insiste tras una respuesta


class FakeVerifier:
    def __init__(self):
        self.detector = FakeDetector(0.9)
        self.started, self.ended = [], []

    def set_callbacks(self, *callbacks):
        pass

    def has_session(self, camera):
        return False

    def alarm_started(self, camera, alarm_id):
        self.started.append(alarm_id)

    def alarm_ended(self, camera):
        self.ended.append(camera)

    def close(self):
        pass


class NodeIdleTests(unittest.TestCase):
    def test_idle_person_creates_reposo_alarm_and_session(self):
        with tempfile.TemporaryDirectory() as folder:
            store = NodeStore(Path(folder) / "node.db")
            verifier = FakeVerifier()
            node = CamDetectorNode([CAM], "u", "p", store, verifier=verifier,
                                   idle_sources={CAM: FakeSource()})
            self.assertEqual(len(node.idle_checkers), 1)
            node._on_idle_person(CAM, FakeDetector(0.9).detect(FRAME))
            row = store.query("SELECT * FROM alarma")[0]
            self.assertEqual(row["tipo"], "reposo")
            self.assertIn("la cámara no avisó", row["nota"])
            self.assertEqual(verifier.started, [row["id"]])
            self.assertEqual(verifier.ended, [CAM])
            store.close()

    def test_no_idle_checker_without_yolo(self):
        node = CamDetectorNode([CAM], "u", "p", None, idle_sources={CAM: FakeSource()})
        self.assertEqual(node.idle_checkers, [])


if __name__ == "__main__":
    unittest.main()
