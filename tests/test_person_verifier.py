import unittest

from src.person_verifier import FrameResult, VerificationTracker, rtsp_url


def frame(at, *confidences):
    boxes = tuple((0, 0, 10, 10) for _ in confidences)
    return FrameResult(at, tuple(confidences), boxes)


class VerificationTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = VerificationTracker("cam", started_at=100.0,
                                           confirm_frames=2, decide_seconds=4.0)

    def test_confirms_after_consecutive_frames(self):
        self.assertIsNone(self.tracker.add(frame(101.0, 0.8)))
        self.assertEqual(self.tracker.add(frame(101.2, 0.9)), "confirmed")
        self.assertIsNone(self.tracker.add(frame(101.4, 0.9)))
        self.assertEqual(self.tracker.summary.first_person_after, 1.0)

    def test_isolated_detection_does_not_confirm(self):
        self.tracker.add(frame(101.0, 0.8))
        self.tracker.add(frame(101.2))
        self.assertIsNone(self.tracker.add(frame(101.4, 0.8)))
        self.assertIsNone(self.tracker.summary.verdict)

    def test_discards_after_timeout(self):
        self.tracker.add(frame(101.0))
        self.assertEqual(self.tracker.add(frame(104.0)), "discarded")
        self.assertIsNone(self.tracker.add(frame(104.2)))

    def test_seen_person_extends_decision_window(self):
        # 2026-10-07 08:36:39: YOLO vio a alguien una vez (0.52) y se descartó a
        # los 4 s para confirmarse después. Ahora espera hasta 8 s.
        self.assertIsNone(self.tracker.add(frame(101.0, 0.52)))
        self.assertIsNone(self.tracker.add(frame(101.2)))
        self.assertIsNone(self.tracker.check_timeout(105.0))
        self.assertEqual(self.tracker.add(frame(103.0, 0.6)), None)
        self.assertEqual(self.tracker.add(frame(103.2, 0.7)), "confirmed")

    def test_seen_once_then_nothing_discards_at_extended_limit(self):
        self.tracker.add(frame(101.0, 0.52))
        self.assertIsNone(self.tracker.check_timeout(107.9))
        self.assertEqual(self.tracker.check_timeout(108.0), "discarded")

    def test_discarded_can_become_confirmed(self):
        self.tracker.add(frame(104.0))
        self.tracker.add(frame(110.0, 0.7))
        self.assertEqual(self.tracker.add(frame(110.2, 0.7)), "confirmed")

    def test_timeout_without_frames(self):
        self.assertIsNone(self.tracker.check_timeout(103.0))
        self.assertEqual(self.tracker.check_timeout(104.5), "discarded")
        self.assertEqual(self.tracker.summary.frames, 0)

    def test_max_people_and_best_confidence(self):
        self.tracker.add(frame(101.0, 0.5))
        self.tracker.add(frame(101.2, 0.6, 0.9))
        self.tracker.add(frame(101.4, 0.7))
        self.assertEqual(self.tracker.summary.max_people, 2)
        self.assertEqual(self.tracker.summary.best_confidence, 0.9)

    def test_best_frame_prefers_more_people(self):
        self.assertTrue(self.tracker.is_better_frame(frame(1, 0.9)))
        self.assertTrue(self.tracker.is_better_frame(frame(2, 0.5, 0.5)))
        self.assertFalse(self.tracker.is_better_frame(frame(3, 0.95)))


class RtspUrlTests(unittest.TestCase):
    def test_credentials_are_escaped(self):
        self.assertEqual(rtsp_url("10.0.0.5", "admin", "p@ss:1"),
                         "rtsp://admin:p%40ss%3A1@10.0.0.5:554/stream2")


if __name__ == "__main__":
    unittest.main()
