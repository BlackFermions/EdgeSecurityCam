import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from src.analysis import SessionStats
from src.storage import NodeStore
from src.tracking import (
    FLOW_SCALE,
    REASON_ALARM,
    REASON_CONFIRM,
    REASON_INTERVAL,
    REASON_LOST,
    REASON_MOTION,
    REASON_WATCH,
    FlowTracker,
    MotionGate,
    TrackManager,
    camera_alias,
    iou,
    yolo_reason,
)


WALL = datetime(2026, 10, 6, 14, 57, 31)


def textured(height=180, width=320, seed=1):
    return np.random.default_rng(seed).integers(0, 255, (height, width), np.uint8)


def decide(**overrides):
    state = dict(now=10.0, last_yolo=9.9, verdict_pending=False, alarm_pending=False,
                 alarm_active=False, has_tracks=True, motion_outside=False,
                 track_lost=False, tracks_moving=False)
    state.update(overrides)
    return yolo_reason(**state)


class SafetyRuleTests(unittest.TestCase):
    """Ahorrar solo al seguir; ante la duda, YOLO."""

    def test_new_alarm_always_runs_yolo(self):
        self.assertEqual(decide(alarm_pending=True, last_yolo=9.99), REASON_ALARM)

    def test_confirmation_runs_at_full_rate(self):
        self.assertEqual(decide(verdict_pending=True, last_yolo=9.75), REASON_CONFIRM)
        self.assertIsNone(decide(verdict_pending=True, last_yolo=9.9))

    def test_motion_outside_tracks_wakes_yolo(self):
        self.assertEqual(decide(motion_outside=True, last_yolo=9.5), REASON_MOTION)

    def test_motion_trigger_is_rate_limited(self):
        self.assertIsNone(decide(motion_outside=True, last_yolo=9.8))

    def test_lost_track_wakes_yolo(self):
        self.assertEqual(decide(track_lost=True, last_yolo=9.0), REASON_LOST)

    def test_max_interval_moving_and_still(self):
        self.assertIsNone(decide(tracks_moving=True, last_yolo=9.2))
        self.assertEqual(decide(tracks_moving=True, last_yolo=9.0), REASON_INTERVAL)
        self.assertIsNone(decide(last_yolo=7.5))
        self.assertEqual(decide(last_yolo=7.0), REASON_INTERVAL)

    def test_alarm_without_tracks_keeps_watching(self):
        self.assertEqual(decide(has_tracks=False, alarm_active=True, last_yolo=9.0),
                         REASON_WATCH)
        self.assertIsNone(decide(has_tracks=False, alarm_active=False, last_yolo=0.0))


class TrackManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = TrackManager("192.168.100.109")

    def apply(self, boxes, now, reason="confirmacion"):
        return self.manager.apply_detections(None, boxes, [0.8] * len(boxes), now,
                                             WALL, reason, "alarma-1")

    def test_codes_use_time_and_subnumber(self):
        created, _ = self.apply([(0, 0, 10, 20), (100, 0, 110, 20)], 0.0)
        self.assertEqual([t.code for t in created],
                         ["cam109-20261006-145731-1", "cam109-20261006-145731-2"])

    def test_same_person_keeps_track(self):
        first, _ = self.apply([(0, 0, 100, 200)], 0.0)
        created, ended = self.apply([(10, 5, 110, 205)], 1.0)
        self.assertEqual((created, ended), ([], []))
        self.assertEqual(self.manager.tracks[0].id, first[0].id)
        self.assertEqual(self.manager.tracks[0].yolo_hits, 2)

    def test_track_ends_after_misses_and_time(self):
        self.apply([(0, 0, 100, 200)], 0.0)
        self.assertEqual(self.apply([], 1.0), ([], []))
        _, ended = self.apply([], 2.5)
        self.assertEqual(len(ended), 1)
        self.assertEqual(self.manager.tracks, [])

    def test_new_person_gets_new_track_with_reason(self):
        self.apply([(0, 0, 100, 200)], 0.0)
        created, _ = self.apply([(0, 0, 100, 200), (300, 0, 400, 200)], 1.0,
                                REASON_MOTION)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].reason, REASON_MOTION)

    def test_helpers(self):
        self.assertEqual(camera_alias("192.168.100.109"), "cam109")
        self.assertEqual(camera_alias("entrada"), "entrada")
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)


class FlowAndMotionTests(unittest.TestCase):
    def test_flow_follows_shifted_texture(self):
        image = textured()
        tracker = FlowTracker(image, (100 / FLOW_SCALE, 50 / FLOW_SCALE,
                                      160 / FLOW_SCALE, 130 / FLOW_SCALE))
        shifted = np.roll(image, 4, axis=1)
        self.assertTrue(tracker.update(image, shifted))
        self.assertAlmostEqual(tracker.box[0], (100 + 4) / FLOW_SCALE, delta=1.0)

    def test_flow_lost_on_flat_dark_area(self):
        dark = np.full((180, 320), 5, np.uint8)
        tracker = FlowTracker(dark, (0, 0, 200, 200))
        self.assertTrue(tracker.lost)
        self.assertFalse(tracker.update(dark, dark))

    def test_motion_gate_outside_and_inside(self):
        gate = MotionGate()
        base = np.zeros((180, 320), np.uint8)
        gate.update(base, [])
        changed = base.copy()
        changed[20:80, 260:315] = 255             # cambio a la derecha, lejos de la caja
        tracked = [(0, 0, 200 / FLOW_SCALE, 180 / FLOW_SCALE)]   # mitad izquierda
        outside, inside = gate.update(changed, tracked)
        self.assertTrue(outside)
        self.assertFalse(inside)
        changed_left = changed.copy()
        changed_left[60:120, 40:120] = 255         # ahora cambia dentro de la caja
        outside, inside = gate.update(changed_left, tracked)
        self.assertFalse(outside)
        self.assertTrue(inside)


class TrackStorageTests(unittest.TestCase):
    def test_track_and_session_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            store = NodeStore(Path(folder) / "node.db")
            manager = TrackManager("192.168.100.109")
            (track,), _ = manager.apply_detections(None, [(0, 0, 10, 20)], [0.7], 0.0,
                                                   WALL, REASON_CONFIRM, None)
            track.session_id = "s1"
            store.track_entered(track)
            manager.apply_detections(None, [(0, 0, 10, 20)], [0.9], 4.0, WALL,
                                     REASON_INTERVAL, None)
            store.track_left(track, "prueba")
            stats = SessionStats("192.168.100.109", WALL, id="s1", ended_wall=WALL,
                                 frames=50, yolo_runs=10, tracks=1, max_people=1)
            stats.reasons[REASON_CONFIRM] = 10
            store.session_finished(stats)
            row = store.query("SELECT * FROM persona_track")[0]
            self.assertEqual(row["codigo"], "cam109-20261006-145731-1")
            self.assertEqual((row["segundos_visible"], row["max_confianza"]), (4.0, 0.9))
            session = store.query("SELECT * FROM sesion_analisis")[0]
            self.assertEqual(session["ahorro"], 0.8)
            self.assertIn("confirmacion", session["motivos"])
            store.close()


if __name__ == "__main__":
    unittest.main()
