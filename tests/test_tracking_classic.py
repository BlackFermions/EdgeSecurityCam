import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.storage import NodeStore
from src.tracking import (
    FLOW_SCALE,
    REASON_CONFIRM,
    REASON_WATCH,
    FlowTracker,
    MotionModel,
    TrackManager,
    appearance_similarity,
    assign,
    color_signature,
)


WALL = datetime(2026, 10, 6, 16, 0, 0)
RED, BLUE = (30, 30, 220), (220, 60, 30)


def scene(people, size=(360, 640)):
    """Fondo gris con personas como rectángulos del color de su ropa."""
    frame = np.full((*size, 3), 90, np.uint8)
    for (x1, y1, x2, y2), color in people:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    return frame


class KalmanTests(unittest.TestCase):
    def test_learns_velocity_and_extrapolates(self):
        model = MotionModel((0, 0, 50, 100), 0.0)
        for step in range(1, 11):
            t = step * 0.1
            x = 100 * t                       # 100 px/s hacia la derecha
            model.update((x, 0, x + 50, 100), t, MotionModel.YOLO_NOISE, 0.7)
        self.assertAlmostEqual(model.x[2], 100, delta=15)
        predicted = model.extrapolate(2.0)    # 1 s después, sin medir
        self.assertAlmostEqual(predicted[0], 200, delta=20)

    def test_extrapolation_is_capped(self):
        model = MotionModel((0, 0, 50, 100), 0.0)
        model.x[2] = 100.0
        self.assertAlmostEqual(model.extrapolate(60.0)[0], 150, delta=1)


class FlowScaleTests(unittest.TestCase):
    def test_box_grows_when_texture_zooms(self):
        rng = np.random.default_rng(3)
        image = cv2.GaussianBlur(rng.integers(0, 255, (180, 320), np.uint8), (3, 3), 0)
        zoomed = cv2.resize(image, None, fx=1.05, fy=1.05)
        offset_y = (zoomed.shape[0] - 180) // 2
        offset_x = (zoomed.shape[1] - 320) // 2
        zoomed = zoomed[offset_y:offset_y + 180, offset_x:offset_x + 320]
        box = (130 / FLOW_SCALE, 60 / FLOW_SCALE, 190 / FLOW_SCALE, 120 / FLOW_SCALE)
        tracker = FlowTracker(image, box)
        self.assertTrue(tracker.update(image, zoomed))
        self.assertGreater(tracker.box[2] - tracker.box[0], box[2] - box[0])


class AppearanceTests(unittest.TestCase):
    def test_same_and_different_clothes(self):
        frame = scene([((100, 50, 180, 300), RED), ((300, 50, 380, 300), BLUE),
                       ((450, 50, 530, 300), RED)])
        red1 = color_signature(frame, (100, 50, 180, 300))
        blue = color_signature(frame, (300, 50, 380, 300))
        red2 = color_signature(frame, (450, 50, 530, 300))
        self.assertGreater(appearance_similarity(red1, red2), 0.9)
        self.assertLess(appearance_similarity(red1, blue), 0.3)

    def test_no_color_in_ir(self):
        gray = cv2.cvtColor(cv2.cvtColor(scene([((100, 50, 180, 300), RED)]),
                                         cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        self.assertIsNone(color_signature(gray, (100, 50, 180, 300)))
        self.assertIsNone(appearance_similarity(None, None))


class AssignmentTests(unittest.TestCase):
    def test_hungarian_beats_greedy(self):
        # Voraz tomaría (0,0)=0.9 y dejaría a la fila 1 con 0.1; óptimo: 0.8+0.8.
        score = np.array([[0.9, 0.8], [0.8, 0.1]])
        self.assertEqual(sorted(assign(score)), [(0, 1), (1, 0)])


class ClassicTrackerTests(unittest.TestCase):
    def setUp(self):
        self.manager = TrackManager("192.168.100.109")
        self.manager.frame_size = (640, 360)

    def apply(self, boxes, now, frame=None, reason=REASON_CONFIRM):
        return self.manager.apply_detections(None, boxes, [0.8] * len(boxes), now, WALL,
                                             reason, None, frame=frame)

    def test_walking_person_hidden_recovered_at_predicted_spot(self):
        first = None
        for step in range(6):                 # camina 100 px/s
            x = 150 + 20 * step
            created, _ = self.apply([(x, 100, x + 60, 260)], step * 0.2)
            first = first or created[0]
        for t in (1.5, 3.2):                  # deja de verse (oculta)
            self.apply([], t)
        self.assertEqual(len(self.manager.hidden()), 1)
        # Reaparece lejos de la última caja, pero cerca de la posición predicha.
        created, _ = self.apply([(400, 100, 460, 260)], 4.0, reason=REASON_WATCH)
        self.assertEqual(created, [])
        self.assertEqual(self.manager.tracks[0].code, first.code)
        self.assertEqual(self.manager.last_recovered[0][1], "posicion")

    def test_hidden_person_reappears_far_away_by_appearance(self):
        frame = scene([((100, 50, 180, 300), RED)])
        (first,), _ = self.apply([(100, 50, 180, 300)], 0.0, frame)
        self.apply([], 1.0)
        self.apply([], 2.5)
        far = scene([((480, 50, 560, 300), RED)])
        created, _ = self.apply([(480, 50, 560, 300)], 20.0, far, REASON_WATCH)
        self.assertEqual(created, [])
        self.assertEqual(self.manager.tracks[0].code, first.code)
        self.assertEqual(self.manager.last_recovered[0][1], "apariencia")
        self.assertEqual(self.manager.tracks[0].recoveries_by_appearance, 1)

    def test_far_reappearance_with_other_clothes_is_new_person(self):
        self.apply([(100, 50, 180, 300)], 0.0, scene([((100, 50, 180, 300), RED)]))
        self.apply([], 1.0)
        self.apply([], 2.5)
        created, _ = self.apply([(480, 50, 560, 300)], 20.0,
                                scene([((480, 50, 560, 300), BLUE)]), REASON_WATCH)
        self.assertEqual(len(created), 1)

    def test_crossing_people_keep_codes_by_appearance(self):
        both = scene([((200, 50, 280, 300), RED), ((320, 50, 400, 300), BLUE)])
        created, _ = self.apply([(200, 50, 280, 300), (320, 50, 400, 300)], 0.0, both)
        codes = {created[0].code: "rojo", created[1].code: "azul"}
        # Se cruzan: ahora el rojo está a la derecha del azul, con cajas cercanas.
        crossed = scene([((240, 50, 320, 300), BLUE), ((300, 50, 380, 300), RED)])
        self.apply([(240, 50, 320, 300), (300, 50, 380, 300)], 0.5, crossed)
        by_position = sorted(self.manager.tracks, key=lambda t: t.box[0])
        self.assertEqual([codes[t.code] for t in by_position], ["azul", "rojo"])


class MigrationTests(unittest.TestCase):
    def test_v2_database_gets_new_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.db"
            db = sqlite3.connect(path)
            db.executescript("""
                CREATE TABLE persona_track (id TEXT PRIMARY KEY, codigo TEXT, camara TEXT,
                    alarma_id TEXT, sesion_id TEXT, entrada TEXT, ultima_vista TEXT,
                    salida TEXT, segundos_visible REAL, max_confianza REAL,
                    detecciones_yolo INTEGER, motivo_deteccion TEXT, nota TEXT,
                    actualizado TEXT, enviado TEXT);
                CREATE TABLE sesion_analisis (id TEXT PRIMARY KEY, camara TEXT, inicio TEXT,
                    fin TEXT, fotogramas INTEGER, inferencias_yolo INTEGER, ahorro REAL,
                    motivos TEXT, nuevas_por_motivo TEXT, trayectorias INTEGER,
                    max_personas INTEGER, error TEXT, enviado TEXT);
                PRAGMA user_version=2;""")
            db.close()
            store = NodeStore(path)
            columns = {row[1] for row in store.query("PRAGMA table_info(persona_track)")}
            self.assertIn("recuperaciones_apariencia", columns)
            self.assertEqual(store.query("PRAGMA user_version")[0][0], 3)
            store.close()


if __name__ == "__main__":
    unittest.main()
