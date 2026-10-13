import unittest

import numpy as np

from src.faces.base import Face
from src.faces.gallery import Gallery
from src.faces.quality import FaceQuality, expand_box, frontality, head_band


def face(nose_x=50.0):
    landmarks = ((40.0, 40.0), (60.0, 40.0), (nose_x, 50.0), (42.0, 60.0), (58.0, 60.0))
    raw = np.zeros(15, np.float32)
    return Face(30, 30, 40, 40, landmarks, 0.9, raw)


def unit(*values):
    vector = np.array(values, np.float32)
    return vector / np.linalg.norm(vector)


class QualityTests(unittest.TestCase):
    def test_head_band_is_top_fraction_full_width(self):
        self.assertEqual(head_band((100, 50, 300, 450), 0.30), (100, 50, 300, 170))

    def test_head_band_grows_for_wide_boxes(self):
        # Persona sentada o cortada por el borde: caja más ancha que alta.
        self.assertEqual(head_band((0, 100, 400, 325), 0.30, 0.5), (0, 100, 400, 300))
        self.assertEqual(head_band((0, 0, 100, 300), 0.30, 0.5), (0, 0, 100, 90))

    def test_frontality(self):
        self.assertEqual(frontality(face(50)), 1.0)       # nariz centrada entre los ojos
        self.assertEqual(frontality(face(60)), 0.0)       # nariz sobre un ojo: perfil
        self.assertAlmostEqual(frontality(face(55)), 0.5)

    def test_problems(self):
        q = FaceQuality(size_px=30, frontal=0.2, sharpness=5, brightness=20)
        self.assertEqual(q.problems(48, 0.5, 20, 40), ["pequeña", "de perfil", "borrosa", "oscura"])
        self.assertEqual(FaceQuality(90, 0.9, 500, 120).problems(48, 0.5, 20, 40), [])

    def test_shift_moves_box_and_landmarks(self):
        shifted = face().shifted(100, 10)
        self.assertEqual(shifted.box, (130, 40, 170, 80))
        self.assertEqual(shifted.landmarks[0], (140.0, 50.0))
        self.assertEqual(float(shifted.raw[4]), 100.0)    # fila cruda desplazada para alinear

    def test_expand_box(self):
        self.assertEqual(expand_box((10, 10, 110, 210), 0.1), (0.0, -10.0, 120.0, 230.0))


class GalleryMatchTests(unittest.TestCase):
    def setUp(self):
        self.gallery = Gallery()
        self.gallery.embeddings = {"Brayan": np.vstack([unit(1, 0, 0), unit(1, 0.2, 0)]),
                                   "Diego": np.vstack([unit(0, 1, 0)])}

    def test_match_uses_best_photo(self):
        result = self.gallery.match(unit(1, 0.21, 0), 0.5, 0.08)
        self.assertEqual((result.name, result.reason), ("Brayan", "coincide"))
        self.assertGreater(result.score, 0.99)

    def test_below_threshold_has_no_name_but_reports_best(self):
        result = self.gallery.match(unit(0.3, 0.3, 1), 0.5, 0.08)
        self.assertIsNone(result.name)
        self.assertEqual(result.reason, "bajo_umbral")
        self.assertIsNotNone(result.best)

    def test_ambiguous_between_two_people(self):
        result = self.gallery.match(unit(1, 1.2, 0), 0.5, 0.08)   # 0,78 frente a 0,77
        self.assertEqual((result.name, result.reason), (None, "ambiguo"))

    def test_empty_gallery(self):
        self.assertEqual(Gallery().match(unit(1, 0, 0), 0.5, 0.08).reason, "galeria_vacia")


if __name__ == "__main__":
    unittest.main()
