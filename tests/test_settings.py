import tempfile
import unittest
from pathlib import Path

from src.analysis.policy import yolo_reason
from src.analysis.verdict import VerificationTracker
from src.settings import SCHEMA, CameraSettings, SettingsError, load_settings, markdown_table
from src.tracking import MotionGate, TrackManager


CAM, OTHER = "192.168.100.109", "192.168.100.110"


def write(text):
    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / "camdetector.toml"
    path.write_text(text, encoding="utf-8")
    return folder, path


class DefaultsTests(unittest.TestCase):
    """Los valores por defecto deben ser exactamente los que estaban en el código."""

    def test_defaults_match_previous_constants(self):
        s = CameraSettings(CAM)
        manager, reference = TrackManager.from_settings(CAM, s), TrackManager(CAM)
        for attr in ("match_iou", "center_gate", "max_misses", "forget_seconds",
                     "grace_seconds", "hidden_seconds", "edge_margin", "MAX_GATE",
                     "EXIT_SPEED", "APPEARANCE_VETO", "APPEARANCE_RECOVERY",
                     "APPEARANCE_MARGIN", "STRONG_OVERLAP"):
            self.assertEqual(getattr(manager, attr), getattr(reference, attr), attr)
        gate = MotionGate()
        self.assertEqual((s.get("puerta", "umbral"), s.get("puerta", "fraccion_min"),
                          s.get("puerta", "margen")),
                         (gate.threshold, gate.min_fraction, gate.margin))
        verdict = VerificationTracker(CAM, 0.0)
        self.assertEqual((s.get("veredicto", "confirmar_fotogramas"),
                          s.get("veredicto", "decidir_s"),
                          s.get("veredicto", "decidir_con_persona_s")),
                         (verdict.confirm_frames, verdict.decide_seconds,
                          verdict.seen_decide_seconds))
        defaults = yolo_reason.__kwdefaults__
        self.assertEqual(s.get("politica", "quieto_s"), defaults["still_interval"])
        self.assertEqual(s.get("politica", "disparo_min_s"), defaults["trigger_interval"])

    def test_missing_file_gives_defaults(self):
        settings = load_settings(Path("no-existe.toml"), [CAM])
        self.assertEqual(settings[CAM].get("yolo", "tamano"), 416)
        self.assertEqual(settings[CAM].sources, {})


class FileTests(unittest.TestCase):
    def test_defaults_and_per_camera_override(self):
        folder, path = write('''
[defecto.yolo]
confianza = 0.45

[camaras."192.168.100.109"]
nombre = "Sala de reuniones"
[camaras."192.168.100.109".seguimiento]
margen_borde = 0.10
''')
        with folder:
            settings = load_settings(path, [CAM, OTHER])
        self.assertEqual(settings[CAM].get("yolo", "confianza"), 0.45)
        self.assertEqual(settings[OTHER].get("yolo", "confianza"), 0.45)
        self.assertEqual(settings[CAM].get("seguimiento", "margen_borde"), 0.10)
        self.assertEqual(settings[OTHER].get("seguimiento", "margen_borde"), 0.08)
        self.assertEqual(settings[CAM].name, "Sala de reuniones")
        self.assertEqual(settings[CAM].sources["seguimiento.margen_borde"],
                         'camaras."192.168.100.109"')

    def test_command_line_wins(self):
        folder, path = write('[defecto.yolo]\ntamano = 640\n')
        with folder:
            settings = load_settings(path, [CAM], {("yolo", "tamano"): 320})
        self.assertEqual(settings[CAM].get("yolo", "tamano"), 320)
        self.assertEqual(settings[CAM].sources["yolo.tamano"], "línea de comandos")

    def test_typo_is_an_error(self):
        folder, path = write('[defecto.seguimiento]\nmargen_bordes = 0.1\n')
        with folder, self.assertRaisesRegex(SettingsError, "margen_bordes"):
            load_settings(path, [CAM])

    def test_unknown_group_is_an_error(self):
        folder, path = write('[defecto.rostros]\numbral = 0.5\n')
        with folder, self.assertRaisesRegex(SettingsError, "rostros"):
            load_settings(path, [CAM])

    def test_out_of_range_and_wrong_type(self):
        for text, message in (('[defecto.yolo]\nconfianza = 1.5\n', "máximo"),
                              ('[defecto.yolo]\ntamano = 500\n', "uno de"),
                              ('[defecto.veredicto]\nconfirmar_fotogramas = 2.5\n', "entero"),
                              ('[defecto.reposo]\ncada_s = 1\n', "al menos 3"),
                              ('[defecto.yolo]\nconfianza = "alta"\n', "número")):
            folder, path = write(text)
            with folder, self.assertRaisesRegex(SettingsError, message):
                load_settings(path, [CAM])

    def test_cross_rules(self):
        folder, path = write('[defecto.veredicto]\ndecidir_s = 10\ndecidir_con_persona_s = 8\n')
        with folder, self.assertRaisesRegex(SettingsError, "decidir_con_persona_s"):
            load_settings(path, [CAM])

    def test_idle_can_be_disabled(self):
        folder, path = write('[defecto.reposo]\ncada_s = 0\n')
        with folder:
            self.assertEqual(load_settings(path, [CAM])[CAM].get("reposo", "cada_s"), 0)

    def test_example_file_is_valid(self):
        example = Path(__file__).resolve().parents[1] / "config" / "camdetector.example.toml"
        load_settings(example, [CAM])


class DocumentationTests(unittest.TestCase):
    def test_every_parameter_is_documented(self):
        table = markdown_table()
        for group, params in SCHEMA.items():
            for name, spec in params.items():
                self.assertIn(f"`{name}`", table)
                self.assertTrue(spec.description, f"{group}.{name} sin descripción")


if __name__ == "__main__":
    unittest.main()
