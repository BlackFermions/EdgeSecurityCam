import unittest
from types import SimpleNamespace

from src.analysis.face_policy import (
    IDENTIFIED,
    PENDING,
    UNKNOWN,
    FaceState,
    current_wait,
    register_attempt,
    should_attempt,
)
from src.faces.gallery import Match
from src.settings import CameraSettings


PARAMS = CameraSettings("cam").section("rostro")      # votos 2, intentos_max 5, espera 1,5 s


def attempt(name=None, score=0.55, valid=True):   # < umbral_seguro (0,60)
    if not valid:
        return SimpleNamespace(counts=False, match=None)
    reason = "coincide" if name else "bajo_umbral"
    return SimpleNamespace(counts=True, match=Match(name, score, name or "X", None, 0.0, reason))


class ShouldAttemptTests(unittest.TestCase):
    def test_pending_person_is_attempted(self):
        self.assertTrue(should_attempt(FaceState(), 10.0, 120, PARAMS))

    def test_waits_between_attempts(self):
        state = FaceState(last_attempt=9.0)
        self.assertFalse(should_attempt(state, 10.0, 120, PARAMS))
        self.assertTrue(should_attempt(state, 10.6, 120, PARAMS))

    def test_small_person_is_not_attempted(self):
        self.assertFalse(should_attempt(FaceState(), 10.0, 30, PARAMS))

    def test_one_request_at_a_time(self):
        self.assertFalse(should_attempt(FaceState(requested=True), 10.0, 120, PARAMS))

    def test_identified_only_rechecked_periodically_or_on_doubt(self):
        state = FaceState(status=IDENTIFIED, name="Brayan", decided_at=100.0, last_attempt=100.0)
        self.assertFalse(should_attempt(state, 130.0, 120, PARAMS))
        self.assertTrue(should_attempt(state, 160.0, 120, PARAMS))
        state.recheck = True                               # reapareció tras perderse
        self.assertTrue(should_attempt(state, 102.0, 120, PARAMS))


class BackoffTests(unittest.TestCase):
    """2026-10-13: personas de espaldas o mirando la laptop; sin espera
    creciente se abría el vídeo HD cada ~3 s sin parar."""

    def test_wait_doubles_without_face_up_to_max(self):
        state = FaceState()
        waits = []
        for t in range(8):
            register_attempt(state, attempt(valid=False), float(t), PARAMS)
            waits.append(current_wait(state, PARAMS))
        self.assertEqual(waits[:5], [3.0, 6.0, 12.0, 24.0, 30.0])
        self.assertEqual(waits[-1], 30.0)

    def test_valid_face_resets_wait(self):
        state = FaceState()
        for t in range(4):
            register_attempt(state, attempt(valid=False), float(t), PARAMS)
        register_attempt(state, attempt(None, 0.3), 10.0, PARAMS)
        self.assertEqual(current_wait(state, PARAMS), 1.5)


class RecheckTests(unittest.TestCase):
    def test_recheck_is_consumed_and_backoff_resumes(self):
        # 2026-10-13 12:21: Persona 2 reapareció y, sin cara, se intentó cada
        # ~4 s durante más de un minuto (36 ráfagas HD en 4 minutos).
        state = FaceState()
        for t in range(4):
            register_attempt(state, attempt(valid=False), float(t), PARAMS)
        state.recheck = True
        self.assertTrue(should_attempt(state, 10.0, 120, PARAMS))
        self.assertFalse(state.recheck)
        state.requested = True
        register_attempt(state, attempt(valid=False), 10.0, PARAMS)
        waits = []
        for t in (13.1, 19.2, 31.3):
            self.assertTrue(should_attempt(state, t, 120, PARAMS))
            register_attempt(state, attempt(valid=False), t, PARAMS)
            waits.append(current_wait(state, PARAMS))
        self.assertEqual(waits, [6.0, 12.0, 24.0])
        self.assertFalse(should_attempt(state, 40.0, 120, PARAMS))


class CandidateTests(unittest.TestCase):
    def test_log_2026_10_13_1739_is_not_unknown(self):
        # Brayan: 0,26 / 0,45 / 0,70 / 0,28 / 0,34 (las bajas con la cara cortada).
        state = FaceState()
        params = dict(PARAMS, umbral_seguro=0.99)        # sin la regla de coincidencia segura
        for t, (name, score) in enumerate([(None, 0.26), (None, 0.449), ("Brayan", 0.70),
                                           (None, 0.28), (None, 0.34)]):
            self.assertIsNone(register_attempt(state, attempt(name, score), float(t), params))
        self.assertEqual(state.status, PENDING)          # sigue siendo candidato

    def test_strong_single_match_identifies(self):
        state = FaceState()
        event = register_attempt(state, attempt("Brayan", 0.70), 1.0, PARAMS)
        self.assertEqual(event, (IDENTIFIED, "Brayan", 0.70))

    def test_candidate_gives_up_after_double_attempts(self):
        state = FaceState()
        params = dict(PARAMS, umbral_seguro=0.99)
        register_attempt(state, attempt("Brayan", 0.50), 0.0, params)
        events = [register_attempt(state, attempt(None, 0.3), float(t), params)
                  for t in range(1, 10)]
        self.assertEqual(events[-1][0], UNKNOWN)
        self.assertTrue(all(e is None for e in events[:-1]))


class RegisterAttemptTests(unittest.TestCase):
    def test_two_votes_identify(self):
        state = FaceState()
        self.assertIsNone(register_attempt(state, attempt("Brayan", 0.55), 1.0, PARAMS))
        event = register_attempt(state, attempt("Brayan", 0.61), 3.0, PARAMS)
        self.assertEqual(event, (IDENTIFIED, "Brayan", 0.61))
        self.assertEqual(state.status, IDENTIFIED)

    def test_invalid_attempts_do_not_count(self):
        state = FaceState()
        for t in range(10):                                # sin cara, de perfil, unida...
            self.assertIsNone(register_attempt(state, attempt(valid=False), t, PARAMS))
        self.assertEqual((state.status, state.valid_attempts, state.attempts), (PENDING, 0, 10))

    def test_unknown_after_max_valid_attempts(self):
        state = FaceState()
        events = [register_attempt(state, attempt(None, 0.22 + i / 100), i, PARAMS)
                  for i in range(5)]
        self.assertEqual(events[:4], [None] * 4)
        self.assertEqual(events[4][0], UNKNOWN)
        self.assertAlmostEqual(events[4][2], 0.26)

    def test_one_lucky_match_does_not_identify(self):
        state = FaceState()
        register_attempt(state, attempt("Brayan"), 1.0, PARAMS)
        for t in range(2, 6):
            register_attempt(state, attempt(None, 0.3), t, PARAMS)
        self.assertEqual(state.status, PENDING)      # candidato: sigue intentando
        self.assertIsNone(state.name)
        for t in range(6, 11):
            register_attempt(state, attempt(None, 0.3), t, PARAMS)
        self.assertEqual(state.status, UNKNOWN)      # al doble de intentos se rinde
        self.assertIsNone(state.name)

    def test_identity_can_change_with_enough_votes(self):
        state = FaceState()
        register_attempt(state, attempt("Diego"), 1.0, PARAMS)
        register_attempt(state, attempt("Diego"), 2.0, PARAMS)
        register_attempt(state, attempt("Brayan"), 70.0, PARAMS)
        event = register_attempt(state, attempt("Brayan"), 72.0, PARAMS)
        self.assertEqual(event[:2], (IDENTIFIED, "Brayan"))

    def test_reconfirming_same_name_emits_nothing(self):
        state = FaceState()
        register_attempt(state, attempt("Diego"), 1.0, PARAMS)
        register_attempt(state, attempt("Diego"), 2.0, PARAMS)
        register_attempt(state, attempt("Diego"), 70.0, PARAMS)
        self.assertIsNone(register_attempt(state, attempt("Diego"), 72.0, PARAMS))
        self.assertEqual(state.decided_at, 72.0)


if __name__ == "__main__":
    unittest.main()
