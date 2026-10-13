"""Cuándo intentar reconocer a una persona y cómo decidir quién es.

Principio: reconocer UNA VEZ por trayectoria. El tracker ya sabe que la
persona del código ...-1 es la misma de hace 5 minutos; basta con ponerle
nombre una vez y volver a mirar solo ante duda.

Estados de cada persona seguida:

    pendiente ──(votos con el mismo nombre)──► identificada
        └────(intentos con cara válida sin coincidencia)──► desconocida

- Un intento "válido" es el que llegó a compararse con la galería. Los que
  no tienen cara, la tienen de mala calidad o encuentran dos caras en la
  franja (caja unida) no cuentan: solo esperan el siguiente turno.
- Identificada o desconocida, se vuelve a verificar cada
  ``reverificar_cada_s`` o de inmediato si la trayectoria se recuperó tras
  perderse (pudo cambiar de persona).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

PENDING, IDENTIFIED, UNKNOWN = "pendiente", "identificada", "desconocida"


@dataclass
class FaceState:
    status: str = PENDING
    name: str | None = None
    score: float = 0.0
    votes: Counter = field(default_factory=Counter)
    best_scores: dict = field(default_factory=dict)
    valid_attempts: int = 0
    attempts: int = 0
    last_attempt: float = float("-inf")
    decided_at: float | None = None
    requested: bool = False
    recheck: bool = False            # forzar verificación (reaparición, cruce)
    misses_in_row: int = 0           # intentos seguidos sin cara utilizable


def current_wait(state: FaceState, params: dict) -> float:
    """Espera antes del siguiente intento: se duplica con cada intento sin cara
    utilizable (persona de espaldas o mirando abajo), hasta ``espera_max_s``."""
    return min(params["espera_entre_intentos_s"] * 2 ** state.misses_in_row,
               params["espera_max_s"])


def should_attempt(state: FaceState, now: float, person_height_px: float,
                   params: dict) -> bool:
    if state.requested or person_height_px < params["persona_min_alto_px"]:
        return False
    if state.recheck:                 # reapareció: la espera acumulada no aplica
        state.misses_in_row = 0
    if now - state.last_attempt < current_wait(state, params):
        return False
    if state.recheck or state.status == PENDING:
        return True
    return (state.decided_at is not None
            and now - state.decided_at >= params["reverificar_cada_s"])


def register_attempt(state: FaceState, attempt, now: float, params: dict
                     ) -> tuple[str, str | None, float] | None:
    """Aplica el resultado de un intento. Devuelve un evento si cambió algo:
    ("identificada", nombre, similitud) o ("desconocida", None, mejor similitud).
    """
    state.requested = False
    state.attempts += 1
    state.last_attempt = now
    if not attempt.counts:
        state.misses_in_row += 1
        return None
    state.misses_in_row = 0
    state.valid_attempts += 1
    match = attempt.match
    if match.name is not None:
        state.votes[match.name] += 1
        state.best_scores[match.name] = max(state.best_scores.get(match.name, 0.0), match.score)
        if state.votes[match.name] >= params["votos"]:
            changed = state.status != IDENTIFIED or state.name != match.name
            state.status, state.name = IDENTIFIED, match.name
            state.score = state.best_scores[match.name]
            state.decided_at, state.recheck = now, False
            state.votes.clear()
            if changed:
                return (IDENTIFIED, match.name, state.score)
        return None
    if state.status == PENDING and state.valid_attempts >= params["intentos_max"]:
        state.status, state.decided_at, state.recheck = UNKNOWN, now, False
        state.score = max([match.score, *state.best_scores.values()])
        return (UNKNOWN, None, state.score)
    if state.status != PENDING and state.recheck:
        # Verificación tras una reaparición sin coincidencia: se mantiene la
        # decisión anterior pero se vuelve a programar normalmente.
        state.recheck = False
        state.decided_at = now
    elif state.status != PENDING:
        state.decided_at = now
    return None
