"""Seguimiento corporal clásico (sin redes neuronales).

- ``flow``: flujo óptico Lucas-Kanade con escala.
- ``kalman``: velocidad constante; predice dónde está quien dejó de verse.
- ``assignment``: asignación húngara.
- ``gate``: diferencia de fotogramas (movimiento nuevo).
- ``manager``: trayectorias, gracia, ocultas, continuidad y salidas.
"""

from src.tracking.assignment import assign
from src.tracking.flow import FLOW_SCALE, FlowTracker
from src.tracking.gate import MotionGate
from src.tracking.geometry import Box, iou
from src.tracking.kalman import MotionModel
from src.tracking.manager import PersonTrack, TrackManager, camera_alias, match_score

__all__ = ["assign", "FLOW_SCALE", "FlowTracker", "MotionGate", "Box", "iou",
           "MotionModel", "PersonTrack", "TrackManager", "camera_alias", "match_score"]
