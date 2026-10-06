"""Seguimiento corporal de bajo consumo con algoritmos clásicos.

YOLO (poco frecuente) crea las trayectorias y corrige su posición. Entre dos
detecciones, sin redes neuronales:

- **flujo óptico Lucas-Kanade** mueve y escala cada caja;
- un **filtro de Kalman** de velocidad constante combina el flujo (ruidoso,
  cada fotograma) con YOLO (preciso, ocasional) y predice dónde está una
  persona que dejó de verse;
- la **asignación húngara** reparte detecciones y trayectorias de forma óptima;
- un **histograma de color del torso** (apariencia clásica) evita intercambiar
  personas con ropa distinta y recupera a quien reaparece en otro sitio. Con
  IR u oscuridad no hay color y solo cuenta la posición.

Una puerta de movimiento por diferencia de fotogramas despierta a YOLO si algo
se mueve fuera de las personas ya seguidas.

Regla de seguridad: el ahorro solo aplica a seguir a quien ya fue detectado;
ante cualquier duda (movimiento nuevo, trayectoria perdida, alarma nueva o
demasiado tiempo sin YOLO) se ejecuta YOLO.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np


Box = tuple[float, float, float, float]       # x1, y1, x2, y2 en píxeles

# Motivos por los que se ejecuta YOLO (se registran para auditar el ahorro).
REASON_CONFIRM = "confirmacion"       # alarma sin veredicto todavía
REASON_ALARM = "alarma"               # la cámara inició una alarma nueva
REASON_MOTION = "movimiento_nuevo"    # movimiento fuera de las personas seguidas
REASON_LOST = "trayectoria_perdida"   # el flujo óptico perdió a alguien
REASON_INTERVAL = "intervalo"         # tiempo máximo sin YOLO
REASON_WATCH = "vigilancia"           # alarma activa sin personas seguidas

FLOW_SCALE = 0.5            # el flujo y la puerta trabajan a media resolución


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def camera_alias(camera: str) -> str:
    """``192.168.100.109`` → ``cam109``; otros nombres se usan tal cual."""
    last = camera.rsplit(".", 1)[-1]
    return f"cam{last}" if last.isdigit() else camera


def _center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _size(box: Box) -> tuple[float, float]:
    return max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)


def _box_from(cx: float, cy: float, width: float, height: float) -> Box:
    return (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


# --- flujo óptico -----------------------------------------------------------

class FlowTracker:
    """Sigue una caja moviendo puntos característicos con Lucas-Kanade.

    Trabaja sobre imágenes en gris a escala ``FLOW_SCALE``. Usa verificación
    ida y vuelta: los puntos que no regresan a su origen se descartan. Estima
    desplazamiento (mediana) y escala (mediana de cambios de distancia al
    centro de los puntos, como Median Flow). Si quedan muy pocos puntos (zona
    oscura, oclusión) la trayectoria se marca perdida y YOLO decide.
    """

    MAX_POINTS = 30
    MIN_POINTS = 5
    MAX_FB_ERROR = 1.5
    MAX_SCALE_STEP = 0.08       # cambio de tamaño máximo por fotograma

    def __init__(self, gray: np.ndarray, box: Box) -> None:
        self.box = box
        self.lost = False
        self.moved = 0.0
        self._points = self._seed(gray, box)
        self._initial = len(self._points)
        if self._initial < self.MIN_POINTS:
            self.lost = True

    @staticmethod
    def _seed(gray: np.ndarray, box: Box) -> np.ndarray:
        height, width = gray.shape[:2]
        x1, y1, x2, y2 = (int(round(v * FLOW_SCALE)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return np.empty((0, 1, 2), np.float32)
        points = cv2.goodFeaturesToTrack(gray[y1:y2, x1:x2], FlowTracker.MAX_POINTS,
                                         0.01, 3)
        if points is None:
            return np.empty((0, 1, 2), np.float32)
        return points.astype(np.float32) + np.float32([x1, y1])

    def update(self, previous: np.ndarray, current: np.ndarray) -> bool:
        """Mueve y escala la caja; devuelve False si la trayectoria se perdió."""
        if self.lost or len(self._points) == 0:
            self.lost = True
            return False
        params = dict(winSize=(15, 15), maxLevel=2)
        new, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, self._points,
                                                  None, **params)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, new,
                                                        None, **params)
        error = np.linalg.norm((self._points - back).reshape(-1, 2), axis=1)
        good = ((status.ravel() == 1) & (back_status.ravel() == 1)
                & (error < self.MAX_FB_ERROR))
        if good.sum() < max(self.MIN_POINTS, 0.3 * self._initial):
            self.lost = True
            return False
        old_points = self._points.reshape(-1, 2)[good]
        new_points = new.reshape(-1, 2)[good]
        shift = np.median(new_points - old_points, axis=0)
        old_spread = np.linalg.norm(old_points - np.median(old_points, axis=0), axis=1)
        new_spread = np.linalg.norm(new_points - np.median(new_points, axis=0), axis=1)
        valid = old_spread > 2.0
        scale = float(np.median(new_spread[valid] / old_spread[valid])) if valid.any() else 1.0
        scale = float(np.clip(scale, 1 - self.MAX_SCALE_STEP, 1 + self.MAX_SCALE_STEP))
        dx, dy = float(shift[0]) / FLOW_SCALE, float(shift[1]) / FLOW_SCALE
        cx, cy = _center(self.box)
        width, height = _size(self.box)
        self.box = _box_from(cx + dx, cy + dy, width * scale, height * scale)
        self.moved = float(np.hypot(dx, dy))
        self._points = new_points.reshape(-1, 1, 2)
        return True


# --- filtro de Kalman -------------------------------------------------------

class MotionModel:
    """Kalman de velocidad constante sobre el centro de la caja.

    El tamaño se suaviza aparte (media móvil). Las mediciones del flujo óptico
    se consideran más ruidosas que las de YOLO.
    """

    FLOW_NOISE = 8.0            # px, desviación de la medición por flujo
    YOLO_NOISE = 3.0            # px, desviación de la medición por YOLO
    ACCELERATION = 300.0        # px/s², cambios de velocidad de una persona

    def __init__(self, box: Box, now: float) -> None:
        cx, cy = _center(box)
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([25.0, 25.0, 400.0, 400.0])
        self.width, self.height = _size(box)
        self.t = now

    def predict(self, now: float) -> None:
        dt = now - self.t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.ACCELERATION ** 2
        self.t = now

    def update(self, box: Box, now: float, noise: float, size_weight: float) -> None:
        self.predict(now)
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        z = np.array(_center(box))
        S = H @ self.P @ H.T + np.eye(2) * noise ** 2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        width, height = _size(box)
        self.width += size_weight * (width - self.width)
        self.height += size_weight * (height - self.height)

    def box(self) -> Box:
        return _box_from(self.x[0], self.x[1], self.width, self.height)

    def extrapolate(self, now: float, horizon: float = 1.5) -> Box:
        """Posición esperada sin modificar el estado. La velocidad se aplica
        como máximo ``horizon`` s: una persona oculta suele detenerse."""
        dt = min(max(now - self.t, 0.0), horizon)
        return _box_from(self.x[0] + self.x[2] * dt, self.x[1] + self.x[3] * dt,
                         self.width, self.height)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))


# --- apariencia clásica -----------------------------------------------------

def color_signature(frame: np.ndarray | None, box: Box) -> np.ndarray | None:
    """Histograma Hue-Saturation del torso, o None si no hay color útil.

    Se usa la franja central (20–60% de la altura, 60% del ancho) para evitar
    fondo, cabeza y piernas. Con IR, oscuridad o colores apagados casi ningún
    píxel tiene saturación y la apariencia no se usa.
    """
    if frame is None:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    tx1, tx2 = int(max(0, x1 + 0.2 * bw)), int(min(width, x2 - 0.2 * bw))
    ty1, ty2 = int(max(0, y1 + 0.2 * bh)), int(min(height, y1 + 0.6 * bh))
    if tx2 - tx1 < 6 or ty2 - ty1 < 6:
        return None
    hsv = cv2.cvtColor(frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
    if cv2.countNonZero(mask) < 0.15 * mask.size:
        return None
    histogram = cv2.calcHist([hsv], [0, 1], mask, [16, 8], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, 1.0, 0.0, cv2.NORM_L1)
    return histogram


def appearance_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """1 = mismo color, 0 = nada en común; None si alguno no tiene color."""
    if a is None or b is None:
        return None
    return 1.0 - float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def blend_signature(old: np.ndarray | None, new: np.ndarray | None,
                    weight: float = 0.3) -> np.ndarray | None:
    if new is None:
        return old
    if old is None:
        return new
    mixed = (1 - weight) * old + weight * new
    cv2.normalize(mixed, mixed, 1.0, 0.0, cv2.NORM_L1)
    return mixed


# --- puerta de movimiento ---------------------------------------------------

class MotionGate:
    """Diferencia de fotogramas a baja resolución.

    Indica si hay movimiento fuera de las cajas seguidas (posible persona
    nueva) y si las personas seguidas se mueven.
    """

    def __init__(self, threshold: int = 25, min_fraction: float = 0.004,
                 margin: float = 0.15) -> None:
        self.threshold = threshold
        self.min_fraction = min_fraction
        self.margin = margin
        self._previous: np.ndarray | None = None

    def update(self, gray: np.ndarray, boxes: list[Box]) -> tuple[bool, bool]:
        """Devuelve (movimiento fuera de las cajas, movimiento dentro)."""
        small = cv2.GaussianBlur(cv2.resize(gray, (160, 90),
                                            interpolation=cv2.INTER_AREA), (5, 5), 0)
        previous, self._previous = self._previous, small
        if previous is None:
            return False, False
        mask = cv2.absdiff(small, previous) > self.threshold
        inside = np.zeros_like(mask)
        sx = 160 / (gray.shape[1] / FLOW_SCALE)
        sy = 90 / (gray.shape[0] / FLOW_SCALE)
        for x1, y1, x2, y2 in boxes:
            mx, my = (x2 - x1) * self.margin, (y2 - y1) * self.margin
            inside[max(0, int((y1 - my) * sy)):int((y2 + my) * sy) + 1,
                   max(0, int((x1 - mx) * sx)):int((x2 + mx) * sx) + 1] = True
        minimum = self.min_fraction * mask.size
        outside_count = int((mask & ~inside).sum())
        inside_count = int((mask & inside).sum())
        return outside_count >= minimum, inside_count >= minimum / 2


# --- trayectorias -----------------------------------------------------------

@dataclass
class PersonTrack:
    camera: str
    code: str                     # legible: cam109-20261006-145731-1
    started_wall: datetime
    first_seen: float             # time.monotonic()
    last_seen: float              # última vez que YOLO la vio
    box: Box
    confidence: float
    reason: str                   # motivo de la ejecución de YOLO que la creó
    alarm_id: str | None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    # Cómo terminó: "borde" (salió del encuadre), "interior" (desapareció
    # dentro de la imagen: oclusión, oscuridad o falla) o "sesion".
    exit_kind: str | None = None
    hidden_since: float | None = None
    max_confidence: float = 0.0
    last_seen_wall: datetime | None = None
    yolo_hits: int = 1
    misses: int = 0
    flow: FlowTracker | None = None
    model: MotionModel | None = None
    signature: np.ndarray | None = None
    recoveries: int = 0
    recoveries_by_appearance: int = 0

    @property
    def visible_seconds(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


def match_score(track_box: Box, detection: Box, match_iou: float,
                center_gate: float) -> float:
    """Afinidad geométrica entre una trayectoria y una detección (0 = no).

    Prioriza la superposición; si no alcanza, acepta detecciones cercanas al
    centro de la caja (en tamaños de caja).
    """
    overlap = iou(track_box, detection)
    if overlap >= match_iou:
        return 1.0 + overlap
    tx, ty = _center(track_box)
    dx, dy = _center(detection)
    size = max(*_size(track_box))
    distance = float(np.hypot(tx - dx, ty - dy)) / size
    return max(0.0, 1.0 - distance / center_gate) if distance < center_gate else 0.0


def assign(score: np.ndarray) -> list[tuple[int, int]]:
    """Asignación óptima (húngara) que maximiza la afinidad total."""
    if score.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:                       # respaldo: voraz por afinidad
        pairs, used_rows, used_cols = [], set(), set()
        for flat in np.argsort(-score, axis=None):
            row, col = np.unravel_index(flat, score.shape)
            if score[row, col] <= 0:
                break
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            pairs.append((int(row), int(col)))
        return pairs
    rows, cols = linear_sum_assignment(-score)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if score[r, c] > 0]


class TrackManager:
    """Asocia las detecciones de YOLO con las trayectorias (estilo ByteTrack).

    1. Las detecciones fuertes se asignan a las trayectorias activas por
       posición predicha (Kalman), corregida por la apariencia.
    2. Las detecciones débiles solo mantienen vivas trayectorias activas que
       quedaron sin pareja; nunca crean personas.
    3. Las detecciones fuertes sobrantes intentan recuperar una trayectoria
       en gracia: primero por posición extrapolada; si una persona oculta en el
       interior reaparece lejos, por apariencia con un umbral estricto.
    4. Una trayectoria que YOLO deja de ver pasa a gracia. Nadie desaparece de
       una casa: solo se sale por un borde de la imagen. Si se perdió junto a
       un borde, "salió" tras ``grace_seconds``; si se perdió en el interior
       queda oculta (mueble, oscuridad, falla) y se espera
       ``hidden_seconds`` antes de registrar que "desapareció".
    """

    APPEARANCE_VETO = 0.30       # ropa claramente distinta: no es la misma
    APPEARANCE_RECOVERY = 0.80   # reaparición lejana: exige ropa muy parecida
    APPEARANCE_MARGIN = 0.10     # y que no haya otra candidata casi igual
    STRONG_OVERLAP = 0.6         # superposición que pesa más que el color
    EXIT_SPEED = 20.0            # px/s hacia el borde para considerar que sale
    MAX_GATE = 1.25              # radio máximo por posición (en tamaños de caja)

    def __init__(self, camera: str, match_iou: float = 0.25, center_gate: float = 0.75,
                 max_misses: int = 2, forget_seconds: float = 2.0,
                 grace_seconds: float = 10.0, hidden_seconds: float = 120.0,
                 edge_margin: float = 0.08) -> None:
        self.camera = camera
        self.alias = camera_alias(camera)
        self.match_iou = match_iou
        self.center_gate = center_gate
        self.max_misses = max_misses
        self.forget_seconds = forget_seconds
        self.grace_seconds = grace_seconds
        self.hidden_seconds = hidden_seconds
        self.edge_margin = edge_margin
        self.frame_size: tuple[int, int] | None = None   # ancho, alto en píxeles
        self.tracks: list[PersonTrack] = []     # activas (se siguen y dibujan)
        self.grace: list[PersonTrack] = []      # sin ver hace poco; recuperables
        # Recuperadas en la última llamada: (trayectoria, "posicion" o
        # "apariencia", segundos sin verse).
        self.last_recovered: list[tuple[PersonTrack, str, float]] = []
        self._codes_in_second: dict[str, int] = {}

    # --- utilidades ---------------------------------------------------------

    def near_edge(self, box: Box) -> bool:
        """¿La caja toca la franja del borde por donde se puede salir?"""
        if self.frame_size is None:
            return True
        width, height = self.frame_size
        mx, my = width * self.edge_margin, height * self.edge_margin
        x1, y1, x2, y2 = box
        return x1 <= mx or y1 <= my or x2 >= width - mx or y2 >= height - my

    def leaving(self, track: PersonTrack) -> bool:
        """¿Se perdió saliendo? Exige estar junto a un borde **y** moverse
        hacia él: quien está quieto junto al borde (p. ej. sentado en una zona
        oscura) no se fue, está oculto."""
        if self.frame_size is None:
            return True
        if not self.near_edge(track.box):
            return False
        if track.model is None:
            return True
        width, height = self.frame_size
        x1, y1, x2, y2 = track.box
        vx, vy = float(track.model.x[2]), float(track.model.x[3])
        outward = max(-vx if x1 <= width * self.edge_margin else 0.0,
                      vx if x2 >= width * (1 - self.edge_margin) else 0.0,
                      -vy if y1 <= height * self.edge_margin else 0.0,
                      vy if y2 >= height * (1 - self.edge_margin) else 0.0)
        return outward >= self.EXIT_SPEED

    def hidden(self) -> list[PersonTrack]:
        return [track for track in self.grace if track.exit_kind == "interior"]

    def new_code(self, moment: datetime) -> str:
        stamp = moment.strftime("%Y%m%d-%H%M%S")
        number = self._codes_in_second.get(stamp, 0) + 1
        self._codes_in_second = {stamp: number}   # solo importa el segundo actual
        return f"{self.alias}-{stamp}-{number}"

    # --- entre detecciones ---------------------------------------------------

    def flow_step(self, previous: np.ndarray, current: np.ndarray,
                  now: float | None = None) -> tuple[bool, bool]:
        """Mueve las cajas entre detecciones. Devuelve (alguna perdida, movimiento).

        Con flujo válido, Kalman lo usa como medición; sin flujo (perdida),
        la caja sigue la predicción de Kalman en vez de quedarse congelada.
        """
        any_lost, moving = False, False
        for track in self.tracks:
            flow_ok = track.flow is not None and track.flow.update(previous, current)
            if track.flow is not None and not flow_ok:
                any_lost = True
            if track.model is not None and now is not None:
                if flow_ok:
                    track.model.update(track.flow.box, now, MotionModel.FLOW_NOISE, 0.5)
                else:
                    track.model.predict(now)
                track.box = track.model.box()
                if flow_ok:
                    track.flow.box = track.box
                moving = moving or track.model.speed > 20.0
            elif flow_ok:
                track.box = track.flow.box
            if flow_ok:
                moving = moving or track.flow.moved > 1.0
        return any_lost, moving

    # --- con cada inferencia de YOLO ------------------------------------------

    def _expected_box(self, track: PersonTrack, now: float) -> Box:
        if track.model is not None and track.exit_kind is not None:
            return track.model.extrapolate(now)
        return track.box

    def _score(self, track: PersonTrack, box: Box, signature: np.ndarray | None,
               now: float, ambiguous: bool = True) -> float:
        gate = self.center_gate
        if track.exit_kind is not None:
            # La incertidumbre crece con el tiempo sin ver a la persona, con un
            # tope: más lejos solo se la recupera por apariencia.
            gate = min(gate * (1 + 0.25 * (now - track.last_seen)), self.MAX_GATE)
        geometric = match_score(self._expected_box(track, now), box, self.match_iou, gate)
        if geometric <= 0:
            return 0.0
        similarity = appearance_similarity(track.signature, signature)
        if similarity is None:
            return geometric
        if (ambiguous and similarity < self.APPEARANCE_VETO
                and geometric < 1.0 + self.STRONG_OVERLAP):
            # Ropa claramente distinta. Solo decide si hay con quién confundir
            # (cruces): con una sola persona, un torso parcial o en sombra
            # cambia de color sin cambiar de persona.
            return 0.0
        return geometric * (0.4 + 0.6 * similarity)

    def _match(self, tracks: list[PersonTrack], boxes: list[Box],
               signatures: list[np.ndarray | None], candidates: list[int],
               now: float) -> list[tuple[PersonTrack, int]]:
        if not tracks or not candidates:
            return []
        ambiguous = len(tracks) > 1 or len(candidates) > 1
        score = np.array([[self._score(track, boxes[d], signatures[d], now, ambiguous)
                           for d in candidates] for track in tracks])
        return [(tracks[row], candidates[col]) for row, col in assign(score)]

    def _refresh(self, track: PersonTrack, box: Box, confidence: float, now: float,
                 wall: datetime, gray: np.ndarray | None,
                 signature: np.ndarray | None) -> None:
        if track.model is None:
            track.model = MotionModel(box, now)
        elif track.exit_kind is not None and now - track.last_seen > 0.5:
            # Reaparición (estilo OC-SORT): la velocidad se reestima con la
            # última observación, no con la predicción acumulada a ciegas.
            dt = now - track.last_seen
            (lx, ly), (nx, ny) = _center(track.box), _center(box)
            track.model = MotionModel(box, now)
            track.model.x[2:] = [(nx - lx) / dt * 0.5, (ny - ly) / dt * 0.5]
        else:
            track.model.update(box, now, MotionModel.YOLO_NOISE, 0.7)
        track.exit_kind = None
        track.hidden_since = None
        track.box = box
        track.confidence = confidence
        track.max_confidence = max(track.max_confidence, confidence)
        track.last_seen = now
        track.last_seen_wall = wall
        track.yolo_hits += 1
        track.misses = 0
        track.signature = blend_signature(track.signature, signature)
        track.flow = FlowTracker(gray, box) if gray is not None else None

    def apply_detections(self, gray: np.ndarray | None, boxes: list[Box],
                         confidences: list[float], now: float, wall: datetime,
                         reason: str, alarm_id: str | None,
                         weak_boxes: list[Box] | None = None,
                         weak_confidences: list[float] | None = None,
                         frame: np.ndarray | None = None,
                         ) -> tuple[list[PersonTrack], list[PersonTrack]]:
        """Actualiza con una ejecución de YOLO. Devuelve (nuevas, terminadas)."""
        weak_boxes = weak_boxes or []
        weak_confidences = weak_confidences or []
        signatures = [color_signature(frame, box) for box in boxes]
        self.last_recovered = []
        matched: set[str] = set()
        remaining = list(range(len(boxes)))

        # 1. Fuertes con activas.
        for track, d in self._match(self.tracks, boxes, signatures, remaining, now):
            self._refresh(track, boxes[d], confidences[d], now, wall, gray, signatures[d])
            matched.add(track.id)
            remaining.remove(d)

        # 2. Débiles con activas sin pareja: solo las mantienen vivas.
        unmatched = [track for track in self.tracks if track.id not in matched]
        weak_signatures: list[np.ndarray | None] = [None] * len(weak_boxes)
        for track, d in self._match(unmatched, weak_boxes, weak_signatures,
                                    list(range(len(weak_boxes))), now):
            self._refresh(track, weak_boxes[d], weak_confidences[d], now, wall, gray,
                          None)
            matched.add(track.id)

        # 3a. Fuertes sobrantes: recuperar en gracia por posición extrapolada.
        for track, d in self._match(self.grace, boxes, signatures, list(remaining), now):
            self._recover(track, "posicion", boxes[d], confidences[d], now, wall, gray,
                          signatures[d])
            matched.add(track.id)
            remaining.remove(d)

        # 3b. Ocultas en el interior que reaparecen lejos: solo por apariencia.
        for d in list(remaining):
            candidates = sorted(
                ((appearance_similarity(track.signature, signatures[d]) or 0.0, i)
                 for i, track in enumerate(self.hidden())), reverse=True)
            if not candidates or candidates[0][0] < self.APPEARANCE_RECOVERY:
                continue
            if (len(candidates) > 1
                    and candidates[1][0] > candidates[0][0] - self.APPEARANCE_MARGIN):
                continue                  # ambiguo: dos ocultas con ropa parecida
            track = self.hidden()[candidates[0][1]]
            self._recover(track, "apariencia", boxes[d], confidences[d], now, wall,
                          gray, signatures[d])
            matched.add(track.id)
            remaining.remove(d)

        # 3c. Continuidad: las personas nuevas entran por los bordes. Una
        #     detección en el interior, habiendo alguien perdido o sin pareja,
        #     es esa persona (el flujo o YOLO la perdieron, p. ej. en la
        #     oscuridad), salvo ropa claramente distinta.
        for d in sorted(remaining, key=lambda index: -confidences[index]):
            if self.near_edge(boxes[d]):
                continue
            pool = ([track for track in self.tracks if track.id not in matched]
                    + self.grace)
            options = []
            for track in pool:
                similarity = appearance_similarity(track.signature, signatures[d])
                if similarity is not None and similarity < self.APPEARANCE_VETO:
                    continue
                (tx, ty), (dx, dy) = _center(self._expected_box(track, now)), _center(boxes[d])
                options.append((float(np.hypot(tx - dx, ty - dy)), track))
            if not options:
                continue
            track = min(options, key=lambda option: option[0])[1]
            if track in self.grace:
                self._recover(track, "continuidad", boxes[d], confidences[d], now, wall,
                              gray, signatures[d])
            else:
                unseen = now - track.last_seen
                self._refresh(track, boxes[d], confidences[d], now, wall, gray,
                              signatures[d])
                self.last_recovered.append((track, "continuidad", unseen))
            matched.add(track.id)
            remaining.remove(d)

        # 4. Activas sin ver: a gracia tras varias ausencias.
        for track in list(self.tracks):
            if track.id in matched:
                continue
            track.misses += 1
            if (track.misses >= self.max_misses
                    and now - track.last_seen >= self.forget_seconds):
                track.flow = None
                track.exit_kind = "borde" if self.leaving(track) else "interior"
                track.hidden_since = now
                self.tracks.remove(track)
                self.grace.append(track)

        # 5. Gracia vencida: salió por el borde o desapareció en el interior.
        ended = [track for track in self.grace
                 if now - track.last_seen >= (self.grace_seconds
                                              if track.exit_kind == "borde"
                                              else self.hidden_seconds)]
        self.grace = [track for track in self.grace if track not in ended]

        # 6. Fuertes que no son nadie conocido: persona nueva.
        created: list[PersonTrack] = []
        for d in remaining:
            track = PersonTrack(
                camera=self.camera, code=self.new_code(wall), started_wall=wall,
                first_seen=now, last_seen=now, box=boxes[d],
                confidence=confidences[d], reason=reason, alarm_id=alarm_id,
                max_confidence=confidences[d], last_seen_wall=wall,
                flow=FlowTracker(gray, boxes[d]) if gray is not None else None,
                model=MotionModel(boxes[d], now), signature=signatures[d])
            self.tracks.append(track)
            created.append(track)
        return created, ended

    def _recover(self, track: PersonTrack, how: str, box: Box, confidence: float,
                 now: float, wall: datetime, gray: np.ndarray | None,
                 signature: np.ndarray | None) -> None:
        unseen = now - track.last_seen
        self._refresh(track, box, confidence, now, wall, gray, signature)
        track.recoveries += 1
        if how == "apariencia":
            track.recoveries_by_appearance += 1
        self.grace.remove(track)
        self.tracks.append(track)
        self.last_recovered.append((track, how, unseen))

    def close_all(self) -> list[PersonTrack]:
        ended, self.tracks, self.grace = self.tracks + self.grace, [], []
        for track in ended:
            if track.exit_kind is None:
                track.exit_kind = "sesion"
        return ended


def yolo_reason(*, now: float, last_yolo: float, verdict_pending: bool,
                alarm_pending: bool, alarm_active: bool, has_tracks: bool,
                searching: bool = False,
                motion_outside: bool, track_lost: bool, tracks_moving: bool,
                confirm_interval: float = 0.2, trigger_interval: float = 0.5,
                moving_interval: float = 1.0, still_interval: float = 3.0,
                watch_interval: float = 1.0) -> str | None:
    """Decide si este fotograma necesita YOLO y por qué (None: basta el flujo).

    ``trigger_interval`` limita los disparos por movimiento o trayectoria
    perdida: una cortina que se mueve sin parar no debe ejecutar YOLO en cada
    fotograma. Es también la demora máxima para detectar a una persona nueva.
    """
    elapsed = now - last_yolo
    if alarm_pending:
        return REASON_ALARM
    if verdict_pending:
        return REASON_CONFIRM if elapsed >= confirm_interval else None
    if motion_outside and elapsed >= trigger_interval:
        return REASON_MOTION
    if track_lost and elapsed >= trigger_interval:
        return REASON_LOST
    if has_tracks:
        limit = moving_interval if tracks_moving else still_interval
        return REASON_INTERVAL if elapsed >= limit else None
    if alarm_active and elapsed >= watch_interval:
        return REASON_WATCH
    if searching and elapsed >= still_interval:
        return REASON_WATCH       # personas en gracia u ocultas: buscarlas
    return None
