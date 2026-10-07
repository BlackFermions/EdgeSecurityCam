"""Trayectorias de personas y su asociación con las detecciones (estilo
ByteTrack, con Kalman, asignación húngara y apariencia intercambiable)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.identity.base import AppearanceModel
from src.identity.color import ColorHistogramAppearance
from src.tracking.assignment import assign
from src.tracking.flow import FlowTracker
from src.tracking.geometry import Box, center, iou, size
from src.tracking.kalman import MotionModel


def camera_alias(camera: str) -> str:
    """``192.168.100.109`` → ``cam109``; otros nombres se usan tal cual."""
    last = camera.rsplit(".", 1)[-1]
    return f"cam{last}" if last.isdigit() else camera


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
    tx, ty = center(track_box)
    dx, dy = center(detection)
    box_size = max(*size(track_box))
    distance = float(np.hypot(tx - dx, ty - dy)) / box_size
    return max(0.0, 1.0 - distance / center_gate) if distance < center_gate else 0.0


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
                 edge_margin: float = 0.08,
                 appearance: AppearanceModel | None = None) -> None:
        self.camera = camera
        # Identidad corporal intercambiable: color del torso hoy, Re-ID mañana.
        self.appearance: AppearanceModel = appearance or ColorHistogramAppearance()
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
        similarity = self.appearance.similarity(track.signature, signature)
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
            (lx, ly), (nx, ny) = center(track.box), center(box)
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
        track.signature = self.appearance.blend(track.signature, signature)
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
        signatures = [self.appearance.signature(frame, box) for box in boxes]
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
                ((self.appearance.similarity(track.signature, signatures[d]) or 0.0, i)
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
                similarity = self.appearance.similarity(track.signature, signatures[d])
                if similarity is not None and similarity < self.APPEARANCE_VETO:
                    continue
                (tx, ty), (dx, dy) = center(self._expected_box(track, now)), center(boxes[d])
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
