"""Recepción de alarmas de la cámara por eventos ONVIF (PullPoint).

La cámara detecta personas y movimiento en su propio procesador y publica el
resultado como eventos ONVIF. Este módulo mantiene una suscripción PullPoint,
la renueva, se recupera de cortes de red y convierte cada notificación en una
``CameraAlarm``. ``AlarmStateTracker`` reduce los eventos repetidos a
transiciones de inicio y fin.
"""

from __future__ import annotations

import base64
import hashlib
import os
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable


SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
NS = {
    "s": SOAP_NS,
    "tds": "http://www.onvif.org/ver10/device/wsdl",
    "tev": "http://www.onvif.org/ver10/events/wsdl",
    "tt": "http://www.onvif.org/ver10/schema",
    "wsnt": "http://docs.oasis-open.org/wsn/b-2",
    "wsa": "http://www.w3.org/2005/08/addressing",
}
WSSE_NS = ("http://docs.oasis-open.org/wss/2004/01/"
           "oasis-200401-wss-wssecurity-secext-1.0.xsd")
WSU_NS = ("http://docs.oasis-open.org/wss/2004/01/"
          "oasis-200401-wss-wssecurity-utility-1.0.xsd")
PULL_ACTION = ("http://www.onvif.org/ver10/events/wsdl/"
               "PullPointSubscription/PullMessagesRequest")
RENEW_ACTION = ("http://docs.oasis-open.org/wsn/bw-2/"
                "SubscriptionManager/RenewRequest")

# Temas conocidos de la Vatilon W51-TY. Se comparan sin prefijos de espacio
# de nombres (``tns1:``), por eso otras marcas con el mismo tema también sirven.
HUMAN_TOPICS = ("UserAlarm/IVA/HumanShapeDetect",)
MOTION_TOPICS = ("RuleEngine/CellMotionDetector/Motion", "VideoSource/MotionAlarm")
STATE_ITEMS = ("State", "IsMotion")

# Modelos cuyo firmware publica la detección de personas con el tema estándar
# de movimiento. Verificado en la W51-TY con Motion Detect apagado y Human
# Detect encendido: mover objetos no genera eventos, una persona sí.
MOTION_AS_HUMAN_MODELS = frozenset({"W51-TY"})
MOTION_MODES = ("auto", "human", "motion")

SUBSCRIPTION_SECONDS = 60
RENEW_EVERY_SECONDS = 30.0
PULL_TIMEOUT_SECONDS = 10
MAX_BACKOFF_SECONDS = 30.0


class CameraEventError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraAlarm:
    camera: str
    kind: str                 # "human", "motion" u "other"
    active: bool | None       # None si el evento no trae estado
    topic: str
    operation: str            # Initialized, Changed o Deleted
    received_at: datetime     # reloj del PC; el de la cámara puede estar mal
    camera_time: str | None
    items: dict[str, str] = field(default_factory=dict)


def normalize_topic(raw: str) -> str:
    """``tns1:UserAlarm/tnsvat:IVA/X`` → ``UserAlarm/IVA/X``."""
    return "/".join(part.split(":", 1)[-1] for part in raw.strip().split("/") if part)


def classify_topic(topic: str, motion_as_human: bool = False) -> str:
    if topic in HUMAN_TOPICS:
        return "human"
    if topic in MOTION_TOPICS:
        return "human" if motion_as_human else "motion"
    return "other"


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in ("true", "1", "on", "active"):
        return True
    if lowered in ("false", "0", "off", "inactive"):
        return False
    return None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_notifications(root: ET.Element, camera: str,
                        received_at: datetime | None = None,
                        motion_as_human: bool = False) -> list[CameraAlarm]:
    moment = received_at or datetime.now(timezone.utc).astimezone()
    alarms: list[CameraAlarm] = []
    for notification in root.iter(f"{{{NS['wsnt']}}}NotificationMessage"):
        topic_element = notification.find("wsnt:Topic", NS)
        topic = normalize_topic(topic_element.text or "") if topic_element is not None else ""
        message = next((e for e in notification.iter() if _local(e.tag) == "Message"
                        and "UtcTime" in e.attrib), None)
        items = {
            element.attrib["Name"]: element.attrib.get("Value", "")
            for element in notification.iter()
            if _local(element.tag) == "SimpleItem" and "Name" in element.attrib
        }
        state = next((items[name] for name in STATE_ITEMS if name in items), None)
        alarms.append(CameraAlarm(
            camera=camera,
            kind=classify_topic(topic, motion_as_human),
            active=_parse_bool(state),
            topic=topic,
            operation=(message.attrib.get("PropertyOperation", "") if message is not None
                       else ""),
            received_at=moment,
            camera_time=message.attrib.get("UtcTime") if message is not None else None,
            items=items,
        ))
    return alarms


@dataclass(frozen=True)
class AlarmTransition:
    camera: str
    kind: str
    active: bool
    at: datetime
    duration: float | None    # segundos activos; None si no se vio el inicio
    initial: bool             # estado informado al suscribirse (Initialized)


class AlarmStateTracker:
    """Convierte eventos repetidos en transiciones inicio/fin por cámara y tipo."""

    def __init__(self) -> None:
        self._active_since: dict[tuple[str, str], datetime] = {}
        self._known: set[tuple[str, str]] = set()

    def update(self, alarm: CameraAlarm) -> AlarmTransition | None:
        if alarm.active is None or alarm.kind == "other":
            return None
        key = (alarm.camera, alarm.kind)
        was_active = key in self._active_since
        first = key not in self._known
        self._known.add(key)
        initial = alarm.operation == "Initialized"
        if alarm.active and not was_active:
            self._active_since[key] = alarm.received_at
            return AlarmTransition(alarm.camera, alarm.kind, True, alarm.received_at,
                                   None, initial)
        if not alarm.active and was_active:
            started = self._active_since.pop(key)
            return AlarmTransition(alarm.camera, alarm.kind, False, alarm.received_at,
                                   (alarm.received_at - started).total_seconds(), initial)
        if first and not alarm.active:
            # Inactivo informado al suscribirse, o fin de una alarma que empezó
            # antes de la suscripción (sin duración conocida).
            return AlarmTransition(alarm.camera, alarm.kind, False, alarm.received_at,
                                   None, initial)
        return None

    def forget_camera(self, camera: str) -> None:
        """Tras perder la conexión el estado es desconocido; se vuelve a aprender."""
        for key in [key for key in self._known if key[0] == camera]:
            self._known.discard(key)
            self._active_since.pop(key, None)


def _security_header(username: str, password: str) -> str:
    nonce = os.urandom(16)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = base64.b64encode(
        hashlib.sha1(nonce + created.encode() + password.encode()).digest()
    ).decode()
    return (
        f'<wsse:Security xmlns:wsse="{WSSE_NS}" xmlns:wsu="{WSU_NS}">'
        f"<wsse:UsernameToken><wsse:Username>{username}</wsse:Username>"
        '<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/'
        'oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
        f"{digest}</wsse:Password>"
        f"<wsse:Nonce>{base64.b64encode(nonce).decode()}</wsse:Nonce>"
        f"<wsu:Created>{created}</wsu:Created></wsse:UsernameToken>"
        "</wsse:Security>"
    )


class OnvifEventClient:
    """Cliente SOAP mínimo; añade WS-Security solo si la cámara lo exige."""

    def __init__(self, host: str, username: str | None = None,
                 password: str | None = None, port: int = 80,
                 motion_mode: str = "auto") -> None:
        if motion_mode not in MOTION_MODES:
            raise ValueError("motion_mode debe ser auto, human o motion")
        self.host = host
        self.device_url = f"http://{host}:{port}/onvif/device_service"
        self.events_url = f"http://{host}:{port}/onvif/Events"
        self._username, self._password = username, password
        self._use_auth = False
        self.motion_mode = motion_mode
        self.model: str | None = None
        self.motion_detect_enabled: bool | None = None
        self.motion_as_human = motion_mode == "human"

    def call(self, url: str, body: str, header: str = "",
             action: str | None = None, timeout: float = 8.0) -> ET.Element:
        for attempt in range(2):
            security = ""
            if self._use_auth and self._username and self._password:
                security = _security_header(self._username, self._password)
            namespaces = " ".join(f'xmlns:{key}="{value}"' for key, value in NS.items())
            envelope = (f"<s:Envelope {namespaces}><s:Header>{header}{security}"
                        f"</s:Header><s:Body>{body}</s:Body></s:Envelope>")
            content_type = "application/soap+xml; charset=utf-8"
            if action:
                content_type += f'; action="{action}"'
            request = urllib.request.Request(
                url, data=envelope.encode("utf-8"), method="POST",
                headers={"Content-Type": content_type})
            can_retry = attempt == 0 and not self._use_auth and bool(self._username)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    root = ET.fromstring(response.read(2_000_000))
            except urllib.error.HTTPError as error:
                if can_retry and error.code in (400, 401, 500):
                    self._use_auth = True
                    continue
                raise CameraEventError(f"HTTP {error.code}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                raise CameraEventError("la cámara no respondió") from error
            except ET.ParseError as error:
                raise CameraEventError("respuesta ONVIF inválida") from error
            if root.find(".//s:Fault", NS) is None:
                return root
            if can_retry:
                self._use_auth = True
                continue
            raise CameraEventError("la cámara rechazó la solicitud ONVIF")
        raise CameraEventError("sin respuesta válida")

    def resolve_profile(self) -> None:
        """Consulta el modelo y decide cómo interpretar el tema de movimiento."""
        if self.model is not None:
            return
        try:
            root = self.call(self.device_url, "<tds:GetDeviceInformation/>")
        except CameraEventError:
            return  # se reintenta en la siguiente reconexión
        model = root.find(".//tds:Model", NS)
        self.model = (model.text or "").strip() if model is not None else ""
        self._update_interpretation()

    def set_profile(self, model: str | None, motion_detect_enabled: bool | None) -> bool:
        """Actualiza modelo y estado de Motion Detect (leídos de la web).

        Devuelve True si cambió la interpretación del tema de movimiento.
        """
        if model:
            self.model = model
        self.motion_detect_enabled = motion_detect_enabled
        return self._update_interpretation()

    def _update_interpretation(self) -> bool:
        if self.motion_mode != "auto":
            return False
        # Con Motion Detect activo el tema mezcla personas y cualquier
        # movimiento: solo es "persona" si Motion Detect está apagado.
        as_human = (self.model in MOTION_AS_HUMAN_MODELS
                    and self.motion_detect_enabled is False)
        changed = as_human != self.motion_as_human
        self.motion_as_human = as_human
        return changed

    def subscribe(self) -> str:
        root = self.call(
            self.events_url,
            "<tev:CreatePullPointSubscription><tev:InitialTerminationTime>"
            f"PT{SUBSCRIPTION_SECONDS}S</tev:InitialTerminationTime>"
            "</tev:CreatePullPointSubscription>")
        address = root.find(".//tev:SubscriptionReference/wsa:Address", NS)
        if address is None or not (address.text or "").strip():
            raise CameraEventError("la cámara no devolvió la suscripción")
        return address.text.strip()

    def pull(self, subscription: str) -> ET.Element:
        header = f"<wsa:Action>{PULL_ACTION}</wsa:Action><wsa:To>{subscription}</wsa:To>"
        return self.call(
            subscription,
            f"<tev:PullMessages><tev:Timeout>PT{PULL_TIMEOUT_SECONDS}S</tev:Timeout>"
            "<tev:MessageLimit>50</tev:MessageLimit></tev:PullMessages>",
            header=header, action=PULL_ACTION, timeout=PULL_TIMEOUT_SECONDS + 6)

    def renew(self, subscription: str) -> None:
        header = f"<wsa:Action>{RENEW_ACTION}</wsa:Action><wsa:To>{subscription}</wsa:To>"
        self.call(subscription,
                  f"<wsnt:Renew><wsnt:TerminationTime>PT{SUBSCRIPTION_SECONDS}S"
                  "</wsnt:TerminationTime></wsnt:Renew>",
                  header=header, action=RENEW_ACTION)

    def unsubscribe(self, subscription: str) -> None:
        self.call(subscription, "<wsnt:Unsubscribe/>",
                  header=f"<wsa:To>{subscription}</wsa:To>", timeout=3)


class CameraEventListener:
    """Hilo que mantiene viva la suscripción de una cámara y entrega alarmas.

    ``on_alarm`` recibe cada ``CameraAlarm``; ``on_status`` recibe mensajes de
    conexión legibles. Ambos se llaman desde el hilo del listener.
    """

    def __init__(self, client: OnvifEventClient,
                 on_alarm: Callable[[CameraAlarm], None],
                 on_status: Callable[[str, bool], None]) -> None:
        self.client = client
        self._on_alarm = on_alarm
        self._on_status = on_status
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"onvif-events-{client.host}", daemon=True)

    def start(self) -> "CameraEventListener":
        self._thread.start()
        return self

    def _run(self) -> None:
        backoff = 2.0
        connected = False
        while not self._stop.is_set():
            subscription = None
            try:
                first_profile = self.client.model is None
                self.client.resolve_profile()
                subscription = self.client.subscribe()
                if first_profile:
                    meaning = ("persona" if self.client.motion_as_human
                               else "movimiento")
                    self._on_status(
                        f"modelo {self.client.model or 'desconocido'}; el tema de "
                        f"movimiento se interpreta como {meaning}", True)
                self._on_status("suscrito a eventos ONVIF", True)
                connected = True
                backoff = 2.0
                renewed_at = time.monotonic()
                while not self._stop.is_set():
                    root = self.client.pull(subscription)
                    for alarm in parse_notifications(
                            root, self.client.host,
                            motion_as_human=self.client.motion_as_human):
                        self._on_alarm(alarm)
                    if time.monotonic() - renewed_at >= RENEW_EVERY_SECONDS:
                        try:
                            self.client.renew(subscription)
                        except CameraEventError:
                            # Muchas cámaras renuevan solas con cada PullMessages;
                            # si no fuera así, el siguiente pull fallará y se
                            # recreará la suscripción.
                            pass
                        renewed_at = time.monotonic()
            except CameraEventError as error:
                if connected:
                    self._on_status(f"conexión perdida ({error})", False)
                    connected = False
                else:
                    self._on_status(f"sin conexión ({error}); reintento en "
                                    f"{backoff:.0f} s", False)
            finally:
                if subscription is not None and self._stop.is_set():
                    try:
                        self.client.unsubscribe(subscription)
                    except CameraEventError:
                        pass
            self._stop.wait(backoff)
            backoff = min(MAX_BACKOFF_SECONDS, backoff * 2)

    def request_stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=PULL_TIMEOUT_SECONDS + 8)
