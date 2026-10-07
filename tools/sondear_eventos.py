"""Diagnóstico de solo lectura: ¿la cámara publica eventos de movimiento/persona?

Uso (desde la carpeta del proyecto):
    .\\.venv\\Scripts\\python tools\\sondear_eventos.py 192.168.1.51 --seconds 60

Pasos:
  1. ONVIF GetCapabilities: ¿existe el servicio de eventos/analítica?
  2. ONVIF GetEventProperties: lista de temas que la cámara declara.
  3. ONVIF PullPoint: escucha eventos reales; camina frente a la cámara.
  4. Interfaz web (web.cgi): consulta módulos de alarma conocidos.
No modifica ninguna configuración de la cámara.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.credentials import read_credentials  # noqa: E402
from src.sources.camera_web import CameraWeb, CameraWebError  # noqa: E402


SOAP = "http://www.w3.org/2003/05/soap-envelope"
NS = {
    "s": SOAP,
    "tds": "http://www.onvif.org/ver10/device/wsdl",
    "tev": "http://www.onvif.org/ver10/events/wsdl",
    "tt": "http://www.onvif.org/ver10/schema",
    "wsnt": "http://docs.oasis-open.org/wsn/b-2",
    "wstop": "http://docs.oasis-open.org/wsn/t-1",
    "wsa": "http://www.w3.org/2005/08/addressing",
}
WSSE = ("http://docs.oasis-open.org/wss/2004/01/"
        "oasis-200401-wss-wssecurity-secext-1.0.xsd")
WSU = ("http://docs.oasis-open.org/wss/2004/01/"
       "oasis-200401-wss-wssecurity-utility-1.0.xsd")
PULL_ACTION = ("http://www.onvif.org/ver10/events/wsdl/"
               "PullPointSubscription/PullMessagesRequest")
INTERESTING = ("motion", "people", "person", "human", "humanoid", "object",
               "field", "line", "intrusion", "vehicle", "face", "pir")
DETECTION_MODULES = {"pd": "Human Detect", "md": "Motion Detect"}
DETECTION_FIELDS = ("enable", "sensitivity", "threshold", "duration", "rect_num",
                    "show_human", "output")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _security_header(username: str, password: str) -> str:
    nonce = os.urandom(16)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = base64.b64encode(
        hashlib.sha1(nonce + created.encode() + password.encode()).digest()
    ).decode()
    return (
        f'<wsse:Security xmlns:wsse="{WSSE}" xmlns:wsu="{WSU}">'
        f"<wsse:UsernameToken><wsse:Username>{username}</wsse:Username>"
        '<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/'
        'oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
        f"{digest}</wsse:Password>"
        f"<wsse:Nonce>{base64.b64encode(nonce).decode()}</wsse:Nonce>"
        f"<wsu:Created>{created}</wsu:Created></wsse:UsernameToken>"
        "</wsse:Security>"
    )


class Onvif:
    def __init__(self, host: str, username: str, password: str) -> None:
        self.host, self.username, self.password = host, username, password
        self.use_auth = False

    def call(self, url: str, body: str, extra_header: str = "",
             action: str | None = None, timeout: float = 8.0) -> ET.Element:
        for attempt in range(2):
            header = extra_header + (
                _security_header(self.username, self.password) if self.use_auth else ""
            )
            namespaces = " ".join(f'xmlns:{k}="{v}"' for k, v in NS.items())
            envelope = (f"<s:Envelope {namespaces}><s:Header>{header}</s:Header>"
                        f"<s:Body>{body}</s:Body></s:Envelope>")
            content_type = "application/soap+xml; charset=utf-8"
            if action:
                content_type += f'; action="{action}"'
            request = urllib.request.Request(
                url, data=envelope.encode(), method="POST",
                headers={"Content-Type": content_type})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    root = ET.fromstring(response.read(2_000_000))
            except urllib.error.HTTPError as error:
                if not self.use_auth and attempt == 0 and error.code in (400, 401, 500):
                    self.use_auth = True
                    continue
                raise RuntimeError(f"HTTP {error.code}") from None
            fault = root.find(".//s:Fault", NS)
            if fault is None:
                return root
            if not self.use_auth and attempt == 0:
                self.use_auth = True
                continue
            reason = " ".join(t.strip() for t in fault.itertext() if t.strip())
            raise RuntimeError(f"SOAP Fault: {reason[:200]}")
        raise RuntimeError("sin respuesta válida")


def step_capabilities(onvif: Onvif) -> str | None:
    print("\n[1] Servicios ONVIF")
    root = onvif.call(f"http://{onvif.host}/onvif/device_service",
                      "<tds:GetCapabilities><tds:Category>All</tds:Category>"
                      "</tds:GetCapabilities>")
    events_url = None
    for element in root.iter():
        name = _local(element.tag)
        if name in ("Events", "Analytics", "Media", "PTZ", "Device"):
            xaddr = next((c.text for c in element if _local(c.tag) == "XAddr"), None)
            if xaddr:
                print(f"  {name:<10} {xaddr}")
                if name == "Events":
                    events_url = xaddr
    if events_url is None:
        print("  La cámara NO declara servicio de eventos ONVIF.")
    print(f"  Autenticación WS-Security: {'sí' if onvif.use_auth else 'no requerida'}")
    return events_url


def _topic_paths(element: ET.Element, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for child in element:
        name = _local(child.tag)
        if name in ("MessageDescription", "Documentation"):
            continue
        path = f"{prefix}/{name}" if prefix else name
        if child.attrib.get(f"{{{NS['wstop']}}}topic") == "true":
            items = [i.attrib.get("Name", "?") for i in child.iter()
                     if _local(i.tag) in ("SimpleItemDescription",)]
            paths.append(f"{path}  {items}" if items else path)
        paths.extend(_topic_paths(child, path))
    return paths


def step_topics(onvif: Onvif, events_url: str) -> None:
    print("\n[2] Temas de eventos declarados")
    root = onvif.call(events_url, "<tev:GetEventProperties/>")
    topic_set = root.find(".//wstop:TopicSet", NS)
    paths = _topic_paths(topic_set) if topic_set is not None else []
    if not paths:
        print("  Ningún tema declarado.")
    for path in paths:
        mark = "  <==" if any(w in path.lower() for w in INTERESTING) else ""
        print(f"  {path}{mark}")


def step_pull(onvif: Onvif, events_url: str, seconds: int) -> None:
    print(f"\n[3] Escuchando eventos reales durante {seconds} s "
          "(camina frente a la cámara, luego quédate quieto)...")
    root = onvif.call(
        events_url,
        "<tev:CreatePullPointSubscription>"
        "<tev:InitialTerminationTime>PT120S</tev:InitialTerminationTime>"
        "</tev:CreatePullPointSubscription>")
    address = root.find(".//tev:SubscriptionReference/wsa:Address", NS)
    if address is None or not address.text:
        print("  La cámara no devolvió dirección de suscripción.")
        return
    url = address.text.strip()
    header = (f"<wsa:Action>{PULL_ACTION}</wsa:Action>"
              f"<wsa:To>{url}</wsa:To>")
    deadline = time.monotonic() + seconds
    count = 0
    try:
        while time.monotonic() < deadline:
            result = onvif.call(
                url,
                "<tev:PullMessages><tev:Timeout>PT5S</tev:Timeout>"
                "<tev:MessageLimit>20</tev:MessageLimit></tev:PullMessages>",
                extra_header=header, action=PULL_ACTION, timeout=12)
            for message in result.findall(".//wsnt:NotificationMessage", NS):
                topic = message.find("wsnt:Topic", NS)
                topic_text = (topic.text or "").strip() if topic is not None else "?"
                items = {i.attrib.get("Name"): i.attrib.get("Value")
                         for i in message.iter() if _local(i.tag) == "SimpleItem"}
                count += 1
                print(f"  {datetime.now():%H:%M:%S}  {topic_text}  {items}")
    finally:
        try:
            onvif.call(url, "<wsnt:Unsubscribe/>",
                       extra_header=f"<wsa:To>{url}</wsa:To>")
        except RuntimeError:
            pass
    print(f"  Eventos recibidos: {count}")


def step_web(host: str, username: str, password: str) -> None:
    print("\n[4] Configuración de detección (web.cgi)")
    web = CameraWeb(host, username, password)
    try:
        device = web.get("device")
    except CameraWebError as error:
        print(f"  No se pudo consultar la web: {error}")
        return
    print(f"  Modelo {device.get('devtype')} · firmware {device.get('version')}")
    for module, label in DETECTION_MODULES.items():
        try:
            data = web.get(module)
        except CameraWebError as error:
            print(f"  {label}: {error}")
            continue
        fields = {name: data[name] for name in DETECTION_FIELDS if name in data}
        state = "ACTIVADO" if data.get("enable") else "desactivado"
        print(f"  {label:<14} {state:<12} {fields}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("host")
    parser.add_argument("--credentials-file", type=Path,
                        default=Path(__file__).resolve().parents[1] / "claves.txt")
    parser.add_argument("--seconds", type=int, default=60)
    args = parser.parse_args()
    username, password = read_credentials(args.credentials_file)
    onvif = Onvif(args.host, username, password)
    events_url = None
    try:
        events_url = step_capabilities(onvif)
        if events_url:
            step_topics(onvif, events_url)
            step_pull(onvif, events_url, args.seconds)
    except (RuntimeError, OSError) as error:
        print(f"  ONVIF falló: {error}")
    step_web(args.host, username, password)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
