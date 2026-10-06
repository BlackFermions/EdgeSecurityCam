"""Descubrimiento ONVIF de la cámara en todas las interfaces IPv4 locales."""

from __future__ import annotations

import select
import socket
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import urlsplit


WS_DISCOVERY_ADDRESS = ("239.255.255.250", 3702)
DEFAULT_CAMERA_MAC = "E8:B7:23:47:95:27"


@dataclass(frozen=True)
class DiscoveredCamera:
    host: str
    endpoint: str
    scopes: str
    local_address: str


def _compact(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def local_ipv4_addresses() -> tuple[str, ...]:
    """Devuelve direcciones utilizables sin depender de herramientas del sistema."""
    addresses: set[str] = set()
    try:
        records = socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET, socket.SOCK_DGRAM
        )
    except socket.gaierror:
        records = []
    for record in records:
        address = record[4][0]
        if not address.startswith(("127.", "169.254.")):
            addresses.add(address)
    return tuple(sorted(addresses))


def _probe_message() -> bytes:
    message_id = uuid.uuid4()
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
        'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
        'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
        f'<e:Header><w:MessageID>uuid:{message_id}</w:MessageID>'
        '<w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
        '<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>'
        '</e:Header><e:Body><d:Probe>'
        '<d:Types>dn:NetworkVideoTransmitter</d:Types>'
        '</d:Probe></e:Body></e:Envelope>'
    ).encode("utf-8")


def _text(element: ET.Element, local_name: str) -> str:
    for child in element.iter():
        if child.tag.rsplit("}", 1)[-1] == local_name and child.text:
            return child.text.strip()
    return ""


def parse_probe_matches(payload: bytes, local_address: str) -> tuple[DiscoveredCamera, ...]:
    """Extrae hosts ONVIF de una respuesta WS-Discovery."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return ()
    matches: list[DiscoveredCamera] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "ProbeMatch":
            continue
        xaddrs = _text(element, "XAddrs")
        endpoint = _text(element, "Address")
        scopes = _text(element, "Scopes")
        for address in xaddrs.split():
            host = urlsplit(address).hostname
            if host:
                matches.append(DiscoveredCamera(host, endpoint, scopes, local_address))
    return tuple(matches)


def _matches_mac(camera: DiscoveredCamera, mac_address: str) -> bool:
    needle = _compact(mac_address)
    haystack = _compact(f"{camera.endpoint} {camera.scopes}")
    return bool(needle and needle in haystack)


def discover_onvif(
    target_mac: str = DEFAULT_CAMERA_MAC,
    timeout: float = 2.0,
    local_addresses: tuple[str, ...] | None = None,
) -> tuple[DiscoveredCamera, ...]:
    """Emite una sonda por cada interfaz y prioriza la MAC solicitada."""
    addresses = local_addresses if local_addresses is not None else local_ipv4_addresses()
    sockets: list[socket.socket] = []
    results: dict[tuple[str, str], DiscoveredCamera] = {}
    probe = _probe_message()
    try:
        for address in addresses:
            connection: socket.socket | None = None
            try:
                connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                connection.setsockopt(
                    socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(address)
                )
                connection.bind((address, 0))
                connection.setblocking(False)
                connection.sendto(probe, WS_DISCOVERY_ADDRESS)
                sockets.append(connection)
            except OSError:
                if connection is not None:
                    connection.close()
        deadline = time.monotonic() + max(0.2, timeout)
        while sockets and time.monotonic() < deadline:
            readable, _, _ = select.select(
                sockets, [], [], max(0.0, deadline - time.monotonic())
            )
            if not readable:
                break
            for connection in readable:
                try:
                    payload, _ = connection.recvfrom(65_535)
                except (BlockingIOError, OSError):
                    continue
                local_address = connection.getsockname()[0]
                for camera in parse_probe_matches(payload, local_address):
                    results[(camera.host, camera.endpoint)] = camera
                    if _matches_mac(camera, target_mac):
                        return (camera,)
    finally:
        for connection in sockets:
            connection.close()
    return tuple(results.values())


def host_is_reachable(host: str, timeout: float = 0.35) -> bool:
    """Comprueba rápidamente los servicios característicos de la cámara."""
    for port in (554, 80):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def resolve_camera_host(
    preferred_host: str,
    target_mac: str = DEFAULT_CAMERA_MAC,
    timeout: float = 2.0,
) -> tuple[str, str]:
    """Devuelve host y origen: IP configurada, ONVIF por MAC o respaldo."""
    cameras = discover_onvif(target_mac=target_mac, timeout=timeout)
    for camera in cameras:
        if _matches_mac(camera, target_mac):
            return camera.host, f"ONVIF por {camera.local_address}"
    if len(cameras) == 1:
        camera = cameras[0]
        return camera.host, f"única cámara ONVIF por {camera.local_address}"
    if preferred_host and host_is_reachable(preferred_host):
        return preferred_host, "configurada"
    return preferred_host, "respaldo sin respuesta"
