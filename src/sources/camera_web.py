"""Lectura de la configuración de la interfaz web de la cámara (web.cgi).

Solo consulta: sirve para revisar cómo están configuradas las detecciones
(``pd`` = Human Detect, ``md`` = Motion Detect) sin modificar nada.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request


class CameraWebError(RuntimeError):
    pass


class CameraWeb:
    def __init__(self, host: str, username: str, password: str) -> None:
        self.host = host
        self.username = username
        self.password = password
        self.session_id: str | None = None

    def _request(self, query: dict[str, str],
                 headers: dict[str, str] | None = None) -> tuple[dict[str, object], object]:
        url = f"http://{self.host}/cgi-bin/web.cgi?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=6) as response:
                body = response.read(500_000)
                response_headers = response.headers
        except (urllib.error.URLError, TimeoutError) as error:
            raise CameraWebError("la interfaz web de la cámara no respondió") from error
        try:
            result = json.loads(body)
        except ValueError:
            # El firmware responde texto plano en algunos errores, p. ej.
            # "param num error" cuando sobra un parámetro.
            raise CameraWebError(f"respuesta no JSON: {body[:80]!r}") from None
        if not isinstance(result, dict):
            raise CameraWebError("respuesta web inválida")
        return result, response_headers

    def login(self) -> None:
        info, _ = self._request({"mod": "session", "cmd": "get_auth_info"})
        realm, nonce, qop = (info.get(key) for key in ("realm", "nonce", "qop"))
        if not all(isinstance(item, str) and item for item in (realm, nonce, qop)):
            raise CameraWebError("autenticación web no compatible")
        cnonce = secrets.token_hex(8)
        md5 = lambda value: hashlib.md5(value.encode("utf-8")).hexdigest()
        uri = "/cgi-bin/web.cgi?mod=account&cmd=check"
        ha1 = md5(f"{self.username}:{realm}:{self.password}")
        ha2 = md5(f"GET:{uri}")
        digest = md5(f"{ha1}:{nonce}:00000001:{cnonce}:{qop}:{ha2}")
        auth = (f'Digest username="{self.username}",realm="{realm}",nonce="{nonce}",'
                f'uri="{uri}",cnonce="{cnonce}",nc=00000001,qop="{qop}",'
                f'response="{digest}"')
        result, headers = self._request(
            {"mod": "session", "cmd": "login1"}, headers={"Authorization": auth})
        session_id = headers.get("Session-Id")
        if result.get("status") != "ok" or not session_id:
            raise CameraWebError("credenciales web rechazadas")
        self.session_id = session_id

    def get(self, mod: str) -> dict[str, object]:
        """Lee un módulo de configuración: ``pd``, ``md``, ``device``..."""
        for attempt in range(2):
            if self.session_id is None:
                self.login()
            result, _ = self._request({"mod": mod, "cmd": "get"},
                                      headers={"Session-Id": self.session_id or ""})
            if result.get("status") == "expired" and attempt == 0:
                self.session_id = None
                continue
            break
        if result.get("status") in ("error", "expired"):
            raise CameraWebError(f"la cámara rechazó la consulta {mod}")
        return result
