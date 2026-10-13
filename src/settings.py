"""Parámetros del nodo, configurables por cámara.

Orden de prioridad (el último gana):

1. valores por defecto de este módulo;
2. sección ``[defecto.<grupo>]`` del archivo ``config/camdetector.toml``;
3. sección ``[camaras."<ip>".<grupo>]`` de ese archivo;
4. opciones de la línea de comandos (se aplican a todas las cámaras).

Cada parámetro declara rango, unidad y descripción: con eso se valida el
archivo (un nombre mal escrito es un error, no se ignora) y se genera
``docs/parametros.md`` (``python tools/parametros.py --markdown``).
"""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Param:
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    unit: str = ""
    description: str = ""
    choices: tuple | None = None
    calibrate: str = ""            # cómo se calibra


# grupo -> nombre -> definición. Los nombres son los del archivo TOML.
SCHEMA: dict[str, dict[str, Param]] = {
    "camara": {
        "config_cada_s": Param(60.0, 10, 3600, "s", "Cada cuánto se relee la configuración web de la cámara."),
        "modo_movimiento": Param("auto", choices=("auto", "human", "motion"),
                                 description="Cómo interpretar el tema de movimiento ONVIF: auto según modelo y Motion Detect."),
    },
    "yolo": {
        "confianza": Param(0.40, 0.05, 0.95, "", "Confianza mínima de una detección fuerte (cuenta personas, confirma alarmas).",
                           calibrate="Falsas detecciones frente a personas perdidas en las fotos de las alarmas."),
        "confianza_debil": Param(0.15, 0.01, 0.90, "", "Detecciones débiles: solo mantienen viva una trayectoria existente."),
        "tamano": Param(416, choices=(320, 416, 480, 640), unit="px",
                        description="Tamaño de entrada de YOLO. Mayor = personas lejanas, más CPU.",
                        calibrate="Subir si la cámara ve personas pequeñas o lejanas."),
    },
    "veredicto": {
        "confirmar_fotogramas": Param(2, 1, 10, "", "Inferencias seguidas con persona para confirmar una alarma."),
        "decidir_s": Param(4.0, 1, 30, "s", "Tiempo sin ver a nadie para descartar una alarma."),
        "decidir_con_persona_s": Param(8.0, 1, 60, "s", "Tiempo máximo para decidir si YOLO ya vio a alguien una vez."),
    },
    "politica": {
        "confirmacion_s": Param(0.2, 0.05, 2, "s", "Intervalo de YOLO mientras una alarma no tiene veredicto."),
        "disparo_min_s": Param(0.5, 0.1, 5, "s", "Mínimo entre disparos por movimiento nuevo o trayectoria perdida (= demora máxima para ver a alguien nuevo)."),
        "movimiento_s": Param(1.0, 0.2, 10, "s", "Máximo sin YOLO con personas en movimiento."),
        "quieto_s": Param(3.0, 0.5, 30, "s", "Máximo sin YOLO con personas quietas; también búsqueda de ocultas."),
        "vigilancia_s": Param(1.0, 0.2, 10, "s", "Intervalo de YOLO con la alarma activa y nadie seguido."),
    },
    "seguimiento": {
        "iou_min": Param(0.25, 0.05, 0.9, "", "Superposición mínima para asociar por cajas."),
        "cercania": Param(0.75, 0.1, 3, "tamaños de caja", "Distancia máxima entre centros para asociar sin superposición."),
        "radio_max": Param(1.25, 0.3, 5, "tamaños de caja", "Radio máximo de búsqueda por posición de alguien sin ver."),
        "ausencias_max": Param(2, 1, 20, "inferencias", "Inferencias sin ver a alguien antes de pasarlo a gracia."),
        "olvidar_s": Param(2.0, 0.5, 30, "s", "Tiempo mínimo sin ver a alguien antes de pasarlo a gracia."),
        "gracia_s": Param(10.0, 1, 120, "s", "Espera antes de registrar que salió por un borde."),
        "ocultas_s": Param(120.0, 5, 3600, "s", "Espera antes de registrar que DESAPARECIÓ en el interior."),
        "margen_borde": Param(0.08, 0.0, 0.4, "fracción", "Franja de cada borde por donde se puede salir.",
                              calibrate="Ajustar a la geometría de la escena (puertas cerca de un borde)."),
        "salida_px_s": Param(20.0, 0, 500, "px/s", "Velocidad hacia el borde para considerar que la persona sale."),
        "debil_max_s": Param(5.0, 0.5, 60, "s",
                             "Tiempo máximo que solo detecciones débiles sostienen una trayectoria (evita que quede 'pegada' a una silla u objeto)."),
        "duplicado_contencion": Param(0.70, 0.3, 1.0, "fracción",
                                      "Una caja nueva contenida en esta fracción dentro de una persona ya seguida, y de altura parecida (≥ 70 %), es un duplicado, no otra persona."),
    },
    "apariencia": {
        "veto": Param(0.30, 0.0, 1.0, "", "Similitud de color bajo la cual dos personas se consideran distintas (solo con ambigüedad)."),
        "recuperacion": Param(0.80, 0.0, 1.0, "", "Similitud de color para recuperar lejos a alguien oculto."),
        "margen": Param(0.10, 0.0, 1.0, "", "Diferencia mínima con la segunda candidata para recuperar por color."),
        "superposicion_fuerte": Param(0.6, 0.0, 1.0, "", "Superposición que pesa más que el color."),
    },
    "puerta": {
        "umbral": Param(25, 1, 255, "niveles de gris", "Cambio de brillo que cuenta como movimiento."),
        "fraccion_min": Param(0.004, 0.0001, 0.5, "fracción", "Área mínima cambiada para considerar movimiento.",
                              calibrate="Subir si la puerta dispara YOLO por ruido (IR, compresión)."),
        "margen": Param(0.15, 0.0, 1.0, "fracción", "Margen alrededor de cada persona que no cuenta como movimiento nuevo."),
    },
    "sesion": {
        "procesar_cada_s": Param(0.1, 0.03, 1.0, "s", "Intervalo de procesamiento (flujo óptico y puerta)."),
        "max_s": Param(900.0, 30, 7200, "s", "Duración máxima de una sesión de análisis."),
    },
    "rostro": {
        "activo": Param(False, description="Reconocimiento facial (requiere galería). Apagado hasta calibrar cada cámara."),
        "galeria": Param("face_gallery", description="Carpeta de la galería: una subcarpeta con fotos por persona (relativa a CamDetector).",
                         unit="ruta"),
        "hd_ruta": Param("/stream1", description="Ruta RTSP del vídeo principal (HD) del que se toma la cara.", unit="ruta",
                         calibrate="La foto ONVIF de la W51-TY es solo 640x360: la cara sale del vídeo HD en ráfagas."),
        "franja_cabeza": Param(0.30, 0.1, 1.0, "fracción", "Parte superior de la caja de la persona donde se busca la cara.",
                               calibrate="¿Caen dentro las caras de personas sentadas y de pie? (F1: sí en HD)."),
        "franja_min_ancho": Param(0.50, 0.0, 2.0, "fracción del ancho",
                                  "Altura mínima de la franja de la cabeza respecto al ancho de la persona (sentados o cortados por el borde)."),
        "yunet_puntuacion": Param(0.80, 0.3, 0.99, "", "Confianza mínima de YuNet para aceptar una cara.",
                                  calibrate="Bajar si se pierden caras inclinadas; vigilar caras falsas en rostro_intento."),
        "cara_min_px": Param(48, 16, 400, "px", "Ancho mínimo de la cara (en HD) para intentar reconocerla.",
                             calibrate="Desde qué tamaño la similitud de una misma persona se mantiene alta."),
        "frontal_min": Param(0.50, 0.0, 1.0, "", "Frontalidad mínima (1 = de frente) según la simetría de ojos y nariz."),
        "nitidez_min": Param(20.0, 0, 10000, "var. Laplaciano", "Nitidez mínima de la cara (descarta movidas o desenfocadas)."),
        "brillo_min": Param(40.0, 0, 255, "nivel medio", "Brillo medio mínimo de la cara (descarta caras en oscuridad)."),
        "umbral": Param(0.45, 0.1, 0.95, "similitud", "Similitud mínima con la galería para dar un nombre.",
                        calibrate="Distribución misma persona frente a distinta (F1: Brayan 0,43 con galería de webcam, desconocidos 0,22–0,24)."),
        "margen_segundo": Param(0.08, 0.0, 0.5, "similitud", "Diferencia mínima con la segunda persona más parecida."),
        "umbral_seguro": Param(0.60, 0.1, 1.0, "similitud",
                               "Una sola coincidencia por encima de este valor identifica sin esperar más votos.",
                               calibrate="2026-10-13: Brayan 0,57–0,70 frente a 0,21–0,32 de otras personas."),
        "votos": Param(2, 1, 10, "coincidencias", "Coincidencias con la misma persona para identificar."),
        "intentos_max": Param(5, 1, 50, "intentos", "Intentos con cara válida sin coincidencia antes de marcar 'desconocida'."),
        "espera_entre_intentos_s": Param(1.5, 0.2, 60, "s", "Tiempo mínimo entre intentos para la misma persona."),
        "espera_max_s": Param(30.0, 1, 600, "s",
                              "Sin cara (de espaldas, mirando abajo) la espera se duplica en cada intento hasta este máximo: evita abrir el vídeo HD sin parar.",
                              calibrate="Cada ráfaga HD cuesta ~0,4 s de CPU: subir si las personas pasan mucho tiempo sin mirar a la cámara."),
        "reverificar_cada_s": Param(60.0, 5, 3600, "s", "Volver a verificar a una persona identificada."),
        "persona_min_alto_px": Param(60, 10, 1000, "px", "Altura mínima de la persona en el substream para intentar (lejos = sin cara útil)."),
        "presupuesto_sface_s": Param(3.0, 0.2, 50, "por segundo", "Máximo de reconocimientos SFace por segundo en la cámara."),
        "guardar_recortes": Param(True, description="Guardar el recorte de cada cara intentada (capturas/rostros) para calibrar."),
    },
    "reposo": {
        "cada_s": Param(10.0, 0, 600, "s", "Verificación en reposo: un fotograma cada N s aunque la cámara no avise (0 = desactivada)."),
        "confianza_min": Param(0.50, 0.05, 0.95, "", "Confianza de YOLO para abrir sesión desde reposo."),
        "espera_s": Param(60.0, 0, 3600, "s", "Espera tras un disparo en reposo antes de volver a disparar."),
    },
}


class SettingsError(ValueError):
    pass


def _validate(group: str, name: str, value: Any) -> Any:
    try:
        spec = SCHEMA[group][name]
    except KeyError:
        known = ", ".join(SCHEMA.get(group, {})) or ", ".join(SCHEMA)
        raise SettingsError(f"parámetro desconocido '{group}.{name}' (válidos: {known})") from None
    if spec.choices is not None:
        if value not in spec.choices:
            raise SettingsError(f"{group}.{name} = {value!r}: debe ser uno de {spec.choices}")
        return value
    if isinstance(spec.default, bool):
        if not isinstance(value, bool):
            raise SettingsError(f"{group}.{name} = {value!r}: debe ser true o false")
        return value
    if isinstance(spec.default, str):
        if not isinstance(value, str) or not value.strip():
            raise SettingsError(f"{group}.{name} = {value!r}: debe ser un texto no vacío")
        return value
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SettingsError(f"{group}.{name} = {value!r}: debe ser un número")
    if isinstance(spec.default, int) and not float(value).is_integer():
        raise SettingsError(f"{group}.{name} = {value!r}: debe ser entero")
    if group == "reposo" and name == "cada_s" and value != 0 and value < 3:
        raise SettingsError("reposo.cada_s debe ser 0 (desactivada) o al menos 3")
    if spec.minimum is not None and value < spec.minimum:
        raise SettingsError(f"{group}.{name} = {value}: mínimo {spec.minimum}")
    if spec.maximum is not None and value > spec.maximum:
        raise SettingsError(f"{group}.{name} = {value}: máximo {spec.maximum}")
    return type(spec.default)(value)


class CameraSettings:
    """Parámetros efectivos de una cámara: ``settings.get("yolo", "confianza")``."""

    def __init__(self, camera: str, name: str | None = None) -> None:
        self.camera = camera
        self.name = name or camera
        self._values = {group: {key: spec.default for key, spec in params.items()}
                        for group, params in SCHEMA.items()}
        self.sources: dict[str, str] = {}      # "grupo.nombre" -> de dónde salió

    def get(self, group: str, name: str) -> Any:
        return self._values[group][name]

    def section(self, group: str) -> dict[str, Any]:
        return dict(self._values[group])

    def set(self, group: str, name: str, value: Any, source: str) -> None:
        self._values[group][name] = _validate(group, name, value)
        self.sources[f"{group}.{name}"] = source

    def check(self) -> None:
        """Reglas entre parámetros."""
        if self.get("yolo", "confianza_debil") > self.get("yolo", "confianza"):
            raise SettingsError("yolo.confianza_debil no puede superar yolo.confianza")
        if self.get("veredicto", "decidir_con_persona_s") < self.get("veredicto", "decidir_s"):
            raise SettingsError("veredicto.decidir_con_persona_s debe ser >= veredicto.decidir_s")

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._values)


def load_settings(path: Path | None, cameras: list[str],
                  overrides: dict[tuple[str, str], Any] | None = None
                  ) -> dict[str, CameraSettings]:
    """Lee el TOML (si existe) y devuelve los parámetros efectivos por cámara."""
    data: dict[str, Any] = {}
    if path is not None and path.exists():
        try:
            with path.open("rb") as handle:
                data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as error:
            raise SettingsError(f"{path}: TOML inválido ({error})") from None
    unknown = set(data) - {"defecto", "camaras"}
    if unknown:
        raise SettingsError(f"secciones desconocidas: {sorted(unknown)} (válidas: defecto, camaras)")

    def apply(target: CameraSettings, groups: dict[str, Any], source: str) -> None:
        for group, values in groups.items():
            if group == "nombre":
                continue
            if group not in SCHEMA:
                raise SettingsError(f"[{source}] grupo desconocido '{group}' "
                                    f"(válidos: {', '.join(SCHEMA)})")
            if not isinstance(values, dict):
                raise SettingsError(f"[{source}] '{group}' debe ser una sección")
            for name, value in values.items():
                target.set(group, name, value, source)

    defaults = data.get("defecto", {})
    per_camera = data.get("camaras", {})
    result: dict[str, CameraSettings] = {}
    for camera in cameras:
        own = per_camera.get(camera, {})
        settings = CameraSettings(camera, own.get("nombre"))
        apply(settings, defaults, "defecto")
        apply(settings, own, f'camaras."{camera}"')
        for (group, name), value in (overrides or {}).items():
            settings.set(group, name, value, "línea de comandos")
        settings.check()
        result[camera] = settings
    return result


def _span(spec: Param) -> str:
    if spec.choices is not None:
        return " / ".join(str(choice) for choice in spec.choices)
    if isinstance(spec.default, bool):
        return "true / false"
    if isinstance(spec.default, str):
        return "texto"
    return f"{spec.minimum}–{spec.maximum}"


def markdown_table() -> str:
    """Documentación de todos los parámetros (docs/parametros.md)."""
    lines = []
    for group, params in SCHEMA.items():
        lines.append(f"\n### `[{group}]`\n")
        lines.append("| Parámetro | Defecto | Rango | Unidad | Qué hace | Cómo calibrar |")
        lines.append("|---|---|---|---|---|---|")
        for name, spec in params.items():
            span = _span(spec)
            lines.append(f"| `{name}` | {spec.default} | {span} | {spec.unit} | "
                         f"{spec.description} | {spec.calibrate} |")
    return "\n".join(lines) + "\n"
