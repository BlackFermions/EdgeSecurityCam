#!/usr/bin/env bash
# Instalador de CamDetector para Raspberry Pi OS de 64 bits (y otros Linux).
# Equivalente a instalar.bat. Se puede volver a ejecutar sin problema.
#
# Uso (desde la carpeta CamDetector):
#   bash raspberry/instalar.sh                                   # solo instalar
#   bash raspberry/instalar.sh --servicio --camara 192.168.100.109   # + arranque automático
#
# Opciones:
#   --servicio        instala y habilita el servicio systemd "camdetector"
#   --camara IP       cámara del servicio (por defecto 192.168.100.109)
#   --sin-apt         no instala paquetes del sistema (si ya están o no hay sudo)
#   --sin-exportar    no exporta YOLO a ONNX/NCNN (CamDetector usará PyTorch)
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RAIZ"

SERVICIO=0
CAMARA="192.168.100.109"
CON_APT=1
EXPORTAR=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --servicio) SERVICIO=1 ;;
        --camara) CAMARA="${2:?falta la IP después de --camara}"; shift ;;
        --sin-apt) CON_APT=0 ;;
        --sin-exportar) EXPORTAR=0 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Opción desconocida: $1 (ver --help)" >&2; exit 2 ;;
    esac
    shift
done

paso() { echo; echo "==> $*"; }
aviso() { echo "AVISO: $*" >&2; }

# --- 1. Equipo -------------------------------------------------------------------
paso "Comprobando el equipo"
ARQ="$(uname -m)"
case "$ARQ" in
    aarch64|x86_64) echo "Arquitectura: $ARQ" ;;
    armv7l|armv6l)
        echo "Sistema de 32 bits ($ARQ): PyTorch no existe para él." >&2
        echo "Instala Raspberry Pi OS de 64 bits (Raspberry Pi Imager → 64-bit)." >&2
        exit 1 ;;
    *) aviso "arquitectura $ARQ no probada; se intenta igual." ;;
esac
[[ -r /proc/device-tree/model ]] && echo "Equipo: $(tr -d '\0' < /proc/device-tree/model)"

if ! command -v python3 >/dev/null; then
    echo "No se encontró python3." >&2
    exit 1
fi
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    echo "Hace falta Python 3.11 o superior (hay $(python3 --version))." >&2
    exit 1
fi
echo "$(python3 --version)"

# --- 2. Paquetes del sistema -----------------------------------------------------
if [[ $CON_APT -eq 1 ]] && command -v apt-get >/dev/null; then
    paso "Paquetes del sistema (pide la contraseña de sudo)"
    # opencv-python necesita libGL y GLib aunque no se abra ninguna ventana.
    GLIB=libglib2.0-0
    apt-cache show libglib2.0-0t64 >/dev/null 2>&1 && GLIB=libglib2.0-0t64
    sudo apt-get update
    sudo apt-get install -y python3-venv python3-pip libgl1 "$GLIB"
fi

# --- 3. Entorno de Python y dependencias ---------------------------------------------
paso "Entorno de Python (.venv)"
[[ -x .venv/bin/python ]] || python3 -m venv .venv
PY="$RAIZ/.venv/bin/python"
"$PY" -m pip install --upgrade pip

if [[ "$ARQ" == "x86_64" ]]; then
    # En PC con Linux, PyPI trae PyTorch con CUDA (~3 GB): se usa la versión CPU.
    paso "PyTorch para CPU (x86_64)"
    "$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

paso "Dependencias de CamDetector (requirements.txt; ~1 GB la primera vez)"
"$PY" -m pip install -r requirements.txt

paso "Motores para YOLO exportado (raspberry/requirements-pi.txt)"
FALLARON=()
while read -r linea; do
    linea="${linea%%#*}"
    linea="$(echo "$linea" | xargs)"
    [[ -z "$linea" ]] && continue
    if ! "$PY" -m pip install "$linea"; then
        FALLARON+=("$linea")
    fi
done < raspberry/requirements-pi.txt
if [[ ${#FALLARON[@]} -gt 0 ]]; then
    aviso "no se pudieron instalar: ${FALLARON[*]}. CamDetector funciona igual" \
          "con los formatos que sí estén (o con PyTorch)."
fi

# --- 4. Modelos ------------------------------------------------------------------------
paso "Modelos: YOLO, YuNet y SFace"
mkdir -p models
(cd models && "$PY" -c "from ultralytics import YOLO; YOLO('yolo26n.pt')")
"$PY" -c "
from pathlib import Path
from src.faces.models import ensure_models
ensure_models(Path('models'))
print('YuNet y SFace listos.')"

if [[ $EXPORTAR -eq 1 ]]; then
    paso "Exportando YOLO a ONNX y NCNN (320 y 416 px) y validando"
    if ! "$PY" tools/exportar_yolo.py --tamanos 320 416; then
        aviso "alguna exportación falló o no pasó la validación (ver arriba)." \
              "CamDetector usa PyTorch en esos tamaños. Alternativa: exportar en el PC" \
              "con 'python tools/exportar_yolo.py' y copiar los modelos de models/."
    fi
fi

# --- 5. Configuración y carpetas ----------------------------------------------------
paso "Configuración"
mkdir -p logs data capturas
if [[ ! -f config/camdetector.toml ]]; then
    cp config/camdetector.example.toml config/camdetector.toml
    echo "Se creó config/camdetector.toml: parámetros por cámara (ver docs/parametros.md)."
fi
if [[ ! -f claves.txt ]]; then
    cp claves.example.txt claves.txt
    echo "Se creó claves.txt: escribe el usuario y la contraseña de la cámara."
fi
chmod 600 claves.txt       # contraseña de la cámara: solo este usuario

# La galería de rostros debe existir en ESTE equipo (en Windows apuntaba a
# "../camaracaptor new/face_gallery").
"$PY" - <<'EOF'
import tomllib
from pathlib import Path

config = tomllib.loads(Path("config/camdetector.toml").read_text(encoding="utf-8"))
groups = [config.get("defecto", {})] + list(config.get("camaras", {}).values())
for group in groups:
    folder = group.get("rostro", {}).get("galeria")
    if folder and not Path(folder).resolve().is_dir():     # relativa a CamDetector
        print(f"AVISO: la galería '{folder}' de config/camdetector.toml no existe en "
              "este equipo. Copia face_gallery/ a CamDetector y usa galeria = \"face_gallery\".")
EOF

# --- 6. Servicio -------------------------------------------------------------------------
if [[ $SERVICIO -eq 1 ]]; then
    paso "Servicio systemd 'camdetector' (cámara $CAMARA)"
    USUARIO="$(id -un)"
    sed -e "s|@USUARIO@|$USUARIO|g" -e "s|@RAIZ@|$RAIZ|g" -e "s|@CAMARA@|$CAMARA|g" \
        raspberry/camdetector.service | sudo tee /etc/systemd/system/camdetector.service >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable camdetector
    if grep -q "CAMBIAR" claves.txt; then
        echo "Servicio habilitado, pero NO iniciado: falta la contraseña en claves.txt."
        echo "Edítala (nano claves.txt) y luego: sudo systemctl start camdetector"
    else
        sudo systemctl restart camdetector
        echo "Servicio iniciado. Estado: sudo systemctl status camdetector"
    fi
fi

paso "Instalación terminada"
echo "Probar a mano:          bash raspberry/ejecutar.sh $CAMARA"
echo "Medir el rendimiento:   .venv/bin/python tools/rendimiento.py --camera $CAMARA"
[[ $SERVICIO -eq 1 ]] && echo "Registro:               tail -f logs/camdetector.log"
exit 0
