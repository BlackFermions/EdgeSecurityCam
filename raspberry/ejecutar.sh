#!/usr/bin/env bash
# Inicia CamDetector en Linux (equivalente a ejecutar.bat).
#
# Uso:  bash raspberry/ejecutar.sh                      # cámara de CAMDETECTOR_CAMARA
#       bash raspberry/ejecutar.sh 192.168.100.109      # otra cámara
#       bash raspberry/ejecutar.sh 192.168.100.109 --raw
#
# El servicio systemd (instalar.sh --servicio) usa este mismo archivo y fija
# CAMDETECTOR_CAMARA con la IP indicada al instalar.
set -euo pipefail
cd "$(dirname "$0")/.."

CAMARA="${CAMDETECTOR_CAMARA:-192.168.100.109}"
if [[ $# -gt 0 && "$1" != -* ]]; then
    CAMARA="$1"
    shift
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "Falta el entorno de Python. Ejecuta primero: bash raspberry/instalar.sh" >&2
    exit 1
fi

# Que los hilos de inferencia no consuman CPU mientras esperan.
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export PYTHONUNBUFFERED=1

exec .venv/bin/python escuchar_alarmas.py --camera "$CAMARA" \
    --log-file logs/camdetector.log "$@"
