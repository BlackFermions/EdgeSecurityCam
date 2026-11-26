#!/usr/bin/env bash
# Monitor independiente: no reinicia ni modifica el servicio de la cámara.
set -euo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
PY="$RAIZ/.venv/bin/python"
USUARIO="$(id -un)"

if [[ "$EUID" -eq 0 ]]; then
    echo "Ejecuta con tu usuario habitual, sin sudo delante de bash." >&2
    exit 1
fi
[[ -x "$PY" ]] || { echo "No se encontró el entorno: $PY" >&2; exit 1; }

# Instalar únicamente la dependencia del monitor, conservando PyTorch CPU.
"$PY" -m pip install 'psutil>=5.9,<8'
"$PY" "$RAIZ/tools/monitor_raspberry.py" --help >/dev/null
if ! systemctl is-active --quiet camdetector-monitor.service; then
    "$PY" "$RAIZ/tools/monitor_raspberry.py" --una-vez
fi
sudo -v
sudo tee /etc/systemd/system/camdetector-monitor.service >/dev/null <<EOF
[Unit]
Description=Registro diario de rendimiento de la Raspberry
After=local-fs.target

[Service]
Type=simple
User=$USUARIO
WorkingDirectory=$RAIZ
ExecStart="$PY" -u "$RAIZ/tools/monitor_raspberry.py"
Restart=on-failure
RestartSec=10
TimeoutStopSec=20
Nice=10
UMask=0077

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable camdetector-monitor.service
sudo systemctl restart camdetector-monitor.service
echo "Monitor instalado. Archivos: $RAIZ/logs/rendimiento/AAAA-MM-DD.csv y .log"
systemctl status camdetector-monitor.service --no-pager
