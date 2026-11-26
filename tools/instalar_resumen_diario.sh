#!/usr/bin/env bash
# Instala solo el resumen periódico; no modifica el servicio de la cámara.
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
PY="$RAIZ/.venv/bin/python"
USUARIO="$(id -un)"

if [[ "$EUID" -eq 0 ]]; then
    echo "Ejecuta este script con tu usuario habitual, sin sudo delante de bash." >&2
    exit 1
fi
if [[ ! -x "$PY" ]]; then
    echo "No se encontró el entorno de Python: $PY" >&2
    exit 1
fi

# Verifica la base, la zona horaria y la escritura antes de instalar el timer.
"$PY" "$RAIZ/tools/resumen_diario.py" --recientes

# Autenticar antes de pasar el contenido de las unidades por la entrada estándar.
sudo -v
sudo tee /etc/systemd/system/camdetector-resumen.service >/dev/null <<EOF
[Unit]
Description=Resumen diario de CamDetector

[Service]
Type=oneshot
User=$USUARIO
WorkingDirectory=$RAIZ
ExecStart="$PY" "$RAIZ/tools/resumen_diario.py" --recientes
UMask=0077
EOF

sudo tee /etc/systemd/system/camdetector-resumen.timer >/dev/null <<'EOF'
[Unit]
Description=Actualizar el resumen de CamDetector cada cinco minutos

[Timer]
OnCalendar=*-*-* *:0/5:00
AccuracySec=1s
Persistent=true
Unit=camdetector-resumen.service

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now camdetector-resumen.timer
echo "Resumen configurado. Archivos: $RAIZ/logs/resumen_diario/AAAA-MM-DD.log"
systemctl list-timers camdetector-resumen.timer --no-pager
