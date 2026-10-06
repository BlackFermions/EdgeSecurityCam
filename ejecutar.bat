@echo off
setlocal
cd /d "%~dp0"

rem IP de la camara; sin parametro se usa esta. Uso: ejecutar.bat 192.168.100.109
set "CAMERA_HOST=192.168.100.109"
if not "%~1"=="" set "CAMERA_HOST=%~1"

if not exist ".venv\Scripts\python.exe" (
    echo Falta el entorno de Python. Ejecuta instalar.bat primero.
    pause
    exit /b 1
)

rem Evita que los hilos de inferencia consuman CPU mientras esperan.
if not defined KMP_BLOCKTIME set "KMP_BLOCKTIME=0"
if not defined OMP_WAIT_POLICY set "OMP_WAIT_POLICY=PASSIVE"

".venv\Scripts\python.exe" escuchar_alarmas.py ^
    --camera "%CAMERA_HOST%" ^
    --log-file "logs\camdetector.log"
