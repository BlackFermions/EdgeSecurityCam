@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 goto usar_py

where python >nul 2>nul
if errorlevel 1 (
    echo No se encontro Python. Instala Python 3.10 o superior y vuelve a ejecutar este archivo.
    pause
    exit /b 1
)
python -m venv .venv
goto entorno_creado

:usar_py
py -3 -m venv .venv

:entorno_creado
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

rem Descarga y prueba el modelo YOLO una sola vez.
if not exist models mkdir models
pushd models
"..\.venv\Scripts\python.exe" -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
if errorlevel 1 (
    popd
    exit /b 1
)
popd

if not exist claves.txt (
    copy claves.example.txt claves.txt >nul
    echo Se creo claves.txt: escribe el usuario y la contrasena de la camara.
)

echo.
echo Instalacion terminada. Ejecuta ejecutar.bat para iniciar CamDetector.
pause
