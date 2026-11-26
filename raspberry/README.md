# CamDetector en Raspberry Pi

Es el mismo programa que corre en el servidor Windows: el código es el mismo
para los dos. Esta carpeta solo tiene lo propio de Linux:

| Archivo | Para qué |
|---|---|
| `instalar.sh` | Instala todo en la Pi (equivale a `instalar.bat`) |
| `ejecutar.sh` | Inicia CamDetector a mano (equivale a `ejecutar.bat`) |
| `camdetector.service` | Arranque automático al encender y reinicio si falla |
| `requirements-pi.txt` | Motores extra para que YOLO vaya rápido en la Pi |

La configuración (`config/camdetector.toml`) es la misma que en Windows.

## Qué se necesita

- Raspberry Pi 5 (8 GB probada en el análisis) o Pi 4 de 4 GB o más.
- **Raspberry Pi OS Lite de 64 bits.** Raspberry Pi OS es Linux (Debian).
- SSD (HAT M.2 o por USB 3) mejor que tarjeta SD: la base de datos escribe a menudo.
- Ventilador: Active Cooler oficial o caja con ventilador. Sin él, la Pi baja su velocidad.
- Fuente oficial (27 W en la Pi 5) y cable de red en la misma red que la cámara.

## 1. Preparar el disco (desde el PC)

1. Instalar **Raspberry Pi Imager** (raspberrypi.com/software).
2. Elegir el modelo de Pi, **Raspberry Pi OS Lite (64-bit)** y el SSD o la tarjeta.
3. En "Editar ajustes": nombre del equipo (p. ej. `camdetector-sala`), usuario
   y contraseña, y **activar SSH**.
4. Grabar, conectar el disco a la Pi, conectar el cable de red y encender.

Desde el PC, entrar a la Pi con PowerShell:

```powershell
ssh usuario@camdetector-sala.local
```

## 2. Copiar el proyecto a la Pi

El código se baja de GitHub; los archivos privados se copian aparte (no están
en git a propósito).

En la Pi:

```bash
sudo apt-get install -y git
git clone -b Facerecognition https://github.com/BlackFermions/EdgeSecurityCam.git CamDetector
```

> Si el repositorio es privado, git pedirá usuario y un *token* de GitHub
> (Settings → Developer settings → Personal access tokens) en lugar de la contraseña.

Desde el PC (PowerShell, en la carpeta `CamDetector`), copiar los archivos privados:

```powershell
scp claves.txt usuario@camdetector-sala.local:CamDetector/
scp config\camdetector.toml usuario@camdetector-sala.local:CamDetector/config/
scp -r face_gallery usuario@camdetector-sala.local:CamDetector/
```

**Importante:** en `config/camdetector.toml` de la Pi, la galería debe ser
`galeria = "face_gallery"`. En el servidor Windows apuntaba a
`../camaracaptor new/face_gallery`, que en la Pi no existe. El instalador avisa si falta.

## 3. Instalar

En la Pi:

```bash
cd CamDetector
bash raspberry/instalar.sh --servicio --camara 192.168.100.109
```

Tarda unos 15–20 minutos y baja ~1 GB. Hace lo siguiente:

1. Comprueba que el sistema sea de 64 bits y que haya Python 3.11 o superior.
2. Instala lo necesario y crea el entorno `.venv`.
3. Descarga YOLO, YuNet y SFace.
4. Exporta YOLO a ONNX y NCNN y valida cada exportación.
5. Crea `config/camdetector.toml` y `claves.txt` si no existen.
6. Instala el servicio `camdetector`.

Si `claves.txt` todavía dice `CAMBIAR`, el servicio no arranca hasta que se edite:

```bash
nano claves.txt
sudo systemctl start camdetector
```

## 4. Comprobar que funciona

```bash
sudo systemctl status camdetector     # "active (running)"
tail -f logs/camdetector.log          # registro en vivo (Ctrl+C para salir)
```

Al arrancar, el registro dice qué modelo de YOLO usa, por ejemplo
`YOLO 416 px: yolo26n_416.onnx`.

## 5. Medir el rendimiento (una vez)

Con el servicio detenido y, si se puede, alguien delante de la cámara:

```bash
sudo systemctl stop camdetector
.venv/bin/python tools/rendimiento.py --camera 192.168.100.109
sudo systemctl start camdetector
```

Deja un informe en `logs/rendimiento_<fecha>.md` con lo que tarda YOLO en cada
formato, el seguimiento, las ráfagas HD y el rostro, más un veredicto sobre si
el rostro se puede encender. Con ese informe se ajustan los parámetros y se
documenta en `docs/calibracion/`.

## Uso diario

| Qué | Comando |
|---|---|
| Ver estado | `sudo systemctl status camdetector` |
| Reiniciar tras cambiar la configuración | `sudo systemctl restart camdetector` |
| Detener / iniciar | `sudo systemctl stop camdetector` / `start` |
| Registro | `tail -f logs/camdetector.log` (rota solo: 10 archivos de 5 MB) |
| Errores graves al arrancar | `journalctl -u camdetector -n 50` |
| Resumen de la base | `.venv/bin/python tools/resumen.py` |
| Actualizar el código | `git pull && sudo systemctl restart camdetector` |

Si una actualización cambia `requirements.txt`, volver a ejecutar
`bash raspberry/instalar.sh` (no toca la configuración ni la contraseña).

## Formato de YOLO

`escuchar_alarmas.py --yolo-formato` (por defecto `auto`):

| Formato | Detecciones | Velocidad |
|---|---|---|
| `pt` (PyTorch) | las calibradas | la más lenta |
| `onnx` | **idénticas** a PyTorch (verificado: diferencia 0,000) | rápida; la que usa `auto` primero |
| `ncnn` | cambian algo: Ultralytics exporta YOLO26 a NCNN con otra cabeza (con NMS) | suele ser la más rápida en ARM |

`auto` usa ONNX si está exportado para el tamaño configurado, si no NCNN, si
no PyTorch. Para forzar NCNN (si la prueba de rendimiento lo justifica),
añadir `--yolo-formato ncnn` en `raspberry/ejecutar.sh` y revisar la
calibración de `[yolo]`.

## Varias Pi

Cuando la primera funcione y esté calibrada, se puede clonar su disco a las
demás (Raspberry Pi Imager o `dd`). En cada copia solo cambian el nombre del
equipo, `claves.txt` y la IP de la cámara:

```bash
sudo hostnamectl set-hostname camdetector-otra
bash raspberry/instalar.sh --servicio --sin-exportar --camara 192.168.1.50
```

Con muchas Pi o actualizaciones frecuentes conviene pasar a Docker; el
instalador sirve de base para la imagen.

## Licencia

Ultralytics (YOLO) usa la licencia AGPL-3.0. Revisarla antes de vender el nodo.
