# YOLO exportado: ONNX y NCNN frente a PyTorch

## 1. Qué se calibró

Si YOLO26n exportado a ONNX y NCNN (preparación para la Raspberry Pi) da las
mismas detecciones que PyTorch y cuánto CPU ahorra. Cámara 192.168.100.109
(W51-TY), substream 640×360.

## 2. Datos

- 2026-11-26, PC de desarrollo (Intel Core i7-8550U, 8 hilos, Windows 11),
  Python 3.12.7, Ultralytics 8.4.56, torch 2.9.1+cpu, ONNX Runtime 1.32.0,
  ncnn 1.0.20261020.
- Fidelidad: 27 fotos de `capturas/` (21 del substream 640×360 y 6
  redimensionadas a 640×360), confianza mínima 0,15.
  `python tools/exportar_yolo.py`
- Velocidad: cámara real, sala vacía, 30 inferencias por formato.
  `python tools/rendimiento.py --camera 192.168.100.109 --segundos 10 --rafagas 3 --inferencias 30`
- **No se midió en la Raspberry Pi.** Las cifras de este informe son del PC.

## 3. Resultados (medidos)

### Fidelidad de la exportación

| Comparación | Imágenes con distinto número de detecciones | Mayor Δ confianza |
|---|---|---|
| ONNX frente a PyTorch | 0 % | 0,000 |
| NCNN frente a PyTorch con la cabeza con NMS | 0 % | 0,000 |
| NCNN frente a PyTorch tal como se calibró | 22 % (320 px) / 33 % (416 px) | 0,215 / 0,277 |

- **ONNX conserva la cabeza sin NMS de YOLO26** (end2end): mismas
  detecciones que lo calibrado.
- **NCNN no la admite.** Ultralytics la desactiva al exportar ("NCNN export
  does not support end2end models") y usa la cabeza con NMS del mismo modelo.
  La exportación es fiel, pero las detecciones cambian: confianzas
  ~0,02–0,06 distintas en personas claras y detecciones débiles de más.
- Un modelo exportado tiene la **forma de entrada fija**. Exportado cuadrado
  (416×416), procesa más relleno que PyTorch, que adapta la entrada a la foto
  (416×256), y daba resultados distintos. Por eso se exporta con la forma
  16:9 del substream: 320 → 192×320, 416 → 256×416. Además, a un modelo
  exportado no hay que pasarle `imgsz`: Ultralytics lo cambiaría a cuadrado
  a partir de la segunda llamada.

### Espera activa de ONNX Runtime

10 inferencias separadas por 100 ms (como en una sesión):

| | CPU en las esperas |
|---|---|
| PyTorch con `OMP_WAIT_POLICY=PASSIVE` | 0 ms |
| NCNN | 0–31 ms |
| ONNX Runtime por defecto | **3.016 ms** (~3 núcleos girando) |
| ONNX Runtime con `allow_spinning = 0` | 16 ms |

### Velocidad (PC, cámara real)

| Formato | 320 px: mediana / CPU por inferencia | 416 px: mediana / CPU por inferencia |
|---|---|---|
| PyTorch | 81 ms / 143 ms | 138 ms / 235 ms |
| ONNX | 29 ms / 55 ms | 51 ms / 93 ms |
| NCNN | 76 ms / 86 ms | 80 ms / 132 ms |

Otros valores medidos en la misma prueba:
- Decodificar el substream (26 fps): 0,05 núcleos.
- Seguimiento: 2,9 ms de CPU por fotograma.
- Ráfaga HD: 1,16 s hasta el fotograma y 0,41 s de CPU.
- SFace: 47 ms de CPU por cara.
- RAM del proceso: ~450 MB (pico 608 MB).

En x86, ONNX es el más rápido. En ARM suele ganar NCNN: **falta medirlo en la Pi**.

## 4. Decisión

- `--yolo-formato auto` (por defecto): ONNX si está exportado para el tamaño,
  si no NCNN, si no PyTorch. Se prefiere ONNX porque no cambia la calibración.
- NCNN solo si en la Pi resulta claramente más rápido que ONNX, y revisando
  entonces `[yolo] confianza` y `confianza_debil`.
- ONNX Runtime sin espera activa (`src/detection/yolo.py`, `quiet_onnxruntime`).
- `tools/exportar_yolo.py` borra una exportación que no reproduce su
  referencia (Δ confianza > 0,01 o distinto número de detecciones en > 5 % de
  las fotos).

## 5. Cambio aplicado

Ninguno en `config/camdetector.toml`. El modelo usado queda en
`config_camara.datos` (`ia.modelo`, p. ej. `yolo26n_416.onnx`): pasar de
PyTorch a ONNX crea una nueva versión de configuración.

Pendiente: repetir `tools/rendimiento.py` en la Raspberry Pi 5 y decidir
formato, `yolo.tamano` y `rostro.activo` (`AAAA-MM-DD_pi_rendimiento.md`).
