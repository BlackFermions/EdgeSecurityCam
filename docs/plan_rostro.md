# Plan: reconocimiento facial de bajo consumo

Objetivo: identificar a cada persona **una vez por trayectoria**, con la cara
tomada de la foto HD bajo demanda, en lugar de reconocer en cada fotograma como
CameraCaptor. Cada paso: código + pruebas → herramienta de medición →
medición con la cámara real → parámetros por cámara → informe en
`docs/calibracion/` → commit.

## Mediciones de partida (F1, 2026-10-07, PC de desarrollo, CPU)

| Operación | Substream 640×360 | HD 2880×1616 |
|---|---|---|
| YuNet en la caja entera | 1,5–4,9 ms | 20–70 ms |
| YuNet en la franja superior del 30 % | 0,7–1,6 ms | 4–15 ms |
| SFace (alinear + huella) | ~36 ms por cara | igual |
| Decodificar foto HD (1,2 MB) | — | 29 ms (19 ms a ½) |
| Tamaño de la cara | ~18 px (inservible) | 81–98 px |

Conclusiones: SFace es lo caro (limitar cuántas veces corre); el substream no
sirve para reconocer a esta distancia (la cara HD bajo demanda es necesaria);
la franja del 30 % ahorra 4–5× en HD. Con la galería actual (webcam/WhatsApp):
Brayan 0,43 (rechazado con umbral 0,50), Diego 0,52 (aceptado), dos
desconocidos 0,22–0,24 (rechazados).

## Pasos

| Paso | Contenido | Estado |
|---|---|---|
| 0 | Parámetros por cámara (`config/camdetector.toml`, `docs/parametros.md`) | ✅ hecho 2026-10-13 |
| 1 | `faces/`: interfaz `FaceDetector`, YuNet, calidad de la cara (tamaño, frontalidad, nitidez, brillo) | ✅ hecho 2026-10-13 (herramienta `tools/rostros.py`: pendiente) |
| 2 | Cara en HD: `sources/hd.py` (ráfaga del vídeo principal), coordenadas substream→HD, franja de la cabeza | ✅ hecho 2026-10-13 (puerta YuNet en el substream: pendiente) |
| 3 | `FaceRecognizer` (SFace) + galería con varias huellas por persona (+ promedio) y caché | ✅ hecho 2026-10-13 (herramienta `tools/galeria.py`: pendiente) |
| 4 | Planificador del rostro (`analysis/face_policy.py`): estados, votos, desconocida, espera creciente sin cara, reverificación | ✅ hecho 2026-10-13 (regla del 70 % para cajas unidas: pendiente; 2 caras en la franja ya se marcan como unión) |
| 5 | Integración: trabajador aparte (`analysis/face_worker.py`), base v4 (`rostro_intento`, identidad en `persona_track`), registro, `--ver` | ✅ hecho 2026-10-13 |
| 6 | Galería con capturas HD de esta cámara (día e IR) | pendiente |
| 7 | Calibración del umbral por cámara (`tools/calibrar_rostro.py`) | pendiente |

## Hallazgos del 2026-10-13 (pasos 1–5)

- **La foto ONVIF de la W51-TY es solo 640×360** en todos los perfiles y su
  resolución no se puede cambiar (`snapshot_res`: `res_cnt = 0`). La cara se
  toma del vídeo principal `/stream1` (2880×1616) en **ráfagas**: ~1,1 s hasta
  el primer fotograma y ~0,35 s de CPU; tenerlo abierto costaría ~91 % de un
  núcleo.
- **Una ráfaga sirve a todas las personas pendientes** (una apertura del HD por
  turno, no una por persona).
- En la sala de reuniones la gente **casi nunca mira a la cámara** (de
  espaldas, mirando la laptop o el celular): la mayoría de intentos son
  `sin_cara`. Por eso **la espera entre intentos se duplica sin cara** (1,5 →
  3 → 6 … hasta 30 s): antes se abría el HD cada ~3 s sin parar.
- **Franja de la cabeza en personas sentadas o cortadas por el borde:** el
  30 % del alto quedaba por debajo de los ojos; ahora la franja mide al menos
  el 50 % del ancho de la persona (`franja_min_ancho`).
- **Primera identificación en vivo:** Brayan, similitud 0,61, en 9 intentos
  (3 coincidencias, 3 bajo umbral, 2 de mala calidad, 6 sin cara en total en
  la sesión). Ráfaga HD ~1,1 s; análisis del rostro 23–68 ms.

## Reglas acordadas

- Reconocer una vez por trayectoria; reintentar solo ante duda (reaparición,
  cruce) o cada `reverificar_cada_s`.
- YuNet en la franja superior del 30 % de la caja, a todo el ancho; en la caja
  entera solo si se une una trayectoria con altura < 70 % de la caja unida o
  la caja es demasiado ancha para una persona. Dos caras en la caja: no se
  asigna nombre hasta que se separen.
- Desconocidos: "Persona N" (numeración diaria por cámara) + código interno
  `camXXX-AAAAMMDD-HHMMSS-n`.
- Nombres y caras solo en el nodo; al NOC, un seudónimo.
