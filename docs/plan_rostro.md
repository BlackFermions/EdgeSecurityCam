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
| 1 | `faces/`: interfaz `FaceDetector`, YuNet, calidad de la cara (tamaño, frontalidad, nitidez, brillo) + `tools/rostros.py medir` | pendiente |
| 2 | Cara desde la foto HD (perfil HD, coordenadas, recorte de la franja) + puerta YuNet en el substream + `tools/rostros.py hd` | pendiente |
| 3 | `FaceRecognizer` (SFace) + galería con varias huellas por persona y caché + `tools/galeria.py` | pendiente |
| 4 | Planificador del rostro (`analysis/face_policy.py`): estados, votos, desconocida, presupuesto, unión de cajas (regla del 70 %) | pendiente |
| 5 | Integración: trabajador aparte, base v4 (`rostro_intento`, identidad en `persona_track`), registro, `--ver` | pendiente |
| 6 | Galería con capturas HD de esta cámara (día e IR) | pendiente |
| 7 | Calibración del umbral por cámara (`tools/calibrar_rostro.py`) | pendiente |

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
