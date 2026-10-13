# Parámetros de CamDetector

Todos los valores ajustables del nodo. Se configuran **por cámara** en
`config/camdetector.toml` (plantilla: `config/camdetector.example.toml`).

## Prioridad

1. Valor por defecto (columna *Defecto*).
2. `[defecto.<grupo>]`: común a todas las cámaras.
3. `[camaras."<ip>".<grupo>]`: solo esa cámara.
4. Opción de línea de comandos (todas las cámaras): `--yolo-size`,
   `--yolo-confidence`, `--idle-check-seconds`, `--idle-min-confidence`,
   `--config-interval`, `--motion-mode`.

Un nombre o grupo mal escrito, un valor fuera de rango o de tipo incorrecto
detiene el arranque con un mensaje claro.

## Trazabilidad

Los parámetros efectivos de cada cámara se guardan en `config_camara.datos`
(clave `ia.parametros`). Cambiar un parámetro crea una nueva versión de
configuración: cada alarma y cada persona quedan ligadas a los valores con que
se procesaron, lo que permite comparar calibraciones con datos.

## Comprobar un cambio antes de reiniciar

```powershell
.\.venv\Scripts\python tools\parametros.py --camera 192.168.100.109
```

Muestra cada valor y, si no es el de por defecto, de dónde sale.

## Tabla de parámetros

Generada desde `src/settings.py` (`python tools/parametros.py --markdown`).
Parámetros internos de los algoritmos (ruido del filtro de Kalman, puntos del
flujo óptico) no se exponen: dependen del algoritmo, no de la cámara.

### `[camara]`

| Parámetro | Defecto | Rango | Unidad | Qué hace | Cómo calibrar |
|---|---|---|---|---|---|
| `config_cada_s` | 60.0 | 10–3600 | s | Cada cuánto se relee la configuración web de la cámara. |  |
| `modo_movimiento` | auto | auto / human / motion |  | Cómo interpretar el tema de movimiento ONVIF: auto según modelo y Motion Detect. |  |

### `[yolo]`

| Parámetro | Defecto | Rango | Unidad | Qué hace | Cómo calibrar |
|---|---|---|---|---|---|
| `confianza` | 0.4 | 0.05–0.95 |  | Confianza mínima de una detección fuerte (cuenta personas, confirma alarmas). | Falsas detecciones frente a personas perdidas en las fotos de las alarmas. |
| `confianza_debil` | 0.15 | 0.01–0.9 |  | Detecciones débiles: solo mantienen viva una trayectoria existente. |  |
| `tamano` | 416 | 320 / 416 / 480 / 640 | px | Tamaño de entrada de YOLO. Mayor = personas lejanas, más CPU. | Subir si la cámara ve personas pequeñas o lejanas. |

### `[veredicto]`

| Parámetro | Defecto | Rango | Unidad | Qué hace | Cómo calibrar |
|---|---|---|---|---|---|
| `confirmar_fotogramas` | 2 | 1–10 |  | Inferencias seguidas con persona para confirmar una alarma. |  |
| `decidir_s` | 4.0 | 1–30 | s | Tiempo sin ver a nadie para descartar una alarma. |  |
| `decidir_con_persona_s` | 8.0 | 1–60 | s | Tiempo máximo para decidir si YOLO ya vio a alguien una vez. |  |

### `[politica]`

| Parámetro | Defecto | Rango | Unidad | Qué hace | Cómo calibrar |
|---|---|---|---|---|---|
| `confirmacion_s` | 0.2 | 0.05–2 | s | Intervalo de YOLO mientras una alarma no tiene veredicto. |  |
| `disparo_min_s` | 0.5 | 0.1–5 | s | Mínimo entre disparos por movimiento nuevo o trayectoria perdida (= demora máxima para ver a alguien nuevo). |  |
| `movimiento_s` | 1.0 | 0.2–10 | s | Máximo sin YOLO con personas en movimiento. |  |
| `quieto_s` | 3.0 | 0.5–30 | s | Máximo sin YOLO con personas quietas; también búsqueda de ocultas. |  |
| `vigilancia_s` | 1.0 | 0.2–10 | s | Intervalo de YOLO con la alarma activa y nadie seguido. |  |

### `[seguimiento]`

| Parámetro | Defecto | Rango | Unidad | Qué hace | Cómo calibrar |
|---|---|---|---|---|---|
| `iou_min` | 0.25 | 0.05–0.9 |  | Superposición mínima para asociar por cajas. |  |
| `cercania` | 0.75 | 0.1–3 | tamaños de caja | Distancia máxima entre centros para asociar sin superposición. |  |
| `radio_max` | 1.25 | 0.3–5 | tamaños de caja | Radio máximo de búsqueda por posición de alguien sin ver. |  |
| `ausencias_max` | 2 | 1–20 | inferencias | Inferencias sin ver a alguien antes de pasarlo a gracia. |  |
| `olvidar_s` | 2.0 | 0.5–30 | s | Tiempo mínimo sin ver a alguien antes de pasarlo a gracia. |  |
| `gracia_s` | 10.0 | 1–120 | s | Espera antes de registrar que salió por un borde. |  |
| `ocultas_s` | 120.0 | 5–3600 | s | Espera antes de registrar que DESAPARECIÓ en el interior. |  |
| `margen_borde` | 0.08 | 0.0–0.4 | fracción | Franja de cada borde por donde se puede salir. | Ajustar a la geometría de la escena (puertas cerca de un borde). |
| `salida_px_s` | 20.0 | 0–500 | px/s | Velocidad hacia el borde para considerar que la persona sale. |  |
| `debil_max_s` | 5.0 | 0.5–60 | s | Tiempo máximo que solo detecciones débiles sostienen una trayectoria (evita que quede 'pegada' a una silla u objeto). |  |
| `duplicado_contencion` | 0.7 | 0.3–1.0 | fracción | Una caja nueva contenida en esta fracción dentro de una persona ya seguida, y de altura parecida (≥ 70 %), es un duplicado, no otra persona. |  |

### `[apariencia]`

| Parámetro | Defecto | Rango | Unidad | Qué hace | Cómo calibrar |
|---|---|---|---|---|---|
| `veto` | 0.3 | 0.0–1.0 |  | Similitud de color bajo la cual dos personas se consideran distintas (solo con ambigüedad). |  |
| `recuperacion` | 0.8 | 0.0–1.0 |  | Similitud de color para recuperar lejos a alguien oculto. |  |
| `margen` | 0.1 | 0.0–1.0 |  | Diferencia mínima con la segunda candidata para recuperar por color. |  |
| `superposicion_fuerte` | 0.6 | 0.0–1.0 |  | Superposición que pesa más que el color. |  |

### `[puerta]`

| Parámetro | Defecto | Rango | Unidad | Qué hace | Cómo calibrar |
|---|---|---|---|---|---|
| `umbral` | 25 | 1–255 | niveles de gris | Cambio de brillo que cuenta como movimiento. |  |
| `fraccion_min` | 0.004 | 0.0001–0.5 | fracción | Área mínima cambiada para considerar movimiento. | Subir si la puerta dispara YOLO por ruido (IR, compresión). |
| `margen` | 0.15 | 0.0–1.0 | fracción | Margen alrededor de cada persona que no cuenta como movimiento nuevo. |  |

### `[sesion]`

| Parámetro | Defecto | Rango | Unidad | Qué hace | Cómo calibrar |
|---|---|---|---|---|---|
| `procesar_cada_s` | 0.1 | 0.03–1.0 | s | Intervalo de procesamiento (flujo óptico y puerta). |  |
| `max_s` | 900.0 | 30–7200 | s | Duración máxima de una sesión de análisis. |  |

### `[rostro]`

| Parámetro | Defecto | Rango | Unidad | Qué hace | Cómo calibrar |
|---|---|---|---|---|---|
| `activo` | False | true / false |  | Reconocimiento facial (requiere galería). Apagado hasta calibrar cada cámara. |  |
| `galeria` | face_gallery | texto | ruta | Carpeta de la galería: una subcarpeta con fotos por persona (relativa a CamDetector). |  |
| `hd_ruta` | /stream1 | texto | ruta | Ruta RTSP del vídeo principal (HD) del que se toma la cara. | La foto ONVIF de la W51-TY es solo 640x360: la cara sale del vídeo HD en ráfagas. |
| `franja_cabeza` | 0.3 | 0.1–1.0 | fracción | Parte superior de la caja de la persona donde se busca la cara. | ¿Caen dentro las caras de personas sentadas y de pie? (F1: sí en HD). |
| `franja_min_ancho` | 0.5 | 0.0–2.0 | fracción del ancho | Altura mínima de la franja de la cabeza respecto al ancho de la persona (sentados o cortados por el borde). |  |
| `yunet_puntuacion` | 0.8 | 0.3–0.99 |  | Confianza mínima de YuNet para aceptar una cara. | Bajar si se pierden caras inclinadas; vigilar caras falsas en rostro_intento. |
| `cara_min_px` | 48 | 16–400 | px | Ancho mínimo de la cara (en HD) para intentar reconocerla. | Desde qué tamaño la similitud de una misma persona se mantiene alta. |
| `frontal_min` | 0.5 | 0.0–1.0 |  | Frontalidad mínima (1 = de frente) según la simetría de ojos y nariz. |  |
| `nitidez_min` | 20.0 | 0–10000 | var. Laplaciano | Nitidez mínima de la cara (descarta movidas o desenfocadas). |  |
| `brillo_min` | 40.0 | 0–255 | nivel medio | Brillo medio mínimo de la cara (descarta caras en oscuridad). |  |
| `umbral` | 0.45 | 0.1–0.95 | similitud | Similitud mínima con la galería para dar un nombre. | Distribución misma persona frente a distinta (F1: Brayan 0,43 con galería de webcam, desconocidos 0,22–0,24). |
| `margen_segundo` | 0.08 | 0.0–0.5 | similitud | Diferencia mínima con la segunda persona más parecida. |  |
| `umbral_seguro` | 0.6 | 0.1–1.0 | similitud | Una sola coincidencia por encima de este valor identifica sin esperar más votos. | 2026-10-13: Brayan 0,57–0,70 frente a 0,21–0,32 de otras personas. |
| `votos` | 2 | 1–10 | coincidencias | Coincidencias con la misma persona para identificar. |  |
| `intentos_max` | 5 | 1–50 | intentos | Intentos con cara válida sin coincidencia antes de marcar 'desconocida'. |  |
| `espera_entre_intentos_s` | 1.5 | 0.2–60 | s | Tiempo mínimo entre intentos para la misma persona. |  |
| `espera_max_s` | 30.0 | 1–600 | s | Sin cara (de espaldas, mirando abajo) la espera se duplica en cada intento hasta este máximo: evita abrir el vídeo HD sin parar. | Cada ráfaga HD cuesta ~0,4 s de CPU: subir si las personas pasan mucho tiempo sin mirar a la cámara. |
| `reverificar_cada_s` | 60.0 | 5–3600 | s | Volver a verificar a una persona identificada. |  |
| `persona_min_alto_px` | 60 | 10–1000 | px | Altura mínima de la persona en el substream para intentar (lejos = sin cara útil). |  |
| `presupuesto_sface_s` | 3.0 | 0.2–50 | por segundo | Máximo de reconocimientos SFace por segundo en la cámara. |  |
| `guardar_recortes` | True | true / false |  | Guardar el recorte de cada cara intentada (capturas/rostros) para calibrar. |  |
| `recortes_margen` | 0.4 | 0.0–2.0 | fracción | Margen alrededor de la cara en los recortes guardados (cabeza completa, revisable a ojo y útil para la galería). |  |
| `recortes_dias` | 7 | 0–3650 | días | Días que se conservan los recortes de rostro (datos biométricos); 0 = no se borran. | Mientras se calibra conviene más; en producción, lo mínimo necesario. |

### `[reposo]`

| Parámetro | Defecto | Rango | Unidad | Qué hace | Cómo calibrar |
|---|---|---|---|---|---|
| `cada_s` | 10.0 | 0–600 | s | Verificación en reposo: un fotograma cada N s aunque la cámara no avise (0 = desactivada). |  |
| `confianza_min` | 0.5 | 0.05–0.95 |  | Confianza de YOLO para abrir sesión desde reposo. |  |
| `espera_s` | 60.0 | 0–3600 | s | Espera tras un disparo en reposo antes de volver a disparar. |  |
