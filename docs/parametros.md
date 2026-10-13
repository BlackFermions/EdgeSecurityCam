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

### `[reposo]`

| Parámetro | Defecto | Rango | Unidad | Qué hace | Cómo calibrar |
|---|---|---|---|---|---|
| `cada_s` | 10.0 | 0–600 | s | Verificación en reposo: un fotograma cada N s aunque la cámara no avise (0 = desactivada). |  |
| `confianza_min` | 0.5 | 0.05–0.95 |  | Confianza de YOLO para abrir sesión desde reposo. |  |
| `espera_s` | 60.0 | 0–3600 | s | Espera tras un disparo en reposo antes de volver a disparar. |  |
