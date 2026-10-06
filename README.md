# CamDetector

Servicio del **nodo de sitio**: recibe las alarmas de detección que la cámara
calcula en su propio procesador, las verifica con YOLO, **sigue a cada persona
mientras está en la imagen** y guarda todo en una base SQLite local junto con
la configuración de la cámara vigente en ese momento. Sin interfaz gráfica:
registro en consola y en `logs/`.

Objetivo de esta etapa: **calibrar la detección de la cámara** (qué detecta,
con qué retraso, cuántas falsas alarmas), clasificar cada alarma como persona
confirmada o falsa alarma y registrar cada aparición de una persona con un
código propio. Más adelante, la base local será la cola de envío hacia el NOC.

## Instalación en el servidor (Windows)

Requisitos: Python 3.10 o superior (probado con 3.12) y acceso de red a la
cámara.

```powershell
# 1. Copiar la carpeta CamDetector al servidor y abrir PowerShell en ella
.\instalar.bat          # crea .venv, instala dependencias y descarga yolo26n

# 2. Editar claves.txt con el usuario y la contraseña de la cámara

# 3. Iniciar (la IP por defecto está en ejecutar.bat)
.\ejecutar.bat
.\ejecutar.bat 192.168.100.109
```

`ejecutar.bat` guarda el registro en `logs\camdetector.log`, la base en
`data\camdetector.db` y las fotos en `capturas\`.

La instalación descarga ~1 GB (PyTorch en CPU). Sin YOLO, la recepción de
alarmas y la base no necesitan dependencias: `python escuchar_alarmas.py --no-yolo`.

## Uso directo

```powershell
.\.venv\Scripts\python escuchar_alarmas.py --camera 192.168.100.109
.\.venv\Scripts\python escuchar_alarmas.py --camera 192.168.100.109 --raw   # cada evento
.\.venv\Scripts\python escuchar_alarmas.py --camera 192.168.100.109 --no-yolo

# Ventana con el vídeo analizado en tiempo real (q o Esc para cerrar)
.\.venv\Scripts\python escuchar_alarmas.py --camera 192.168.100.109 --ver

# Resumen de la base para calibrar
.\.venv\Scripts\python tools\resumen.py
.\.venv\Scripts\python tools\resumen.py --desde 2026-10-06T20:00 --ultimas 30

# Diagnóstico de la cámara: servicios ONVIF, temas, eventos y configuración
.\.venv\Scripts\python tools\sondear_eventos.py 192.168.100.109 --seconds 60
```

Con `--ver` se abre una ventana por cámara (también `.\ejecutar.bat 192.168.100.109 --ver`):

- **verde:** posición puesta por YOLO en ese fotograma;
- **celeste:** posición estimada con flujo óptico (sin YOLO);
- **rojo:** el flujo óptico la perdió; marco rojo: movimiento nuevo fuera de las personas;
- **gris fino:** en gracia: `oculta` (perdida en el interior) o `saliendo` (junto a un borde);
- abajo, el motivo por el que se ejecutó YOLO; arriba, fotogramas, inferencias
  y porcentaje sin YOLO;
- en reposo muestra la cámara en vivo con el aviso "REPOSO: YOLO dormido".

Para ver la cámara en reposo, la ventana abre **su propia conexión RTSP**
(solo con `--ver`): el análisis sigue sin abrir el vídeo hasta que la cámara
avisa, igual que sin ventana. Sirve para comprobar a simple vista si la
cámara deja de avisar con alguien en la imagen (p. ej. en zonas oscuras).

Necesita escritorio (no funciona en un servidor sin pantalla) y consume más
CPU (decodificar el vídeo continuo y dibujar); es para depurar y calibrar.

Sin `--camera`, busca la cámara por su MAC con ONVIF. Varias cámaras se indican
repitiendo `--camera`. Otras opciones: `--db`, `--no-db`, `--config-interval`,
`--yolo-size` (416 por defecto), `--yolo-confidence`, `--model`,
`--snapshots-dir`, `--motion-mode`.

Salida típica:

```
[192.168.100.109] Human Detect  ACTIVADO · sensibilidad 90 · duración 20 s · región toda la imagen
[192.168.100.109] Motion Detect ACTIVADO · sensibilidad 80 · duración 2 s · región toda la imagen
[192.168.100.109] las alarmas se registran como movimiento (Motion Detect activo: puede ser cualquier movimiento)
[192.168.100.109] suscrito a eventos ONVIF
[192.168.100.109] >>> MOVIMIENTO DETECTADO
[192.168.100.109]     persona cam109-20261006-145731-1 entró (confianza 0.88 · YOLO por confirmacion)
[192.168.100.109]     ✔ PERSONA CONFIRMADA · 1 persona · confianza 0.88 · a los 1.4 s
[192.168.100.109] <<< fin de movimiento (6.2 s)
[192.168.100.109]     persona cam109-20261006-145740-1 entró (confianza 0.74 · YOLO por movimiento_nuevo)
[192.168.100.109]     persona cam109-20261006-145731-1 salió (34.2 s visible · 21 detecciones YOLO)
[192.168.100.109]     persona cam109-20261006-145740-1 salió (12.0 s visible · 9 detecciones YOLO)
[192.168.100.109]     resumen YOLO: máximo 2 persona(s) · 31 fotogramas · foto capturas/2026-10-06/...jpg
[192.168.100.109]     sesión: 412 fotogramas · 52 YOLO (87% sin YOLO) · 2 persona(s) seguida(s) · motivos: ...
[192.168.100.109] CONFIGURACIÓN CAMBIADA: Motion Detect: sensitivity 80 → 60
```

## Cómo funciona

```
REPOSO                 ALARMA DE LA CÁMARA          PERSONAS PRESENTES
YOLO dormido,          YOLO ~5 fps hasta el         YOLO ~1 fps (3 s si están quietas)
CPU ~0                 veredicto (≤ 4 s)            + flujo óptico entre detecciones
      └── la cámara avisa ──►     │                 + salvaguardas que despiertan a YOLO
                                  └── persona confirmada ──►│
                                                            └─ nadie y sin alarma ─► REPOSO
```

1. **Configuración:** al arrancar y cada 60 s lee Human Detect, Motion Detect,
   modelo y firmware por la web de la cámara. Si cambió, crea una nueva versión
   en la base y lo muestra en el registro.
2. **Alarmas:** se suscribe a los eventos ONVIF (PullPoint); la conexión la
   inicia el nodo, sin abrir puertos. La cámara reenvía el estado ~5 veces por
   segundo; el programa lo reduce a **inicio** y **fin** con duración.
3. **Interpretación:** en la W51-TY, Human Detect y Motion Detect salen por el
   mismo tema. Con Motion Detect apagado la alarma se registra como persona;
   con Motion Detect activo, como movimiento. Se ajusta solo si la
   configuración cambia.
4. **Sesión de análisis:** la primera alarma abre el substream RTSP
   (`/stream2`); las alarmas siguientes se unen a la misma sesión. La sesión
   dura mientras la alarma siga activa **o haya personas seguidas**, y se
   cierra sola después.
5. **Verificación de cada alarma** (YOLO26 nano, 416 px, solo clase persona):
   - **confirmada:** persona en 2 inferencias seguidas;
   - **descartada:** 4 s sin confirmar (puede pasar a confirmada si alguien
     aparece después);
   - **sin_video:** no se pudo abrir el vídeo;
   - aunque la alarma dure 1–2 s, se analiza hasta tener veredicto.
6. **Seguimiento corporal (algoritmos clásicos, sin redes neuronales):** YOLO
   crea las trayectorias y corrige su posición; entre dos inferencias:
   - **flujo óptico Lucas-Kanade** a media resolución mueve y **escala** cada
     caja (se agranda o achica cuando la persona se acerca o se aleja);
   - un **filtro de Kalman** de velocidad constante combina el flujo (ruidoso)
     con YOLO (preciso) y **predice dónde está** quien dejó de verse;
   - la **asignación húngara** reparte detecciones y personas de forma óptima;
   - un **histograma de color del torso** (0,14 ms por persona) impide
     intercambiar personas con ropa distinta y recupera a quien reaparece en
     otro sitio. Con IR u oscuridad no hay color y solo cuenta la posición.

    Cada aparición recibe un código legible
   `cam109-AAAAMMDD-HHMMSS-n` (cámara, fecha y hora de aparición, subnúmero si
   aparecen varias en el mismo segundo) y un UUID. Para no cambiar el código
   de una misma persona (*ID switch*) se usan las ideas de ByteTrack:
   - **asociación por superposición o cercanía:** si el flujo óptico se quedó
     atrás y la caja ya no se superpone, una detección cercana sigue siendo la
     misma persona;
   - **detecciones débiles** (confianza 0,15–0,40): mantienen viva una
     trayectoria existente, pero nunca crean personas ni confirman alarmas;
   - **gracia:** quien deja de verse no se da por ido de inmediato; si
     reaparece cerca de donde Kalman lo esperaba (el radio crece con el tiempo
     hasta 1,25 tamaños de caja) recupera su código **por posición**; si
     estaba oculto en el interior y reaparece lejos, lo recupera **por
     apariencia** solo si la ropa es muy parecida (≥ 0,80) y sin otra
     candidata similar;
   - **cruces:** si hay varias personas, la ropa claramente distinta
     (similitud < 0,30) impide intercambiar sus códigos, salvo superposición
     muy alta. Con una sola persona el color no veta nada: un torso parcial o
     en sombra cambia de color sin cambiar de persona;
   - **continuidad:** las personas nuevas entran por los bordes. Una
     detección que aparece **en el interior** habiendo alguien perdido o sin
     pareja es esa misma persona (salvo ropa claramente distinta); solo una
     detección junto a un borde puede ser alguien que entra.
7. **Salir o desaparecer.** Nadie desaparece de una casa: solo se sale por un
   borde de la imagen, moviéndose hacia él. Si YOLO deja de ver a alguien
   (2 inferencias y 2 s):

   | Última situación | Estado | Si no reaparece |
   |---|---|---|
   | Junto a un borde (8% del ancho/alto) **y moviéndose hacia él** (≥ 20 px/s, según Kalman) | saliendo | a los 10 s: **salió por el borde** |
   | En el interior, **o quieta junto a un borde** | **oculta** (mueble, zona oscura, agachada, falla) | se la busca con YOLO cada 3 s; a los 2 min: **DESAPARECIÓ**, anomalía a revisar |

   Mientras haya personas en gracia u ocultas la sesión sigue abierta.
8. **Salvaguardas.** El ahorro solo aplica a seguir a quien ya fue detectado;
   detectar lo nuevo nunca se retrasa más de ~0,5 s. YOLO se ejecuta si:

   | Motivo | Cuándo |
   |---|---|
   | `alarma` | la cámara inicia una alarma nueva (de inmediato) |
   | `confirmacion` | hay una alarma sin veredicto (cada 0,2 s) |
   | `movimiento_nuevo` | la diferencia de fotogramas muestra cambios fuera de las personas seguidas (máx. 1 cada 0,5 s; el aviso queda retenido hasta que YOLO corra) |
   | `trayectoria_perdida` | el flujo óptico pierde a alguien, p. ej. en zonas oscuras (máx. 1 cada 0,5 s) |
   | `intervalo` | 1 s sin YOLO con personas en movimiento, 3 s si están quietas |
   | `vigilancia` | alarma activa sin personas seguidas (1 por segundo) o personas ocultas/saliendo por encontrar (cada 3 s) |

   La base registra cuántas inferencias hubo por motivo y **por qué motivo se
   detectó cada persona**. Si aparecen personas nuevas detectadas por
   `intervalo`, la puerta de movimiento no las vio a tiempo y hay que ajustarla.
9. **Base local:** alarmas, veredictos, personas seguidas y estadísticas de
   cada sesión, con la versión de configuración vigente.

### Consumo medido (PC de pruebas, 8 núcleos, CPU, sin GPU)

| | RAM | Tiempo |
|---|---|---|
| Python + OpenCV | ~35 MB | — |
| + PyTorch + YOLO26n cargado | ~320–390 MB | carga ~8 s |
| Inferencia a 640 / 416 / 320 px | | ~132 / ~68 / ~55 ms |

Sesión de 15 s con personas reales recortadas de capturas de esta cámara
moviéndose sobre la sala vacía (vídeo sintético a 10 fps):

| | Inferencias YOLO | CPU |
|---|---|---|
| YOLO en cada fotograma | 44 de 44 (no alcanza a procesar todos) | 87% de un núcleo |
| **YOLO + flujo óptico + salvaguardas** | **13 de 79 (84% sin YOLO)** | **31% de un núcleo** |

La persona que aparece a mitad del vídeo se detectó ~0,6 s después, por
`movimiento_nuevo`. La RAM no cambia: la ocupa PyTorch. Exportar el modelo a
ONNX y quitar PyTorch es la siguiente optimización.

`escuchar_alarmas.py` define `KMP_BLOCKTIME=0` y `OMP_WAIT_POLICY=PASSIVE`: sin
ellas los hilos de PyTorch siguen ocupando CPU después de cada inferencia.

## Base de datos del nodo (`data/camdetector.db`, esquema v3)

| Tabla | Una fila por | Contenido |
|---|---|---|
| `config_camara` | versión de configuración de una cámara | `vigente_desde`/`vigente_hasta`, Human/Motion Detect (activo, sensibilidad, duración), modelo, firmware, parámetros de IA y seguimiento, configuración completa en JSON (`datos`) |
| `alarma` | alarma de la cámara | UUID, cámara, `config_id`, tipo, inicio, fin, duración, veredicto, máximo de personas, confianza, fotogramas, segundos hasta la primera persona, foto, nota |
| `persona_track` | aparición de una persona | UUID, `codigo` legible, cámara, alarma y sesión, entrada, última vista, salida, segundos visible, confianza máxima, detecciones YOLO, `motivo_deteccion`, `recuperaciones` y `recuperaciones_apariencia`, nota (salió por el borde / DESAPARECIÓ en el centro / sesión terminada) |
| `sesion_analisis` | sesión de análisis | inicio, fin, fotogramas, inferencias YOLO, ahorro, inferencias y personas nuevas por motivo, reapariciones por método (JSON) |
| `evento_sistema` | evento del nodo | inicio, fin, conexión, desconexión, cambios de configuración, errores |

- Horas en UTC (ISO 8601); `tools/resumen.py` las muestra en hora local.
- `enviado` queda en NULL hasta que el NOC confirme la recepción; cualquier
  actualización de la fila la vuelve a marcar como pendiente (patrón outbox).
- Los identificadores son UUID generados en el nodo: el NOC podrá descartar
  reenvíos duplicados.
- Una base anterior se actualiza sola al arrancar (tablas y columnas nuevas).
- El código de una persona identifica **una aparición**, no a la persona: si
  sale y vuelve a entrar recibe otro código. Unir apariciones es trabajo de
  Re-ID y del reconocimiento facial (siguientes etapas).

## Hallazgos sobre la Vatilon W51-TY (firmware V1.15.29)

- Declara dos temas: `UserAlarm/IVA/HumanShapeDetect` y
  `RuleEngine/CellMotionDetector/Motion`. En la práctica solo envía el segundo,
  para Human Detect y Motion Detect a la vez.
- Con Motion Detect desactivado, mover una silla o un folder no dispara alarma;
  una persona sí.
- Human Detect **no detecta bien a personas de espaldas**; por eso Motion
  Detect queda activo como disparador y YOLO confirma.
- En **zonas oscuras** la cámara no detecta movimiento: si la cámara no avisa,
  YOLO no se despierta. Revisar el cambio a modo noche/IR o la iluminación.
- Cada detección tiene su Alarm Duration: la duración de la alarma da una pista
  de qué la disparó (≈2 s movimiento, ≥20 s persona).
- Con Motion Detect activo (sensibilidad 80) hubo falsas alarmas sin nadie en
  la sala; YOLO las descartó todas (3 de 3 el 2026-10-06).
- La ventana del lado derecho puede generar falsas alarmas: conviene excluirla
  en **Region** en ambas detecciones.

## Calibración pendiente

| Prueba | Qué medir |
|---|---|
| Persona de frente / lado / espaldas | Alarmas y confirmaciones de YOLO por orientación |
| Persona quieta / sentada | ¿La alarma se mantiene o se corta? ¿El seguimiento la conserva? |
| Dos personas que se cruzan | ¿Se mantienen sus códigos o se intercambian? |
| Persona que se mueve rápido o se agacha | ¿Conserva su código? (antes cambiaba: 3 códigos en 2 min) |
| Persona en la zona oscura | ¿Queda `oculta` y recupera su código al volver a verse? |
| Caminar detrás de un mueble y salir por otro lado | ¿Reaparece con su código (por posición o por apariencia)? |
| Sentarse junto al borde, en la zona oscura | ¿Queda oculta y conserva su código, sin "salió por el borde"? |
| Entrar por el borde (cuerpo parcial → completo) | ¿Un solo código desde la entrada? |
| Entrada mientras ya hay alguien | ¿Se detecta por `movimiento_nuevo` y con qué retraso? |
| Objetos, cortinas, cambios de luz | Falsas alarmas de la cámara y cuántas descarta YOLO |
| Noche con IR y zonas oscuras | Detección, falsas alarmas y trayectorias perdidas |
| Distancia y ángulo | Hasta dónde detecta; personas parciales en el borde |

## Siguientes etapas

1. **Re-ID ligero** (p. ej. OSNet x0.25, pocas huellas por trayectoria) para
   unir apariciones de la misma persona: reentradas y cruces.
2. **Reconocimiento facial** (YuNet + SFace de CameraCaptor) para poner nombre
   a la trayectoria.
3. **Modelo en ONNX sin PyTorch** para bajar la RAM en hardware edge.

## Estructura

```
escuchar_alarmas.py      punto de entrada (argumentos, registro, arranque)
instalar.bat             crea .venv, instala dependencias y descarga el modelo
ejecutar.bat             inicia el nodo con registro en logs/
requirements.txt
claves.example.txt       plantilla de credenciales (claves.txt no se versiona)
src/node.py              servicio del nodo: une configuración, alarmas, análisis y base
src/camera_events.py     suscripción ONVIF, parseo e inicio/fin de alarmas
src/camera_config.py     lectura, huella y cambios de la configuración
src/camera_web.py        acceso de solo lectura a la web de la cámara
src/analysis.py          sesión de análisis por cámara: verificación + seguimiento
src/tracking.py          flujo óptico, puerta de movimiento, trayectorias y salvaguardas
src/person_verifier.py   detector YOLO y veredicto de cada alarma
src/storage.py           base SQLite del nodo
src/discovery.py         búsqueda de la cámara por MAC (WS-Discovery)
src/config.py            credenciales
tools/resumen.py         resumen de la base para calibrar
tools/sondear_eventos.py diagnóstico de la cámara
tests/                   pruebas: python -m unittest discover -s tests -t .
data/ logs/ capturas/ models/   datos locales (no se versionan)
```

YOLO (Ultralytics) se distribuye bajo AGPL-3.0: revisar la licencia antes de
un uso comercial.
