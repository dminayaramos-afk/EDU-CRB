# EDU-CRB

Un modelo de lenguaje  **construido enteramente desde cero
con NumPy**. Sin PyTorch, sin TensorFlow, sin autograd, sin APIs, sin modelos
preentrenados de nadie: tokenizador, atención, retropropagación, optimizador,
entrenamiento, memoria y agente están escritos aquí, se pueden leer de principio a
fin y funcionan sin internet.

```
python3 main.py        # habla con el modelo que ya viene entrenado
```

## Página web (GitHub Pages)

`index.html` es una página que muestra Cerebro como una red neuronal en movimiento: cada columna es una pieza real del modelo y, al pasar el ratón o tocar, aparece una descripción corta. Para verla en línea: Settings → Pages → Deploy from a branch → `main` / `(root)`. Quedará en `https://dminayaramos-afk.github.io/cerebro/`. También funciona abriendo el archivo directamente en el navegador.

## Página web interactiva (GitHub Pages)

`index.html` es una página de un solo archivo que muestra cómo funciona Cerebro por dentro:

- Una red neuronal animada cuyas columnas son las piezas reales del modelo (tokenizador, embeddings, atención, MLP, logits, muestreo). Al pasar el ratón o tocar una pieza aparece su descripción.
- Doce preguntas para elegir. Al pulsar una, el modelo la responde **en tu navegador con los pesos reales de `best.npz`** (mismo tokenizador BPE, misma pasada hacia adelante y mismo muestreo que el código Python; los logits coinciden con una diferencia de ~1e-5 y la generación voraz es idéntica) y la animación recorre cada paso: tokens, atención por capa y cabeza, y probabilidades del siguiente token.
- Opciones del modelo: temperatura, top-k, top-p, penalización de repetición y velocidad de la animación.

No necesita servidor ni dependencias: los pesos (~0,8 MB) van incrustados en el propio HTML, así que también funciona abriéndolo con doble clic.

### Publicarla

1. Sube `index.html` (y `.nojekyll`) a la raíz del repositorio.
2. En GitHub: Settings → Pages → Build and deployment → Deploy from a branch → rama `main`, carpeta `/ (root)` → Save.
3. En uno o dos minutos estará en `https://dminayaramos-afk.github.io/Educa-Cerebro/`. Si ves una versión vieja, recarga con Ctrl+F5.

### Si reentrenas el modelo

```bash
python3 train.py                 # dentro de cerebro-main/
cd ..
python3 exportar_web.py          # vuelve a incrustar best.npz en index.html
```

Después sube el `index.html` nuevo. Solo hace falta numpy.

## Alcance honesto

Cerebro **no** es un LLM competitivo ni sabe cosas del mundo. El perfil original tiene ~190 mil
parámetros y aprendió de un corpus propio de unos 80 mil tokens. Lo que sí hace bien:

- Contesta con fluidez las preguntas de su corpus y sus reformulaciones
  (incluida la escritura informal: «que es la atencion»).
- Ejecuta cuentas y consulta la hora con herramientas seguras, sin inventárselas.
- Recuerda datos entre sesiones (memoria de largo plazo).

Lo que **no** hace: responder sobre temas que no están en su corpus. Si le
preguntas algo desconocido contestará con la respuesta memorizada que más se
parezca, con total seguridad y sin sentido. Es el comportamiento esperable de un
modelo así de pequeño y `evaluate.py` lo mide (ver «Métricas»), no lo esconde.
Mejorarlo pasa por darle más texto (`data/corpus/`) y usar un perfil mayor.

## Qué hay dentro (todo real, todo verificado)

| Pieza | Dónde | Detalle |
|---|---|---|
| Transformer | `core/transformer/optimized_llm.py` | Pre-LayerNorm, atención multi-cabeza causal, MLP con GELU, embeddings de posición, dropout, pesos compartidos entrada/salida, entrenamiento por lotes |
| Retropropagación | mismo archivo | Derivada **a mano** con la regla de la cadena. Las pruebas la comparan con diferencias finitas (error ~1e-6) |
| Tokenizador BPE | `core/tokenizer.py` | Byte-level: nunca hay palabras desconocidas y decodifica tildes, ñ y emojis sin pérdida. Se conserva además el tokenizador por palabras original |
| Optimizadores | `core/optim.py` | AdamW (decaimiento desacoplado), SGD con momentum, recorte por norma global, learning rate con calentamiento + coseno |
| Muestreo | `core/sampling.py` | Temperatura, top-k, top-p, penalización de repetición, generación reproducible con semilla |
| Datos | `core/data.py` | Documentos separados por línea en blanco, validación aparte, **pérdida solo sobre las respuestas** (las líneas del usuario se enmascaran, como en el ajuste supervisado de un chatbot) |
| Checkpoints | `core/checkpoint.py` | Un `.npz` con pesos + arquitectura + tokenizador; escritura atómica; sin pickle |
| Entrenamiento | `train.py`, `trainer_engine.py` | Validación real, mejor checkpoint por bits/byte, `--resume`, no sobrescribe un modelo mejor |
| Evolución | `evolve.py` | Algoritmo evolutivo real: población, selección, cruce y mutación de hiperparámetros, con presupuesto de parámetros |
| Chat | `main.py`, `chat.py` | Comandos, memoria de corto plazo en el prompt, enrutador de herramientas |
| Agente | `agent/` | Límites de pasos y llamadas, **permisos por herramienta**, workspace cerrado |
| Memoria | `memory/advanced_memory.py` | Corto plazo + largo plazo persistente (JSON atómico) con búsqueda |
| Datos extra | `aumentar_corpus.py` | Reformulaciones, escritura informal y conversaciones multi-turno a partir del corpus |
| Reanudación reproducible | `trainer_engine.py`, `core/checkpoint.py` | Guarda RNG del modelo + muestreador de batches + AdamW; continuar produce los mismos pasos |
| Entrenamiento eficiente | `trainer_engine.py` | Acumulación de gradientes, scheduler basado en paso global y early stopping configurable |
| Evaluación | `data/test.txt`, `evaluate.py` | Conjunto TEST final que nunca selecciona checkpoints; huellas SHA-256 de train/validación |
| Pruebas | `tests/tests_final.py` | 27 pruebas de integridad, sin frameworks |

## Instalación

```bash
git clone https://github.com/dminayaramos-afk/cerebro.git
cd cerebro
pip install -r requirements.txt      # solo numpy
python3 tests/tests_final.py         # comprueba que todo funciona
python3 main.py                      # a conversar
```

Única dependencia real: `numpy`. `psutil` es opcional.

## Uso

### Conversar
```bash
python3 main.py                          # interactivo
python3 main.py --una-vez "hola"         # un mensaje, para scripts
python3 main.py --semilla 7              # respuestas reproducibles
```
Comandos: `/ayuda /estado /memoria /recordar clave = valor /olvidar clave /calc 12*7
/hora /archivos /leer f /escribir f texto /temp 0.5 /limpiar /salir`.
También entiende «cuánto es 12 por 7» y «qué hora es» (por reglas, no por el modelo).

### Entrenar
```bash
python3 train.py                     # 8000 pasos, perfil "pequeno"
CEREBRO_PERFIL=v2 python3 train.py   # perfil V2: 4 capas, dim 96, contexto 192, vocab 768
python3 train.py --perfil v2 --pasos 8000
python3 train.py --pasos 2000        # o --epocas 50
python3 train.py --resume            # continúa desde checkpoints/last.npz
python3 train.py --corpus mis_textos/    # tu propio corpus (archivo o carpeta)
CEREBRO_PERFIL=diminuto python3 train.py
python3 train.py --batch 8 --pasos 2000
```
Al terminar, `best.npz` solo se reemplaza si el modelo nuevo es **mejor** que el
existente (comparados hoy, sobre la misma validación, en bits por byte). Con
`--forzar` se sobrescribe sin comparar.

### Evolucionar hiperparámetros
```bash
python3 evolve.py --poblacion 6 --generaciones 3 --pasos-candidato 250 --pasos-final 3000
```

### Evaluar
```bash
python3 evaluate.py
```

### Añadir conocimiento
Cerebro solo sabe lo que hay en su corpus. Para enseñarle algo:

1. Escribe diálogos en `data/corpus/lo_que_sea.txt`, separados por una línea en blanco:
   ```
   usuario: ¿Qué es un byte?
   cerebro: Es un grupo de ocho bits.

   usuario: ¿Y un bit?
   cerebro: Es la unidad mínima de información.
   ```
   (La prosa suelta también vale.)
2. `python3 aumentar_corpus.py` (opcional, genera variantes de las preguntas).
3. `python3 train.py`.

## Datos y métricas: cómo leer los números

Hay tres conjuntos que **nunca se mezclan**:

- `data/corpus.txt` + `data/corpus/*.txt` — entrenamiento.
- `data/validacion.txt` — reformulaciones *nuevas* de cosas que el modelo sí debería
  saber. Con esto se elige el mejor checkpoint.
- `data/generalizacion.txt` — diálogos sobre temas que jamás vio. Mide conocimiento
  nuevo y, en un modelo de este tamaño, sale mal a propósito de mostrarlo.
- `data/test.txt` — conjunto final separado. Solo se informa; nunca decide qué checkpoint es mejor.

Cada checkpoint guarda además hashes SHA-256 de train/validación para detectar cambios de datos.

Métricas (las imprime `python3 evaluate.py`):

- **bits por byte (bpb)**: el error del modelo por cada byte de texto. Permite
  comparar modelos con tokenizadores distintos. Menor es mejor.
- **Respuestas exactas**: % de preguntas reformuladas que el modelo contesta idéntico
  al corpus con generación voraz. Es lo que se nota al chatear.

Resultado del modelo incluido (`checkpoints/best.npz`):

| Métrica | Valor |
|---|---|
| Parámetros | 190.528 (3 capas × 4 cabezas, dim 64, contexto 112, vocabulario BPE de 512) |
| Validación (reformulaciones nuevas) | 0,100 bits/byte (perplejidad 1,17) |
| Respuestas exactas a preguntas reformuladas | 47 % (similitud media 0,61) |
| Generalización (diálogos nunca vistos) | 4,51 bits/byte (perplejidad ~995): no sabe nada fuera de su corpus |
| Velocidad medida (CPU del sandbox, 1 núcleo) | ~26 000 tok/s entrenando, ~1 100 tok/s generando, ~46 MB de RAM |

Entrenamiento: `python3 train.py` (6000 pasos, ~6 min).

La validación es buena porque mide reformulaciones de lo aprendido; la
generalización es mala porque el modelo no sabe nada más. Los dos números son
verdad a la vez.

## Configuración (`config.py`)

| Qué | Dónde |
|---|---|
| Tamaño del modelo | `PERFILES` (`diminuto`, `pequeno`, `mediano`) o `CEREBRO_PERFIL=...` |
| Entrenamiento | `BATCH_SIZE`, `TASA_APRENDIZAJE`, `WEIGHT_DECAY`, `GRAD_CLIP`, `OPTIMIZADOR`, `PASOS_POR_DEFECTO` |
| Generación | `GEN_TEMPERATURA`, `GEN_TOP_K`, `GEN_TOP_P`, `GEN_PENALIZACION_REPETICION` |
| Agente | `MAX_AGENT_STEPS`, `MAX_TOOL_CALLS`, `PERMISOS_AGENTE`, `MAX_FILE_BYTES`, `WORKSPACE_DIR` |

Para un agente de solo lectura quita `"WRITE"` de `PERMISOS_AGENTE`.

## Seguridad del agente

Todo esto lo comprueban las pruebas:

- La calculadora **no usa `eval`**: analiza la expresión como árbol y solo admite
  números, operadores y `sqrt/abs/sin/...`. Rechaza código, nombres, cadenas y
  bombas de cómputo (`9**9**9**9`, `10**100000`).
- Los archivos viven en `data/workspace/`. La ruta se resuelve con `realpath` y
  se comprueba con `commonpath`: bloquea `../`, rutas absolutas, carpetas
  vecinas con el mismo prefijo y enlaces simbólicos. Hay tope de tamaño.
- Cada herramienta exige un permiso (`READ`, `WRITE`, `CALCULATE`); sin él, no se ejecuta.
- Un plan que falla se detiene. Nada de archivos se activa por lenguaje natural:
  solo con `/leer` y `/escribir`.

## Estructura

```
config.py               configuración central y perfiles
main.py  chat.py        chat interactivo y su lógica
train.py                entrenamiento (CLI)
trainer_engine.py       bucle de entrenamiento y evaluación
evolve.py               búsqueda evolutiva de hiperparámetros
evaluate.py             informe de evaluación
aumentar_corpus.py      aumento de datos del corpus
core/
  tokenizer.py          BPE por bytes + tokenizador por palabras
  transformer/optimized_llm.py   el modelo (forward + backward a mano)
  optim.py  sampling.py  data.py  checkpoint.py
agent/                  agente, herramientas y enrutador de intenciones
memory/                 memoria de corto y largo plazo
benchmarks/             medición de velocidad y RAM
tests/tests_final.py    pruebas
data/                   corpus, validación, generalización, workspace
checkpoints/best.npz    modelo entrenado incluido
```

## Cómo seguir creciendo

El código está pensado para extenderse sin romper lo demás:

- **Nueva arquitectura**: un bloque son solo `_bloque_forward` y `_bloque_backward`
  en `optimized_llm.py`. `ModeloConfig` ignora campos desconocidos al cargar
  checkpoints, así que se pueden añadir opciones. **Regla de oro:** después de
  tocar el backward, ejecuta `python3 tests/tests_final.py -k gradientes`.
- **Nuevo tokenizador**: añádelo a `TOKENIZADORES` en `core/tokenizer.py`.
- **Nueva herramienta**: `ToolRegistry.registrar(nombre, desc, func, schema, permiso)`.
- **Modelo mayor**: crea un perfil en `config.py`. Más datos ayudan más que más parámetros.
- **Ideas siguientes**: caché de claves/valores para generar más rápido, RoPE en vez
  de posiciones aprendidas, muestreo por haces, recuperación de memoria por
  similitud de embeddings.


