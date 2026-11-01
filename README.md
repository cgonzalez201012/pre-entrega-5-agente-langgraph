# Agente de Gobierno de Datos con LangGraph y memoria persistente

Pre-entrega 5 del curso de AI Engineering (Coderhouse): un agente de razonamiento cíclico (ReAct) construido con **LangGraph** que decide por sí mismo cuándo usar herramientas, razona en varios pasos y recuerda la conversación gracias a un checkpointer en **SQLite**.

El caso es de Gobierno de Datos, que es mi área de trabajo: el agente responde preguntas sobre los incidentes y las reglas de calidad de los maestros de datos (artículos, proveedores y sucursales) de **Mercado Austral S.A.**, una empresa ficticia de retail. Los datos son inventados y viven en `data/`; las herramientas los consultan como si fueran una base de datos.

## Qué hace

```
Pregunta 1: ¿Cuántos incidentes críticos abiertos tiene el maestro de artículos y quién es el responsable del más antiguo?
Agente:     → decide usar buscar_incidentes_calidad(maestro="articulos", estado="abierto", severidad="critica")
            → recibe 2 incidentes (INC-1001, INC-1002) y ve cuál tiene más días desde el alta
            → decide usar detalle_incidente(incidente_id="INC-1001")
            → recibe el responsable y ya tiene todo → responde

Pregunta 2: ¿Y qué regla incumple ese incidente?     (mismo thread_id: el agente recuerda cuál era "ese incidente")
```

La respuesta necesita dos consultas encadenadas: la segunda depende de un dato que salió de la primera. Ese es el razonamiento multi-paso que pide la consigna.

## Cómo está armado

```
        START ──► agente ──(¿pidió herramientas?)──► tools
                    ▲            │                      │
                    │            └── no ──► END         │
                    └─────────────────────────────────────┘
```

| Archivo | Qué hace |
|---|---|
| `agente.py` | El `StateGraph` (estado sobre `MessagesState`, nodo del modelo, `ToolNode`, arista condicional `tools_condition`), el checkpointer `AsyncSqliteSaver`, y `preguntar()` que ejecuta un turno con `thread_id` y `recursion_limit`. |
| `herramientas.py` | Las 3 herramientas con `@tool`, esquemas Pydantic y docstrings descriptivos. |
| `repositorio.py` | La "base de datos" simulada (lee los JSON de `data/` sin bloquear el bucle de eventos). |
| `prompts.py` | Las reglas de conducta del agente (no inventar datos, qué hacer ante un error). |
| `trazas.py` | Convierte los mensajes de un turno en pasos del ciclo ReAct y los guarda como JSON y log. |
| `main.py` | Línea de comandos: una pregunta, chat, ver historial de un hilo y la corrida de demostración. |
| `config.py`, `errores.py` | Configuración por variables de entorno y errores con mensajes claros. |
| `tests/` | 79 tests que corren sin claves ni red. |

Las tres herramientas son de solo lectura:

- `buscar_incidentes_calidad(maestro, estado, severidad)`: lista incidentes de un maestro. No trae el responsable.
- `detalle_incidente(incidente_id)`: ficha completa con responsable, regla incumplida y SLA.
- `consultar_regla_calidad(regla_id)`: qué exige una regla, su umbral y su medición actual.

## Cómo levantar el entorno

Necesito Python 3.12 o superior y una clave de Anthropic.

```bash
git clone https://github.com/cgonzalez201012/pre-entrega-5-agente-langgraph.git
cd pre-entrega-5-agente-langgraph

python3.13 -m venv .venv             # o python3.12: probé el proyecto con las dos versiones
source .venv/bin/activate            # en Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env                 # y completo mi ANTHROPIC_API_KEY en el .env
```

Las variables que lee el programa están en `.env.example`. La clave nunca se escribe en el código y `.env` está en el `.gitignore`.

### Correr los tests (no necesitan clave)

```bash
pytest
mypy .
```

## Uso

```bash
# Una pregunta (el hilo por defecto se llama "consola")
python main.py "¿Cuántos incidentes críticos abiertos tiene el maestro de artículos y quién es el responsable del más antiguo?"

# Con -v se ve cada paso del ciclo ReAct
python main.py -v --thread auditoria-1 "¿Cuántos incidentes abiertos tiene el maestro de proveedores?"

# Misma sesión: el agente recuerda lo anterior, incluso en otra ejecución del programa
python main.py --thread auditoria-1 "¿Y cuál es el más grave?"

# Conversación interactiva
python main.py --chat --thread auditoria-1

# Ver qué quedó guardado en la base para un hilo (no necesita clave)
python main.py --ver-hilo auditoria-1

# Corrida completa de 4 escenarios: guarda la traza en trazas/
python main.py --demo
```

La memoria queda en `memoria/agente.sqlite` (ignorada por git). Cambiando el `--thread` se empieza una conversación nueva.

## Traza de ejecución

`python main.py --demo` hace una corrida real con el modelo y guarda la traza en `trazas/traza-demo.json` y `trazas/traza-demo.log`. Son 4 escenarios:

1. **Razonamiento multi-paso**: la pregunta de arriba. El agente llama a la herramienta al menos dos veces.
2. **Memoria con el mismo `thread_id`**: "¿Y qué regla incumple ese incidente?", sin repetir de qué incidente se habla.
3. **Ciclo de retorno ante un error**: se le pide el detalle de `INC-1010`, que no existe. La herramienta devuelve el error con una ayuda (los IDs válidos) y el agente tiene que reintentar con otra consulta o pedir una aclaración.
4. **Memoria tras reiniciar**: se cierra la conexión a SQLite y se abre de nuevo (equivale a cortar el programa y volver a correrlo), y el agente igual recuerda de qué se venía hablando.

El `.log` usa el mismo formato de flechas que muestra la consigna:

```
Usuario: "..."
→ El agente decide usar la herramienta: nombre(argumentos)
→ La herramienta devuelve: {...}
→ El agente ya tiene lo que necesita → responde.
Respuesta: "..."
```

El `.json` tiene lo mismo, paso a paso, más el modelo usado, el `recursion_limit` y la cantidad de herramientas llamadas en cada turno. La redacción de las respuestas puede variar entre corridas; los datos salen siempre de las herramientas.

**Lo que pasó en mi corrida** (`claude-haiku-4-5`, traza incluida en el repo):

1. Llamó a `buscar_incidentes_calidad` y después a `detalle_incidente(INC-1001)`: 2 llamadas encadenadas.
2. Con el mismo `thread_id` entendió que "ese incidente" era INC-1001, consultó `consultar_regla_calidad(R-ART-01)` y respondió que no cumple el umbral (96,8 % medido contra 99,5 % exigido).
3. Ante `INC-1010` leyó el error, listó los IDs que sí existen (INC-1001 a INC-1005) y pidió que se elija uno.
4. Tras cerrar y reabrir la base respondió con 0 llamadas a herramientas: recordó INC-1001 y su responsable.

## Cómo cumple la consigna

| Criterio de aceptación / checklist | Dónde está |
|---|---|
| **Autonomía**: el agente decide cuándo llamar a una herramienta, sin rutas `if/else` | `agente.py`: `llm.bind_tools()` y `tools_condition`. En el código no hay ninguna condición sobre el contenido de la pregunta: el enrutado lo hace `tools_condition` según lo que devuelva el modelo. |
| **Ciclo de retorno**: ante un error, segundo intento o pedir aclaración | Las herramientas devuelven `{"error", "ayuda"}` como texto y la arista `tools → agente` le devuelve el resultado al modelo. El prompt le indica qué hacer. Escenario 3 de la traza. |
| **Resiliencia de estado**: con `thread_id` recuerda la sesión | `AsyncSqliteSaver` + `thread_id` en `preguntar()`. Escenarios 2 y 4 de la traza; `tests/test_memoria.py`. |
| **Código limpio**: Python 3.12, type hints, `asyncio` | El agente, las herramientas y la persistencia son asíncronos (`async`/`await`, `ainvoke`). Todas las funciones tienen anotaciones de tipo (`mypy .` sin errores; un test revisa que no falte ninguna). |
| `StateGraph` sobre `MessagesState`, nodo de modelo + nodo de herramientas + `tools_condition` | `agente.py`: `EstadoAgente(MessagesState)`, nodos `agente` y `tools`. |
| Al menos 1 herramienta con `@tool` y docstring descriptivo | `herramientas.py` (3 herramientas). |
| Persistencia con SqliteSaver + `thread_id` | `agente.py` (ver la decisión sobre `AsyncSqliteSaver` abajo). |
| Prueba multi-paso (≥2 llamadas) y `recursion_limit` definido | Escenario 1 de la traza; `recursion_limit=10` en `config.py`, pasado en cada invocación. |
| Ejemplo de traza (.json o log) | `trazas/traza-demo.json` y `trazas/traza-demo.log`. |
| Repo público, sin API keys, README con el entorno | Este README, `.env.example`, `.gitignore`; un test busca claves en todos los archivos. |

## Cómo cumple la rúbrica

| Criterio (peso) | Qué hice y dónde está |
|---|---|
| **Arquitectura del grafo y gestión de estado** (30 %) | `EstadoAgente(MessagesState)` con el reducer `add_messages`; nodos `agente` y `tools`; arista condicional `tools_condition` y arista de retorno `tools → agente`; recorte del historial para que el estado no crezca sin control. `agente.py` y `tests/test_grafo.py`. |
| **Integración de herramientas y razonamiento multi-paso** (35 %) | Tres herramientas `@tool` asíncronas con esquemas Pydantic y descripciones largas, enlazadas con `bind_tools`; los errores vuelven al modelo como texto. Escenarios 1 y 3 de la traza. `herramientas.py` y `tests/test_herramientas.py`. |
| **Persistencia con SqliteSaver** (20 %) | `AsyncSqliteSaver` con `thread_id`; la memoria queda en `memoria/agente.sqlite`. Escenarios 2 y 4 de la traza, `--ver-hilo` y `tests/test_memoria.py`. |
| **Calidad del repositorio y entorno profesional** (15 %) | Un módulo por responsabilidad, type hints verificados con `mypy`, `requirements.txt` con versiones fijas, `pyproject.toml`, `.env.example` y `.gitignore`, un test que busca claves en todo el repo y 79 tests. |

## Decisiones de diseño

- **`AsyncSqliteSaver` en vez de `SqliteSaver`.** La consigna pide `SqliteSaver` y también código asíncrono. Probé las dos cosas juntas: el `SqliteSaver` síncrono falla con `NotImplementedError: The SqliteSaver does not support async methods` apenas se usa `ainvoke`. `AsyncSqliteSaver` es su versión asíncrona, viene en el mismo paquete (`langgraph-checkpoint-sqlite`), guarda en el mismo formato de base SQLite y se usa igual. Elegí respetar el requisito de `asyncio`.
- **Claude Haiku 4.5, sin `temperature`.** La consigna acepta OpenAI o Anthropic; uso Anthropic. No envío `temperature` porque la API la rechaza con este modelo (error 400 `temperature: Extra inputs are not permitted`), así que queda el valor por defecto del servicio. Por eso la redacción de las respuestas, y a veces la elección de herramientas, puede variar entre corridas.
- **Los errores de las herramientas viajan como texto, no como excepciones.** Un ID inexistente devuelve `{"error": ..., "ayuda": ...}`. Así el modelo lo lee y decide si corrige o pregunta. Para las excepciones inesperadas, el `ToolNode` está con `handle_tool_errors=True` y el texto del error también vuelve al modelo.
- **Herramientas de solo lectura y sin SQL libre.** El modelo nunca arma una consulta: elige una herramienta y completa argumentos validados con Pydantic (principio de mínimo privilegio).
- **`recursion_limit=10`.** Cada vuelta modelo → herramienta cuenta 2 pasos y la respuesta final 1, así que entran hasta 4 rondas de herramientas. Si el modelo se pasa, el turno se corta con un aviso en vez de gastar API sin fin. Al cortar, el hilo se deja en un estado válido (cada pedido de herramienta pendiente se cierra con un mensaje de error); si no, la pregunta siguiente del mismo hilo fallaría en Anthropic por una llamada sin respuesta.
- **Estado que no crece sin control.** Al modelo le mando solo el final del historial (por defecto, los últimos 30 mensajes, siempre desde un mensaje del usuario para no dejar respuestas de herramientas huérfanas). El historial completo sigue guardado en SQLite.
- **El estado no tiene campos extra.** Todo lo que el agente necesita recordar (preguntas, llamadas, resultados) ya está en los mensajes, así que no agregué campos solo para cumplir.

## Límites

- Los datos son ficticios y de juguete: armé 12 incidentes y 12 reglas.
- Mis tests prueban el grafo con un modelo falso que sigue un guion: verifican el enrutamiento, la ejecución de herramientas, la memoria y los límites, **no** que Claude decida bien. Eso se ve en la corrida real de `--demo`, cuya traza está en `trazas/`.
- Recorto el historial contando mensajes, no tokens.
- SQLite local es para desarrollo, como indica la consigna. Para producción lo cambiaría por un checkpointer de Postgres sin tocar el grafo.
