"""Agente ReAct con LangGraph: grafo, persistencia y ejecucion de un turno.

El grafo tiene solo dos nodos y un ciclo:

        START -> agente --(¿pidio herramientas?)--> tools
                   ^              |                    |
                   |              +-- no --> END       |
                   +-----------------------------------+

- `agente`: llama al modelo con las herramientas enlazadas (`bind_tools`).
- `tools`:  ejecuta las herramientas que pidio el modelo (`ToolNode`).
- Arista condicional `tools_condition`: si el ultimo mensaje trae `tool_calls` va a
  `tools`; si no, termina. La decision de usar una herramienta es del modelo: en el
  codigo no hay ningun if/else sobre el contenido de la pregunta.
"""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import Configuracion
from herramientas import HERRAMIENTAS
from prompts import PROMPT_SISTEMA
from trazas import Turno, extraer_pasos, texto_de

NODO_MODELO = "agente"
NODO_HERRAMIENTAS = "tools"

AVISO_LIMITE = (
    "Me quedé sin pasos para resolver esto (alcancé el límite de recursión). "
    "Probá con una pregunta más acotada."
)


class EstadoAgente(MessagesState):
    """Estado del grafo.

    Hereda `messages` de MessagesState: una lista de mensajes con el reducer
    `add_messages`, que ANEXA lo que devuelve cada nodo en lugar de pisar la lista.
    No agrego campos propios porque todo lo que el agente necesita recordar
    (preguntas, llamadas a herramientas, resultados) ya vive en los mensajes.
    """


def recortar_historial(mensajes: Sequence[BaseMessage], maximo: int) -> list[BaseMessage]:
    """Se queda con los ultimos `maximo` mensajes para no crecer sin limite ("estado sucio").

    Solo recorta lo que se le MANDA al modelo; el historial completo sigue guardado en
    el checkpoint. Siempre arranca en un mensaje del usuario, asi nunca queda una
    respuesta de herramienta huerfana (sin la llamada que la origino), que Anthropic
    rechaza.
    """
    recortados = trim_messages(
        list(mensajes),
        max_tokens=maximo,
        token_counter=len,  # cuenta mensajes, no tokens
        strategy="last",
        start_on="human",
        allow_partial=False,
    )
    if recortados:
        return list(recortados)
    # El turno actual es mas largo que el maximo: se conserva completo desde la ultima pregunta.
    ultimo = max((i for i, m in enumerate(mensajes) if isinstance(m, HumanMessage)), default=0)
    return list(mensajes[ultimo:])


def construir_grafo(
    llm: BaseChatModel, checkpointer: BaseCheckpointSaver | None, max_mensajes_contexto: int
) -> CompiledStateGraph:
    """Arma y compila el StateGraph ReAct con el modelo y el checkpointer que se le pasen."""
    llm_con_herramientas = llm.bind_tools(HERRAMIENTAS)

    async def llamar_modelo(estado: EstadoAgente) -> dict[str, Any]:
        contexto = [SystemMessage(content=PROMPT_SISTEMA), *recortar_historial(estado["messages"], max_mensajes_contexto)]
        respuesta = await llm_con_herramientas.ainvoke(contexto)
        return {"messages": [respuesta]}

    grafo = StateGraph(EstadoAgente)
    # Los tipos de langgraph 1.2 no reconocen una funcion async con TypedDict como nodo
    # (mypy), aunque es el uso documentado: por eso el ignore acotado a esta linea.
    grafo.add_node(NODO_MODELO, llamar_modelo)  # type: ignore[call-overload]
    # handle_tool_errors=True: si una herramienta lanza una excepcion inesperada, el
    # texto del error vuelve al modelo como ToolMessage en vez de romper el grafo.
    grafo.add_node(NODO_HERRAMIENTAS, ToolNode(HERRAMIENTAS, handle_tool_errors=True))
    grafo.add_edge(START, NODO_MODELO)
    grafo.add_conditional_edges(NODO_MODELO, tools_condition, {"tools": NODO_HERRAMIENTAS, END: END})
    grafo.add_edge(NODO_HERRAMIENTAS, NODO_MODELO)  # <- el ciclo: observar y volver a razonar
    return grafo.compile(checkpointer=checkpointer)


def crear_llm(config: Configuracion) -> ChatAnthropic:
    """Modelo de Anthropic.

    No se envia `temperature`: la API de Anthropic la rechaza con este modelo
    (error 400 "temperature: Extra inputs are not permitted"), asi que se deja el valor del servicio.
    """
    return ChatAnthropic(  # type: ignore[call-arg]
        model=config.modelo,
        api_key=config.anthropic_api_key,  # type: ignore[arg-type]
        max_tokens=1024,
    )


@asynccontextmanager
async def abrir_agente(config: Configuracion, llm: BaseChatModel | None = None) -> AsyncIterator[CompiledStateGraph]:
    """Abre la base SQLite de memoria y entrega el agente listo para usar.

    Uso asincrono (`AsyncSqliteSaver`, la version async de `SqliteSaver`: el SqliteSaver
    sincrono lanza NotImplementedError con `ainvoke`). Al salir del bloque se cierra la
    conexion; la memoria queda en el archivo y la retoma cualquier proceso posterior.
    """
    config.ruta_db.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(config.ruta_db)) as checkpointer:
        yield construir_grafo(llm or crear_llm(config), checkpointer, config.max_mensajes_contexto)


def _configuracion_de_hilo(thread_id: str, recursion_limit: int) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


async def historial(agente: CompiledStateGraph, thread_id: str, recursion_limit: int = 10) -> list[BaseMessage]:
    """Mensajes guardados de un hilo (vacio si el hilo no existe)."""
    instantanea = await agente.aget_state(_configuracion_de_hilo(thread_id, recursion_limit))
    mensajes: list[BaseMessage] = instantanea.values.get("messages", [])
    return mensajes


async def leer_historial_guardado(ruta_db: Path, thread_id: str) -> list[BaseMessage]:
    """Lee de la base SQLite los mensajes de un hilo, sin armar el agente (no necesita clave)."""
    if not ruta_db.exists():
        return []
    async with AsyncSqliteSaver.from_conn_string(str(ruta_db)) as checkpointer:
        guardado = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    if guardado is None:
        return []
    mensajes: list[BaseMessage] = guardado.checkpoint["channel_values"].get("messages", [])
    return mensajes


async def _cerrar_turno_cortado(agente: CompiledStateGraph, configuracion: RunnableConfig) -> None:
    """Deja el hilo en un estado valido despues de cortar por recursion_limit.

    Al cortar, el ultimo mensaje puede ser una peticion de herramienta sin respuesta.
    Si se dejara asi, la siguiente pregunta del mismo hilo haria fallar a Anthropic
    ("tool_use sin tool_result"). Se responde cada peticion pendiente con un
    ToolMessage de error y se agrega un aviso del agente.
    """
    instantanea = await agente.aget_state(configuracion)
    mensajes: list[BaseMessage] = instantanea.values.get("messages", [])
    cierre: list[BaseMessage] = []
    if mensajes and isinstance(mensajes[-1], AIMessage):
        for llamada in mensajes[-1].tool_calls:
            cierre.append(
                ToolMessage(
                    content="Interrumpido: se alcanzó el límite de pasos antes de ejecutar esta herramienta.",
                    tool_call_id=str(llamada["id"]),
                    name=llamada["name"],
                    status="error",
                )
            )
    cierre.append(AIMessage(content=AVISO_LIMITE))
    await agente.aupdate_state(configuracion, {"messages": cierre}, as_node=NODO_MODELO)


async def preguntar(
    agente: CompiledStateGraph,
    pregunta: str,
    thread_id: str,
    recursion_limit: int,
    escenario: str = "consulta",
) -> Turno:
    """Hace UNA pregunta en un hilo y devuelve el turno con su traza.

    Con el mismo `thread_id` el agente retoma el historial guardado por el checkpointer.
    `recursion_limit` es el techo de pasos del grafo para esta pregunta (cada vuelta
    modelo -> herramienta cuenta 2); si se supera, el turno se corta con un aviso.
    """
    configuracion = _configuracion_de_hilo(thread_id, recursion_limit)
    antes = len(await historial(agente, thread_id, recursion_limit))
    limite_alcanzado = False
    try:
        await agente.ainvoke({"messages": [HumanMessage(content=pregunta)]}, configuracion)
    except GraphRecursionError:
        limite_alcanzado = True
        await _cerrar_turno_cortado(agente, configuracion)

    nuevos = (await historial(agente, thread_id, recursion_limit))[antes:]
    respuesta = texto_de(nuevos[-1]) if nuevos else ""
    return Turno(
        escenario=escenario,
        thread_id=thread_id,
        pregunta=pregunta,
        respuesta=respuesta,
        pasos=extraer_pasos(nuevos),
        limite_de_pasos_alcanzado=limite_alcanzado,
    )
