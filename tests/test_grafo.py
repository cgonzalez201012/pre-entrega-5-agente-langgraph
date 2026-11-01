"""El grafo ReAct: estructura, ciclo multi-paso, ciclo de retorno y recursion_limit."""

from typing import get_type_hints

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

from agente import (
    AVISO_LIMITE,
    EstadoAgente,
    abrir_agente,
    construir_grafo,
    historial,
    preguntar,
    recortar_historial,
)
from config import Configuracion
from prompts import PROMPT_SISTEMA
from tests.fakes import ModeloPorPolitica, modelo_con_guion, pedir, pedir_varias, responder

# ---------- estructura ----------


def test_el_estado_hereda_de_messagesstate_con_reducer_add_messages() -> None:
    campos = get_type_hints(EstadoAgente, include_extras=True)
    assert "messages" in campos
    assert set(get_type_hints(MessagesState)) <= set(campos)
    reductor = campos["messages"].__metadata__[0]
    assert reductor is add_messages


def test_el_grafo_tiene_modelo_herramientas_y_aristas_correctas() -> None:
    agente = construir_grafo(modelo_con_guion(), None, 30)
    grafo = agente.get_graph()
    assert {"agente", "tools"} <= set(grafo.nodes)
    aristas = {(a.source, a.target, a.conditional) for a in grafo.edges}
    assert ("__start__", "agente", False) in aristas
    assert ("agente", "tools", True) in aristas  # tools_condition
    assert ("agente", "__end__", True) in aristas
    assert ("tools", "agente", False) in aristas  # el ciclo ReAct


# ---------- ciclo ReAct ----------


async def test_sin_herramientas_responde_directo(config: Configuracion) -> None:
    llm = modelo_con_guion(responder("Ese tema no es mío."))
    async with abrir_agente(config, llm) as agente:
        turno = await preguntar(agente, "¿Qué hora es?", "t", 10)
    assert turno.llamadas_a_herramientas == 0
    assert turno.respuesta == "Ese tema no es mío."


async def test_razonamiento_multi_paso_usa_la_herramienta_dos_veces(config: Configuracion) -> None:
    llm = modelo_con_guion(
        pedir("buscar_incidentes_calidad", maestro="articulos", estado="abierto", severidad="critica"),
        pedir("detalle_incidente", incidente_id="INC-1001"),
        responder("Hay 2 críticos abiertos; el más antiguo es INC-1001 (Gerente Comercial)."),
    )
    async with abrir_agente(config, llm) as agente:
        turno = await preguntar(agente, "¿Cuántos críticos hay y quién responde por el más antiguo?", "t", 10)

    assert turno.llamadas_a_herramientas == 2
    tipos = [p.tipo for p in turno.pasos]
    assert tipos == ["usuario", "decision_herramienta", "observacion", "decision_herramienta", "observacion", "respuesta"]
    herramientas = [p.herramienta for p in turno.pasos if p.tipo == "decision_herramienta"]
    assert herramientas == ["buscar_incidentes_calidad", "detalle_incidente"]
    # la 2da llamada usa un dato que salio de la 1ra observacion
    assert turno.pasos[2].resultado["incidentes"][0]["id"] == turno.pasos[3].argumentos["incidente_id"]


async def test_dos_llamadas_en_paralelo_vuelven_cada_una_con_su_resultado(config: Configuracion) -> None:
    llm = modelo_con_guion(
        pedir_varias(
            ("detalle_incidente", {"incidente_id": "INC-1001"}),
            ("consultar_regla_calidad", {"regla_id": "R-ART-01"}),
        ),
        responder("Listo."),
    )
    async with abrir_agente(config, llm) as agente:
        turno = await preguntar(agente, "Dame el detalle y la regla.", "t", 10)
    observaciones = [p for p in turno.pasos if p.tipo == "observacion"]
    assert [o.herramienta for o in observaciones] == ["detalle_incidente", "consultar_regla_calidad"]
    assert observaciones[0].resultado["id"] == "INC-1001"
    assert observaciones[1].resultado["id"] == "R-ART-01"


async def test_el_modelo_recibe_el_prompt_de_sistema_primero(config: Configuracion) -> None:
    llm = modelo_con_guion(responder("ok"))
    async with abrir_agente(config, llm) as agente:
        await preguntar(agente, "hola", "t", 10)
    primer_contexto = llm.recibidos[0]
    assert isinstance(primer_contexto[0], SystemMessage)
    assert primer_contexto[0].content == PROMPT_SISTEMA


# ---------- ciclo de retorno ante un error ----------


async def test_ante_un_error_de_la_herramienta_el_agente_hace_un_segundo_intento(config: Configuracion) -> None:
    """El modelo falso reacciona al texto del error, igual que se espera del real."""

    def politica(mensajes: list[BaseMessage]) -> AIMessage:
        ultimo = mensajes[-1]
        if isinstance(ultimo, HumanMessage):
            return pedir("detalle_incidente", incidente_id="INC-1010")
        assert isinstance(ultimo, ToolMessage)
        if "error" in str(ultimo.content):
            return pedir("buscar_incidentes_calidad", maestro="articulos", estado="todos")
        return responder("El incidente INC-1010 no existe; los de artículos van de INC-1001 a INC-1005.")

    llm = ModeloPorPolitica(politica=politica)
    async with abrir_agente(config, llm) as agente:
        turno = await preguntar(agente, "Dame el detalle del incidente INC-1010.", "t", 10)

    observaciones = [p for p in turno.pasos if p.tipo == "observacion"]
    assert observaciones[0].es_error  # primer intento: la herramienta dijo que no existe
    assert not observaciones[1].es_error  # segundo intento: la busqueda funciono
    assert turno.llamadas_a_herramientas == 2


# ---------- recursion_limit ----------


def _siempre_pide_herramienta() -> ModeloPorPolitica:
    return ModeloPorPolitica(politica=lambda _: pedir("buscar_incidentes_calidad", maestro="articulos"))


def _sin_peticiones_huerfanas(mensajes: list[BaseMessage]) -> bool:
    """Cada AIMessage con tool_calls tiene un ToolMessage por cada id (lo exige Anthropic)."""
    respondidos = {m.tool_call_id for m in mensajes if isinstance(m, ToolMessage)}
    return all(
        str(c["id"]) in respondidos for m in mensajes if isinstance(m, AIMessage) for c in m.tool_calls
    )


@pytest.mark.parametrize("limite", [3, 4, 6])
async def test_recursion_limit_corta_un_bucle_y_deja_el_hilo_utilizable(
    config: Configuracion, limite: int
) -> None:
    llm = _siempre_pide_herramienta()
    async with abrir_agente(config, llm) as agente:
        turno = await preguntar(agente, "Listame todo, una y otra vez.", "bucle", limite)

        assert turno.limite_de_pasos_alcanzado
        assert turno.respuesta == AVISO_LIMITE
        # el modelo no fue llamado mas veces que las que permite el limite
        assert len(llm.recibidos) <= limite
        mensajes = await historial(agente, "bucle")
        assert _sin_peticiones_huerfanas(mensajes)

        # y la conversacion sigue: otra pregunta en el mismo hilo se procesa normalmente
        llm.politica = lambda _: responder("Seguimos.")
        segundo = await preguntar(agente, "Bueno, ¿seguimos?", "bucle", limite)
    assert not segundo.limite_de_pasos_alcanzado
    assert segundo.respuesta == "Seguimos."


async def test_el_limite_no_corta_un_razonamiento_que_entra(config: Configuracion) -> None:
    llm = modelo_con_guion(pedir("consultar_regla_calidad", regla_id="R-ART-01"), responder("Listo."))
    async with abrir_agente(config, llm) as agente:
        turno = await preguntar(agente, "¿Qué exige R-ART-01?", "t", 5)  # necesita 3 pasos
    assert not turno.limite_de_pasos_alcanzado


# ---------- estado que crece: recorte del contexto ----------


def _conversacion(vueltas: int) -> list[BaseMessage]:
    mensajes: list[BaseMessage] = []
    for i in range(vueltas):
        mensajes.append(HumanMessage(content=f"pregunta {i}"))
        mensajes.append(AIMessage(content=f"respuesta {i}"))
    return mensajes


def test_recortar_historial_respeta_el_maximo_y_arranca_en_un_mensaje_del_usuario() -> None:
    recortados = recortar_historial(_conversacion(20), 7)
    assert len(recortados) <= 7
    assert isinstance(recortados[0], HumanMessage)
    assert recortados[-1].content == "respuesta 19"  # conserva lo mas reciente


def test_recortar_historial_no_deja_respuestas_de_herramienta_huerfanas() -> None:
    llamada = pedir("detalle_incidente", incidente_id="INC-1001")
    mensajes: list[BaseMessage] = [
        *_conversacion(3),
        HumanMessage(content="detalle"),
        llamada,
        ToolMessage(content="{}", tool_call_id=str(llamada.tool_calls[0]["id"])),
        AIMessage(content="ok"),
    ]
    for maximo in range(4, len(mensajes)):
        recortados = recortar_historial(mensajes, maximo)
        assert isinstance(recortados[0], HumanMessage), f"maximo={maximo}"
        assert _sin_peticiones_huerfanas(recortados)


def test_si_el_turno_actual_es_mas_largo_que_el_maximo_se_conserva_entero() -> None:
    llamada = pedir("detalle_incidente", incidente_id="INC-1001")
    turno: list[BaseMessage] = [
        HumanMessage(content="pregunta larga"),
        llamada,
        ToolMessage(content="{}", tool_call_id=str(llamada.tool_calls[0]["id"])),
    ]
    assert recortar_historial([*_conversacion(5), *turno], 2) == turno


async def test_el_modelo_recibe_el_historial_recortado_pero_el_checkpoint_guarda_todo(
    config: Configuracion,
) -> None:
    chica = Configuracion(config.anthropic_api_key, config.modelo, config.ruta_db, 10, 6)
    llm = modelo_con_guion(*[responder(f"r{i}") for i in range(8)])
    async with abrir_agente(chica, llm) as agente:
        for i in range(8):
            await preguntar(agente, f"p{i}", "largo", 10)
        guardados = await historial(agente, "largo")
    assert len(guardados) == 16  # todo sigue guardado
    ultimo_contexto = llm.recibidos[-1]
    assert len(ultimo_contexto) <= 1 + 6  # prompt de sistema + maximo de mensajes
    assert ultimo_contexto[-1].content == "p7"


def test_el_prompt_fija_las_reglas_de_conducta_sin_enrutar_preguntas() -> None:
    assert "herramientas" in PROMPT_SISTEMA and "Nunca lo inventes" in PROMPT_SISTEMA
    assert "No agregues opiniones" in PROMPT_SISTEMA  # la corrida real agrego un juicio que ninguna herramienta informaba
    assert "ayuda" in PROMPT_SISTEMA and "aclaración" in PROMPT_SISTEMA  # que hacer ante un error
    # no hay una tabla "si preguntan X usa la herramienta Y": eso lo decide el modelo
    for nombre in ("buscar_incidentes_calidad", "detalle_incidente", "consultar_regla_calidad"):
        assert nombre not in PROMPT_SISTEMA


def test_el_modelo_se_crea_sin_enviar_temperature_a_la_api(config: Configuracion) -> None:
    """Anthropic rechaza `temperature` con este modelo (400: Extra inputs are not permitted)."""
    from agente import crear_llm

    llm = crear_llm(config)
    assert llm.model == "modelo-falso"
    assert llm.max_tokens == 1024
    carga = llm._get_request_payload([HumanMessage(content="hola")])  # lo que viajaria a la API
    assert "temperature" not in carga


@pytest.mark.parametrize("rondas, corta", [(4, False), (5, True)])
async def test_con_limite_10_caben_4_rondas_de_herramientas_y_la_quinta_corta(
    config: Configuracion, rondas: int, corta: bool
) -> None:
    """Cada ronda (modelo -> herramienta) cuenta 2 pasos y la respuesta final 1: 4*2+1=9 <= 10; 5*2+1=11 > 10."""
    guion = [pedir("consultar_regla_calidad", regla_id="R-ART-01") for _ in range(rondas)] + [responder("Listo.")]
    async with abrir_agente(config, modelo_con_guion(*guion)) as agente:
        turno = await preguntar(agente, "Consultá la regla varias veces.", "rondas", 10)
    assert turno.limite_de_pasos_alcanzado is corta
