"""La traza ReAct: pasos, log con flechas y archivos JSON/log."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agente import abrir_agente, preguntar
from config import Configuracion
from tests.fakes import modelo_con_guion, pedir, responder
from trazas import Paso, Turno, a_log, extraer_pasos, guardar_traza, texto_de


def test_texto_de_acepta_texto_y_bloques_de_claude() -> None:
    assert texto_de(AIMessage(content="  hola ")) == "hola"
    bloques = [{"type": "text", "text": "Voy a buscar."}, {"type": "tool_use", "id": "x", "name": "n", "input": {}}]
    assert texto_de(AIMessage(content=bloques)) == "Voy a buscar."


def test_extraer_pasos_numera_y_enlaza_cada_observacion_con_su_herramienta() -> None:
    llamada = pedir("detalle_incidente", incidente_id="INC-1001")
    mensajes = [
        HumanMessage(content="¿Responsable de INC-1001?"),
        llamada,
        ToolMessage(content='{"responsable": "Gerente Comercial"}', tool_call_id=str(llamada.tool_calls[0]["id"])),
        AIMessage(content="El Gerente Comercial."),
    ]
    pasos = extraer_pasos(mensajes)
    assert [p.numero for p in pasos] == [1, 2, 3, 4]
    assert pasos[1].herramienta == "detalle_incidente" and pasos[1].argumentos == {"incidente_id": "INC-1001"}
    assert pasos[2].herramienta == "detalle_incidente"  # la observacion sabe de que herramienta viene
    assert pasos[2].resultado == {"responsable": "Gerente Comercial"}
    assert not pasos[2].es_error


def test_un_resultado_con_error_se_marca_como_error() -> None:
    llamada = pedir("detalle_incidente", incidente_id="X")
    pasos = extraer_pasos(
        [llamada, ToolMessage(content='{"error": "no existe"}', tool_call_id=str(llamada.tool_calls[0]["id"]))]
    )
    assert pasos[1].es_error


def test_el_log_usa_el_formato_de_flechas_de_la_consigna() -> None:
    turno = Turno(
        escenario="prueba",
        thread_id="t",
        pregunta="p",
        respuesta="r",
        pasos=[
            Paso(1, "usuario", contenido="¿Cuántos?"),
            Paso(2, "decision_herramienta", herramienta="buscar", argumentos={"maestro": "articulos"}),
            Paso(3, "observacion", herramienta="buscar", resultado={"cantidad": 2}),
            Paso(4, "respuesta", contenido="Dos."),
        ],
    )
    log = a_log([turno])
    assert 'Usuario: "¿Cuántos?"' in log
    assert "→ El agente decide usar la herramienta: buscar(maestro='articulos')" in log
    assert '→ La herramienta devuelve: {"cantidad": 2}' in log
    assert 'Respuesta: "Dos."' in log
    assert "[herramientas llamadas en este turno: 1]" in log


async def test_guardar_traza_escribe_json_valido_y_log(config: Configuracion, tmp_path: Path) -> None:
    llm = modelo_con_guion(
        pedir("buscar_incidentes_calidad", maestro="articulos"),
        pedir("detalle_incidente", incidente_id="INC-1001"),
        responder("Listo."),
    )
    async with abrir_agente(config, llm) as agente:
        turno = await preguntar(agente, "pregunta", "t", 10, escenario="multi-paso")

    ruta_json, ruta_log = guardar_traza(tmp_path / "trazas", "demo", [turno], {"modelo": "falso"})
    documento = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert documento["modelo"] == "falso"
    assert "generado_en" in documento
    assert documento["turnos"][0]["llamadas_a_herramientas"] == 2
    assert documento["turnos"][0]["pasos"][1]["herramienta"] == "buscar_incidentes_calidad"
    assert "El agente decide usar la herramienta" in ruta_log.read_text(encoding="utf-8")


def test_el_texto_previo_a_las_herramientas_se_muestra_una_sola_vez() -> None:
    mensaje = AIMessage(
        content="Voy a consultar dos cosas.",
        tool_calls=[
            {"name": "detalle_incidente", "args": {"incidente_id": "INC-1001"}, "id": "a", "type": "tool_call"},
            {"name": "consultar_regla_calidad", "args": {"regla_id": "R-ART-01"}, "id": "b", "type": "tool_call"},
        ],
    )
    pasos = extraer_pasos([mensaje])
    assert [p.contenido for p in pasos] == ["Voy a consultar dos cosas.", ""]
