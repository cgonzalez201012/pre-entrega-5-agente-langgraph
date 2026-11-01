"""Persistencia con SQLite y thread_id: la memoria sobrevive entre preguntas y entre aperturas de la base."""

import sqlite3

from langchain_core.messages import HumanMessage

from agente import abrir_agente, historial, leer_historial_guardado, preguntar
from config import Configuracion
from tests.fakes import modelo_con_guion, pedir, responder


async def test_con_el_mismo_thread_id_el_modelo_ve_la_pregunta_anterior(config: Configuracion) -> None:
    llm = modelo_con_guion(
        pedir("detalle_incidente", incidente_id="INC-1001"),
        responder("El responsable es el Gerente Comercial."),
        responder("Es INC-1001."),
    )
    async with abrir_agente(config, llm) as agente:
        await preguntar(agente, "¿Quién es el responsable de INC-1001?", "sesion-a", 10)
        await preguntar(agente, "¿De qué incidente hablábamos?", "sesion-a", 10)

    contexto_2da_pregunta = llm.recibidos[-1]
    textos = [str(m.content) for m in contexto_2da_pregunta]
    assert any("INC-1001" in t and "responsable" in t for t in textos)  # la 1ra pregunta
    assert any("Gerente Comercial" in t for t in textos)  # y su respuesta
    # y tambien la observacion de la herramienta de la 1ra vuelta
    assert any(getattr(m, "tool_call_id", None) for m in contexto_2da_pregunta)


async def test_el_estado_acumula_los_mensajes_en_lugar_de_pisarlos(config: Configuracion) -> None:
    llm = modelo_con_guion(responder("uno"), responder("dos"), responder("tres"))
    async with abrir_agente(config, llm) as agente:
        for pregunta in ("a", "b", "c"):
            await preguntar(agente, pregunta, "acumula", 10)
        mensajes = await historial(agente, "acumula")
    assert [m.content for m in mensajes] == ["a", "uno", "b", "dos", "c", "tres"]


async def test_hilos_distintos_no_comparten_memoria(config: Configuracion) -> None:
    llm = modelo_con_guion(responder("para A"), responder("para B"))
    async with abrir_agente(config, llm) as agente:
        await preguntar(agente, "secreto de A", "hilo-a", 10)
        await preguntar(agente, "hola desde B", "hilo-b", 10)
        mensajes_b = await historial(agente, "hilo-b")
    assert all("secreto de A" not in str(m.content) for m in mensajes_b)
    assert all("secreto de A" not in str(m.content) for m in llm.recibidos[-1])


async def test_la_memoria_sobrevive_al_cierre_y_reapertura_de_la_base(config: Configuracion) -> None:
    """Equivale a cortar el programa y volver a correrlo: otro AsyncSqliteSaver, mismo archivo."""
    primer_modelo = modelo_con_guion(responder("Anotado: el incidente es INC-1001."))
    async with abrir_agente(config, primer_modelo) as agente:
        await preguntar(agente, "Hablemos de INC-1001.", "reinicio", 10)

    assert config.ruta_db.exists()

    segundo_modelo = modelo_con_guion(responder("Hablábamos de INC-1001."))
    async with abrir_agente(config, segundo_modelo) as agente:
        await preguntar(agente, "¿De qué hablábamos?", "reinicio", 10)

    recibido = [str(m.content) for m in segundo_modelo.recibidos[0]]
    assert "Hablemos de INC-1001." in recibido  # el modelo "nuevo" ya conocia la charla anterior


async def test_leer_historial_guardado_no_necesita_el_agente(config: Configuracion) -> None:
    llm = modelo_con_guion(responder("hola"))
    async with abrir_agente(config, llm) as agente:
        await preguntar(agente, "buen día", "lectura", 10)

    mensajes = await leer_historial_guardado(config.ruta_db, "lectura")
    assert [type(m) for m in mensajes][0] is HumanMessage
    assert [m.content for m in mensajes] == ["buen día", "hola"]


async def test_leer_historial_de_un_hilo_o_una_base_inexistente_devuelve_vacio(config: Configuracion) -> None:
    assert await leer_historial_guardado(config.ruta_db, "nada") == []  # la base ni existe
    async with abrir_agente(config, modelo_con_guion()):
        pass
    assert await leer_historial_guardado(config.ruta_db, "nada") == []  # existe pero sin ese hilo


async def test_la_base_es_un_sqlite_real_con_checkpoints(config: Configuracion) -> None:
    llm = modelo_con_guion(responder("ok"))
    async with abrir_agente(config, llm) as agente:
        await preguntar(agente, "hola", "tabla", 10)

    conexion = sqlite3.connect(config.ruta_db)
    try:
        tablas = {fila[0] for fila in conexion.execute("select name from sqlite_master where type='table'")}
        hilos = {fila[0] for fila in conexion.execute("select distinct thread_id from checkpoints")}
    finally:
        conexion.close()
    assert "checkpoints" in tablas
    assert hilos == {"tabla"}
