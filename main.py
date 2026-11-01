"""Linea de comandos del agente de Gobierno de Datos.

Ejemplos:
    python main.py "¿Cuántos incidentes críticos abiertos tiene el maestro de artículos?"
    python main.py --thread auditoria-1 "¿Y quién es el responsable del más antiguo?"
    python main.py --chat --thread auditoria-1
    python main.py --ver-hilo auditoria-1
    python main.py --demo          # corrida completa; guarda la traza en trazas/
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence

import anthropic
from langchain_core.messages import BaseMessage

from agente import abrir_agente, leer_historial_guardado, preguntar
from config import BASE_DIR, Configuracion
from errores import ErrorConfiguracion
from trazas import Turno, a_log, guardar_traza, texto_de

THREAD_POR_DEFECTO = "consola"
CARPETA_TRAZAS = BASE_DIR / "trazas"

# Escenarios de la corrida --demo. Son SOLO preguntas: que herramientas usar y en que
# orden lo decide el modelo.
PREGUNTA_MULTIPASO = (
    "¿Cuántos incidentes críticos abiertos tiene el maestro de artículos y "
    "quién es el responsable del más antiguo?"
)
PREGUNTA_MEMORIA = "¿Y qué regla de calidad incumple ese incidente? ¿Cumple hoy el umbral?"
PREGUNTA_ERROR = "Dame el detalle del incidente INC-1010."
PREGUNTA_TRAS_REINICIO = "Recordame: ¿de qué incidente veníamos hablando y quién era su responsable?"


def _mostrar_turno(turno: Turno, detalle: bool) -> None:
    if detalle:
        print(a_log([turno]))
        return
    print(turno.respuesta)
    aviso = " (se alcanzó el límite de pasos)" if turno.limite_de_pasos_alcanzado else ""
    print(f"\n[herramientas llamadas: {turno.llamadas_a_herramientas}{aviso}]")


async def _una_pregunta(config: Configuracion, pregunta: str, thread_id: str, detalle: bool) -> None:
    async with abrir_agente(config) as agente:
        turno = await preguntar(agente, pregunta, thread_id, config.recursion_limit)
    _mostrar_turno(turno, detalle)


async def _chat(config: Configuracion, thread_id: str, detalle: bool) -> None:
    print(f"Chat con el agente (hilo '{thread_id}'). Escribí 'salir' para terminar.")
    print("Si volvés a abrir el chat con el mismo --thread, el agente recuerda lo anterior.\n")
    async with abrir_agente(config) as agente:
        while True:
            try:
                pregunta = (await asyncio.to_thread(input, "Vos: ")).strip()
            except EOFError:
                break
            if pregunta.lower() in {"salir", "exit", "quit"}:
                break
            if not pregunta:
                continue
            turno = await preguntar(agente, pregunta, thread_id, config.recursion_limit)
            print()
            _mostrar_turno(turno, detalle)
            print()


async def _ver_hilo(config: Configuracion, thread_id: str) -> None:
    mensajes: Sequence[BaseMessage] = await leer_historial_guardado(config.ruta_db, thread_id)
    if not mensajes:
        print(f"No hay historial guardado para el hilo '{thread_id}' en {config.ruta_db}.")
        return
    print(f"Historial del hilo '{thread_id}' ({len(mensajes)} mensajes) en {config.ruta_db}:\n")
    for mensaje in mensajes:
        rol = {"human": "Usuario", "ai": "Agente", "tool": "Herramienta"}.get(mensaje.type, mensaje.type)
        pedidas = getattr(mensaje, "tool_calls", None) or []
        detalle = texto_de(mensaje) or ", ".join(f"{c['name']}({c['args']})" for c in pedidas)
        print(f"[{rol}] {detalle}")


async def _demo(config: Configuracion) -> None:
    """Corrida completa con el modelo real. Guarda la traza como JSON y como log."""
    ruta_demo = config.ruta_db.with_name("demo.sqlite")
    ruta_demo.unlink(missing_ok=True)  # la demo siempre arranca de cero
    config_demo = Configuracion(
        anthropic_api_key=config.anthropic_api_key,
        modelo=config.modelo,
        ruta_db=ruta_demo,
        recursion_limit=config.recursion_limit,
        max_mensajes_contexto=config.max_mensajes_contexto,
    )
    turnos: list[Turno] = []

    async with abrir_agente(config_demo) as agente:
        for escenario, thread_id, pregunta in [
            ("1. Razonamiento multi-paso", "demo-multipaso", PREGUNTA_MULTIPASO),
            ("2. Memoria: mismo thread_id", "demo-multipaso", PREGUNTA_MEMORIA),
            ("3. Ciclo de retorno ante un error", "demo-error", PREGUNTA_ERROR),
        ]:
            print(f"--> {escenario}")
            turno = await preguntar(agente, pregunta, thread_id, config.recursion_limit, escenario)
            turnos.append(turno)
            _mostrar_turno(turno, detalle=False)
            print()

    # Se cierra la base y se abre de nuevo: es lo mismo que cortar el programa y volver a correrlo.
    async with abrir_agente(config_demo) as agente:
        print("--> 4. Memoria tras reabrir la base (equivale a reiniciar el programa)")
        turno = await preguntar(
            agente, PREGUNTA_TRAS_REINICIO, "demo-multipaso", config.recursion_limit, "4. Memoria tras reiniciar"
        )
        turnos.append(turno)
        _mostrar_turno(turno, detalle=False)

    ruta_json, ruta_log = guardar_traza(
        CARPETA_TRAZAS,
        "traza-demo",
        turnos,
        {
            "proyecto": "pre-entrega-5-agente-langgraph",
            "modelo": config.modelo,
            "recursion_limit": config.recursion_limit,
            "checkpointer": "AsyncSqliteSaver",
        },
    )
    print(f"\nTraza guardada en:\n  {ruta_json}\n  {ruta_log}")


def _parsear(argumentos: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agente de Gobierno de Datos (LangGraph + Claude).")
    parser.add_argument("pregunta", nargs="?", help="Pregunta para el agente.")
    parser.add_argument("--thread", default=THREAD_POR_DEFECTO, help="thread_id: misma sesión = misma memoria.")
    parser.add_argument("--chat", action="store_true", help="Conversación interactiva.")
    parser.add_argument("--demo", action="store_true", help="Corrida completa de 4 escenarios; guarda la traza.")
    parser.add_argument("--ver-hilo", metavar="THREAD_ID", help="Muestra el historial guardado de un hilo.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Muestra cada paso del ciclo ReAct.")
    return parser.parse_args(argumentos)


def _ejecutar(args: argparse.Namespace) -> int:
    if args.ver_hilo:
        config = Configuracion.desde_entorno(requerir_clave=False)
        asyncio.run(_ver_hilo(config, args.ver_hilo))
        return 0

    config = Configuracion.desde_entorno()
    if args.demo:
        asyncio.run(_demo(config))
    elif args.chat:
        asyncio.run(_chat(config, args.thread, args.verbose))
    elif args.pregunta:
        asyncio.run(_una_pregunta(config, args.pregunta, args.thread, args.verbose))
    else:
        print("Indicá una pregunta, --chat, --demo o --ver-hilo. Mirá python main.py --help.")
        return 2
    return 0


def main(argumentos: Sequence[str] | None = None) -> int:
    args = _parsear(argumentos)
    try:
        return _ejecutar(args)
    except ErrorConfiguracion as error:
        print(f"Error de configuración: {error}", file=sys.stderr)
    except anthropic.AuthenticationError:
        print("Anthropic rechazó la clave. Revisá ANTHROPIC_API_KEY en el .env.", file=sys.stderr)
    except anthropic.RateLimitError:
        print("Anthropic devolvió límite de uso (429). Esperá un rato y probá de nuevo.", file=sys.stderr)
    except anthropic.APIConnectionError:
        print("No pude conectarme con Anthropic. Revisá tu conexión a internet.", file=sys.stderr)
    except anthropic.APIStatusError as error:
        print(f"Anthropic devolvió un error ({error.status_code}): {error.message}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrumpido. La memoria del hilo queda guardada.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
