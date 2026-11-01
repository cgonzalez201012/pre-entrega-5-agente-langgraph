"""Traza del razonamiento ReAct: convierte los mensajes de un turno en pasos legibles.

Un turno (una pregunta del usuario) deja en el estado esta secuencia de mensajes:

    Human -> AI(con tool_calls) -> Tool -> AI(con tool_calls) -> Tool -> ... -> AI(respuesta)

`extraer_pasos` la traduce a pasos numerados que se guardan como JSON y como log.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

TIPO_USUARIO = "usuario"
TIPO_DECISION = "decision_herramienta"
TIPO_OBSERVACION = "observacion"
TIPO_RESPUESTA = "respuesta"


@dataclass(frozen=True)
class Paso:
    """Un paso del ciclo ReAct."""

    numero: int
    tipo: str
    contenido: str = ""
    herramienta: str | None = None
    argumentos: dict[str, Any] | None = None
    resultado: Any = None
    es_error: bool = False


@dataclass
class Turno:
    """Una pregunta del usuario, con todo lo que hizo el agente para responderla."""

    escenario: str
    thread_id: str
    pregunta: str
    respuesta: str
    pasos: list[Paso] = field(default_factory=list)
    limite_de_pasos_alcanzado: bool = False

    @property
    def llamadas_a_herramientas(self) -> int:
        return sum(1 for paso in self.pasos if paso.tipo == TIPO_DECISION)

    def a_dict(self) -> dict[str, Any]:
        return {
            "escenario": self.escenario,
            "thread_id": self.thread_id,
            "pregunta": self.pregunta,
            "respuesta": self.respuesta,
            "llamadas_a_herramientas": self.llamadas_a_herramientas,
            "limite_de_pasos_alcanzado": self.limite_de_pasos_alcanzado,
            "pasos": [asdict(paso) for paso in self.pasos],
        }


def texto_de(mensaje: BaseMessage) -> str:
    """Texto plano de un mensaje (Claude puede devolver una lista de bloques)."""
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido.strip()
    partes: list[str] = []
    for bloque in contenido:
        if isinstance(bloque, str):
            partes.append(bloque)
        elif isinstance(bloque, dict) and bloque.get("type") == "text":
            partes.append(str(bloque.get("text", "")))
    return "".join(partes).strip()


def _interpretar(contenido: str) -> Any:
    """Si la herramienta devolvio JSON lo muestra estructurado; si no, como texto."""
    try:
        return json.loads(contenido)
    except (json.JSONDecodeError, TypeError):
        return contenido


def extraer_pasos(mensajes: Sequence[BaseMessage]) -> list[Paso]:
    """Traduce los mensajes de UN turno a pasos numerados."""
    pasos: list[Paso] = []
    nombres: dict[str, str] = {}  # tool_call_id -> nombre de la herramienta

    def agregar(**datos: Any) -> None:
        pasos.append(Paso(numero=len(pasos) + 1, **datos))

    for mensaje in mensajes:
        if isinstance(mensaje, HumanMessage):
            agregar(tipo=TIPO_USUARIO, contenido=texto_de(mensaje))
        elif isinstance(mensaje, AIMessage) and mensaje.tool_calls:
            razonamiento = texto_de(mensaje)
            for llamada in mensaje.tool_calls:
                nombres[str(llamada["id"])] = llamada["name"]
                agregar(
                    tipo=TIPO_DECISION,
                    contenido=razonamiento,
                    herramienta=llamada["name"],
                    argumentos=dict(llamada["args"]),
                )
                razonamiento = ""  # el texto previo se muestra una sola vez
        elif isinstance(mensaje, AIMessage):
            agregar(tipo=TIPO_RESPUESTA, contenido=texto_de(mensaje))
        elif isinstance(mensaje, ToolMessage):
            resultado = _interpretar(texto_de(mensaje))
            es_error = mensaje.status == "error" or (isinstance(resultado, dict) and "error" in resultado)
            agregar(
                tipo=TIPO_OBSERVACION,
                herramienta=nombres.get(mensaje.tool_call_id),
                resultado=resultado,
                es_error=es_error,
            )
    return pasos


def a_log(turnos: Sequence[Turno]) -> str:
    """Version en texto, con el mismo formato de flechas que muestra la consigna."""
    lineas: list[str] = []
    for turno in turnos:
        lineas.append(f"=== {turno.escenario} (thread_id={turno.thread_id}) ===")
        for paso in turno.pasos:
            if paso.tipo == TIPO_USUARIO:
                lineas.append(f'Usuario: "{paso.contenido}"')
            elif paso.tipo == TIPO_DECISION:
                if paso.contenido:
                    lineas.append(f"   (el agente razona: {paso.contenido})")
                argumentos = ", ".join(f"{k}={v!r}" for k, v in (paso.argumentos or {}).items())
                lineas.append(f"→ El agente decide usar la herramienta: {paso.herramienta}({argumentos})")
            elif paso.tipo == TIPO_OBSERVACION:
                marca = " [ERROR]" if paso.es_error else ""
                texto = json.dumps(paso.resultado, ensure_ascii=False)
                lineas.append(f"→ La herramienta devuelve{marca}: {texto}")
            elif paso.tipo == TIPO_RESPUESTA:
                lineas.append("→ El agente ya tiene lo que necesita → responde.")
                lineas.append(f'Respuesta: "{paso.contenido}"')
        if turno.limite_de_pasos_alcanzado:
            lineas.append("!! Se alcanzó el límite de pasos (recursion_limit) y el turno se cortó.")
        lineas.append(f"[herramientas llamadas en este turno: {turno.llamadas_a_herramientas}]")
        lineas.append("")
    return "\n".join(lineas)


def guardar_traza(
    carpeta: Path, nombre: str, turnos: Sequence[Turno], metadatos: dict[str, Any]
) -> tuple[Path, Path]:
    """Escribe `<nombre>.json` y `<nombre>.log` en la carpeta y devuelve sus rutas."""
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_json = carpeta / f"{nombre}.json"
    ruta_log = carpeta / f"{nombre}.log"
    documento = {
        **metadatos,
        "generado_en": datetime.now(UTC).isoformat(timespec="seconds"),
        "turnos": [turno.a_dict() for turno in turnos],
    }
    ruta_json.write_text(json.dumps(documento, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ruta_log.write_text(a_log(turnos), encoding="utf-8")
    return ruta_json, ruta_log
