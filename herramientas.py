"""Herramientas del agente (contrato: nombre + descripcion + esquema de entrada).

El modelo decide si usar una herramienta y con que argumentos leyendo SOLO el
docstring y el esquema de cada una; por eso las descripciones son largas y dicen
cuando usarla, cuando no, y que devuelve.

Reglas de las tres herramientas:
- Son de solo lectura (minimo privilegio): no hay SQL libre ni escrituras.
- Un error de la consulta NO levanta una excepcion: se devuelve como JSON con
  "error" y "ayuda", asi el modelo lo lee y decide si corrige o pregunta.
"""

import json
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from errores import ErrorConsulta
from repositorio import obtener_repositorio


def _a_texto(resultado: dict[str, Any]) -> str:
    """Las herramientas devuelven JSON en texto: es lo que el modelo lee."""
    return json.dumps(resultado, ensure_ascii=False)


def _error_a_texto(error: ErrorConsulta) -> str:
    return _a_texto({"error": error.mensaje, "ayuda": error.ayuda})


class BuscarIncidentesArgs(BaseModel):
    """Argumentos de buscar_incidentes_calidad."""

    maestro: str = Field(
        description="Maestro de datos a consultar. Valores validos: 'articulos', 'proveedores' o 'sucursales'."
    )
    estado: str = Field(
        default="abierto",
        description="Estado de los incidentes: 'abierto', 'en_curso', 'cerrado' o 'todos'. Por defecto 'abierto'.",
    )
    severidad: str | None = Field(
        default=None,
        description="Filtro opcional por severidad: 'critica', 'alta', 'media' o 'baja'. Si se omite trae todas.",
    )


class DetalleIncidenteArgs(BaseModel):
    """Argumentos de detalle_incidente."""

    incidente_id: str = Field(description="ID del incidente, con formato INC-NNNN (por ejemplo 'INC-1001').")


class ReglaCalidadArgs(BaseModel):
    """Argumentos de consultar_regla_calidad."""

    regla_id: str = Field(description="ID de la regla de calidad, con formato R-XXX-NN (por ejemplo 'R-ART-01').")


@tool("buscar_incidentes_calidad", args_schema=BuscarIncidentesArgs)
async def buscar_incidentes_calidad(maestro: str, estado: str = "abierto", severidad: str | None = None) -> str:
    """Lista los incidentes de calidad de datos de UN maestro de Mercado Austral S.A.

    Usala para saber CUÁNTOS incidentes hay y cuáles son (ID, titulo, severidad,
    estado, fecha de alta y dias transcurridos desde el alta). Se consulta un solo
    maestro por llamada: 'articulos', 'proveedores' o 'sucursales'. Se puede filtrar
    por estado (abierto, en_curso, cerrado o todos) y por severidad (critica, alta,
    media o baja).

    NO devuelve el responsable ni la regla incumplida: para eso hay que llamar a
    detalle_incidente con el ID de cada incidente que interese.

    Devuelve un JSON con 'cantidad' y 'incidentes'. Si el maestro, el estado o la
    severidad no existen, devuelve un JSON con 'error' y 'ayuda' (valores validos).
    """
    try:
        resultado = await obtener_repositorio().buscar_incidentes(maestro, estado, severidad)
    except ErrorConsulta as error:
        return _error_a_texto(error)
    return _a_texto(resultado)


@tool("detalle_incidente", args_schema=DetalleIncidenteArgs)
async def detalle_incidente(incidente_id: str) -> str:
    """Devuelve la ficha completa de UN incidente de calidad de datos a partir de su ID.

    Usala cuando ya tenés el ID (formato INC-NNNN, por ejemplo INC-1001) y necesitás
    datos que el listado no trae: responsable (Data Owner), Data Steward, regla de
    calidad incumplida (regla_id), cantidad de registros afectados y plazo de
    resolucion (SLA en horas habiles segun la severidad).

    Si no conocés el ID, primero obtenelo con buscar_incidentes_calidad: no lo
    inventes. Si el ID no existe, devuelve un JSON con 'error' y 'ayuda' que lista
    los IDs existentes.
    """
    try:
        resultado = await obtener_repositorio().detalle_incidente(incidente_id)
    except ErrorConsulta as error:
        return _error_a_texto(error)
    return _a_texto(resultado)


@tool("consultar_regla_calidad", args_schema=ReglaCalidadArgs)
async def consultar_regla_calidad(regla_id: str) -> str:
    """Devuelve una regla de calidad de datos: qué exige, su umbral y su medicion actual.

    Usala para explicar QUÉ regla incumple un incidente o si un maestro cumple el
    umbral de aceptacion. Necesita el ID de la regla (formato R-XXX-NN, por ejemplo
    R-ART-01); ese ID figura en el campo regla_id del detalle de un incidente
    (detalle_incidente), no en el listado de buscar_incidentes_calidad.

    Devuelve un JSON con nombre, descripcion, umbral_aceptacion_pct,
    medicion_actual_pct y cumple_umbral. Si el ID no existe, devuelve un JSON con
    'error' y 'ayuda'.
    """
    try:
        resultado = await obtener_repositorio().regla(regla_id)
    except ErrorConsulta as error:
        return _error_a_texto(error)
    return _a_texto(resultado)


HERRAMIENTAS: list[BaseTool] = [buscar_incidentes_calidad, detalle_incidente, consultar_regla_calidad]
