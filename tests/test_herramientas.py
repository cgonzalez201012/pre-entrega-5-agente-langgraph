"""Las tres herramientas: contrato, resultados y errores devueltos como texto."""

import json
from typing import Any

import pytest

import herramientas
from herramientas import (
    HERRAMIENTAS,
    buscar_incidentes_calidad,
    consultar_regla_calidad,
    detalle_incidente,
)


async def _llamar(herramienta: Any, **argumentos: Any) -> dict[str, Any]:
    """Invoca la herramienta como lo hace el ToolNode y parsea el JSON que devuelve."""
    texto = await herramienta.ainvoke(argumentos)
    resultado: dict[str, Any] = json.loads(texto)
    return resultado


# ---------- contrato ----------


def test_hay_al_menos_una_herramienta_con_nombre_y_descripcion_larga() -> None:
    assert len(HERRAMIENTAS) >= 1
    for h in HERRAMIENTAS:
        assert h.name
        # "descripciones vagas" es el error comun de la consigna: exijo una descripcion de verdad
        assert len(h.description) > 250, f"{h.name}: descripcion demasiado corta"


def test_cada_argumento_tiene_descripcion() -> None:
    for h in HERRAMIENTAS:
        propiedades = h.get_input_schema().model_json_schema()["properties"]
        for nombre, definicion in propiedades.items():
            assert definicion.get("description"), f"{h.name}.{nombre} sin descripcion"


def test_las_herramientas_son_asincronas() -> None:
    for h in HERRAMIENTAS:
        assert h.coroutine is not None, f"{h.name} no es async"


def test_las_descripciones_dicen_como_encadenar_las_herramientas() -> None:
    # El listado NO trae el responsable: la descripcion tiene que decirlo para que el
    # modelo sepa que necesita una segunda llamada (razonamiento multi-paso).
    assert "detalle_incidente" in buscar_incidentes_calidad.description
    assert "buscar_incidentes_calidad" in detalle_incidente.description
    assert "regla_id" in consultar_regla_calidad.description


# ---------- buscar_incidentes_calidad ----------


async def test_buscar_criticos_abiertos_de_articulos() -> None:
    r = await _llamar(buscar_incidentes_calidad, maestro="articulos", estado="abierto", severidad="critica")
    assert r["cantidad"] == 2
    assert [i["id"] for i in r["incidentes"]] == ["INC-1001", "INC-1002"]


async def test_buscar_calcula_los_dias_desde_el_alta() -> None:
    r = await _llamar(buscar_incidentes_calidad, maestro="articulos", severidad="critica")
    dias = {i["id"]: i["dias_desde_el_alta"] for i in r["incidentes"]}
    assert dias == {"INC-1001": 47, "INC-1002": 11}  # corte 2026-10-31


@pytest.mark.parametrize("escrito", ["Artículos", " ARTICULOS ", "articulo"])
async def test_buscar_normaliza_el_maestro(escrito: str) -> None:
    r = await _llamar(buscar_incidentes_calidad, maestro=escrito)
    assert r["maestro"] == "articulos"
    assert r["cantidad"] == 3  # abiertos por defecto: 1001, 1002, 1003


async def test_buscar_sin_resultados_no_es_un_error() -> None:
    r = await _llamar(buscar_incidentes_calidad, maestro="sucursales", severidad="critica")
    assert r["cantidad"] == 0
    assert r["incidentes"] == []
    assert "error" not in r


async def test_buscar_estado_todos_trae_los_cerrados_tambien() -> None:
    r = await _llamar(buscar_incidentes_calidad, maestro="proveedores", estado="todos")
    assert r["cantidad"] == 4


async def test_maestro_inexistente_devuelve_error_con_ayuda() -> None:
    r = await _llamar(buscar_incidentes_calidad, maestro="productos")
    assert "productos" in r["error"]
    for valido in ("articulos", "proveedores", "sucursales"):
        assert valido in r["ayuda"]


async def test_estado_y_severidad_invalidos_devuelven_error() -> None:
    estado = await _llamar(buscar_incidentes_calidad, maestro="articulos", estado="pendiente")
    severidad = await _llamar(buscar_incidentes_calidad, maestro="articulos", severidad="gravisima")
    assert "estado" in estado["error"] and "abierto" in estado["ayuda"]
    assert "severidad" in severidad["error"] and "critica" in severidad["ayuda"]


# ---------- detalle_incidente ----------


async def test_detalle_trae_responsable_regla_y_sla() -> None:
    r = await _llamar(detalle_incidente, incidente_id="INC-1001")
    assert r["responsable"].startswith("Gerente Comercial")
    assert r["regla_id"] == "R-ART-01"
    assert r["sla_horas_habiles"] == 4  # severidad critica
    assert r["registros_afectados"] == 312


async def test_detalle_acepta_minusculas_y_espacios() -> None:
    r = await _llamar(detalle_incidente, incidente_id=" inc-2001 ")
    assert r["id"] == "INC-2001"


async def test_detalle_inexistente_lista_los_ids_del_maestro_correspondiente() -> None:
    r = await _llamar(detalle_incidente, incidente_id="INC-1010")
    assert "INC-1010" in r["error"]
    assert "INC-1001" in r["ayuda"] and "INC-1005" in r["ayuda"]
    assert "INC-2001" not in r["ayuda"]  # solo los de articulos


async def test_detalle_con_formato_raro_lista_todos_los_ids() -> None:
    r = await _llamar(detalle_incidente, incidente_id="XYZ")
    assert "INC-1001" in r["ayuda"] and "INC-3003" in r["ayuda"]


# ---------- consultar_regla_calidad ----------


async def test_regla_incumplida_y_regla_cumplida() -> None:
    incumple = await _llamar(consultar_regla_calidad, regla_id="R-ART-01")
    cumple = await _llamar(consultar_regla_calidad, regla_id="r-art-05")
    assert incumple["cumple_umbral"] is False
    assert incumple["medicion_actual_pct"] < incumple["umbral_aceptacion_pct"]
    assert cumple["cumple_umbral"] is True


async def test_regla_inexistente_devuelve_error_con_ayuda() -> None:
    r = await _llamar(consultar_regla_calidad, regla_id="R-XXX-99")
    assert "R-XXX-99" in r["error"]
    assert "regla_id" in r["ayuda"]


# ---------- una herramienta que se rompe de verdad ----------


async def test_excepcion_inesperada_en_la_herramienta_no_rompe_el_grafo(
    monkeypatch: pytest.MonkeyPatch, config: Any
) -> None:
    from agente import abrir_agente, preguntar
    from tests.fakes import modelo_con_guion, pedir, responder

    class RepositorioRoto:
        async def buscar_incidentes(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("la base no responde")

    monkeypatch.setattr(herramientas, "obtener_repositorio", lambda: RepositorioRoto())
    llm = modelo_con_guion(
        pedir("buscar_incidentes_calidad", maestro="articulos"),
        responder("No pude consultar la base en este momento."),
    )
    async with abrir_agente(config, llm) as agente:
        turno = await preguntar(agente, "¿Cuántos incidentes hay?", "roto", 10)

    observacion = next(p for p in turno.pasos if p.tipo == "observacion")
    assert observacion.es_error
    assert "la base no responde" in str(observacion.resultado)
    assert turno.respuesta == "No pude consultar la base en este momento."
