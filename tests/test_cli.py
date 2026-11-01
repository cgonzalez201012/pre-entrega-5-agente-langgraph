"""La linea de comandos, sin red: el modelo se reemplaza por uno con guion."""

import json
from pathlib import Path
from typing import Any

import pytest

import agente as modulo_agente
import main as modulo_main
from agente import abrir_agente, preguntar
from config import Configuracion
from tests.fakes import modelo_con_guion, pedir, responder
from trazas import Turno


@pytest.fixture(autouse=True)
def entorno_aislado(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Sin .env real, sin clave del entorno y con la base en una carpeta temporal."""
    monkeypatch.setattr("config.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cli.sqlite"))


def test_sin_clave_el_programa_avisa_y_no_se_cae(capsys: pytest.CaptureFixture[str]) -> None:
    codigo = modulo_main.main(["hola"])
    assert codigo == 1
    assert "Falta ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_sin_argumentos_explica_como_usarlo(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    assert modulo_main.main([]) == 2
    assert "--demo" in capsys.readouterr().out


def test_ver_hilo_funciona_sin_clave_y_avisa_si_no_hay_historial(capsys: pytest.CaptureFixture[str]) -> None:
    assert modulo_main.main(["--ver-hilo", "no-existe"]) == 0
    assert "No hay historial guardado" in capsys.readouterr().out


async def test_ver_hilo_muestra_los_mensajes_guardados(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = Configuracion("x", "m", tmp_path / "cli.sqlite", 10, 30)
    llm = modelo_con_guion(pedir("detalle_incidente", incidente_id="INC-1001"), responder("Es el Gerente Comercial."))
    async with abrir_agente(config, llm) as agente:
        await preguntar(agente, "¿Responsable de INC-1001?", "visible", 10)

    await modulo_main._ver_hilo(config, "visible")
    salida = capsys.readouterr().out
    assert "[Usuario] ¿Responsable de INC-1001?" in salida
    assert "detalle_incidente" in salida
    assert "[Herramienta]" in salida
    assert "[Agente] Es el Gerente Comercial." in salida


def test_mostrar_turno_resume_o_detalla(capsys: pytest.CaptureFixture[str]) -> None:
    turno = Turno("e", "t", "p", "La respuesta.")
    modulo_main._mostrar_turno(turno, detalle=False)
    assert "La respuesta." in capsys.readouterr().out
    turno.limite_de_pasos_alcanzado = True
    modulo_main._mostrar_turno(turno, detalle=False)
    assert "límite de pasos" in capsys.readouterr().out


async def test_demo_corre_los_cuatro_escenarios_y_guarda_la_traza(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # El modelo falso sigue el guion de la demo: 4 escenarios, dos aperturas de la base.
    llm = modelo_con_guion(
        pedir("buscar_incidentes_calidad", maestro="articulos", estado="abierto", severidad="critica"),
        pedir("detalle_incidente", incidente_id="INC-1001"),
        responder("Hay 2 críticos; el más antiguo es INC-1001 (Gerente Comercial)."),
        pedir("consultar_regla_calidad", regla_id="R-ART-01"),
        responder("Incumple R-ART-01 y no cumple el umbral."),
        pedir("detalle_incidente", incidente_id="INC-1010"),
        responder("INC-1010 no existe. ¿Querés que liste los de artículos?"),
        responder("Veníamos hablando de INC-1001; responde el Gerente Comercial."),
    )
    monkeypatch.setattr(modulo_agente, "crear_llm", lambda config: llm)
    monkeypatch.setattr(modulo_main, "CARPETA_TRAZAS", tmp_path / "trazas")
    config = Configuracion("clave-falsa", "modelo-falso", tmp_path / "memoria" / "agente.sqlite", 10, 30)

    await modulo_main._demo(config)

    documento: dict[str, Any] = json.loads((tmp_path / "trazas" / "traza-demo.json").read_text(encoding="utf-8"))
    turnos = documento["turnos"]
    assert [t["escenario"][0] for t in turnos] == ["1", "2", "3", "4"]
    assert turnos[0]["llamadas_a_herramientas"] >= 2  # requisito de la consigna
    assert documento["recursion_limit"] == 10
    assert turnos[3]["llamadas_a_herramientas"] == 0  # tras el "reinicio" responde con lo que recuerda
    assert (tmp_path / "trazas" / "traza-demo.log").exists()
    # la demo usa su propia base y no toca la de uso diario
    assert (tmp_path / "memoria" / "demo.sqlite").exists()
    assert not (tmp_path / "memoria" / "agente.sqlite").exists()
    assert "Traza guardada en" in capsys.readouterr().out


async def test_la_demo_arranca_siempre_de_cero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = Configuracion("clave-falsa", "modelo-falso", tmp_path / "memoria" / "agente.sqlite", 10, 30)
    # Una corrida anterior dejo historial en el mismo hilo de la demo
    viejo = modelo_con_guion(responder("respuesta vieja"))
    async with abrir_agente(Configuracion("x", "m", config.ruta_db.with_name("demo.sqlite"), 10, 30), viejo) as agente:
        await preguntar(agente, "mensaje de una corrida anterior", "demo-multipaso", 10)

    nuevo = modelo_con_guion(
        pedir("buscar_incidentes_calidad", maestro="articulos"),
        pedir("detalle_incidente", incidente_id="INC-1001"),
        responder("a"),
        responder("b"),
        responder("c"),
        responder("d"),
    )
    monkeypatch.setattr(modulo_agente, "crear_llm", lambda config: nuevo)
    monkeypatch.setattr(modulo_main, "CARPETA_TRAZAS", tmp_path / "trazas")

    await modulo_main._demo(config)

    assert all("mensaje de una corrida anterior" not in str(m.content) for m in nuevo.recibidos[0])
