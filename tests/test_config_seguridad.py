"""Configuracion, seguridad de credenciales y calidad del repositorio."""

import ast
import re
from pathlib import Path

import pytest

import config as modulo_config
from config import BASE_DIR, Configuracion
from errores import ErrorConfiguracion

VARIABLES = ("ANTHROPIC_API_KEY", "MODELO_LLM", "DB_PATH", "RECURSION_LIMIT", "MAX_MENSAJES_CONTEXTO")


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch: pytest.MonkeyPatch) -> None:
    for nombre in VARIABLES:
        monkeypatch.delenv(nombre, raising=False)


# ---------- configuracion ----------


def test_sin_clave_falla_con_un_mensaje_que_dice_que_hacer() -> None:
    with pytest.raises(ErrorConfiguracion) as error:
        Configuracion.desde_entorno(cargar_dotenv=False)
    assert "ANTHROPIC_API_KEY" in str(error.value) and ".env" in str(error.value)


def test_con_clave_usa_los_valores_por_defecto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    c = Configuracion.desde_entorno(cargar_dotenv=False)
    assert c.modelo == "claude-haiku-4-5"
    assert c.recursion_limit == 10
    assert c.max_mensajes_contexto == 30
    assert c.ruta_db == BASE_DIR / "memoria" / "agente.sqlite"  # relativa a la carpeta del proyecto


def test_requerir_clave_false_permite_comandos_que_no_llaman_al_modelo() -> None:
    assert Configuracion.desde_entorno(requerir_clave=False, cargar_dotenv=False).anthropic_api_key == ""


def test_los_valores_se_pueden_cambiar_por_variables_de_entorno(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    monkeypatch.setenv("RECURSION_LIMIT", "15")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "otra.sqlite"))
    c = Configuracion.desde_entorno(cargar_dotenv=False)
    assert c.recursion_limit == 15
    assert c.ruta_db == tmp_path / "otra.sqlite"


@pytest.mark.parametrize("valor", ["diez", "2"])
def test_recursion_limit_invalido_se_rechaza(monkeypatch: pytest.MonkeyPatch, valor: str) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    monkeypatch.setenv("RECURSION_LIMIT", valor)
    with pytest.raises(ErrorConfiguracion, match="RECURSION_LIMIT"):
        Configuracion.desde_entorno(cargar_dotenv=False)


def test_el_dotenv_se_busca_junto_al_codigo_y_no_pisa_el_entorno(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    llamadas: list[dict[str, object]] = []

    def falso_load_dotenv(ruta: Path, override: bool = False) -> None:
        llamadas.append({"ruta": ruta, "override": override})

    monkeypatch.setattr(modulo_config, "load_dotenv", falso_load_dotenv)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    Configuracion.desde_entorno()
    assert llamadas == [{"ruta": BASE_DIR / ".env", "override": False}]


# ---------- seguridad de credenciales ----------

PATRONES_DE_CLAVES = re.compile(
    r"sk-ant-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|pcsk_[0-9A-Za-z_]{20,}|ghp_[0-9A-Za-z]{30,}|sk-[A-Za-z0-9]{40,}"
)
CARPETAS_IGNORADAS = {".git", "memoria", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}


def _archivos_del_repo() -> list[Path]:
    """Archivos que viajarian a GitHub: ignora el .env local, las bases y los entornos virtuales."""
    encontrados: list[Path] = []
    for ruta in BASE_DIR.rglob("*"):
        if not ruta.is_file():
            continue
        partes = set(ruta.relative_to(BASE_DIR).parts)
        if partes & CARPETAS_IGNORADAS or ruta.name == ".env":
            continue
        if any((padre / "pyvenv.cfg").exists() for padre in ruta.parents if padre != BASE_DIR.parent):
            continue  # un entorno virtual (con cualquier nombre)
        if ruta.suffix in {".py", ".md", ".txt", ".json", ".log", ".toml", ".example", ""}:
            encontrados.append(ruta)
    return encontrados


def test_no_hay_claves_escritas_en_ningun_archivo_del_repo() -> None:
    archivos = _archivos_del_repo()
    assert len(archivos) > 10  # el recorrido encontro el proyecto
    for ruta in archivos:
        texto = ruta.read_text(encoding="utf-8", errors="ignore")
        assert not PATRONES_DE_CLAVES.search(texto), f"parece haber una clave en {ruta.relative_to(BASE_DIR)}"


def test_gitignore_excluye_el_env_y_las_bases_de_memoria() -> None:
    lineas = {linea.strip() for linea in (BASE_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert ".env" in lineas
    assert "memoria/" in lineas
    assert "!.env.example" in lineas


def test_env_example_no_trae_una_clave_real() -> None:
    texto = (BASE_DIR / ".env.example").read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=pega-aca-tu-clave" in texto
    assert not PATRONES_DE_CLAVES.search(texto)


# ---------- calidad del repositorio ----------


def test_las_dependencias_estan_fijadas_a_una_version_exacta() -> None:
    lineas = [
        l.strip()
        for l in (BASE_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    assert lineas
    assert all("==" in linea for linea in lineas), lineas
    nombres = {linea.split("==")[0] for linea in lineas}
    assert {"langgraph", "langgraph-checkpoint-sqlite", "aiosqlite", "langchain-anthropic"} <= nombres


MODULOS = ["agente", "config", "errores", "herramientas", "main", "prompts", "repositorio", "trazas"]


@pytest.mark.parametrize("modulo", MODULOS)
def test_todas_las_funciones_tienen_type_hints(modulo: str) -> None:
    arbol = ast.parse((BASE_DIR / f"{modulo}.py").read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assert nodo.returns is not None, f"{modulo}.{nodo.name}: falta el tipo de retorno"
        argumentos = [*nodo.args.posonlyargs, *nodo.args.args, *nodo.args.kwonlyargs]
        for argumento in argumentos:
            if argumento.arg in {"self", "cls"}:
                continue
            assert argumento.annotation is not None, f"{modulo}.{nodo.name}({argumento.arg}): falta el tipo"


def test_el_codigo_del_agente_no_enruta_con_if_sobre_la_pregunta() -> None:
    """Criterio de autonomia: la decision de usar una herramienta es del modelo, no del codigo."""
    codigo = (BASE_DIR / "agente.py").read_text(encoding="utf-8")
    assert "tools_condition" in codigo
    for prohibido in ("if \"incidente\" in", "if 'incidente' in", ".lower().startswith(", "re.search("):
        assert prohibido not in codigo
