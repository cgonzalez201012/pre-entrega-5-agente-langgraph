"""Configuracion del agente: todo sale de variables de entorno (archivo .env)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from errores import ErrorConfiguracion

BASE_DIR = Path(__file__).resolve().parent

MODELO_POR_DEFECTO = "claude-haiku-4-5"
RUTA_DB_POR_DEFECTO = "memoria/agente.sqlite"
RECURSION_LIMIT_POR_DEFECTO = 10
MAX_MENSAJES_POR_DEFECTO = 30


@dataclass(frozen=True)
class Configuracion:
    """Valores con los que se arma el agente."""

    anthropic_api_key: str
    modelo: str
    ruta_db: Path
    recursion_limit: int
    max_mensajes_contexto: int

    @classmethod
    def desde_entorno(
        cls, *, requerir_clave: bool = True, cargar_dotenv: bool = True
    ) -> "Configuracion":
        """Lee la configuracion del entorno.

        `requerir_clave=False` sirve para comandos que no llaman al modelo
        (por ejemplo, mirar el historial guardado de un hilo).
        """
        if cargar_dotenv:
            load_dotenv(BASE_DIR / ".env", override=False)

        clave = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if requerir_clave and not clave:
            raise ErrorConfiguracion(
                "Falta ANTHROPIC_API_KEY. Copiá .env.example a .env y pegá tu clave "
                "de Anthropic (o exportala como variable de entorno)."
            )

        ruta_db = Path(os.environ.get("DB_PATH", RUTA_DB_POR_DEFECTO))
        if not ruta_db.is_absolute():
            ruta_db = BASE_DIR / ruta_db

        return cls(
            anthropic_api_key=clave,
            modelo=os.environ.get("MODELO_LLM", MODELO_POR_DEFECTO).strip(),
            ruta_db=ruta_db,
            recursion_limit=_entero("RECURSION_LIMIT", RECURSION_LIMIT_POR_DEFECTO, minimo=3),
            max_mensajes_contexto=_entero("MAX_MENSAJES_CONTEXTO", MAX_MENSAJES_POR_DEFECTO, minimo=6),
        )


def _entero(nombre: str, por_defecto: int, *, minimo: int) -> int:
    """Lee un entero del entorno y valida que no sea absurdamente chico."""
    texto = os.environ.get(nombre, "").strip()
    if not texto:
        return por_defecto
    try:
        valor = int(texto)
    except ValueError as error:
        raise ErrorConfiguracion(f"{nombre} tiene que ser un numero entero (vale {texto!r}).") from error
    if valor < minimo:
        raise ErrorConfiguracion(f"{nombre} tiene que ser al menos {minimo} (vale {valor}).")
    return valor
