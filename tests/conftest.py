"""Fixtures compartidas. Ningun test necesita claves ni red."""

from pathlib import Path

import pytest

from config import Configuracion


@pytest.fixture
def config(tmp_path: Path) -> Configuracion:
    """Configuracion de prueba: base SQLite en una carpeta temporal."""
    return Configuracion(
        anthropic_api_key="clave-falsa-de-prueba",
        modelo="modelo-falso",
        ruta_db=tmp_path / "memoria" / "prueba.sqlite",
        recursion_limit=10,
        max_mensajes_contexto=30,
    )
