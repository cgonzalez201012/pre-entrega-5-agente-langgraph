"""Modelos de chat falsos para probar el GRAFO sin claves ni red.

Ojo con lo que prueban: verifican que el grafo enruta, ejecuta herramientas,
acumula mensajes y persiste. NO prueban que Claude decida bien: eso se ve con una
corrida real (python main.py --demo) y queda en trazas/.
"""

from collections.abc import Callable, Sequence
from itertools import count
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_ids = count(1)


def pedir(herramienta: str, **argumentos: Any) -> AIMessage:
    """Mensaje del modelo que pide UNA herramienta."""
    return AIMessage(
        content="",
        tool_calls=[{"name": herramienta, "args": argumentos, "id": f"call_{next(_ids)}", "type": "tool_call"}],
    )


def pedir_varias(*llamadas: tuple[str, dict[str, Any]]) -> AIMessage:
    """Mensaje del modelo que pide VARIAS herramientas a la vez (llamadas paralelas)."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": nombre, "args": argumentos, "id": f"call_{next(_ids)}", "type": "tool_call"}
            for nombre, argumentos in llamadas
        ],
    )


def responder(texto: str) -> AIMessage:
    """Mensaje final del modelo (sin herramientas)."""
    return AIMessage(content=texto)


class ModeloPorPolitica(BaseChatModel):
    """Modelo cuyo siguiente mensaje lo decide una funcion de lo que recibe."""

    politica: Callable[[list[BaseMessage]], AIMessage]
    recibidos: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "falso-por-politica"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ModeloPorPolitica":
        return self  # las herramientas no hacen falta: el modelo falso ya sabe que pedir

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.recibidos.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.politica(list(messages)))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop)


def modelo_con_guion(*mensajes: AIMessage) -> ModeloPorPolitica:
    """Modelo que devuelve, en orden, los mensajes dados (y falla si se le piden mas)."""
    pendientes = list(mensajes)

    def politica(_: list[BaseMessage]) -> AIMessage:
        if not pendientes:
            raise AssertionError("El modelo falso se quedo sin mensajes del guion.")
        return pendientes.pop(0)

    return ModeloPorPolitica(politica=politica)
