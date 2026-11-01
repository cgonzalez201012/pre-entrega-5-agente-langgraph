"""Errores propios del proyecto, con mensajes pensados para quien ejecuta el agente."""


class ErrorConfiguracion(Exception):
    """Falta o es invalida una variable de entorno necesaria para arrancar."""


class ErrorConsulta(Exception):
    """Una consulta a la base simulada no se pudo resolver.

    Lleva un mensaje y una ayuda: ambos viajan de vuelta al agente como texto
    para que pueda corregir la llamada o pedirle una aclaracion al usuario.
    """

    def __init__(self, mensaje: str, ayuda: str) -> None:
        super().__init__(mensaje)
        self.mensaje = mensaje
        self.ayuda = ayuda
