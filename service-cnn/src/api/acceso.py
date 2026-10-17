"""Control de acceso del servicio: límite de uso por cliente y clave de API opcional.

El servicio es público y de solo lectura, así que por defecto solo limita la cantidad de consultas por cliente.
La clave es opcional (se activa con `QUINUAPROD_API_KEY`) y sirve para quien publique el servicio de forma restringida.

Una clave dentro de la interfaz web NO es secreta (cualquiera puede leerla en el código que descarga el navegador).
Sirve para llamadas entre servidores, o cuando un proxy inverso con autenticación propia agrega la cabecera por el
usuario. Para una interfaz pública, la defensa real es el límite de uso y CORS.

El límite se cuenta en memoria y por proceso: con varios procesos o réplicas cada uno cuenta por separado.
"""
from __future__ import annotations

import math
import secrets
import time
from collections import deque
from collections.abc import Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .problemas import MEDIA_TYPE

VENTANA_SEGUNDOS = 60.0
MAX_CLIENTES_EN_MEMORIA = 10_000
RUTAS_LIBRES = frozenset({"/salud"})  # el chequeo de salud no cuenta ni exige clave


def _problema(status: int, title: str, detail: str, cabeceras: dict[str, str] | None = None) -> JSONResponse:
    cuerpo = {"type": "about:blank", "title": title, "status": status, "detail": detail}
    return JSONResponse(cuerpo, status_code=status, media_type=MEDIA_TYPE, headers=cabeceras)


def _cabecera(scope: Scope, nombre: bytes) -> str | None:
    for clave, valor in scope.get("headers", []):
        if clave == nombre:
            return valor.decode("latin-1")
    return None


class ControlDeAcceso:
    """Middleware ASGI. Orden: primero el límite (así también frena los intentos con claves erróneas), luego la clave."""

    def __init__(
        self,
        app: ASGIApp,
        clave_api: str | None = None,
        limite_por_minuto: int = 0,
        confiar_proxy: bool = False,
        reloj: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app = app
        self.clave_api = clave_api.encode("utf-8") if clave_api else None
        self.limite = limite_por_minuto
        self.confiar_proxy = confiar_proxy
        self.reloj = reloj
        self._consultas: dict[str, deque[float]] = {}

    def _cliente(self, scope: Scope) -> str:
        if self.confiar_proxy:
            reenviado = _cabecera(scope, b"x-forwarded-for")
            if reenviado:
                return reenviado.split(",")[0].strip()  # el cliente original; los demás son proxies
        cliente = scope.get("client")
        return cliente[0] if cliente else "desconocido"

    def _segundos_de_espera(self, cliente: str) -> int:
        """0 si la consulta cabe en el límite (y la cuenta); si no, los segundos que faltan para poder consultar."""
        ahora = self.reloj()
        registro = self._consultas.setdefault(cliente, deque())
        while registro and ahora - registro[0] >= VENTANA_SEGUNDOS:
            registro.popleft()
        if len(registro) >= self.limite:
            return max(1, math.ceil(VENTANA_SEGUNDOS - (ahora - registro[0])))
        registro.append(ahora)
        if len(self._consultas) > MAX_CLIENTES_EN_MEMORIA:  # evita que la memoria crezca sin tope
            for k in [k for k, v in self._consultas.items() if not v or ahora - v[-1] >= VENTANA_SEGUNDOS]:
                del self._consultas[k]
        return 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Las pre-solicitudes CORS (OPTIONS) no llevan clave y no son consultas de datos.
        if scope["type"] != "http" or scope["path"] in RUTAS_LIBRES or scope["method"] == "OPTIONS":
            return await self.app(scope, receive, send)

        if self.limite > 0:
            espera = self._segundos_de_espera(self._cliente(scope))
            if espera:
                respuesta = _problema(
                    429, "Demasiadas consultas",
                    f"Se superó el límite de {self.limite} consultas por minuto. Reintente en {espera} s.",
                    {"Retry-After": str(espera)},
                )
                return await respuesta(scope, receive, send)

        if self.clave_api is not None:
            recibida = _cabecera(scope, b"x-api-key")
            if recibida is None or not secrets.compare_digest(recibida.encode("utf-8"), self.clave_api):
                respuesta = _problema(401, "No autorizado", "Falta la clave de la API o no es válida (cabecera X-API-Key).")
                return await respuesta(scope, receive, send)

        await self.app(scope, receive, send)
