"""Errores del servicio en formato RFC 9457 (`application/problem+json`), como pide el contrato."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

MEDIA_TYPE = "application/problem+json"

_TITULOS = {
    404: "Recurso no encontrado",
    405: "Método no permitido",
}


class Problema(Exception):
    """Error de negocio con el estado HTTP y el mensaje que verá la interfaz."""

    def __init__(self, status: int, title: str, detail: str | None = None) -> None:
        super().__init__(title)
        self.status = status
        self.title = title
        self.detail = detail


def _respuesta(status: int, title: str, detail: str | None = None) -> JSONResponse:
    cuerpo: dict[str, object] = {"type": "about:blank", "title": title, "status": status}
    if detail:
        cuerpo["detail"] = detail
    return JSONResponse(cuerpo, status_code=status, media_type=MEDIA_TYPE)


def _detalle_validacion(error: RequestValidationError) -> str:
    partes = []
    for e in error.errors():
        ubicacion = ".".join(str(p) for p in e["loc"] if p not in ("query", "body"))
        partes.append(f"{ubicacion}: {e['msg']}" if ubicacion else e["msg"])
    return "; ".join(partes)


def registrar_manejadores(app: FastAPI) -> None:
    @app.exception_handler(Problema)
    async def _problema(_: Request, exc: Problema) -> JSONResponse:
        return _respuesta(exc.status, exc.title, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _validacion(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _respuesta(422, "Solicitud inválida", _detalle_validacion(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _respuesta(exc.status_code, _TITULOS.get(exc.status_code, "Error"), str(exc.detail))

    @app.exception_handler(FileNotFoundError)
    async def _sin_datos(_: Request, exc: FileNotFoundError) -> JSONResponse:
        # Los artefactos de reports/ se regeneran con el pipeline; si faltan, es un problema de despliegue.
        return _respuesta(
            503,
            "Datos no disponibles",
            f"Falta el artefacto «{exc.filename}». Ejecute el piloto (scripts/run_pilot.py) o `dvc repro`.",
        )
