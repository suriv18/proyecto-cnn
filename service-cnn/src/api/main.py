"""Creación de la aplicación FastAPI.

Ejecución local (desde service-cnn/):

    .venv/Scripts/python.exe -m uvicorn src.api.main:app --port 8000

Variables de entorno:
    QUINUAPROD_RAIZ            directorio de service-cnn con reports/, data/ y configs/ (por defecto, este).
    QUINUAPROD_CORS_ORIGINS    orígenes permitidos separados por comas (por defecto, los de Vite).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .problemas import registrar_manejadores
from .repositorio import Repositorio
from .rutas import router

RAIZ_POR_DEFECTO = Path(__file__).resolve().parents[2]
ORIGENES_POR_DEFECTO = "http://localhost:5173,http://localhost:4173"
PREFIJO = "/api/v1"


def _contrato(raiz: Path) -> dict | None:
    """El OpenAPI de la interfaz es la fuente de verdad; si está en el repositorio, /docs lo muestra tal cual."""
    ruta = raiz.parent / "app-quinuaprod" / "api" / "openapi.yaml"
    return yaml.safe_load(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def create_app(raiz: Path | None = None, origenes: list[str] | None = None) -> FastAPI:
    raiz = Path(raiz or os.environ.get("QUINUAPROD_RAIZ") or RAIZ_POR_DEFECTO)
    app = FastAPI(title="API de app-quinuaprod", version="0.1.0", description="Solo lectura.")
    app.state.repositorio = Repositorio(raiz)

    origenes = origenes or os.environ.get("QUINUAPROD_CORS_ORIGINS", ORIGENES_POR_DEFECTO).split(",")
    # Solo lectura: no se necesitan credenciales ni métodos de escritura salvo la inferencia (POST).
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origenes if o.strip()], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    registrar_manejadores(app)
    app.include_router(router, prefix=PREFIJO)

    @app.get("/salud", include_in_schema=False)
    def salud() -> dict[str, str]:
        return {"estado": "ok"}

    contrato = _contrato(raiz)
    if contrato is not None:
        app.openapi = lambda: contrato  # type: ignore[method-assign]
    return app


app = create_app()
