"""CLI de la Actividad 2-3: ingesta de las 5 fuentes de datos (Tabla 7).

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.9.

Estado actual: NO ejecutable de punta a punta todavía. Requiere:
  1. El esquema real confirmado del archivo MIDAGRI/SIEA (columnas exactas de
     datosabiertos.gob.pe) en configs/column_mapping.yaml — ver la nota en
     ese archivo.
  2. Una cuenta de Google Earth Engine vinculada a un proyecto de Google
     Cloud, para `src.ingestion.gee_client.GeeClient.from_default(project_id)`.

Este script valida que ambos prerrequisitos estén satisfechos y falla con un
mensaje explícito señalando cuál falta, en vez de producir una salida vacía o
simulada silenciosamente.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.midagri_loader import load_column_mapping  # noqa: E402


class IngestionNotReadyError(RuntimeError):
    """Un prerrequisito de la ingesta real todavía no está disponible."""


def _verificar_mapeo_midagri(ruta_mapeo: Path) -> None:
    mapeo = load_column_mapping(ruta_mapeo)
    columnas_placeholder = {
        "DEPARTAMENTO",
        "PROVINCIA",
        "ANIO_COSECHA",
        "CULTIVO",
        "SUPERFICIE_SEMBRADA",
        "SUPERFICIE_COSECHADA",
        "PRODUCCION",
        "RENDIMIENTO",
    }
    valores_actuales = set(mapeo.columnas_requeridas())
    if valores_actuales == columnas_placeholder:
        raise IngestionNotReadyError(
            f"{ruta_mapeo} todavía contiene los nombres de columna placeholder "
            "(no confirmados contra el archivo real de MIDAGRI/SIEA). Descargue "
            "el dataset de datosabiertos.gob.pe manualmente, confirme los "
            "nombres de columna reales y actualice el archivo antes de "
            "ejecutar la ingesta real."
        )


def _verificar_credenciales_gee(project_id: str | None) -> None:
    if project_id is None:
        raise IngestionNotReadyError(
            "No se proporcionó --gee-project-id. La ingesta de CHIRPS/"
            "ERA5-Land/MODIS requiere una cuenta de Google Earth Engine "
            "vinculada a un proyecto de Google Cloud (ver "
            "docs/02-arquitectura-tecnica.md §2)."
        )
    try:
        from src.ingestion.gee_client import GeeClient

        GeeClient.from_default(project_id=project_id)
    except ImportError as exc:
        raise IngestionNotReadyError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--column-mapping",
        type=Path,
        default=Path("configs/column_mapping.yaml"),
    )
    parser.add_argument(
        "--gee-project-id",
        type=str,
        default=None,
        help="ID del proyecto de Google Cloud vinculado a Earth Engine.",
    )
    args = parser.parse_args()

    try:
        _verificar_mapeo_midagri(args.column_mapping)
        _verificar_credenciales_gee(args.gee_project_id)
    except IngestionNotReadyError as exc:
        print(f"INGESTA NO LISTA: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        "Prerrequisitos verificados. La descarga real (MIDAGRI, CHIRPS, "
        "ERA5-Land, MODIS) se implementa cuando ambos estén confirmados."
    )


if __name__ == "__main__":
    main()
