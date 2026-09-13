"""CLI de la Actividad 3-4: máscara agrícola y alineación fenológica.

Referencia: secciones 4.10.1-4.10.2 de la tesis.

Estado actual: NO ejecutable todavía — depende de `data/raw/` con los datos
crudos de la ingesta real (CHIRPS, ERA5-Land, MODIS, MIDAGRI), que a su vez
depende de `run_ingestion.py` (ver ese script para los prerrequisitos
pendientes: esquema MIDAGRI confirmado, cuenta de Google Earth Engine).

Este script queda como punto de integración documentado: cuando
`data/raw/` exista, aquí se invocan en orden:
  1. `src.preprocessing.mask.apply_agricultural_mask` (con umbrales
     calibrados vía `calibrate_ndvi_amplitude_threshold` sobre entrenamiento)
     — pendiente: requiere cobertura de suelo + MDE por celda, aún no
     extraídos de Earth Engine (ver data/manifest/gee_series_climaticas_
     espectrales.yaml, sección `pendiente`).
  2. `src.preprocessing.phenology.estimate_sowing_date` + `build_phase_
     windows`, por provincia-campaña — YA EJECUTABLE de punta a punta con
     datos reales vía `scripts/run_phenology_alignment.py` (siembra mensual
     real de MIDAGRI + serie NDVI real de data/raw/gee_series_crudo.parquet),
     resultado en data/interim/fenologia_alineada.parquet. Aún no integrado
     a este script porque falta (1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


class PreprocessingNotReadyError(RuntimeError):
    """Los datos crudos de entrada todavía no existen."""


def _verificar_datos_crudos(directorio_raw: Path) -> None:
    archivos_reales = [
        p for p in directorio_raw.iterdir() if p.is_file() and p.name != ".gitkeep"
    ] if directorio_raw.exists() else []
    if not archivos_reales:
        raise PreprocessingNotReadyError(
            f"{directorio_raw} no contiene datos reales (solo el placeholder "
            ".gitkeep, si existe). El preprocesamiento requiere los datos "
            "crudos de la ingesta real (ver scripts/run_ingestion.py para "
            "los prerrequisitos pendientes: esquema MIDAGRI confirmado y "
            "cuenta de Google Earth Engine)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos-raw", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--salida", type=Path, default=Path("data/interim/series_alineadas")
    )
    args = parser.parse_args()

    try:
        _verificar_datos_crudos(args.datos_raw)
    except PreprocessingNotReadyError as exc:
        print(f"PREPROCESAMIENTO NO LISTO: {exc}", file=sys.stderr)
        sys.exit(1)

    raise NotImplementedError(
        "La orquestación de máscara + alineación fenológica sobre datos "
        "reales se completa cuando exista data/raw/ (Actividad 6 del "
        "roadmap, docs/02-arquitectura-tecnica.md §9). Las funciones que se "
        "invocarán ya están implementadas y probadas en "
        "src/preprocessing/mask.py y src/preprocessing/phenology.py."
    )


if __name__ == "__main__":
    main()
