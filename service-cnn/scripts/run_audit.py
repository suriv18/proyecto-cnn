"""CLI de la Actividad 1 (Hito H1): auditoría de cobertura y elegibilidad.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, secciones 4.5.2-4.5.7.

Uso:
    python scripts/run_audit.py \
        --provincias configs/provincias.csv \
        --produccion-documentada configs/produccion_documentada.csv \
        --n-campanas 19 \
        --salida reports/E1_matriz_cobertura.csv

Estado actual: requiere `configs/provincias.csv` con la clasificación
territorial oficial de las 80 provincias de los 8 departamentos delimitados
(Tabla 3-4 de la tesis) — archivo aún no creado porque la clasificación
definitiva depende de la fuente territorial oficial que se seleccione en la
propia auditoría H1 (sección 4.5.3). Este script produce N0-N3 (calculables
sin datos de producción reales) y deja N4-N5 explícitamente pendientes hasta
que existan `--celdas-con-produccion` y `--celdas-con-calidad` reales
(sección 4.5.6: nunca se estiman con tasas supuestas).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.audit import build_coverage_cascade  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provincias",
        type=Path,
        required=True,
        help="CSV con columnas provincia_id, departamento, clasificacion "
        "(Tabla 4 de la tesis).",
    )
    parser.add_argument("--n-campanas", type=int, default=19)
    parser.add_argument(
        "--produccion-documentada",
        type=Path,
        default=None,
        help="CSV con columnas provincia_id, produccion_documentada (Tabla 3). "
        "Si se omite, N3 queda igual a N2.",
    )
    parser.add_argument(
        "--celdas-con-produccion",
        type=Path,
        default=None,
        help="CSV con columnas provincia_id, campana_id, produccion_ton. "
        "Requerido para calcular N4 (sección 4.5.6: no se estima sin datos "
        "reales).",
    )
    parser.add_argument(
        "--celdas-con-calidad",
        type=Path,
        default=None,
        help="CSV con columnas provincia_id, campana_id, cumple_calidad. "
        "Requerido para calcular N5.",
    )
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()

    provincias = pd.read_csv(args.provincias)
    produccion_documentada = (
        pd.read_csv(args.produccion_documentada)
        if args.produccion_documentada
        else None
    )
    celdas_con_produccion = (
        pd.read_csv(args.celdas_con_produccion)
        if args.celdas_con_produccion
        else None
    )
    celdas_con_calidad = (
        pd.read_csv(args.celdas_con_calidad) if args.celdas_con_calidad else None
    )

    cascada = build_coverage_cascade(
        provincias=provincias,
        n_campanas=args.n_campanas,
        produccion_documentada=produccion_documentada,
        celdas_con_produccion=celdas_con_produccion,
        celdas_con_calidad=celdas_con_calidad,
    )

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(cascada, indent=2), encoding="utf-8")

    for nivel, valor in cascada.items():
        estado = valor if valor is not None else "PENDIENTE (sin datos reales)"
        print(f"{nivel}: {estado}")


if __name__ == "__main__":
    main()
