"""Deriva la altitud de la superficie de cultivo de cada provincia (sección 4.5.3).

Lee las celdas de cultivo muestreadas por `calibrate_mask_amplitude.py` (WorldCover
clase 40, SRTM) y escribe `configs/altitud_cultivo_por_provincia.csv` con la
mediana de altitud y la fracción de celdas dentro del rango de la máscara.

Uso: es una variable de ESTRATIFICACIÓN para el análisis de sensibilidad
(4.10.1), no un criterio de exclusión por defecto: mide la altitud de todo el
cultivo, no la de la quinua, que no se cultiva sobre ~4 000 m (INIA).

    python scripts/derive_altitud_cultivo.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.mask import (  # noqa: E402
    summarize_agricultural_surface_altitude,
    summarize_cropland_altitude,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--celdas", type=Path, default=Path("data/interim/calibracion_mascara_celdas.csv"))
    parser.add_argument("--salida", type=Path, default=Path("configs/altitud_cultivo_por_provincia.csv"))
    parser.add_argument("--altitud-min-m", type=float, default=2500)
    parser.add_argument("--altitud-max-m", type=float, default=4000)
    parser.add_argument("--pendiente-max-grados", type=float, default=14.04)
    parser.add_argument(
        "--salida-media",
        type=Path,
        default=Path("configs/altitud_media_superficie_agricola.csv"),
        help="Covariable estática del tensor (4.10.3): altitud media del cultivo que pasa la máscara.",
    )
    args = parser.parse_args()

    celdas = pd.read_csv(args.celdas)
    resumen = summarize_cropland_altitude(celdas, args.altitud_min_m, args.altitud_max_m)
    resumen = resumen.round({"altitud_mediana_m": 0, "frac_en_rango": 3}).sort_values("provincia_id")
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(args.salida, index=False)
    print(f"{args.salida}: {len(resumen)} provincias")

    media = summarize_agricultural_surface_altitude(
        celdas, args.altitud_min_m, args.altitud_max_m, args.pendiente_max_grados
    ).round({"altitud_media_m": 1}).sort_values("provincia_id")
    media.to_csv(args.salida_media, index=False)
    print(f"{args.salida_media}: {len(media)} provincias (la media usa solo celdas de cultivo dentro de la máscara)")


if __name__ == "__main__":
    main()
