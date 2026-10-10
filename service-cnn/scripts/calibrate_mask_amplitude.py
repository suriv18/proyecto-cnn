"""Calibra `amplitud_ndvi_min` de la máscara agrícola (sección 4.10.1, C-F2-1).

Muestrea celdas de cultivo (ESA WorldCover clase 40) de cada provincia y calcula
para cada campaña de ENTRENAMIENTO la amplitud NDVI (máximo - mínimo sobre la
ventana de campaña, MODIS MOD13Q1 V061, mismo cálculo que
`gee_mask_pipeline.build_agricultural_mask_image`). El resumen se obtiene con
`src.preprocessing.mask.summarize_amplitude_calibration`.

Solo se usan campañas de entrenamiento. Por defecto 2016-2020: con origen móvil
(sección 4.12.1) esas campañas forman parte del entrenamiento de cualquier
pliego con ventana inicial >= 5, así que la calibración no usa información de
campañas de prueba.

No modifica el manifiesto: el valor se registra manualmente en
`data/manifest/mascara_agricola.yaml` tras la aprobación del asesor.

Uso:
    python scripts/calibrate_mask_amplitude.py --campanas 2016 2017 2018 2019 2020
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.gee_config import get_collection_spec  # noqa: E402
from src.ingestion.gee_extraction_pipeline import campana_a_ventana_extraccion  # noqa: E402
from src.ingestion.province_geometries import (  # noqa: E402
    build_province_feature_collection,
    departamentos_a_nombres_gaul,
)
from src.preprocessing.mask import summarize_amplitude_calibration  # noqa: E402

_WORLDCOVER_ID = "ESA/WorldCover/v200/2021"
_SRTM_ID = "USGS/SRTMGL1_003"
_CLASE_CULTIVO = 40


def _imagen_compuesta(ee, campanas: list[int]):
    spec = get_collection_spec("ndvi")
    elevacion = ee.Image(_SRTM_ID).select("elevation").rename(["altitud_m"])
    imagen = (
        ee.Image(_WORLDCOVER_ID).select("Map").rename(["cobertura"])
        .addBands(elevacion)
        .addBands(ee.Terrain.slope(elevacion).rename(["pendiente_grados"]))
    )
    for campana in campanas:
        ini, fin = campana_a_ventana_extraccion(campana)
        col = (
            ee.ImageCollection(spec.collection_id)
            .filterDate(ini.isoformat(), fin.isoformat())
            .select(spec.band)
        )
        imagen = imagen.addBands(col.max().multiply(spec.factor_escala).rename([f"max_{campana}"]))
        imagen = imagen.addBands(col.min().multiply(spec.factor_escala).rename([f"min_{campana}"]))
    return imagen


def _a_formato_largo(crudo: pd.DataFrame, campanas: list[int]) -> pd.DataFrame:
    partes = []
    for campana in campanas:
        parte = crudo[
            ["provincia_id", "departamento", "celda_id", "altitud_m", "pendiente_grados"]
        ].copy()
        parte["campana_id"] = campana
        parte["cobertura_suelo"] = "cultivo"
        parte["ndvi_max"] = crudo[f"max_{campana}"]
        parte["ndvi_min"] = crudo[f"min_{campana}"]
        partes.append(parte)
    largo = pd.concat(partes, ignore_index=True)
    return largo.dropna(subset=["ndvi_max", "ndvi_min", "altitud_m", "pendiente_grados"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campanas", type=int, nargs="+", default=[2016, 2017, 2018, 2019, 2020])
    parser.add_argument("--n-por-provincia", type=int, default=60)
    parser.add_argument("--escala-m", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--percentiles", type=float, nargs="+", default=[1, 5, 10, 25])
    parser.add_argument("--provincias", type=Path, default=Path("configs/provincias.csv"))
    parser.add_argument("--manifiesto", type=Path, default=Path("data/manifest/mascara_agricola.yaml"))
    parser.add_argument("--gee-project-id", default="cnn-sentinel")
    parser.add_argument("--salida-celdas", type=Path, default=Path("data/interim/calibracion_mascara_celdas.csv"))
    parser.add_argument("--salida-resumen", type=Path, default=Path("reports/calibracion_mascara_amplitud.json"))
    args = parser.parse_args()

    import ee  # type: ignore[import-not-found]

    ee.Initialize(project=args.gee_project_id)

    umbrales = yaml.safe_load(args.manifiesto.read_text(encoding="utf-8"))["umbrales"]
    provincias = pd.read_csv(args.provincias)
    departamentos_gaul = departamentos_a_nombres_gaul(sorted(provincias["departamento"].unique()))
    fc = build_province_feature_collection(ee, list(provincias["provincia_id"]), departamentos_gaul)
    imagen = _imagen_compuesta(ee, args.campanas)
    depto_de = dict(zip(provincias["provincia_id"], provincias["departamento"]))

    filas, fallos = [], []
    inicio = time.time()
    for feature in fc.toList(fc.size()).getInfo():
        provincia_id = feature["properties"]["provincia_id"]
        try:
            muestra = imagen.stratifiedSample(
                numPoints=0,
                classBand="cobertura",
                region=ee.Geometry(feature["geometry"]),
                scale=args.escala_m,
                seed=args.seed,
                classValues=[_CLASE_CULTIVO],
                classPoints=[args.n_por_provincia],
                geometries=False,
            ).getInfo()["features"]
        except Exception as exc:  # noqa: BLE001 - se registra y se continúa con el resto
            fallos.append({"provincia_id": provincia_id, "error": str(exc)[:200]})
            print(f"FALLO {provincia_id}: {exc}", file=sys.stderr, flush=True)
            continue
        for i, fe in enumerate(muestra):
            filas.append({**fe["properties"], "provincia_id": provincia_id,
                          "departamento": depto_de[provincia_id], "celda_id": f"{provincia_id}_{i}"})
        print(f"{provincia_id}: {len(muestra)} celdas ({round(time.time() - inicio)} s)", flush=True)

    largo = _a_formato_largo(pd.DataFrame(filas), args.campanas)
    args.salida_celdas.parent.mkdir(parents=True, exist_ok=True)
    largo.to_csv(args.salida_celdas, index=False)

    resumen = summarize_amplitude_calibration(
        largo,
        percentiles=args.percentiles,
        altitud_min_m=umbrales["altitud_min_m"],
        altitud_max_m=umbrales["altitud_max_m"],
        pendiente_max_grados=umbrales["pendiente_max_grados"],
        campanas_entrenamiento=args.campanas,
    )
    resumen["metadatos"] = {
        "fecha": date.today().isoformat(),
        "n_por_provincia": args.n_por_provincia,
        "escala_m": args.escala_m,
        "seed": args.seed,
        "cobertura": _WORLDCOVER_ID,
        "ndvi": "MODIS/061/MOD13Q1 (max - min por ventana de campaña, sin filtro de calidad, igual que la máscara de producción)",
        "provincias_con_fallo": fallos,
        "n_provincias": int(largo["provincia_id"].nunique()),
    }
    args.salida_resumen.parent.mkdir(parents=True, exist_ok=True)
    args.salida_resumen.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Celdas-campaña: {len(largo)} | elegibles: {resumen['n_elegibles']} | fallos: {len(fallos)}")
    for p, r in resumen["umbrales"].items():
        print(f"p{p}: umbral={r['umbral']:.3f} excluye {r['excluidas_pct_total']:.1f}%")


if __name__ == "__main__":
    main()
