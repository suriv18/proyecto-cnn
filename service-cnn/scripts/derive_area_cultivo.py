"""Área de cultivo por provincia (denominador de la covariable dinámica, 4.10.3).

Para cada una de las 80 provincias calcula con Google Earth Engine:
  - `area_cultivo_ha`: superficie con cobertura Cropland (ESA WorldCover v200 2021,
    clase 40), a 10 m. Es la «superficie agrícola» de la provincia.
  - `area_cultivo_en_mascara_ha`: la parte de esa superficie que cumple los criterios
    de altitud y pendiente de la máscara agrícola (`data/manifest/mascara_agricola.yaml`),
    a 30 m (resolución del SRTM).
  - `frac_area_en_mascara`: su cociente. Mide qué parte del cultivo de la provincia
    representa el NDVI enmascarado (sección 4.10.1), mejor que la mediana de altitud
    de una muestra.

Escribe `configs/area_cultivo_por_provincia.csv`. WorldCover es de 2021 y se asume
estático para las 10 campañas (misma aproximación declarada en el manifiesto de la máscara).

    python scripts/derive_area_cultivo.py
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.province_geometries import (  # noqa: E402
    build_province_feature_collection,
    departamentos_a_nombres_gaul,
)

_WORLDCOVER = "ESA/WorldCover/v200/2021"
_SRTM = "USGS/SRTMGL1_003"
_CLASE_CULTIVO = 40


def _area_ha(ee, imagen_mascara, geom, escala: int) -> float:
    r = (
        ee.Image.pixelArea().divide(10000).updateMask(imagen_mascara)
        .reduceRegion(reducer=ee.Reducer.sum(), geometry=geom, scale=escala, maxPixels=int(1e13), tileScale=8)
        .getInfo()
    )
    return float(r.get("area") or 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provincias", type=Path, default=Path("configs/provincias.csv"))
    ap.add_argument("--manifiesto", type=Path, default=Path("data/manifest/mascara_agricola.yaml"))
    ap.add_argument("--salida", type=Path, default=Path("configs/area_cultivo_por_provincia.csv"))
    ap.add_argument("--gee-project-id", default="cnn-sentinel")
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args()

    import ee  # type: ignore[import-not-found]

    ee.Initialize(project=args.gee_project_id)
    u = yaml.safe_load(args.manifiesto.read_text(encoding="utf-8"))["umbrales"]
    provincias = pd.read_csv(args.provincias)
    fc = build_province_feature_collection(
        ee, list(provincias["provincia_id"]), departamentos_a_nombres_gaul(sorted(provincias["departamento"].unique()))
    )

    cultivo = ee.Image(_WORLDCOVER).select("Map").eq(_CLASE_CULTIVO)
    elev = ee.Image(_SRTM).select("elevation")
    en_mascara = (
        cultivo.And(elev.gte(u["altitud_min_m"])).And(elev.lte(u["altitud_max_m"]))
        .And(ee.Terrain.slope(elev).lte(u["pendiente_max_grados"]))
    )

    def una(feature: dict) -> dict:
        pid = feature["properties"]["provincia_id"]
        geom = ee.Geometry(feature["geometry"])
        for intento in range(3):
            try:
                total = _area_ha(ee, cultivo, geom, 10)
                mascara = _area_ha(ee, en_mascara, geom, 30)
                return {"provincia_id": pid, "area_cultivo_ha": total, "area_cultivo_en_mascara_ha": mascara}
            except Exception as exc:  # noqa: BLE001 - se reintenta y se informa
                print(f"reintento {intento + 1} {pid}: {str(exc)[:120]}", file=sys.stderr, flush=True)
                time.sleep(5)
        return {"provincia_id": pid, "area_cultivo_ha": float("nan"), "area_cultivo_en_mascara_ha": float("nan")}

    filas, inicio = [], time.time()
    features = fc.toList(fc.size()).getInfo()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed([pool.submit(una, f) for f in features]):
            r = fut.result()
            filas.append(r)
            print(f"{len(filas):2d}/{len(features)} {r['provincia_id']:22s} {r['area_cultivo_ha']:10.0f} ha  ({time.time() - inicio:.0f} s)", flush=True)

    tabla = pd.DataFrame(filas).sort_values("provincia_id")
    tabla["frac_area_en_mascara"] = (tabla["area_cultivo_en_mascara_ha"] / tabla["area_cultivo_ha"]).round(4)
    tabla["fuente"] = f"{_WORLDCOVER} (clase 40) + {_SRTM}; extraído {date.today().isoformat()}"
    tabla = tabla.round({"area_cultivo_ha": 1, "area_cultivo_en_mascara_ha": 1})
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(args.salida, index=False)
    print(f"{args.salida}: {len(tabla)} provincias, {int(tabla.area_cultivo_ha.isna().sum())} con fallo")


if __name__ == "__main__":
    main()
