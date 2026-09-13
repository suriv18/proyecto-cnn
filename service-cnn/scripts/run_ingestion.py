"""CLI de la Actividad 2-3: ingesta real de las fuentes de datos (Tabla 7).

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.9;
data/manifest/midagri_sisagri.yaml (enmienda de periodo 2016-2025).

Ejecuta, contra la API real de Google Earth Engine:
  1. Construcción de las geometrías de las 80 provincias delimitadas
     (FAO/GAUL/2015/level2), a partir de configs/provincias.csv.
  2. Extracción cruda de precipitación (CHIRPS), temperatura máxima/mínima y
     radiación solar (ERA5-Land) y NDVI (MODIS) para cada campaña del periodo
     enmendado (2016-2025 por defecto), en la ventana amplia del ciclo
     agrícola (ver `gee_extraction_pipeline.campana_a_ventana_extraccion`).
  3. Guarda el resultado crudo en `data/raw/gee_series_crudo.parquet` y, si
     hubo fallos aislados, `data/raw/gee_series_fallos.json`.

La alineación fenológica fina (fases BBCH) y la ingesta de MIDAGRI (ya
ejecutada en la Actividad 1/H1, ver scripts/run_audit.py) se implementan en
etapas posteriores del pipeline (preprocess).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.ingestion.gee_extraction_pipeline import (  # noqa: E402
    extract_all_provinces_all_variables,
)
from src.ingestion.midagri_loader import load_column_mapping  # noqa: E402
from src.ingestion.province_geometries import (  # noqa: E402
    build_province_feature_collection,
    departamentos_a_nombres_gaul,
)

_VARIABLES_TABLA_7 = [
    "precipitacion",
    "temperatura_maxima",
    "temperatura_minima",
    "radiacion_solar",
    "ndvi",
]
_CAMPANAS_PERIODO_ENMENDADO = list(range(2016, 2026))


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


def _construir_cliente_gee(project_id: str | None):
    if project_id is None:
        raise IngestionNotReadyError(
            "No se proporcionó --gee-project-id. La ingesta de CHIRPS/"
            "ERA5-Land/MODIS requiere una cuenta de Google Earth Engine "
            "vinculada a un proyecto de Google Cloud (ver "
            "docs/02-arquitectura-tecnica.md §2)."
        )
    try:
        from src.ingestion.gee_client import GeeClient

        cliente = GeeClient.from_default(project_id=project_id)
    except ImportError as exc:
        raise IngestionNotReadyError(str(exc)) from exc

    cliente.initialize()
    return cliente


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--column-mapping",
        type=Path,
        default=Path("configs/column_mapping.yaml"),
    )
    parser.add_argument(
        "--provincias",
        type=Path,
        default=Path("configs/provincias.csv"),
    )
    parser.add_argument(
        "--gee-project-id",
        type=str,
        default=None,
        help="ID del proyecto de Google Cloud vinculado a Earth Engine.",
    )
    parser.add_argument(
        "--campana-inicio",
        type=int,
        default=_CAMPANAS_PERIODO_ENMENDADO[0],
    )
    parser.add_argument(
        "--campana-fin",
        type=int,
        default=_CAMPANAS_PERIODO_ENMENDADO[-1],
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help=(
            "Llamadas a Earth Engine en vuelo simultáneamente. El cuello de "
            "botella real es espera de red (~78s por variable-mes con las "
            "80 provincias, verificado en producción), no cómputo local."
        ),
    )
    args = parser.parse_args()

    try:
        _verificar_mapeo_midagri(args.column_mapping)
        cliente = _construir_cliente_gee(args.gee_project_id)
    except IngestionNotReadyError as exc:
        print(f"INGESTA NO LISTA: {exc}", file=sys.stderr)
        sys.exit(1)

    provincias = pd.read_csv(args.provincias)
    departamentos_gaul = departamentos_a_nombres_gaul(
        sorted(provincias["departamento"].unique())
    )
    geometrias = build_province_feature_collection(
        ee_module=cliente.ee_module,
        provincia_ids=provincias["provincia_id"].tolist(),
        departamentos_gaul=departamentos_gaul,
    )

    n_geometrias = geometrias.size().getInfo()
    if n_geometrias != len(provincias):
        print(
            f"ADVERTENCIA: se esperaban {len(provincias)} geometrías, se "
            f"encontraron {n_geometrias} en FAO/GAUL/2015/level2.",
            file=sys.stderr,
        )

    campanas = list(range(args.campana_inicio, args.campana_fin + 1))
    print(
        f"Extrayendo {len(_VARIABLES_TABLA_7)} variables x {len(campanas)} "
        f"campañas ({campanas[0]}-{campanas[-1]}) para {n_geometrias} "
        "provincias. Esto puede tardar varios minutos contra la API real..."
    )

    resultado, fallos = extract_all_provinces_all_variables(
        cliente=cliente,
        geometrias_por_provincia=geometrias,
        variables=_VARIABLES_TABLA_7,
        campanas=campanas,
        devolver_fallos=True,
        max_workers=args.max_workers,
    )

    args.salida.mkdir(parents=True, exist_ok=True)
    ruta_datos = args.salida / "gee_series_crudo.parquet"
    resultado.to_parquet(ruta_datos, index=False)
    print(f"Guardado: {ruta_datos} ({len(resultado)} filas)")

    if fallos:
        ruta_fallos = args.salida / "gee_series_fallos.json"
        ruta_fallos.write_text(
            json.dumps(fallos, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"ADVERTENCIA: {len(fallos)} combinaciones variable x campaña "
            f"fallaron, ver {ruta_fallos}.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
