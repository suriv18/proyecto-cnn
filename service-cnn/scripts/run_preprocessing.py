"""CLI de las Actividades 4-5: alineación fenológica, tensor V×T y objetivo.

Referencia: secciones 4.10.1-4.10.4 y 4.12.2 de la tesis;
docs/crisp/F2-preparacion-datos.md.

Produce en `data/processed/`: `tensores.npz`, `observaciones.parquet`,
`fenologia.parquet`, `exclusiones.csv`, `diccionario_variables.csv` y
`metadata.json` (ver `src.preprocessing.preprocessing_pipeline`).

Protecciones (el script se detiene antes de leer el SISAGRI, que es lento):
  - El parquet crudo debe tener el esquema NDVI vigente (`valor_mean`/
    `valor_max`); si no, hay que re-extraer el NDVI con `run_ingestion.py`.
  - La máscara agrícola se aplica en la extracción de Earth Engine, no aquí;
    este script exige que `data/manifest/mascara_agricola.yaml` tenga todos
    los umbrales calibrados. `--permitir-sin-mascara` lo omite y marca la
    salida como no válida para la evaluación.
  - Hay que indicar `--campanas-entrenamiento` (respaldo departamental sin
    fuga, sección 4.12.2) o `--exploratorio` (usa todo el histórico).

Uso:
    python scripts/run_preprocessing.py \
        --archivo-sisagri "ruta/a/SISAGRI.xlsx" \
        --campanas-entrenamiento 2016 2017 2018 2019 2020 \
        --dias-antes-de-cosecha 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.mask_manifest import (  # noqa: E402
    AmplitudNdviNoCalibradaError,
    load_mask_thresholds,
)
from src.ingestion.midagri_aggregation import aggregate_to_provincia_campana  # noqa: E402
from src.ingestion.midagri_headerless import (  # noqa: E402
    load_sisagri_headerless,
    load_sisagri_headerless_config,
)
from src.ingestion.midagri_siembra_mensual import (  # noqa: E402
    load_siembra_mensual_por_provincia,
)
from src.preprocessing.phenology_pipeline import load_fenologia_bbch  # noqa: E402
from src.preprocessing.preprocessing_pipeline import (  # noqa: E402
    RawSchemaDesactualizadaError,
    build_processed_dataset,
    save_processed_dataset,
    validate_raw_schema,
)

_HOJAS_DEL_ARCHIVO_REAL = ["Exportar Hoja de Trabajo", "Sheet1", "Sheet2", "Sheet3", "Sheet4"]
_DEPARTAMENTOS_DELIMITADOS = {
    "PUNO", "AYACUCHO", "APURIMAC", "AREQUIPA", "JUNIN", "CUSCO",
    "LA LIBERTAD", "HUANCAVELICA",
}


def _leer_quinua_sisagri(archivo: Path, column_mapping: Path) -> tuple[pd.DataFrame, object]:
    """Lee las 5 hojas del SISAGRI real y conserva solo quinua de los 8 departamentos."""
    config = load_sisagri_headerless_config(column_mapping)
    columnas_originales = [
        config.anio, config.mes, "COD_UBIGEO", config.departamento, config.provincia,
        config.distrito, "COD_PRODUCTO", config.cultivo, config.superficie_sembrada_ha,
        config.superficie_cosechada_ha, config.produccion_ton, "VERDE_ACTUAL", "PRECIO_CHACRA",
    ]
    fragmentos = []
    for i, hoja in enumerate(_HOJAS_DEL_ARCHIVO_REAL):
        print(f"Procesando hoja: {hoja}...", file=sys.stderr)
        crudo = pd.read_excel(
            archivo,
            sheet_name=hoja,
            header=0 if i == 0 else None,
            names=None if i == 0 else columnas_originales,
        )
        filtrado = crudo[crudo[config.departamento].isin(_DEPARTAMENTOS_DELIMITADOS)]
        filtrado = filtrado[filtrado[config.cultivo].astype(str).str.upper().str.contains("QUINUA")]
        if not filtrado.empty:
            fragmentos.append(filtrado)
    return pd.concat(fragmentos, ignore_index=True), config


def _rendimiento_por_provincia_campana(quinua: pd.DataFrame, config) -> pd.DataFrame:
    """Rendimiento (kg/ha) = producción / superficie cosechada, solo celdas con ambas > 0."""
    mensual = load_sisagri_headerless(quinua, config)
    celdas = aggregate_to_provincia_campana(mensual).rename(columns={"provincia": "provincia_id"})
    validas = celdas[(celdas["produccion_ton"] > 0) & (celdas["superficie_cosechada_ha"] > 0)].copy()
    validas["rendimiento_kg_ha"] = (
        validas["produccion_ton"] * 1000.0 / validas["superficie_cosechada_ha"]
    )
    return validas[["provincia_id", "campana_id", "rendimiento_kg_ha"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--archivo-sisagri", type=Path, required=True)
    parser.add_argument("--datos-raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--provincias", type=Path, default=Path("configs/provincias.csv"))
    parser.add_argument("--column-mapping", type=Path, default=Path("configs/column_mapping.yaml"))
    parser.add_argument("--hyperparams", type=Path, default=Path("configs/hyperparams.yaml"))
    parser.add_argument("--manifiesto-mascara", type=Path, default=Path("data/manifest/mascara_agricola.yaml"))
    parser.add_argument("--salida", type=Path, default=Path("data/processed"))
    parser.add_argument("--campanas-entrenamiento", type=int, nargs="+", default=None)
    parser.add_argument("--exploratorio", action="store_true")
    parser.add_argument("--permitir-sin-mascara", action="store_true")
    parser.add_argument("--dias-antes-de-cosecha", type=int, default=None)
    args = parser.parse_args()

    if (args.campanas_entrenamiento is None) == (not args.exploratorio):
        print(
            "Indique exactamente una opción: --campanas-entrenamiento C1 C2 ... "
            "(respaldo sin fuga) o --exploratorio (usa todo el histórico, no "
            "válido para la evaluación).",
            file=sys.stderr,
        )
        sys.exit(2)

    parquet = args.datos_raw / "gee_series_crudo.parquet"
    if not parquet.exists():
        print(f"PREPROCESAMIENTO NO LISTO: no existe {parquet} (ejecute run_ingestion.py).", file=sys.stderr)
        sys.exit(1)
    series_crudas = pd.read_parquet(parquet)
    try:
        validate_raw_schema(series_crudas)
    except RawSchemaDesactualizadaError as exc:
        print(f"PREPROCESAMIENTO NO LISTO: {exc}", file=sys.stderr)
        sys.exit(1)

    mascara_calibrada = True
    try:
        load_mask_thresholds(args.manifiesto_mascara)
    except AmplitudNdviNoCalibradaError as exc:
        if not args.permitir_sin_mascara:
            print(
                f"PREPROCESAMIENTO NO LISTO: {exc}\n"
                "Calibre amplitud_ndvi_min (docs/crisp/F2, C-F2-1) o use "
                "--permitir-sin-mascara para una corrida exploratoria.",
                file=sys.stderr,
            )
            sys.exit(1)
        mascara_calibrada = False

    provincias = pd.read_csv(args.provincias)
    provincia_a_departamento = dict(zip(provincias["provincia_id"], provincias["departamento"]))

    print("Leyendo quinua de SISAGRI...", file=sys.stderr)
    quinua, config = _leer_quinua_sisagri(args.archivo_sisagri, args.column_mapping)
    siembra_mensual = load_siembra_mensual_por_provincia(quinua)
    rendimiento = _rendimiento_por_provincia_campana(quinua, config)
    print(f"Celdas con rendimiento: {len(rendimiento)}", file=sys.stderr)

    dataset = build_processed_dataset(
        series_crudas=series_crudas,
        siembra_mensual=siembra_mensual,
        rendimiento=rendimiento,
        provincia_a_departamento=provincia_a_departamento,
        duracion_dias_por_fase=load_fenologia_bbch(args.hyperparams),
        campanas_entrenamiento=args.campanas_entrenamiento,
        dias_antes_de_cosecha=args.dias_antes_de_cosecha,
    )
    dataset.metadata["mascara_calibrada"] = mascara_calibrada
    save_processed_dataset(dataset, args.salida)

    print(f"Guardado en {args.salida}: X{dataset.X.shape}, {dataset.metadata['n_observaciones']} "
          f"observaciones, {dataset.metadata['n_exclusiones']} exclusiones.")
    if dataset.metadata["exploratorio"] or not mascara_calibrada:
        print("AVISO: salida NO válida para la evaluación definitiva "
              f"(exploratorio={dataset.metadata['exploratorio']}, "
              f"mascara_calibrada={mascara_calibrada}).", file=sys.stderr)


if __name__ == "__main__":
    main()
