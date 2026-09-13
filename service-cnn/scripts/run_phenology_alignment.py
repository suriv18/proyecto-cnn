"""CLI de la Actividad 4: alineación fenológica sobre datos reales.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.10.2.

Conecta:
  1. Superficie sembrada mensual real de MIDAGRI/SISAGRI (nivel primario) —
     lee el archivo real SISAGRI.xlsx igual que scripts/compute_n4_n5.py.
  2. La serie NDVI real ya extraída de Earth Engine, en
     data/raw/gee_series_crudo.parquet (nivel secundario).
  3. Fecha modal departamental EXPLORATORIA (nivel de respaldo) — ver
     ADVERTENCIA en src.preprocessing.phenology_pipeline.
     estimate_departmental_modal_dates_exploratory: no válida para H2, usada
     solo mientras el esquema de validación real no está conectado a este
     pipeline.

Produce, para cada una de las 80 provincias x 10 campañas (2016-2025), la
fuente de alineación usada, la fecha de siembra estimada y las 5 ventanas de
fase BBCH, guardado en data/interim/fenologia_alineada.parquet.

Uso:
    python scripts/run_phenology_alignment.py \
        --archivo-sisagri "ruta/a/SISAGRI.xlsx" \
        --gee-series-crudo data/raw/gee_series_crudo.parquet \
        --provincias configs/provincias.csv \
        --column-mapping configs/column_mapping.yaml \
        --hyperparams configs/hyperparams.yaml \
        --salida data/interim/fenologia_alineada.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.midagri_headerless import (  # noqa: E402
    load_sisagri_headerless_config,
)
from src.ingestion.midagri_siembra_mensual import (  # noqa: E402
    load_siembra_mensual_por_provincia,
)
from src.preprocessing.phenology_pipeline import (  # noqa: E402
    build_all_phase_windows,
    estimate_departmental_modal_dates_exploratory,
    load_fenologia_bbch,
)

_HOJAS_DEL_ARCHIVO_REAL = [
    "Exportar Hoja de Trabajo",
    "Sheet1",
    "Sheet2",
    "Sheet3",
    "Sheet4",
]

_DEPARTAMENTOS_DELIMITADOS = {
    "PUNO", "AYACUCHO", "APURIMAC", "AREQUIPA", "JUNIN", "CUSCO",
    "LA LIBERTAD", "HUANCAVELICA",
}


def _leer_siembra_mensual_real(archivo_sisagri: Path, column_mapping: Path) -> pd.DataFrame:
    config = load_sisagri_headerless_config(column_mapping)
    columnas_originales = [
        config.anio, config.mes, "COD_UBIGEO", config.departamento, config.provincia,
        config.distrito, "COD_PRODUCTO", config.cultivo, config.superficie_sembrada_ha,
        config.superficie_cosechada_ha, config.produccion_ton, "VERDE_ACTUAL", "PRECIO_CHACRA",
    ]

    fragmentos = []
    for hoja in _HOJAS_DEL_ARCHIVO_REAL:
        print(f"Procesando hoja: {hoja}...", file=sys.stderr)
        tiene_cabecera = hoja == _HOJAS_DEL_ARCHIVO_REAL[0]
        crudo = pd.read_excel(
            archivo_sisagri,
            sheet_name=hoja,
            header=0 if tiene_cabecera else None,
            names=None if tiene_cabecera else columnas_originales,
        )
        crudo_filtrado = crudo[crudo[config.departamento].isin(_DEPARTAMENTOS_DELIMITADOS)]
        crudo_filtrado = crudo_filtrado[
            crudo_filtrado[config.cultivo].astype(str).str.upper().str.contains("QUINUA")
        ]
        if not crudo_filtrado.empty:
            fragmentos.append(load_siembra_mensual_por_provincia(crudo_filtrado))

    if not fragmentos:
        return pd.DataFrame(columns=["provincia_id", "campana_id", "mes", "superficie_ha"])

    combinado = pd.concat(fragmentos, ignore_index=True)
    return (
        combinado.groupby(["provincia_id", "campana_id", "mes"], as_index=False)[
            "superficie_ha"
        ]
        .sum()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo-sisagri", type=Path, required=True)
    parser.add_argument("--gee-series-crudo", type=Path, default=Path("data/raw/gee_series_crudo.parquet"))
    parser.add_argument("--provincias", type=Path, default=Path("configs/provincias.csv"))
    parser.add_argument("--column-mapping", type=Path, default=Path("configs/column_mapping.yaml"))
    parser.add_argument("--hyperparams", type=Path, default=Path("configs/hyperparams.yaml"))
    parser.add_argument("--salida", type=Path, default=Path("data/interim/fenologia_alineada.parquet"))
    args = parser.parse_args()

    provincias = pd.read_csv(args.provincias)
    provincia_a_departamento = dict(zip(provincias["provincia_id"], provincias["departamento"]))
    duracion_dias_por_fase = load_fenologia_bbch(args.hyperparams)

    print("Leyendo superficie sembrada mensual real de SISAGRI...", file=sys.stderr)
    siembra_mensual = _leer_siembra_mensual_real(args.archivo_sisagri, args.column_mapping)
    print(f"Registros de siembra mensual (8 deptos, quinua): {len(siembra_mensual)}", file=sys.stderr)

    print(f"Cargando serie NDVI real de {args.gee_series_crudo}...", file=sys.stderr)
    serie_ndvi_cruda = pd.read_parquet(args.gee_series_crudo)

    fechas_modales_departamentales = estimate_departmental_modal_dates_exploratory(
        siembra_mensual=siembra_mensual,
        provincia_a_departamento=provincia_a_departamento,
    )

    campanas = sorted(serie_ndvi_cruda["campana_id"].unique())
    provincia_campanas = [
        (provincia_id, int(campana_id))
        for provincia_id in provincias["provincia_id"]
        for campana_id in campanas
    ]

    print(
        f"Alineando fenológicamente {len(provincia_campanas)} "
        "combinaciones provincia-campaña...",
        file=sys.stderr,
    )
    resultados = build_all_phase_windows(
        provincia_campanas=provincia_campanas,
        siembra_mensual=siembra_mensual,
        serie_ndvi_cruda=serie_ndvi_cruda,
        fechas_modales_departamentales=fechas_modales_departamentales,
        duracion_dias_por_fase=duracion_dias_por_fase,
        provincia_a_departamento=provincia_a_departamento,
    )

    filas_planas = []
    fallos = []
    for r in resultados:
        if r["error"] is not None:
            fallos.append(r)
            continue
        for ventana in r["ventanas"]:
            filas_planas.append(
                {
                    "provincia_id": r["provincia_id"],
                    "campana_id": r["campana_id"],
                    "fuente": r["fuente"].value,
                    "fecha_siembra": r["fecha_siembra"],
                    "es_imputado": r["es_imputado"],
                    "fase": ventana.fase,
                    "inicio": ventana.inicio,
                    "fin": ventana.fin,
                }
            )

    tabla = pd.DataFrame(filas_planas)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(args.salida, index=False)
    print(f"Guardado: {args.salida} ({len(tabla)} filas, {len(resultados) - len(fallos)} provincia-campañas alineadas)")

    if fallos:
        print(
            f"ADVERTENCIA: {len(fallos)} provincia-campañas sin ningún nivel "
            "de alineación disponible (ni siembra mensual, ni NDVI, ni "
            "respaldo departamental).",
            file=sys.stderr,
        )
        for f in fallos[:10]:
            print(f"  - {f['provincia_id']} / {f['campana_id']}: {f['error']}", file=sys.stderr)

    if not tabla.empty:
        print(tabla["fuente"].value_counts())


if __name__ == "__main__":
    main()
