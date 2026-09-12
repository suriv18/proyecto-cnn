"""Calcula N4 (y N5 cuando exista el criterio de calidad) reales del Hito H1.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.5.6.

Procesa el archivo real SISAGRI.xlsx (5 hojas, ~4.3 millones de filas) de
forma incremental por hoja para no cargarlo completo en memoria, filtra
quinua de los 8 departamentos delimitados, deriva campaña agrícola, agrega a
nivel provincia-campaña, y calcula N4 = celdas con producción > 0, aplicando
la cascada N0-N5 completa sobre el periodo enmendado (2016-2025, ver
data/manifest/midagri_sisagri.yaml).

Uso:
    python scripts/compute_n4_n5.py \
        --archivo-sisagri "ruta/a/SISAGRI.xlsx" \
        --provincias configs/provincias.csv \
        --produccion-documentada configs/produccion_documentada.csv \
        --column-mapping configs/column_mapping.yaml \
        --n-campanas 10 \
        --salida-celdas data/interim/celdas_con_produccion.csv \
        --salida-cascada reports/E1_matriz_cobertura_real.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.audit import build_coverage_cascade  # noqa: E402
from src.ingestion.midagri_aggregation import aggregate_to_provincia_campana  # noqa: E402
from src.ingestion.midagri_headerless import (  # noqa: E402
    load_sisagri_headerless,
    load_sisagri_headerless_config,
)

_DEPARTAMENTOS_DELIMITADOS = {
    "PUNO", "AYACUCHO", "APURIMAC", "AREQUIPA", "JUNIN", "CUSCO",
    "LA LIBERTAD", "HUANCAVELICA",
}

_HOJAS_DEL_ARCHIVO_REAL = [
    "Exportar Hoja de Trabajo",
    "Sheet1",
    "Sheet2",
    "Sheet3",
    "Sheet4",
]


def _procesar_hoja(
    ruta_archivo: Path, nombre_hoja: str, config, columnas_originales: list[str]
) -> pd.DataFrame:
    """Lee una hoja del archivo real, filtrando quinua+departamentos delimitados.

    Solo la primera hoja repite la fila de cabecera (ver
    data/manifest/midagri_sisagri.yaml, nota_estructura); las siguientes
    comparten el mismo esquema de columnas sin repetirla.
    """
    tiene_cabecera = nombre_hoja == _HOJAS_DEL_ARCHIVO_REAL[0]
    crudo = pd.read_excel(
        ruta_archivo,
        sheet_name=nombre_hoja,
        header=0 if tiene_cabecera else None,
        names=None if tiene_cabecera else columnas_originales,
    )

    crudo_filtrado = crudo[crudo[config.departamento].isin(_DEPARTAMENTOS_DELIMITADOS)]
    crudo_filtrado = crudo_filtrado[
        crudo_filtrado[config.cultivo].astype(str).str.upper().str.contains("QUINUA")
    ]

    if crudo_filtrado.empty:
        return pd.DataFrame(
            columns=["provincia", "campana_id", "produccion_ton", "superficie_cosechada_ha"]
        )

    return load_sisagri_headerless(crudo_filtrado, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo-sisagri", type=Path, required=True)
    parser.add_argument("--provincias", type=Path, required=True)
    parser.add_argument("--produccion-documentada", type=Path, default=None)
    parser.add_argument("--column-mapping", type=Path, default=Path("configs/column_mapping.yaml"))
    parser.add_argument("--n-campanas", type=int, required=True)
    parser.add_argument("--salida-celdas", type=Path, required=True)
    parser.add_argument("--salida-cascada", type=Path, required=True)
    args = parser.parse_args()

    config = load_sisagri_headerless_config(args.column_mapping)
    columnas_originales = [
        config.anio, config.mes, "COD_UBIGEO", config.departamento, config.provincia,
        config.distrito, "COD_PRODUCTO", config.cultivo, config.superficie_sembrada_ha,
        config.superficie_cosechada_ha, config.produccion_ton, "VERDE_ACTUAL", "PRECIO_CHACRA",
    ]

    fragmentos = []
    for hoja in _HOJAS_DEL_ARCHIVO_REAL:
        print(f"Procesando hoja: {hoja}...", file=sys.stderr)
        fragmentos.append(_procesar_hoja(args.archivo_sisagri, hoja, config, columnas_originales))

    quinua_mensual = pd.concat(fragmentos, ignore_index=True)
    print(f"Registros mensuales de quinua (8 deptos): {len(quinua_mensual)}", file=sys.stderr)

    celdas_con_produccion = aggregate_to_provincia_campana(quinua_mensual)
    args.salida_celdas.parent.mkdir(parents=True, exist_ok=True)
    celdas_con_produccion.to_csv(args.salida_celdas, index=False)
    print(f"Celdas provincia-campaña agregadas: {len(celdas_con_produccion)}", file=sys.stderr)

    provincias = pd.read_csv(args.provincias)
    produccion_documentada = (
        pd.read_csv(args.produccion_documentada) if args.produccion_documentada else None
    )

    celdas_con_produccion_para_cascada = celdas_con_produccion.rename(
        columns={"provincia": "provincia_id"}
    )

    cascada = build_coverage_cascade(
        provincias=provincias,
        n_campanas=args.n_campanas,
        produccion_documentada=produccion_documentada,
        celdas_con_produccion=celdas_con_produccion_para_cascada,
        celdas_con_calidad=None,
    )

    args.salida_cascada.parent.mkdir(parents=True, exist_ok=True)
    args.salida_cascada.write_text(json.dumps(cascada, indent=2), encoding="utf-8")

    for nivel, valor in cascada.items():
        estado = valor if valor is not None else "PENDIENTE (sin datos reales)"
        print(f"{nivel}: {estado}")


if __name__ == "__main__":
    main()
