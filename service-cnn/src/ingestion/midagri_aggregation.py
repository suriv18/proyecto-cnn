"""Agregación de registros mensuales de MIDAGRI a celdas provincia-campaña.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.5.6. El
archivo real de MIDAGRI (SISAGRI.xlsx) reporta un registro por mes; una celda
provincia-campaña agrega los meses de esa campaña ya derivada (sección 4.4,
ver `src.preprocessing.campaign_calendar`).
"""
from __future__ import annotations

import pandas as pd


def aggregate_to_provincia_campana(registros_mensuales: pd.DataFrame) -> pd.DataFrame:
    """Suma producción y superficie cosechada mensual a nivel provincia-campaña.

    Args:
        registros_mensuales: columnas `provincia`, `campana_id`,
            `produccion_ton`, `superficie_cosechada_ha` — salida (ya filtrada
            a quinua) de `src.ingestion.midagri_loader.load_midagri_production`
            o `src.ingestion.midagri_headerless.load_sisagri_headerless`, con
            un registro por mes calendario.

    Returns:
        DataFrame con una fila por combinación única de `provincia` y
        `campana_id`, con `produccion_ton` y `superficie_cosechada_ha`
        sumadas sobre todos los meses de esa campaña.
    """
    if registros_mensuales.empty:
        return registros_mensuales.copy()

    return (
        registros_mensuales.groupby(["provincia", "campana_id"], as_index=False)[
            ["produccion_ton", "superficie_cosechada_ha"]
        ]
        .sum()
    )
