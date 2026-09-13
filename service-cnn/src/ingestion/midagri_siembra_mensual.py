"""Superficie sembrada mensual de quinua por provincia (nivel primario, 4.10.2).

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.10.2 —
jerarquía de 3 niveles de `preprocessing.phenology.estimate_sowing_date`. El
nivel primario necesita `superficie_ha` POR MES dentro de cada
provincia-campaña, para localizar el mes de mayor superficie sembrada.

Se distingue de `midagri_aggregation.aggregate_to_provincia_campana`, que
colapsa TODOS los meses de la campaña en un solo total (para el cálculo de
rendimiento) — aquí se preserva el detalle mensual. También agrega
DISTRITO -> PROVINCIA: el archivo real de SISAGRI.xlsx reporta a nivel
distrito, pero la unidad de análisis de la tesis es provincia (sección 4.4;
ver configs/column_mapping.yaml, nota "PENDIENTE" sobre esta agregación).
"""
from __future__ import annotations

import pandas as pd

from src.ingestion.midagri_loader import _CULTIVO_OBJETIVO, _normalizar_texto
from src.preprocessing.campaign_calendar import derive_campana_from_month

_COLUMNAS_CRUDAS_REQUERIDAS = {
    "AÑO": "anio",
    "MES": "mes",
    "PROVINCIA": "provincia_id",
    "PRODUCTO": "cultivo",
    "SIEMBRA": "superficie_ha",
}


def load_siembra_mensual_por_provincia(datos_crudos: pd.DataFrame) -> pd.DataFrame:
    """Extrae superficie sembrada mensual de quinua, agregada a nivel provincia.

    Args:
        datos_crudos: DataFrame con las columnas reales de SISAGRI.xlsx (AÑO,
            MES, PROVINCIA, PRODUCTO, SIEMBRA como mínimo — ver
            `configs/column_mapping.yaml`, sección `sisagri_headerless`).

    Returns:
        DataFrame con una fila por combinación única de `provincia_id`,
        `campana_id`, `mes`, columna `superficie_ha` (`SIEMBRA` sumada sobre
        todos los distritos de esa provincia y mes). Filtrado a filas de
        quinua únicamente (misma regla de `midagri_loader.load_midagri_
        production`).
    """
    renombrado = datos_crudos.rename(columns=_COLUMNAS_CRUDAS_REQUERIDAS)

    es_quinua = renombrado["cultivo"].apply(_normalizar_texto).str.contains(
        _CULTIVO_OBJETIVO
    )
    filtrado = renombrado[es_quinua].copy()

    filtrado["campana_id"] = [
        derive_campana_from_month(anio=int(anio), mes=int(mes))
        for anio, mes in zip(filtrado["anio"], filtrado["mes"])
    ]

    if filtrado.empty:
        return pd.DataFrame(columns=["provincia_id", "campana_id", "mes", "superficie_ha"])

    return (
        filtrado.groupby(["provincia_id", "campana_id", "mes"], as_index=False)[
            "superficie_ha"
        ]
        .sum()
    )
