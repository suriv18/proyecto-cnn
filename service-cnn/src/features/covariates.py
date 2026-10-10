"""Covariables del tensor (sección 4.10.3).

La altitud media de la superficie agrícola (estática, por provincia) y la
proporción de superficie sembrada con quinua (dinámica, por provincia-campaña)
se mantienen SEPARADAS del tensor V×T: se concatenan después del bloque
convolucional de la CNN-1D.
"""
from __future__ import annotations

import pandas as pd

COVARIABLES = ["altitud_media_m", "prop_quinua"]
ROLES = {"altitud_media_m": "estatica", "prop_quinua": "dinamica"}


def build_covariates(
    observaciones: pd.DataFrame,
    siembra_mensual: pd.DataFrame,
    altitud_media_por_provincia: dict[str, float],
    area_cultivo_ha_por_provincia: dict[str, float],
) -> pd.DataFrame:
    """Calcula las dos covariables para cada observación, en su mismo orden.

    - `altitud_media_m`: altitud media de la superficie agrícola de la provincia.
    - `prop_quinua`: superficie sembrada con quinua en la campaña (suma de todos
      los meses de `siembra_mensual`) dividida por el área de cultivo de la
      provincia (ESA WorldCover, clase 40). No se recorta: una proporción mayor
      que 1 delata una inconsistencia entre fuentes y debe quedar visible.

    Una provincia sin dato de altitud o de área da NaN; no se inventa un valor.

    Args:
        observaciones: columnas `provincia_id`, `campana_id`.
        siembra_mensual: columnas `provincia_id`, `campana_id`, `mes`, `superficie_ha`.

    Returns:
        DataFrame con las columnas de `COVARIABLES`, con el índice de `observaciones`.
    """
    sembrada = (
        siembra_mensual.groupby(["provincia_id", "campana_id"])["superficie_ha"].sum().rename("sembrada_ha")
    )
    con_siembra = observaciones.join(sembrada, on=["provincia_id", "campana_id"])
    area = con_siembra["provincia_id"].map(area_cultivo_ha_por_provincia)
    resultado = pd.DataFrame(index=observaciones.index)
    resultado["altitud_media_m"] = con_siembra["provincia_id"].map(altitud_media_por_provincia).astype(float)
    resultado["prop_quinua"] = (con_siembra["sembrada_ha"] / area).astype(float)
    return resultado
