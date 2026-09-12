"""Agregación de series temporales de Earth Engine por fase fenológica.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, secciones 4.10.2-4.10.4.

Opera sobre datos YA exportados de Earth Engine (tabla provincia x fecha x
valor) — no depende de `ee`/`geemap`, por lo que es 100% testeable sin
credenciales. La capa que sí ejecuta consultas reales vive en `gee_client.py`.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.ingestion.gee_config import AggregationRule

_AGG_FUNC = {
    AggregationRule.SUM: "sum",
    AggregationRule.MEAN: "mean",
    AggregationRule.MAX: "max",
    AggregationRule.MIN: "min",
}


class HorizonteExcedidoError(ValueError):
    """La serie contiene observaciones posteriores al punto de corte del
    horizonte de pronóstico — violaría la regla anti-fuga temporal (4.12.2)."""


def aggregate_by_phase(
    serie: pd.DataFrame,
    fases: pd.DataFrame,
    aggregation: AggregationRule,
    fecha_corte: date | None = None,
) -> pd.DataFrame:
    """Agrega una serie temporal diaria/horaria por fase fenológica.

    Args:
        serie: columnas `provincia_id`, `campana_id`, `fecha`, `valor` — una
            observación por paso temporal, ya recortada a la provincia-campaña
            de interés.
        fases: columnas `provincia_id`, `campana_id`, `fase`, `inicio`, `fin` —
            ventanas de fase fenológica (sección 4.10.2).
        aggregation: regla de agregación de la variable (Tabla 7).
        fecha_corte: si se especifica, ninguna fecha de `serie` puede ser
            posterior a este valor (regla anti-fuga, sección 4.10.4). Si se
            excede, se lanza `HorizonteExcedidoError` en vez de recortar
            silenciosamente — una violación de horizonte debe corregirse en el
            paso de extracción, no enmascararse aquí.

    Returns:
        DataFrame con una fila por fase, columnas `provincia_id`, `campana_id`,
        `fase`, `valor_agregado`. Una fase sin observaciones en la serie
        devuelve NaN, nunca 0 (evita ocultar huecos de datos, sección 4.7).
    """
    serie_fechas = pd.to_datetime(serie["fecha"]).dt.date
    if fecha_corte is not None and (serie_fechas > fecha_corte).any():
        maxima = serie_fechas.max()
        raise HorizonteExcedidoError(
            f"La serie contiene observaciones hasta {maxima}, posteriores al "
            f"punto de corte {fecha_corte}. Esto violaría la regla anti-fuga "
            "temporal (sección 4.12.2) — corrija la extracción, no la agregación."
        )

    funcion = _AGG_FUNC[aggregation]
    filas: list[dict] = []
    for _, fase_row in fases.iterrows():
        en_ventana = (serie_fechas >= fase_row["inicio"]) & (
            serie_fechas <= fase_row["fin"]
        )
        valores_en_fase = serie.loc[en_ventana, "valor"]
        valor_agregado = (
            getattr(valores_en_fase, funcion)() if len(valores_en_fase) > 0 else float("nan")
        )
        filas.append(
            {
                "provincia_id": fase_row["provincia_id"],
                "campana_id": fase_row["campana_id"],
                "fase": fase_row["fase"],
                "valor_agregado": valor_agregado,
            }
        )

    return pd.DataFrame(filas)
