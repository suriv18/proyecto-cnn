"""Orquestación del tensor V×T sobre el conjunto de datos real (sección 4.10.3).

Conecta las funciones puras `ingestion.gee_series_builder.aggregate_by_phase`
y `features.tensor_builder.build_observation_tensor` con las dos fuentes reales:
  - `data/raw/gee_series_crudo.parquet`: series por variable.
  - `data/interim/fenologia_alineada.parquet`: una fila por fase y
    provincia-campaña (salida de `preprocessing.phenology_pipeline`).

Convención de nombres en `orden_variables`: cada entrada es una variable del
catálogo de `gee_config` (ej. `"precipitacion"`). Para las variables con varios
reducers (NDVI: media y máximo, Tabla 7) la entrada puede llevar el sufijo del
reducer (`"ndvi_mean"`, `"ndvi_max"`); sin sufijo se usa el reducer principal
(`aggregation` de la especificación, la media para NDVI).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.tensor_builder import (
    ObservacionIncompletaError,
    build_observation_tensor,
)
from src.ingestion.gee_config import AggregationRule, get_collection_spec
from src.ingestion.gee_series_builder import aggregate_by_phase


def resolve_variable(entrada: str) -> tuple[str, str, AggregationRule]:
    """Devuelve `(variable, columna_de_valor, regla)` para una entrada de `orden_variables`."""
    for regla in AggregationRule:
        sufijo = f"_{regla.value}"
        if entrada.endswith(sufijo):
            base = entrada.removesuffix(sufijo)
            try:
                spec = get_collection_spec(base)
            except KeyError:
                continue
            if regla in spec.reducers:
                return base, f"valor_{regla.value}", regla

    spec = get_collection_spec(entrada)
    if spec.reducers:
        return entrada, f"valor_{spec.aggregation.value}", spec.aggregation
    return entrada, "valor", spec.aggregation


def _orden_de_fases_del_lote(fenologia_alineada: pd.DataFrame) -> list[str]:
    """Orden cronológico de fases válido para todo el lote.

    Se recorre cada provincia-campaña por fecha de inicio y se agregan las
    fases aún no vistas, de modo que una observación truncada por el horizonte
    no recorte las columnas de las demás.
    """
    orden: list[str] = []
    for _, grupo in fenologia_alineada.groupby(["provincia_id", "campana_id"]):
        for fase in grupo.sort_values("inicio")["fase"]:
            if fase not in orden:
                orden.append(fase)
    return orden


def _arrastrar_ultima_observacion(
    agregada: pd.DataFrame,
    fases: pd.DataFrame,
    serie_variable: pd.DataFrame,
    columna: str,
    max_dias: int,
) -> pd.DataFrame:
    """Rellena las fases vacías con la última observación previa (imputación limitada).

    Sección 4.7 (interpolación limitada) sin fuga: para cada fase sin valor se
    toma la observación válida más reciente con `fecha <= fin` de la fase y a
    lo sumo `max_dias` antes de ese fin. Nunca se usan observaciones
    posteriores, así que no se viola el horizonte de pronóstico (4.10.4).

    Devuelve `agregada` con `valor_agregado` completado y la columna booleana
    `imputada`.
    """
    resultado = agregada.copy()
    resultado["imputada"] = False
    if max_dias <= 0:
        return resultado

    serie = serie_variable.assign(_f=pd.to_datetime(serie_variable["fecha"])).dropna(subset=[columna])
    for i, fila in resultado.iterrows():
        if pd.notna(fila["valor_agregado"]):
            continue
        fin = pd.Timestamp(fases.loc[fases["fase"] == fila["fase"], "fin"].iloc[0])
        previas = serie[(serie["_f"] <= fin) & (serie["_f"] >= fin - pd.Timedelta(days=max_dias))]
        if previas.empty:
            continue
        resultado.at[i, "valor_agregado"] = previas.sort_values("_f")[columna].iloc[-1]
        resultado.at[i, "imputada"] = True
    return resultado


def build_all_observation_tensors(
    series_crudas: pd.DataFrame,
    fenologia_alineada: pd.DataFrame,
    orden_variables: list[str],
    orden_fases: list[str] | None = None,
    max_dias_arrastre: int = 0,
) -> list[dict]:
    """Construye el tensor V×T de cada provincia-campaña presente en `fenologia_alineada`.

    Args:
        series_crudas: formato de `gee_series_crudo.parquet` (`provincia_id`,
            `campana_id`, `fecha`, `variable` y `valor` o `valor_mean`/
            `valor_max` según la variable).
        fenologia_alineada: formato de `fenologia_alineada.parquet`
            (`provincia_id`, `campana_id`, `fase`, `inicio`, `fin`).
        orden_variables: filas del tensor, en orden fijo (ver convención del
            módulo).
        orden_fases: columnas del tensor. Por defecto, el orden cronológico
            común a todo el lote. Una fase ausente en una observación (ej.
            excluida por el horizonte, sección 4.10.4) queda como NaN.
        max_dias_arrastre: si es > 0, las celdas vacías de las variables con
            varios reducers (NDVI: compuestos de 16 días) se completan con la
            última observación previa, a lo sumo este número de días antes del
            fin de la fase (ver `_arrastrar_ultima_observacion`). 0 = sin
            imputación.

    Returns:
        Un dict por provincia-campaña con `provincia_id`, `campana_id`,
        `tensor` (array V×T), `imputadas` (array booleano V×T, True en las
        celdas rellenadas por arrastre) y `error`. Si falta por completo una variable, el
        tensor es `None` y `error` describe cuál, sin descartar el resto del
        lote.
    """
    if orden_fases is None:
        orden_fases = _orden_de_fases_del_lote(fenologia_alineada)

    resueltas = [(entrada, *resolve_variable(entrada)) for entrada in orden_variables]

    resultados: list[dict] = []
    for (provincia_id, campana_id), fases in fenologia_alineada.groupby(
        ["provincia_id", "campana_id"], sort=True
    ):
        series_obs = series_crudas[
            (series_crudas["provincia_id"] == provincia_id)
            & (series_crudas["campana_id"] == campana_id)
        ]

        agregadas: list[pd.DataFrame] = []
        for entrada, variable, columna, regla in resueltas:
            serie_variable = series_obs[series_obs["variable"] == variable]
            if serie_variable.empty:
                continue
            agregada = aggregate_by_phase(
                serie=serie_variable,
                fases=fases,
                reglas_por_columna={columna: regla},
            )
            valor = [c for c in agregada.columns if c.startswith("valor_agregado")][0]
            agregada = agregada.rename(columns={valor: "valor_agregado"})
            if get_collection_spec(variable).reducers:
                agregada = _arrastrar_ultima_observacion(
                    agregada, fases, serie_variable, columna, max_dias_arrastre
                )
            else:
                agregada["imputada"] = False
            agregada["variable"] = entrada
            agregadas.append(agregada)

        base = {"provincia_id": provincia_id, "campana_id": campana_id}
        try:
            if not agregadas:
                raise ObservacionIncompletaError(
                    f"Ninguna de las variables {orden_variables} tiene observaciones."
                )
            todas = pd.concat(agregadas, ignore_index=True)
            tensor = build_observation_tensor(
                series_agregadas=todas,
                orden_variables=orden_variables,
                orden_fases=orden_fases,
            )
            banderas = build_observation_tensor(
                series_agregadas=todas.assign(valor_agregado=todas["imputada"].astype(float)),
                orden_variables=orden_variables,
                orden_fases=orden_fases,
            )
        except ObservacionIncompletaError as exc:
            resultados.append({**base, "tensor": None, "imputadas": None, "error": str(exc)})
            continue

        resultados.append(
            {**base, "tensor": tensor, "imputadas": np.nan_to_num(banderas) > 0.5, "error": None}
        )

    return resultados
