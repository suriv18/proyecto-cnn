"""Tratamiento de la variable objetivo: nivel vs. anomalía (sección 4.11).

La variable objetivo es el rendimiento oficial de quinua en kg/ha. Se evalúan
dos representaciones: nivel (kg/ha) y anomalía respecto de una línea base
provincial. La línea base se estima con una tendencia temporal simple
(regresión lineal sobre la campaña) ajustada EXCLUSIVAMENTE con campañas de
entrenamiento — nunca con la campaña externa de prueba. Las provincias con
historia insuficiente emplean una regla de respaldo documentada (media simple).

Las métricas confirmatorias se reportan siempre sobre el rendimiento
reconstruido en kg/ha; las métricas sobre anomalías son un análisis
complementario (sección 4.11). B2 (benchmark) es la predicción de esta misma
línea base.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HISTORIA_MINIMA_CAMPANAS = 5
"""Número mínimo de campañas de entrenamiento para ajustar una tendencia
lineal confiable; por debajo de este umbral se usa la regla de respaldo (media
simple). Valor a confirmar/ajustar en el ensayo piloto (H2, sección 4.11)."""


@dataclass(frozen=True)
class ProvincialBaseline:
    """Línea base provincial ajustada solo con campañas de entrenamiento."""

    provincia_id: str
    pendiente: float
    intercepto: float
    campana_referencia: int
    es_respaldo: bool


def fit_provincial_baseline(
    historia_entrenamiento: pd.DataFrame, provincia_id: str
) -> ProvincialBaseline:
    """Ajusta la línea base de una provincia usando solo su historia de entrenamiento.

    Args:
        historia_entrenamiento: columnas `provincia_id`, `campana_id`,
            `rendimiento_kg_ha` — debe contener EXCLUSIVAMENTE campañas de
            entrenamiento del pliegue actual (regla anti-fuga, sección 4.12.2);
            garantizar esa restricción es responsabilidad del llamador.
        provincia_id: provincia para la que se ajusta la línea base.

    Returns:
        `ProvincialBaseline` con pendiente e intercepto de la regresión lineal
        sobre `campana_id - campana_referencia` (para estabilidad numérica), o
        con pendiente 0 e intercepto igual a la media simple si la historia
        disponible tiene menos de `HISTORIA_MINIMA_CAMPANAS` observaciones
        (regla de respaldo documentada, sección 4.11).

    Raises:
        ValueError: si la provincia no tiene ninguna observación en
            `historia_entrenamiento`.
    """
    historia_provincia = historia_entrenamiento[
        historia_entrenamiento["provincia_id"] == provincia_id
    ].sort_values("campana_id")

    if historia_provincia.empty:
        raise ValueError(
            f"La provincia '{provincia_id}' no tiene ninguna observación en "
            "el conjunto de entrenamiento provisto."
        )

    campana_referencia = int(historia_provincia["campana_id"].iloc[0])

    if len(historia_provincia) < HISTORIA_MINIMA_CAMPANAS:
        media = float(historia_provincia["rendimiento_kg_ha"].mean())
        return ProvincialBaseline(
            provincia_id=provincia_id,
            pendiente=0.0,
            intercepto=media,
            campana_referencia=campana_referencia,
            es_respaldo=True,
        )

    x = (historia_provincia["campana_id"] - campana_referencia).to_numpy(dtype=float)
    y = historia_provincia["rendimiento_kg_ha"].to_numpy(dtype=float)
    pendiente, intercepto = np.polyfit(x, y, deg=1)

    return ProvincialBaseline(
        provincia_id=provincia_id,
        pendiente=float(pendiente),
        intercepto=float(intercepto),
        campana_referencia=campana_referencia,
        es_respaldo=False,
    )


def predict_baseline(baseline: ProvincialBaseline, campana_id: int) -> float:
    """Predice el rendimiento de línea base (B2) para una campaña dada.

    Con la regla de respaldo (`es_respaldo=True`), la predicción es constante
    (la media histórica) para cualquier campaña, ya que la pendiente es 0.
    """
    x = campana_id - baseline.campana_referencia
    return baseline.intercepto + baseline.pendiente * x


def to_anomaly(
    rendimiento_observado: float, baseline: ProvincialBaseline, campana_id: int
) -> float:
    """Calcula la anomalía: rendimiento observado menos la línea base predicha.

    La anomalía es la representación complementaria de modelado (sección
    4.11); las métricas confirmatorias siempre se calculan sobre el
    rendimiento reconstruido en kg/ha, no sobre esta anomalía.
    """
    return rendimiento_observado - predict_baseline(baseline, campana_id)
