"""Benchmark B3: regresión simple con NDVI máximo (Tabla 7-8, sección 3.4).

Cota mínima del aporte espectral: una regresión lineal univariada entre el
NDVI máximo de la campaña y el rendimiento. Ajustada exclusivamente con
campañas de entrenamiento (misma disciplina anti-fuga que B2, ver
`src/features/target_transform.py`). No se optimiza — es una regla de
referencia fija, no uno de los 5 modelos ajustables (sección 4.12.3): B1
(media histórica) y B2 (línea base) ya están en `target_transform.py`; B3
completa el trío de benchmarks de la Tabla 8.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NdviBenchmark:
    """Regresión lineal ndvi_max -> rendimiento_kg_ha, ajustada con entrenamiento."""

    pendiente: float
    intercepto: float


def fit_ndvi_benchmark(historia_entrenamiento: pd.DataFrame) -> NdviBenchmark:
    """Ajusta B3 usando únicamente el historial de entrenamiento del pliegue.

    Args:
        historia_entrenamiento: columnas `ndvi_max`, `rendimiento_kg_ha` — debe
            contener EXCLUSIVAMENTE observaciones de entrenamiento del pliegue
            actual (regla anti-fuga, sección 4.12.2); garantizar esa
            restricción es responsabilidad del llamador.

    Returns:
        `NdviBenchmark` con pendiente e intercepto de la regresión lineal.

    Raises:
        ValueError: si el historial está vacío o tiene menos de 2 observaciones.
    """
    if len(historia_entrenamiento) == 0:
        raise ValueError(
            "El historial de entrenamiento está vacío: no es posible ajustar B3."
        )
    if len(historia_entrenamiento) < 2:
        raise ValueError(
            "Se requieren al menos 2 observaciones para ajustar una regresión "
            f"lineal; se recibieron {len(historia_entrenamiento)}."
        )

    pendiente, intercepto = np.polyfit(
        historia_entrenamiento["ndvi_max"].to_numpy(dtype=float),
        historia_entrenamiento["rendimiento_kg_ha"].to_numpy(dtype=float),
        deg=1,
    )

    return NdviBenchmark(pendiente=float(pendiente), intercepto=float(intercepto))


def predict_ndvi_benchmark(
    benchmark: NdviBenchmark, ndvi_max: float | np.ndarray
) -> float | np.ndarray:
    """Predice el rendimiento de B3 para uno o varios valores de NDVI máximo."""
    return benchmark.intercepto + benchmark.pendiente * ndvi_max
