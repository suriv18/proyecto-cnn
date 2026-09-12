"""Métricas de evaluación de la exactitud predictiva (Tabla 9, sección 4.13).

RMSE y MAE en la unidad original (kg/ha); rRMSE como porcentaje relativo a la
media observada; R² fuera de muestra, que puede ser negativo (sección 2.4.10).
"""
from __future__ import annotations

import numpy as np

_UMBRAL_MEDIA_CERCANA_A_CERO = 1.0
"""Umbral en la unidad de la variable objetivo (kg/ha para nivel, o su
escala equivalente para anomalías). Un valor absoluto de 1 kg/ha es
insignificante frente a rendimientos típicos de cientos o miles de kg/ha, por
lo que una media observada por debajo de este umbral se considera "cercana a
cero" en el sentido de la Tabla 9. Ajustable si la escala de la anomalía en
uso lo justifica."""


def rmse(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """Raíz del error cuadrático medio: penaliza con mayor intensidad los
    errores grandes (sección 2.4.10)."""
    return float(np.sqrt(np.mean((y_real - y_pred) ** 2)))


def mae(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """Error absoluto medio: resume la desviación absoluta media."""
    return float(np.mean(np.abs(y_real - y_pred)))


def rrmse(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE expresado como porcentaje de la media observada.

    Raises:
        ValueError: si la media observada está cerca de cero — Tabla 9: "no se
            calculará sobre anomalías con media cercana a cero", porque el
            resultado sería un porcentaje engañosamente grande o indefinido.
    """
    media_observada = np.mean(y_real)
    if abs(media_observada) < _UMBRAL_MEDIA_CERCANA_A_CERO:
        raise ValueError(
            "La media observada está cercana a cero "
            f"({media_observada:.2e}); rRMSE no debe calcularse en este caso "
            "(Tabla 9: no se calcula sobre anomalías con media cercana a cero)."
        )
    return float(rmse(y_real, y_pred) / media_observada * 100.0)


def r2_fuera_de_muestra(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """R² calculado fuera de muestra: compara el error del modelo con la
    variabilidad del conjunto de prueba. Puede ser negativo si el modelo
    predice peor que la media del conjunto de prueba (sección 2.4.10)."""
    suma_residuos_cuadrados = np.sum((y_real - y_pred) ** 2)
    suma_total_cuadrados = np.sum((y_real - np.mean(y_real)) ** 2)
    return float(1.0 - suma_residuos_cuadrados / suma_total_cuadrados)
