"""Indicadores de las dimensiones X1, X2 e Y1 del Anexo 01 (operacionalización de las variables).

  * Y1, exactitud predictiva: RMSE, MAE, rRMSE y R² fuera de muestra por modelo y campaña.
  * X1, modelado temporal con CNN-1D: campo receptivo temporal, porcentaje de predicciones válidas y
    porcentaje de ejecuciones reproducibles.
  * X2, explicabilidad con SHAP: porcentaje de predicciones explicadas y valor SHAP absoluto medio y
    orden de importancia por predictor.

Y2 (estabilidad, fidelidad y concordancia) está en `he2_evaluation`.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from torch import nn

from src.evaluation.metrics import mae, r2_fuera_de_muestra, rmse, rrmse


def accuracy_indicators(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, rRMSE y R². El rRMSE es NaN si la media observada es cercana a cero (Tabla 9)."""
    y_real, y_pred = np.asarray(y_real, dtype=float), np.asarray(y_pred, dtype=float)
    try:
        relativo = rrmse(y_real, y_pred)
    except ValueError:
        relativo = float("nan")
    return {"rmse": rmse(y_real, y_pred), "mae": mae(y_real, y_pred), "rrmse": relativo, "r2": r2_fuera_de_muestra(y_real, y_pred)}


def accuracy_table(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Indicadores de Y1 por modelo y campaña; las semillas se resumen con la mediana (4.12.4).

    Args:
        predicciones: columnas `modelo`, `campana_prueba`, `semilla`, `y_real`, `y_pred` (rendimiento
            reconstruido en kg/ha).
    """
    filas = []
    for (modelo, campana, semilla), g in predicciones.groupby(["modelo", "campana_prueba", "semilla"]):
        filas.append({"modelo": modelo, "campana_prueba": campana, "semilla": semilla, **accuracy_indicators(g["y_real"], g["y_pred"])})
    por_semilla = pd.DataFrame(filas)
    return por_semilla.groupby(["modelo", "campana_prueba"], as_index=False)[["rmse", "mae", "rrmse", "r2"]].median()


def receptive_field(modelo: nn.Module) -> int:
    """Campo receptivo temporal (en pasos) de la pila de convoluciones y pooling no globales.

    Se recorren las capas `Conv1d` y `MaxPool1d`/`AvgPool1d` en el orden en que están registradas
    y se acumula `r += (k - 1) * dilatación * salto`, con `salto *= stride`. El pooling global
    adaptativo no se cuenta: agrega todo el eje temporal después de la pila convolucional.
    """
    campo, salto = 1, 1
    for capa in modelo.modules():
        if isinstance(capa, nn.Conv1d):
            k, d, s = capa.kernel_size[0], capa.dilation[0], capa.stride[0]
        elif isinstance(capa, (nn.MaxPool1d, nn.AvgPool1d)):
            k = capa.kernel_size if isinstance(capa.kernel_size, int) else capa.kernel_size[0]
            d = getattr(capa, "dilation", 1)
            d = d if isinstance(d, int) else d[0]
            s = capa.stride if isinstance(capa.stride, int) else capa.stride[0]
        else:
            continue
        campo += (k - 1) * d * salto
        salto *= s
    return int(campo)


def valid_prediction_rate(predicciones: np.ndarray) -> float:
    """Fracción de predicciones finitas (ni NaN ni infinitas)."""
    predicciones = np.asarray(predicciones, dtype=float)
    if predicciones.size == 0:
        raise ValueError("No hay predicciones para evaluar su validez.")
    return float(np.mean(np.isfinite(predicciones)))


def reproducibility_rate(ejecutar: Callable[[int], np.ndarray], semillas: Iterable[int], tolerancia: float = 1e-6) -> float:
    """Fracción de semillas cuya ejecución, repetida con la misma semilla, da el mismo resultado."""
    semillas = list(semillas)
    if not semillas:
        raise ValueError("Se requiere al menos una semilla.")
    iguales = sum(
        bool(np.allclose(np.asarray(ejecutar(s), dtype=float), np.asarray(ejecutar(s), dtype=float), atol=tolerancia, rtol=0.0))
        for s in semillas
    )
    return iguales / len(semillas)


def explained_fraction(n_explicadas: int, n_total: int) -> float:
    """Fracción de las predicciones externas para las que se generó una explicación SHAP."""
    if n_total <= 0:
        raise ValueError("El total de predicciones debe ser positivo.")
    if not 0 <= n_explicadas <= n_total:
        raise ValueError(f"Las predicciones explicadas ({n_explicadas}) deben estar entre 0 y el total ({n_total}).")
    return n_explicadas / n_total


def shap_predictor_importance(valores_shap: np.ndarray, nombres_variables: list[str]) -> dict:
    """Valor SHAP absoluto medio por predictor y su orden de importancia.

    Args:
        valores_shap: forma (n_observaciones, n_variables, n_pasos). Por observación se suma el valor
            absoluto sobre los pasos (la misma unidad que la agregación por fase) y se promedia entre
            observaciones.
        nombres_variables: un nombre por variable.

    Returns:
        `{"importancia": {variable: valor}, "orden": [variables de mayor a menor]}`.
    """
    valores_shap = np.asarray(valores_shap, dtype=float)
    if len(nombres_variables) != valores_shap.shape[1]:
        raise ValueError(f"Se requiere un nombre por variable: {len(nombres_variables)} nombres para {valores_shap.shape[1]} variables.")
    por_variable = np.abs(valores_shap).sum(axis=2).mean(axis=0)
    importancia = {n: float(v) for n, v in zip(nombres_variables, por_variable)}
    return {"importancia": importancia, "orden": sorted(importancia, key=importancia.get, reverse=True)}
