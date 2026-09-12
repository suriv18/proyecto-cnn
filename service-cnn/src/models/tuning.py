"""Optimización de hiperparámetros con Optuna (sección 4.12.3).

Los 5 modelos ajustables (CNN-1D, Elastic-Net, Random Forest, XGBoost, SVR)
reciben el MISMO número máximo de evaluaciones de hiperparámetros
("presupuesto de ajuste equiparado"). Optuna (TPE / optimización bayesiana con
pruning) se eligió sobre grid search exhaustivo por eficiencia bajo presupuesto
de cómputo acotado, siguiendo el precedente de Sabo et al. (2023) — ver
docs/02-arquitectura-tecnica.md §1. Los benchmarks (B1, B2, B3) NUNCA se
optimizan (sección 4.12.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna

from src.models.classic_models import build_classic_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

PRESUPUESTO_TRIALS_POR_MODELO: dict[str, int] = {
    "cnn1d": 30,
    "elastic_net": 30,
    "random_forest": 30,
    "xgboost": 30,
    "svr": 30,
}
"""Número máximo de evaluaciones de hiperparámetros, igual para los 5 modelos
ajustables (sección 4.12.3). Valor final a confirmar en el ensayo piloto (H2)
según el presupuesto de cómputo disponible tras H1; lo esencial del diseño es
que los 5 compartan el mismo número, no el valor específico."""


@dataclass
class TuningResult:
    """Resultado de una búsqueda de hiperparámetros."""

    mejores_hiperparametros: dict
    mejor_rmse_validacion: float
    n_trials_ejecutados: int


def sugerir_hiperparametro(
    trial: optuna.Trial, nombre: str, especificacion: tuple
) -> object:
    tipo, minimo, maximo = especificacion
    if tipo == "float":
        return trial.suggest_float(nombre, minimo, maximo)
    if tipo == "int":
        return trial.suggest_int(nombre, minimo, maximo)
    raise ValueError(
        f"Tipo de hiperparámetro '{tipo}' no soportado para '{nombre}'. "
        "Use 'float' o 'int'."
    )


def tune_classic_model(
    nombre_modelo: str,
    espacio_busqueda: dict[str, tuple],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    semilla: int,
) -> TuningResult:
    """Busca los mejores hiperparámetros de un modelo clásico con Optuna.

    Args:
        nombre_modelo: uno de `src.models.classic_models.MODELOS_DISPONIBLES`.
        espacio_busqueda: mapeo `{nombre_hiperparametro: (tipo, min, max)}`,
            con `tipo` en {"float", "int"}. Debe publicarse antes de la
            evaluación definitiva (registro de preespecificación H2, sección
            4.18) y dimensionarse sin favorecer sistemáticamente a una
            familia de modelos (sección 4.12.3).
        x_train, y_train: conjunto de entrenamiento del pliegue (después de
            aplanar el tensor V×T, ver `src/features/flatten.py`).
        x_val, y_val: partición interna de validación para seleccionar
            hiperparámetros — SIEMPRE dentro del conjunto de entrenamiento del
            pliegue, nunca la campaña externa de prueba (regla de anidamiento,
            sección 4.12.2).
        n_trials: número de evaluaciones — debe coincidir entre los 5 modelos
            ajustables (ver `PRESUPUESTO_TRIALS_POR_MODELO`).
        semilla: semilla del muestreador de Optuna, para reproducibilidad.

    Returns:
        `TuningResult` con los mejores hiperparámetros encontrados, el RMSE de
        validación correspondiente y el número de trials realmente ejecutados.
    """

    def objetivo(trial: optuna.Trial) -> float:
        hiperparametros = {
            nombre: sugerir_hiperparametro(trial, nombre, especificacion)
            for nombre, especificacion in espacio_busqueda.items()
        }
        modelo = build_classic_model(nombre_modelo, hiperparametros, semilla)
        modelo.fit(x_train, y_train)
        predicciones = modelo.predict(x_val)
        rmse = float(np.sqrt(np.mean((predicciones - y_val) ** 2)))
        return rmse

    muestreador = optuna.samplers.TPESampler(seed=semilla)
    estudio = optuna.create_study(direction="minimize", sampler=muestreador)
    estudio.optimize(objetivo, n_trials=n_trials)

    return TuningResult(
        mejores_hiperparametros=estudio.best_params,
        mejor_rmse_validacion=estudio.best_value,
        n_trials_ejecutados=len(estudio.trials),
    )
