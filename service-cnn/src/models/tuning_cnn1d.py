"""Optimización de hiperparámetros de la CNN-1D con Optuna (sección 4.12.3).

Mismo contrato de resultado que `src/models/tuning.tune_classic_model`
(`TuningResult`, mismo criterio de RMSE de validación), pero operando sobre
tensores 3D y reutilizando `train_cnn1d` para entrenar cada trial. Necesario
para que la CNN-1D reciba exactamente el mismo tipo de presupuesto de
evaluaciones que los 4 modelos clásicos — la igualdad se garantiza usando la
misma entrada `PRESUPUESTO_TRIALS_POR_MODELO["cnn1d"]` declarada en
`tuning.py`, no un valor separado.
"""
from __future__ import annotations

import numpy as np
import optuna
import torch

from src.models.cnn1d import CNN1DConfig, QuinuaYieldCNN1D
from src.models.cnn1d_trainer import TrainingConfig, train_cnn1d
from src.models.tuning import TuningResult, sugerir_hiperparametro

optuna.logging.set_verbosity(optuna.logging.WARNING)

_HIPERPARAMETROS_ARQUITECTURA = {"filtros_capa1", "filtros_capa2", "kernel_size", "dropout", "unidades_densa"}
_HIPERPARAMETROS_ENTRENAMIENTO = {"tasa_aprendizaje", "weight_decay"}


def tune_cnn1d(
    n_variables: int,
    n_covariables: int,
    espacio_busqueda: dict[str, tuple],
    x_train: torch.Tensor,
    covariables_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    covariables_val: torch.Tensor,
    y_val: torch.Tensor,
    n_trials: int,
    epocas_por_trial: int,
    semilla: int,
) -> TuningResult:
    """Busca los mejores hiperparámetros de la CNN-1D con Optuna.

    Args:
        n_variables, n_covariables: dimensiones fijas del problema (no forman
            parte del espacio de búsqueda).
        espacio_busqueda: mapeo `{nombre: (tipo, min, max)}`. Las claves
            reconocidas en `CNN1DConfig` (filtros_capa1, filtros_capa2,
            kernel_size, dropout, unidades_densa) ajustan la arquitectura;
            las reconocidas en `TrainingConfig` (tasa_aprendizaje,
            weight_decay) ajustan el entrenamiento. Debe publicarse antes de
            la evaluación definitiva (registro de preespecificación H2).
        x_train, covariables_train, y_train: tensores del conjunto de
            entrenamiento del pliegue (ya restringidos a campañas de
            entrenamiento, regla anti-fuga sección 4.12.2).
        x_val, covariables_val, y_val: partición interna de validación para
            seleccionar hiperparámetros — dentro del entrenamiento del
            pliegue, nunca la campaña externa de prueba.
        n_trials: número de evaluaciones; debe coincidir con el presupuesto de
            los 4 modelos clásicos (`PRESUPUESTO_TRIALS_POR_MODELO["cnn1d"]`
            en `tuning.py`), no un valor definido de forma independiente aquí.
        epocas_por_trial: épocas máximas de entrenamiento dentro de cada
            trial (con early stopping propio de `train_cnn1d`).
        semilla: semilla del muestreador de Optuna Y de cada entrenamiento
            individual dentro de los trials, para reproducibilidad total.

    Returns:
        `TuningResult` con los mejores hiperparámetros encontrados (separados
        entre arquitectura y entrenamiento en el mismo diccionario plano), el
        RMSE de validación correspondiente y el número de trials ejecutados.
    """

    def objetivo(trial: optuna.Trial) -> float:
        hiperparametros = {
            nombre: sugerir_hiperparametro(trial, nombre, especificacion)
            for nombre, especificacion in espacio_busqueda.items()
        }

        kwargs_arquitectura = {
            k: v for k, v in hiperparametros.items() if k in _HIPERPARAMETROS_ARQUITECTURA
        }
        kwargs_entrenamiento = {
            k: v for k, v in hiperparametros.items() if k in _HIPERPARAMETROS_ENTRENAMIENTO
        }

        config = CNN1DConfig(
            n_variables=n_variables,
            n_pasos_temporales=x_train.shape[2],
            n_covariables=n_covariables,
            **kwargs_arquitectura,
        )
        modelo = QuinuaYieldCNN1D(config)

        training_config = TrainingConfig(
            epocas_maximas=epocas_por_trial, semilla=semilla, **kwargs_entrenamiento
        )
        train_cnn1d(modelo, x_train, covariables_train, y_train, training_config)

        modelo.eval()
        with torch.no_grad():
            predicciones = modelo(x_val, covariables_val)
        rmse = float(torch.sqrt(torch.mean((predicciones - y_val) ** 2)))
        return rmse

    muestreador = optuna.samplers.TPESampler(seed=semilla)
    estudio = optuna.create_study(direction="minimize", sampler=muestreador)
    estudio.optimize(objetivo, n_trials=n_trials)

    return TuningResult(
        mejores_hiperparametros=estudio.best_params,
        mejor_rmse_validacion=estudio.best_value,
        n_trials_ejecutados=len(estudio.trials),
    )
