"""Bucle de entrenamiento de la CNN-1D (sección 4.12.3-4.12.4).

Optimizador Adam con weight_decay (regularización L2, siguiendo a Sabo et al.
2023), early stopping sobre una partición interna de validación, y
reproducibilidad estricta vía semilla explícita — necesaria para las 10
réplicas por semilla que exige la sección 4.12.4 para modelos estocásticos.

Nota: esta función entrena sobre un conjunto ya perteneciente a un pliegue de
entrenamiento (ver `src/evaluation/splits.py`); no realiza ningún split
temporal — la partición de validación interna aquí es solo para early
stopping, siempre estrictamente dentro del conjunto de entrenamiento del
pliegue, nunca usando la campaña externa de prueba (regla anti-fuga, 4.12.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparámetros del bucle de entrenamiento."""

    epocas_maximas: int
    semilla: int
    tasa_aprendizaje: float = 1e-3
    weight_decay: float = 1e-6  # regularización L2, Sabo et al. (2023)
    paciencia: int = 10
    fraccion_validacion: float = 0.2


@dataclass
class TrainingResult:
    """Resultado del entrenamiento: el modelo ajustado y su historial de pérdida."""

    modelo: nn.Module
    historial_perdida: list[float] = field(default_factory=list)


def train_cnn1d(
    modelo: nn.Module,
    x_secuencial: torch.Tensor,
    covariables: torch.Tensor,
    y: torch.Tensor,
    training_config: TrainingConfig,
) -> TrainingResult:
    """Entrena una `QuinuaYieldCNN1D` con Adam + weight_decay y early stopping.

    Args:
        modelo: instancia de `QuinuaYieldCNN1D` (u otra `nn.Module` con la
            misma firma `forward(x_secuencial, covariables)` y un método
            `_inicializar_pesos()`). Sus pesos se RE-INICIALIZAN bajo la
            semilla de `training_config` al inicio de esta función — la
            reproducibilidad de las 10 réplicas por semilla (sección 4.12.4)
            debe cubrir tanto la inicialización de pesos como el
            entrenamiento, no solo este último; de lo contrario, pesos
            iniciales creados antes de fijar la semilla romperían la
            reproducibilidad entre llamadas.
        x_secuencial, covariables, y: tensores del conjunto de entrenamiento
            del pliegue actual — ya restringidos a campañas de entrenamiento
            por el llamador (regla anti-fuga, sección 4.12.2).
        training_config: hiperparámetros del entrenamiento.

    Returns:
        `TrainingResult` con el modelo ajustado y el historial de pérdida de
        entrenamiento por época (puede ser más corto que `epocas_maximas` si
        el early stopping se activa antes).

    Raises:
        ValueError: si `fraccion_validacion` no está en (0, 1).
    """
    if not (0.0 < training_config.fraccion_validacion < 1.0):
        raise ValueError(
            "fraccion_validacion debe estar en (0, 1); recibido: "
            f"{training_config.fraccion_validacion}"
        )

    torch.manual_seed(training_config.semilla)
    modelo._inicializar_pesos()

    n_muestras = x_secuencial.shape[0]
    n_validacion = max(1, int(n_muestras * training_config.fraccion_validacion))
    indices = torch.randperm(n_muestras)
    indices_validacion = indices[:n_validacion]
    indices_entrenamiento = indices[n_validacion:]

    x_train, cov_train, y_train = (
        x_secuencial[indices_entrenamiento],
        covariables[indices_entrenamiento],
        y[indices_entrenamiento],
    )
    x_val, cov_val, y_val = (
        x_secuencial[indices_validacion],
        covariables[indices_validacion],
        y[indices_validacion],
    )

    optimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=training_config.tasa_aprendizaje,
        weight_decay=training_config.weight_decay,
    )
    funcion_perdida = nn.MSELoss()

    historial_perdida: list[float] = []
    mejor_perdida_validacion = float("inf")
    epocas_sin_mejora = 0

    for _ in range(training_config.epocas_maximas):
        modelo.train()
        optimizador.zero_grad()
        prediccion_train = modelo(x_train, cov_train)
        perdida_train = funcion_perdida(prediccion_train, y_train)
        perdida_train.backward()
        optimizador.step()

        historial_perdida.append(perdida_train.item())

        modelo.eval()
        with torch.no_grad():
            prediccion_val = modelo(x_val, cov_val)
            perdida_val = funcion_perdida(prediccion_val, y_val).item()

        if perdida_val < mejor_perdida_validacion:
            mejor_perdida_validacion = perdida_val
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= training_config.paciencia:
                break

    return TrainingResult(modelo=modelo, historial_perdida=historial_perdida)
