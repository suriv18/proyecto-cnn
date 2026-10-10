"""Bucle de entrenamiento de la CNN-1D (sección 4.12.3-4.12.4).

Optimizador Adam con weight_decay (regularización L2, siguiendo a Sabo et al.
2023), early stopping sobre una partición interna de validación, y
reproducibilidad estricta vía semilla explícita — necesaria para las 10
réplicas por semilla que exige la sección 4.12.4 para modelos estocásticos.

Nota: esta función entrena sobre un conjunto ya perteneciente a un pliegue de
entrenamiento (ver `src/evaluation/splits.py`); la partición de validación
interna es solo para early stopping, siempre estrictamente dentro del conjunto
de entrenamiento del pliegue, nunca usando la campaña externa de prueba (regla
anti-fuga, 4.12.2).

Validación temporal (brecha G1 de la preespecificación): si se pasan las
campañas de cada fila (`grupos_temporales`), la validación es la ÚLTIMA campaña
del entrenamiento, como exige 4.12.2 (orden temporal y solo campañas
anteriores). Sin ellas se conserva la partición aleatoria anterior, que mezcla
provincias de la misma campaña (correlacionadas) y NO es válida para la
evaluación definitiva.

Mejor época (brecha G2): por defecto se restauran los pesos de la época con
menor pérdida de validación, no los de la última.
"""
from __future__ import annotations

import copy
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
    fraccion_validacion: float = 0.2  # solo en la partición aleatoria (sin campañas)
    restaurar_mejor_epoca: bool = True


@dataclass
class TrainingResult:
    """Resultado del entrenamiento: el modelo ajustado y su historial de pérdida."""

    modelo: nn.Module
    historial_perdida: list[float] = field(default_factory=list)
    historial_perdida_validacion: list[float] = field(default_factory=list)
    mejor_epoca: int = 0  # numeración desde 1
    filas_validacion: list[int] = field(default_factory=list)


def train_cnn1d(
    modelo: nn.Module,
    x_secuencial: torch.Tensor,
    covariables: torch.Tensor,
    y: torch.Tensor,
    training_config: TrainingConfig,
    grupos_temporales: torch.Tensor | None = None,
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
        grupos_temporales: campaña de cada fila (vector de longitud N). Si se
            indica, la validación del early stopping es la última campaña
            (temporal, sin aleatoriedad) y el resto es entrenamiento.

    Returns:
        `TrainingResult` con el modelo ajustado y el historial de pérdida de
        entrenamiento y de validación por época (pueden ser más cortos que
        `epocas_maximas` si el early stopping se activa antes), la mejor época
        (desde 1) y las filas usadas como validación.

    Raises:
        ValueError: si `fraccion_validacion` no está en (0, 1) o si
            `grupos_temporales` contiene menos de dos campañas.
    """
    if not (0.0 < training_config.fraccion_validacion < 1.0):
        raise ValueError(
            "fraccion_validacion debe estar en (0, 1); recibido: "
            f"{training_config.fraccion_validacion}"
        )

    torch.manual_seed(training_config.semilla)
    modelo._inicializar_pesos()

    n_muestras = x_secuencial.shape[0]
    if grupos_temporales is not None:
        ultima_campana = grupos_temporales.max()
        es_validacion = grupos_temporales == ultima_campana
        if bool(es_validacion.all()):
            raise ValueError(
                "La validación temporal necesita al menos dos campañas: todas las "
                "filas pertenecen a la misma (no quedaría entrenamiento)."
            )
        indices_validacion = torch.nonzero(es_validacion, as_tuple=True)[0]
        indices_entrenamiento = torch.nonzero(~es_validacion, as_tuple=True)[0]
    else:
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
    historial_perdida_validacion: list[float] = []
    mejor_perdida_validacion = float("inf")
    mejor_epoca = 0
    mejores_pesos = None
    epocas_sin_mejora = 0

    for epoca in range(1, training_config.epocas_maximas + 1):
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
        historial_perdida_validacion.append(perdida_val)

        if perdida_val < mejor_perdida_validacion:
            mejor_perdida_validacion = perdida_val
            mejor_epoca = epoca
            mejores_pesos = copy.deepcopy(modelo.state_dict())
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= training_config.paciencia:
                break

    if training_config.restaurar_mejor_epoca and mejores_pesos is not None:
        modelo.load_state_dict(mejores_pesos)

    return TrainingResult(
        modelo=modelo,
        historial_perdida=historial_perdida,
        historial_perdida_validacion=historial_perdida_validacion,
        mejor_epoca=mejor_epoca,
        filas_validacion=[int(i) for i in indices_validacion],
    )
