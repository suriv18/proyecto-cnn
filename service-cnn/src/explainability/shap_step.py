"""Paso de SHAP del piloto: fondo de referencia, precisión local y contingencia (tarea T6).

Plan de la sección 4.15 y de la preespecificación (E5): `DeepExplainer` es el
explicador principal; si su propiedad de precisión local falla (suma de valores SHAP
+ valor base ≠ predicción del modelo, fricción documentada de `shap` con redes
profundas en series multivariadas), se migra a `KernelExplainer`, que cumple la
eficiencia por construcción. La tolerancia es una fracción de la desviación
estándar de la salida del modelo sobre el fondo de referencia.

El fondo de referencia debe salir SOLO de las campañas de entrenamiento del pliego
(4.12.2): lo garantiza el llamador; aquí solo se muestrea de forma reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap
import torch
from torch import nn

from src.explainability.shap_aggregation import aggregate_shap_by_phase
from src.explainability.shap_explainer import (
    LocalAccuracyError,
    SHAPExplanation,
    explain_with_deep_explainer,
    validate_local_accuracy,
)


def sample_background(
    x: torch.Tensor, covariables: torch.Tensor, n_max: int, semilla: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Muestra reproducible de hasta `n_max` filas (sin reemplazo) para el fondo de referencia."""
    n = x.shape[0]
    if n <= n_max:
        return x, covariables
    generador = torch.Generator().manual_seed(semilla)
    indices = torch.randperm(n, generator=generador)[:n_max]
    return x[indices], covariables[indices]


def explain_with_kernel_explainer(
    modelo: nn.Module,
    x_background: torch.Tensor,
    covariables_background: torch.Tensor,
    x_prueba: torch.Tensor,
    covariables_prueba: torch.Tensor,
    n_muestras: int | str = "auto",
    l1_reg: float | str = 0,
) -> SHAPExplanation:
    """Explicaciones SHAP con `KernelExplainer` sobre la entrada aplanada [V×T, covariables].

    Es la contingencia a `DeepExplainer`: agnóstica al modelo y con eficiencia exacta
    por construcción, a costa de más tiempo de cómputo.

    `l1_reg=0` desactiva la regularización L1 automática de `shap` (`"auto"`), que con
    muchas variables y pocas muestras encoge las atribuciones: en el piloto técnico la suma
    de |SHAP| quedaba a 1/2,2 de la de DeepExplainer (correlación 0,87); con 0 la correlación
    sube a 0,95. Se prefiere una estimación fiel de Shapley.
    """
    n_var, n_pasos = x_background.shape[1], x_background.shape[2]
    n_cov = covariables_background.shape[1]

    def predecir(plano: np.ndarray) -> np.ndarray:
        plano = np.atleast_2d(plano)
        x = torch.tensor(plano[:, : n_var * n_pasos].reshape(-1, n_var, n_pasos), dtype=torch.float32)
        z = torch.tensor(plano[:, n_var * n_pasos :], dtype=torch.float32)
        modelo.eval()
        with torch.no_grad():
            return modelo(x, z).squeeze(1).numpy()

    def aplanar(x: torch.Tensor, z: torch.Tensor) -> np.ndarray:
        return np.concatenate([x.reshape(x.shape[0], -1).numpy(), z.numpy()], axis=1)

    explainer = shap.KernelExplainer(predecir, aplanar(x_background, covariables_background))
    valores = np.asarray(explainer.shap_values(aplanar(x_prueba, covariables_prueba), nsamples=n_muestras, l1_reg=l1_reg, silent=True))
    if valores.ndim == 3:  # algunas versiones devuelven (n, features, 1)
        valores = valores[..., 0]
    esperado = np.asarray(explainer.expected_value).reshape(-1)[0]

    return SHAPExplanation(
        valores_secuenciales=valores[:, : n_var * n_pasos].reshape(-1, n_var, n_pasos),
        valores_covariables=valores[:, n_var * n_pasos : n_var * n_pasos + n_cov],
        valor_base=np.full(x_prueba.shape[0], float(esperado)),
    )


@dataclass(frozen=True)
class ExplanationOutcome:
    explicacion: SHAPExplanation
    variante: str  # "deep_explainer" o "kernel_explainer"
    tolerancia: float
    error_deep_explainer: float  # error máximo de precisión local de DeepExplainer (siempre se mide)


def _error_maximo(explicacion: SHAPExplanation, predicciones: np.ndarray) -> float:
    reconstruida = (
        explicacion.valor_base
        + explicacion.valores_secuenciales.sum(axis=(1, 2))
        + explicacion.valores_covariables.sum(axis=1)
    )
    return float(np.max(np.abs(reconstruida - predicciones)))


def explain_with_contingency(
    modelo: nn.Module,
    x_background: torch.Tensor,
    covariables_background: torch.Tensor,
    x_prueba: torch.Tensor,
    covariables_prueba: torch.Tensor,
    tolerancia_fraccion: float,
) -> ExplanationOutcome:
    """Explica con `DeepExplainer`; si falla la precisión local, migra a `KernelExplainer`.

    Args:
        tolerancia_fraccion: la tolerancia absoluta es esta fracción por la desviación
            estándar de la salida del modelo sobre el fondo de referencia.

    Raises:
        LocalAccuracyError: si tampoco `KernelExplainer` cumple la tolerancia.
    """
    modelo.eval()
    with torch.no_grad():
        desviacion = float(modelo(x_background, covariables_background).std())
        predicciones = modelo(x_prueba, covariables_prueba).squeeze(1).numpy()
    tolerancia = tolerancia_fraccion * max(desviacion, 1e-12)

    profunda = explain_with_deep_explainer(modelo, x_background, covariables_background, x_prueba, covariables_prueba)
    error_deep = _error_maximo(profunda, predicciones)
    try:
        validate_local_accuracy(profunda, predicciones, tolerancia)
        return ExplanationOutcome(profunda, "deep_explainer", tolerancia, error_deep)
    except LocalAccuracyError:
        pass

    kernel = explain_with_kernel_explainer(modelo, x_background, covariables_background, x_prueba, covariables_prueba)
    validate_local_accuracy(kernel, predicciones, tolerancia)  # si falla, propaga LocalAccuracyError
    return ExplanationOutcome(kernel, "kernel_explainer", tolerancia, error_deep)


def campaign_phase_contribution(explicacion: SHAPExplanation, mapeo_paso_a_fase: list[str]) -> dict[str, float]:
    """Contribución absoluta por fase de una campaña: promedio sobre sus observaciones.

    Es la entrada de `stability.evaluate_stability` (una entrada por campaña externa;
    las semillas se promedian antes, 3.3.3).
    """
    por_observacion = aggregate_shap_by_phase(explicacion.valores_secuenciales, mapeo_paso_a_fase)
    fases = por_observacion[0].keys()
    return {fase: float(np.mean([obs[fase] for obs in por_observacion])) for fase in fases}
