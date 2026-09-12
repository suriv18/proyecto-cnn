"""Generación de explicaciones SHAP sobre la CNN-1D (sección 4.15).

Las explicaciones se generan ÚNICAMENTE sobre observaciones externas de
prueba, después del entrenamiento. El conjunto de referencia/background debe
extraerse solo de campañas de entrenamiento (regla anti-fuga, sección 4.12.2)
— responsabilidad del llamador, no de este módulo.

`DeepExplainer` es el explainer principal (más rápido, precedente: Joshi et
al. 2025). Existe fricción documentada (2025-2026) entre `shap` y frameworks
de redes profundas que puede violar la propiedad de "local accuracy" en series
temporales multivariadas (docs/02-arquitectura-tecnica.md §1.1); por eso
`validate_local_accuracy` debe ejecutarse en el ensayo piloto (H2) antes de
comprometerse a DeepExplainer, con `KernelExplainer` como contingencia
documentada (no implementada aún — se añade si la validación falla).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap
import torch
from torch import nn


class LocalAccuracyError(ValueError):
    """La suma de los valores SHAP más el valor base no reproduce la
    predicción real del modelo dentro de la tolerancia — señal de que
    DeepExplainer no es fiable para esta arquitectura/entrada y debe migrarse
    a KernelExplainer (plan de contingencia, sección 4.15)."""


@dataclass(frozen=True)
class SHAPExplanation:
    """Valores SHAP de un lote de observaciones, separados por tipo de entrada."""

    valores_secuenciales: np.ndarray
    """Forma (n_observaciones, n_variables, n_pasos_temporales)."""
    valores_covariables: np.ndarray
    """Forma (n_observaciones, n_covariables)."""
    valor_base: np.ndarray
    """Forma (n_observaciones,) — valor esperado del modelo sobre el
    background, replicado por observación para facilitar `validate_local_accuracy`."""


def explain_with_deep_explainer(
    modelo: nn.Module,
    x_background: torch.Tensor,
    covariables_background: torch.Tensor,
    x_prueba: torch.Tensor,
    covariables_prueba: torch.Tensor,
) -> SHAPExplanation:
    """Genera explicaciones SHAP con `shap.DeepExplainer` sobre observaciones de prueba.

    Args:
        modelo: `QuinuaYieldCNN1D` (u otro `nn.Module` con la misma firma
            `forward(x_secuencial, covariables)`), ya entrenado y en modo eval.
        x_background, covariables_background: conjunto de referencia — debe
            provenir exclusivamente de campañas de entrenamiento del pliegue
            (regla anti-fuga, sección 4.12.2); esta función no lo valida.
        x_prueba, covariables_prueba: observaciones externas de prueba sobre
            las que se generan las explicaciones (sección 4.15: "únicamente
            para observaciones externas de prueba").

    Returns:
        `SHAPExplanation` con los valores SHAP separados por tipo de entrada,
        en la misma forma que los tensores de entrada correspondientes.
    """
    explainer = shap.DeepExplainer(modelo, [x_background, covariables_background])
    valores_shap = explainer.shap_values([x_prueba, covariables_prueba])

    valores_secuenciales, valores_covariables = valores_shap
    valores_secuenciales = np.asarray(valores_secuenciales).squeeze(-1)
    valores_covariables = np.asarray(valores_covariables).squeeze(-1)

    valor_esperado = explainer.expected_value
    if isinstance(valor_esperado, (list, np.ndarray)):
        valor_esperado = np.asarray(valor_esperado).reshape(-1)[0]
    valor_base = np.full(x_prueba.shape[0], float(valor_esperado))

    return SHAPExplanation(
        valores_secuenciales=valores_secuenciales,
        valores_covariables=valores_covariables,
        valor_base=valor_base,
    )


def validate_local_accuracy(
    explicacion: SHAPExplanation, predicciones_reales: np.ndarray, tolerancia: float
) -> None:
    """Valida la propiedad de "local accuracy" de Lundberg y Lee (2017).

    La suma de los valores SHAP (secuenciales + covariables) más el valor
    base debe reproducir la predicción real del modelo para cada observación.
    Esta validación es la condición explícita, documentada en la arquitectura
    técnica, para decidir si `DeepExplainer` es fiable en el ensayo piloto
    (H2) o si debe migrarse a `KernelExplainer`.

    Args:
        explicacion: resultado de `explain_with_deep_explainer` (u otro
            explainer con la misma estructura de salida).
        predicciones_reales: predicción real del modelo (`modelo(x, cov)`)
            para las mismas observaciones, en el mismo orden.
        tolerancia: diferencia absoluta máxima permitida entre la suma
            reconstruida y la predicción real.

    Raises:
        LocalAccuracyError: si alguna observación excede la tolerancia —
            indica que DeepExplainer no es fiable para esta configuración y
            debe activarse el plan de contingencia (KernelExplainer).
    """
    suma_secuencial = explicacion.valores_secuenciales.sum(axis=(1, 2))
    suma_covariables = explicacion.valores_covariables.sum(axis=1)
    prediccion_reconstruida = (
        explicacion.valor_base + suma_secuencial + suma_covariables
    )

    diferencias = np.abs(prediccion_reconstruida - predicciones_reales)
    if np.any(diferencias > tolerancia):
        peor_diferencia = float(diferencias.max())
        raise LocalAccuracyError(
            "La propiedad de local accuracy no se cumple dentro de la "
            f"tolerancia ({tolerancia}): diferencia máxima observada "
            f"{peor_diferencia:.4f}. Active el plan de contingencia "
            "documentado en docs/02-arquitectura-tecnica.md §1.1 "
            "(migrar a KernelExplainer)."
        )
