"""Fidelidad mediante oclusión (HE2b, sección 3.3.4, 4.15).

En el conjunto externo de prueba, se reemplazan las entradas de la ventana de
alta contribución por su valor de referencia (background), SIN reentrenar, y
se compara el incremento del RMSE con el producido por una ventana de control
de igual duración y baja contribución. La fidelidad se sostiene cuando el
incremento por oclusión de alta contribución es mayor que el del control
(sección 3.3.4). El término "ablación" se reserva para el análisis secundario
con reentrenamiento — no implementado aquí.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.evaluation.metrics import rmse


def occlude_phase(
    tensor: torch.Tensor, indices_fase: list[int], valor_referencia: torch.Tensor
) -> torch.Tensor:
    """Reemplaza los pasos temporales de una fase por un valor de referencia.

    Args:
        tensor: forma (batch, n_variables, n_pasos_temporales).
        indices_fase: índices de los pasos temporales a ocluir.
        valor_referencia: forma (n_variables,) — valor de referencia del
            background por variable, aplicado a todos los pasos ocluidos.

    Returns:
        Una copia de `tensor` con los pasos indicados reemplazados; el
        original no se modifica.
    """
    ocluido = tensor.clone()
    for indice in indices_fase:
        ocluido[:, :, indice] = valor_referencia
    return ocluido


@dataclass(frozen=True)
class FidelityResult:
    """Resultado de la evaluación de fidelidad por oclusión (HE2b)."""

    incremento_rmse_alta_contribucion: float
    incremento_rmse_control: float
    fidelidad_sostenida: bool


def evaluate_fidelity_by_occlusion(
    modelo: nn.Module,
    x_prueba: torch.Tensor,
    covariables_prueba: torch.Tensor,
    y_real: torch.Tensor,
    indices_fase_alta_contribucion: list[int],
    indices_fase_control: list[int],
    valor_referencia: torch.Tensor,
) -> FidelityResult:
    """Evalúa la fidelidad de las explicaciones mediante oclusión (HE2b).

    Args:
        modelo: modelo ya entrenado, en modo eval — NUNCA se reentrena aquí.
        x_prueba, covariables_prueba: observaciones externas de prueba.
        y_real: rendimiento observado de esas mismas observaciones.
        indices_fase_alta_contribucion: pasos temporales de la fase con mayor
            contribución SHAP agregada (identificada por
            `fase_de_maxima_contribucion`).
        indices_fase_control: pasos temporales de una fase de igual duración
            y baja contribución, usada como control.
        valor_referencia: valor de referencia del background por variable,
            usado para ocluir (mismo criterio que el conjunto de referencia
            de SHAP, sección 4.15).

    Returns:
        `FidelityResult` con el incremento de RMSE al ocluir cada ventana
        respecto del RMSE sin oclusión, y si la fidelidad se sostiene
        (incremento de alta contribución > incremento de control).

    Raises:
        ValueError: si las dos ventanas no tienen la misma duración (número
            de pasos temporales), condición exigida por la sección 3.3.4 para
            que la comparación sea válida.
    """
    if len(indices_fase_alta_contribucion) != len(indices_fase_control):
        raise ValueError(
            "La ventana de alta contribución y la ventana de control deben "
            "tener la misma duración para que la comparación sea válida "
            f"(sección 3.3.4); recibido {len(indices_fase_alta_contribucion)} "
            f"y {len(indices_fase_control)} pasos respectivamente."
        )

    modelo.eval()
    with torch.no_grad():
        prediccion_base = modelo(x_prueba, covariables_prueba)
        rmse_base = rmse(y_real.numpy(), prediccion_base.numpy())

        x_ocluido_alta = occlude_phase(
            x_prueba, indices_fase_alta_contribucion, valor_referencia
        )
        prediccion_ocluida_alta = modelo(x_ocluido_alta, covariables_prueba)
        rmse_ocluido_alta = rmse(y_real.numpy(), prediccion_ocluida_alta.numpy())

        x_ocluido_control = occlude_phase(
            x_prueba, indices_fase_control, valor_referencia
        )
        prediccion_ocluida_control = modelo(x_ocluido_control, covariables_prueba)
        rmse_ocluido_control = rmse(y_real.numpy(), prediccion_ocluida_control.numpy())

    incremento_alta_contribucion = rmse_ocluido_alta - rmse_base
    incremento_control = rmse_ocluido_control - rmse_base

    return FidelityResult(
        incremento_rmse_alta_contribucion=incremento_alta_contribucion,
        incremento_rmse_control=incremento_control,
        fidelidad_sostenida=incremento_alta_contribucion > incremento_control,
    )


def occlusion_increments_by_phase(
    modelo: nn.Module,
    x_prueba: torch.Tensor,
    covariables_prueba: torch.Tensor,
    y_real: torch.Tensor,
    valor_referencia: torch.Tensor,
    indices_por_fase: dict[str, list[int]],
) -> dict[str, float]:
    """Aumento de RMSE al ocluir cada fase por separado, sin reentrenar.

    Es el insumo de `he2_pvalues.fidelity_pvalue`: permite comparar la fase de mayor
    contribución con todas las demás y no solo con un control. Exige que todas las fases
    tengan la misma duración, como la comparación de 3.3.4.

    Raises:
        ValueError: si hay menos de 2 fases o sus duraciones difieren.
    """
    if len(indices_por_fase) < 2:
        raise ValueError("Se requieren al menos 2 fases para comparar la oclusión.")
    duraciones = {len(indices) for indices in indices_por_fase.values()}
    if len(duraciones) != 1:
        raise ValueError(
            "Todas las fases deben tener la misma duración para que la comparación sea válida "
            f"(sección 3.3.4); recibido {sorted(duraciones)}."
        )

    modelo.eval()
    with torch.no_grad():
        rmse_base = rmse(y_real.numpy(), modelo(x_prueba, covariables_prueba).numpy())
        aumentos = {}
        for fase, indices in indices_por_fase.items():
            ocluido = occlude_phase(x_prueba, indices, valor_referencia)
            aumentos[fase] = float(rmse(y_real.numpy(), modelo(ocluido, covariables_prueba).numpy()) - rmse_base)
    return aumentos


def conditional_occlusion_increments_by_phase(
    modelo: nn.Module,
    x_prueba: torch.Tensor,
    covariables_prueba: torch.Tensor,
    y_real: torch.Tensor,
    x_fondo: torch.Tensor,
    indices_por_fase: dict[str, list[int]],
    k: int = 10,
) -> dict[str, float]:
    """Aumento de RMSE al ocluir cada fase con un reemplazo CONDICIONAL, sin reentrenar.

    En lugar de un valor de referencia fijo (que crea combinaciones fuera de distribución, Hase et
    al., 2021), la fase ocluida de cada observación se rellena con el promedio de esa misma fase en
    las `k` observaciones del fondo más parecidas en el RESTO de las fases (distancia euclídea). Así
    se conserva la dependencia entre fases. El fondo debe provenir solo del entrenamiento.

    Raises:
        ValueError: si hay menos de 2 fases o sus duraciones difieren.
    """
    if len(indices_por_fase) < 2:
        raise ValueError("Se requieren al menos 2 fases para comparar la oclusión.")
    if len({len(i) for i in indices_por_fase.values()}) != 1:
        raise ValueError("Todas las fases deben tener la misma duración para que la comparación sea válida (sección 3.3.4).")

    n_pasos = x_prueba.shape[2]
    k = max(1, min(k, x_fondo.shape[0]))
    modelo.eval()
    with torch.no_grad():
        rmse_base = rmse(y_real.numpy(), modelo(x_prueba, covariables_prueba).numpy())
        aumentos = {}
        for fase, indices in indices_por_fase.items():
            resto = [i for i in range(n_pasos) if i not in indices]
            distancias = torch.cdist(x_prueba[:, :, resto].flatten(1), x_fondo[:, :, resto].flatten(1))
            vecinos = distancias.topk(k, dim=1, largest=False).indices  # (n_prueba, k)
            reemplazo = x_fondo[:, :, indices][vecinos].mean(dim=1)  # (n_prueba, V, len(indices))
            ocluido = x_prueba.clone()
            ocluido[:, :, indices] = reemplazo
            aumentos[fase] = float(rmse(y_real.numpy(), modelo(ocluido, covariables_prueba).numpy()) - rmse_base)
    return aumentos
