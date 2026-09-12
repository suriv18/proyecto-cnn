"""Contrastes estadísticos confirmatorios HE1a y HE1b (sección 3.3.1-3.3.2, 4.14).

HE1a — no inferioridad frente a XGBoost: para cada campaña externa f,
q_f = RMSE(CNN-1D,f)/RMSE(XGBoost,f) - 1. Margen δ=0.05. La no inferioridad se
sostiene si el límite superior unilateral del intervalo de confianza al 95%
de la mediana de q_f es menor que 0.05.

HE1b — superioridad frente a B2: Δ_f = RMSE(CNN-1D,f) - RMSE(B2,f). Se
sostiene si el límite superior unilateral del IC 95% de la mediana de Δ_f es
menor que cero.

Intervalos obtenidos mediante remuestreo (bootstrap) sobre las campañas
externas (bloques temporales, sección 4.14); Wilcoxon pareado unilateral como
contraste de apoyo; se reportan además el estimador de Hodges-Lehmann y la
correlación biserial de rangos para datos pareados.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _validar_misma_longitud(a: np.ndarray, b: np.ndarray) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"Los arreglos deben tener la misma longitud; recibido {len(a)} y "
            f"{len(b)} (una observación por campaña externa)."
        )


def _limite_superior_bootstrap_mediana(
    valores: np.ndarray, n_bootstrap: int, semilla: int
) -> float:
    """Límite superior unilateral del IC 95% de la mediana, vía bootstrap
    sobre las campañas externas (remuestreo por bloques temporales, 4.14)."""
    generador = np.random.default_rng(semilla)
    medianas_bootstrap = np.empty(n_bootstrap)
    n = len(valores)
    for i in range(n_bootstrap):
        muestra = generador.choice(valores, size=n, replace=True)
        medianas_bootstrap[i] = np.median(muestra)
    return float(np.quantile(medianas_bootstrap, 0.95))


def _hodges_lehmann(diferencias: np.ndarray) -> float:
    """Estimador de Hodges-Lehmann: mediana de los promedios de Walsh (todos
    los promedios por pares, incluyendo el promedio de cada valor consigo
    mismo)."""
    n = len(diferencias)
    promedios_walsh = [
        (diferencias[i] + diferencias[j]) / 2.0
        for i in range(n)
        for j in range(i, n)
    ]
    return float(np.median(promedios_walsh))


def _correlacion_biserial_rangos_pareada(diferencias: np.ndarray) -> float:
    """Correlación biserial de rangos para datos pareados (efecto asociado al
    contraste de Wilcoxon): proporción de la suma de rangos que favorece la
    dirección positiva menos la que favorece la negativa, normalizada."""
    diferencias_no_nulas = diferencias[diferencias != 0]
    if len(diferencias_no_nulas) == 0:
        return 0.0
    rangos = stats.rankdata(np.abs(diferencias_no_nulas))
    suma_positivos = rangos[diferencias_no_nulas > 0].sum()
    suma_negativos = rangos[diferencias_no_nulas < 0].sum()
    suma_total = rangos.sum()
    return float((suma_positivos - suma_negativos) / suma_total)


@dataclass(frozen=True)
class NonInferiorityResult:
    """Resultado del contraste HE1a."""

    limite_superior_ic95: float
    no_inferioridad_sostenida: bool
    wilcoxon_p_valor: float
    hodges_lehmann: float
    correlacion_biserial_rangos: float


@dataclass(frozen=True)
class SuperiorityResult:
    """Resultado del contraste HE1b."""

    limite_superior_ic95: float
    superioridad_sostenida: bool
    wilcoxon_p_valor: float
    hodges_lehmann: float
    correlacion_biserial_rangos: float


def evaluate_non_inferiority(
    rmse_cnn: np.ndarray,
    rmse_xgboost: np.ndarray,
    margen: float,
    n_bootstrap: int,
    semilla: int,
) -> NonInferiorityResult:
    """Contrasta HE1a: no inferioridad de CNN-1D frente a XGBoost.

    Args:
        rmse_cnn, rmse_xgboost: RMSE por campaña externa (mismo orden,
            emparejados) de cada modelo — ya agregados por campaña tras las
            10 semillas (mediana entre semillas, sección 3.3.3, para no
            inflar el número de unidades independientes).
        margen: margen de no inferioridad δ (0.05 en la tesis).
        n_bootstrap: número de remuestreos para el intervalo de confianza.
        semilla: semilla del generador aleatorio, para reproducibilidad.

    Returns:
        `NonInferiorityResult` con el límite superior del IC 95% de la
        mediana de q_f, si la no inferioridad se sostiene (límite < margen),
        el p-valor de Wilcoxon pareado unilateral de apoyo, y los estadísticos
        complementarios de Hodges-Lehmann y correlación biserial de rangos.

    Raises:
        ValueError: si los arreglos no tienen la misma longitud.
    """
    _validar_misma_longitud(rmse_cnn, rmse_xgboost)

    q_f = rmse_cnn / rmse_xgboost - 1.0

    limite_superior = _limite_superior_bootstrap_mediana(q_f, n_bootstrap, semilla)

    _, p_valor = stats.wilcoxon(rmse_cnn, rmse_xgboost, alternative="less")

    return NonInferiorityResult(
        limite_superior_ic95=limite_superior,
        no_inferioridad_sostenida=limite_superior < margen,
        wilcoxon_p_valor=float(p_valor),
        hodges_lehmann=_hodges_lehmann(q_f),
        correlacion_biserial_rangos=_correlacion_biserial_rangos_pareada(
            rmse_cnn - rmse_xgboost
        ),
    )


def evaluate_superiority(
    rmse_cnn: np.ndarray,
    rmse_b2: np.ndarray,
    n_bootstrap: int,
    semilla: int,
) -> SuperiorityResult:
    """Contrasta HE1b: superioridad de CNN-1D frente al benchmark B2.

    Args:
        rmse_cnn, rmse_b2: RMSE por campaña externa (emparejados) de CNN-1D y
            del benchmark de línea base provincial.
        n_bootstrap: número de remuestreos para el intervalo de confianza.
        semilla: semilla del generador aleatorio.

    Returns:
        `SuperiorityResult` con el límite superior del IC 95% de la mediana de
        Δ_f = RMSE(CNN-1D) - RMSE(B2); la superioridad se sostiene si ese
        límite es menor que cero.

    Raises:
        ValueError: si los arreglos no tienen la misma longitud.
    """
    _validar_misma_longitud(rmse_cnn, rmse_b2)

    delta_f = rmse_cnn - rmse_b2

    limite_superior = _limite_superior_bootstrap_mediana(delta_f, n_bootstrap, semilla)

    _, p_valor = stats.wilcoxon(rmse_cnn, rmse_b2, alternative="less")

    return SuperiorityResult(
        limite_superior_ic95=limite_superior,
        superioridad_sostenida=limite_superior < 0.0,
        wilcoxon_p_valor=float(p_valor),
        hodges_lehmann=_hodges_lehmann(delta_f),
        correlacion_biserial_rangos=_correlacion_biserial_rangos_pareada(delta_f),
    )
