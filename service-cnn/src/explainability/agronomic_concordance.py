"""Concordancia agronómica (HE2c, sección 3.3.5, 4.15).

Criterio confirmatorio categórico de plausibilidad externa. En el horizonte
principal, la condición se satisface si la fase de máxima contribución
predictiva pertenece al conjunto preespecificado {floración, llenado de
grano} (Vásquez et al., 2024). Para el horizonte temprano, la comparación se
restringe a las fases observables hasta el cierre de la floración. La
coincidencia se interpreta como concordancia entre la explicación del modelo
y el referente agronómico, NUNCA como evidencia causal (sección 2.4.7).
"""
from __future__ import annotations

from dataclasses import dataclass

FASES_CRITICAS_PREESPECIFICADAS = {"floracion", "llenado_grano"}
"""Periodo crítico de rendimiento de la quinua, Vásquez et al. (2024): entre
floración y llenado de grano (sección 2.4.6)."""

_FASES_VALIDAS_POR_HORIZONTE = {
    "principal": FASES_CRITICAS_PREESPECIFICADAS,
    # Horizonte temprano: solo floración es observable hasta el cierre de la
    # floración (sección 3.3.5) — llenado_grano aún no ha ocurrido en ese punto
    # de corte, por lo que no puede contar como concordancia válida aquí.
    "temprano": {"floracion"},
}


@dataclass(frozen=True)
class AgronomicConcordanceResult:
    """Resultado de la evaluación de concordancia agronómica (HE2c)."""

    fase_maxima_contribucion: str
    horizonte: str
    concordancia_sostenida: bool


def evaluate_agronomic_concordance(
    fase_maxima_contribucion: str, horizonte: str
) -> AgronomicConcordanceResult:
    """Evalúa si la fase de máxima contribución concuerda con el referente
    agronómico preespecificado, según el horizonte de pronóstico.

    Args:
        fase_maxima_contribucion: fase con mayor contribución SHAP agregada
            (salida de `fase_de_maxima_contribucion`).
        horizonte: "principal" (30 días antes de cosecha) o "temprano"
            (cierre de floración) — sección 1.5, 4.10.4.

    Returns:
        `AgronomicConcordanceResult` con si la concordancia se sostiene para
        ese horizonte.

    Raises:
        ValueError: si `horizonte` no es "principal" ni "temprano".
    """
    if horizonte not in _FASES_VALIDAS_POR_HORIZONTE:
        raise ValueError(
            f"horizonte debe ser uno de {sorted(_FASES_VALIDAS_POR_HORIZONTE)}; "
            f"recibido '{horizonte}'."
        )

    fases_validas = _FASES_VALIDAS_POR_HORIZONTE[horizonte]

    return AgronomicConcordanceResult(
        fase_maxima_contribucion=fase_maxima_contribucion,
        horizonte=horizonte,
        concordancia_sostenida=fase_maxima_contribucion in fases_validas,
    )
