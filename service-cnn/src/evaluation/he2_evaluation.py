"""Evaluación de HE2 definitiva (tesis v3, secciones 3.3.3 a 3.3.5).

HE2 se respalda cuando se cumple el criterio PRIMARIO de concordancia agronómica (prueba binomial
exacta sobre las campañas externas, sin corrección porque es un único contraste primario). La
fidelidad (aleatorización exacta por rangos de la oclusión de cada fase) y la estabilidad
(permutación de las fases dentro de cada campaña) son criterios SECUNDARIOS: se reportan con su
magnitud, con la corrección de Holm entre ellos, y no condicionan la decisión. Los umbrales W >= 0,70
y J >= 0,50 se conservan solo como referencia descriptiva.

Además se informa la proporción de contribución SHAP que reciben las fases del referente agronómico,
que debe superar la que cabría por azar (fases del referente / fases totales).

La unidad es la campaña externa. Las contribuciones por fase y los aumentos de RMSE por oclusión
llegan ya agregados por campaña (las semillas se resumen antes, 3.3.3).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.explainability.agronomic_concordance import _FASES_VALIDAS_POR_HORIZONTE
from src.explainability.he2_pvalues import concordance_pvalue, fidelity_pvalue, he2_family_decision, stability_pvalue
from src.explainability.shap_aggregation import fase_de_maxima_contribucion
from src.explainability.stability import UMBRAL_JACCARD, UMBRAL_W_KENDALL


@dataclass(frozen=True)
class He2Result:
    campanas: list[int]
    n_campanas: int
    fases_maximas: dict[int, str]
    p_concordancia: float
    rechaza_concordancia: bool
    w_kendall: float
    jaccard_promedio: float
    umbrales_de_estabilidad_cumplidos: bool
    p_valores: dict[str, float]
    p_ajustados_secundarios: dict[str, float]
    rechazos_secundarios: dict[str, bool]
    secundarios_cumplidos: bool
    proporcion_contribucion_referente: float
    proporcion_referente_supera_azar: bool
    he2_respaldada: bool


def evaluate_he2(
    contribuciones_por_campana: dict[int, dict[str, float]],
    aumentos_rmse_por_campana: dict[int, dict[str, float]],
    horizonte: str,
    n_fases_principales: int,
    n_permutaciones: int,
    semilla: int,
    alfa: float = 0.05,
) -> He2Result:
    """Contrasta HE2 sobre las campañas externas.

    Args:
        contribuciones_por_campana: `{campaña: {fase: contribución SHAP absoluta agregada}}`.
        aumentos_rmse_por_campana: `{campaña: {fase: aumento de RMSE al ocluir esa fase}}`.
        horizonte: "principal" o "temprano" (define el referente agronómico).

    Raises:
        ValueError: si hay menos de 2 campañas, si las campañas de ambas tablas no coinciden
            o si las fases no son las mismas en todas las campañas.
    """
    if set(contribuciones_por_campana) != set(aumentos_rmse_por_campana):
        raise ValueError(
            "Las contribuciones y los aumentos de RMSE deben tener las mismas campañas; "
            f"recibido {sorted(contribuciones_por_campana)} y {sorted(aumentos_rmse_por_campana)}."
        )
    campanas = sorted(contribuciones_por_campana)
    if len(campanas) < 2:
        raise ValueError("HE2 requiere al menos 2 campañas externas.")
    fases = {frozenset(d) for d in list(contribuciones_por_campana.values()) + list(aumentos_rmse_por_campana.values())}
    if len(fases) != 1:
        raise ValueError("Todas las campañas deben tener las mismas fases en contribuciones y aumentos de RMSE.")
    n_fases = len(next(iter(fases)))

    contribuciones = [contribuciones_por_campana[c] for c in campanas]
    aumentos = [aumentos_rmse_por_campana[c] for c in campanas]
    maximas = [fase_de_maxima_contribucion(c) for c in contribuciones]

    estabilidad = stability_pvalue(contribuciones, n_fases_principales, n_permutaciones, semilla)
    p = {
        "estabilidad": estabilidad.p,
        "fidelidad": fidelity_pvalue(aumentos, maximas),
        "concordancia": concordance_pvalue(maximas, horizonte, n_fases),
    }
    decision = he2_family_decision(p["estabilidad"], p["fidelidad"], alfa)
    umbrales = estabilidad.w_kendall >= UMBRAL_W_KENDALL and estabilidad.jaccard_promedio >= UMBRAL_JACCARD
    rechaza_concordancia = p["concordancia"] <= alfa
    referente = _FASES_VALIDAS_POR_HORIZONTE[horizonte]
    proporciones = [sum(v for f, v in c.items() if f in referente) / sum(c.values()) for c in contribuciones]
    proporcion = float(sum(proporciones) / len(proporciones))
    return He2Result(
        campanas=[int(c) for c in campanas],
        n_campanas=len(campanas),
        fases_maximas=dict(zip((int(c) for c in campanas), maximas)),
        p_concordancia=p["concordancia"],
        rechaza_concordancia=bool(rechaza_concordancia),
        w_kendall=estabilidad.w_kendall,
        jaccard_promedio=estabilidad.jaccard_promedio,
        umbrales_de_estabilidad_cumplidos=bool(umbrales),
        p_valores=p,
        p_ajustados_secundarios=decision.p_ajustados,
        rechazos_secundarios=decision.rechazos,
        secundarios_cumplidos=decision.secundarios_cumplidos,
        proporcion_contribucion_referente=proporcion,
        proporcion_referente_supera_azar=bool(proporcion > len(referente) / n_fases),
        he2_respaldada=bool(rechaza_concordancia),
    )
