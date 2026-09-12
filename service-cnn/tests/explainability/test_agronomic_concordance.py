"""Pruebas de concordancia agronómica (HE2c, sección 3.3.5, 4.15).

Criterio confirmatorio categórico: en el horizonte principal, la fase de
máxima contribución predictiva debe pertenecer al conjunto preespecificado
{floración, llenado de grano} (Vásquez et al., 2024). En el horizonte
temprano, la comparación se restringe a las fases observables hasta el cierre
de la floración. La coincidencia se interpreta como concordancia, no como
evidencia causal (sección 2.4.7).
"""
import pytest

from src.explainability.agronomic_concordance import (
    FASES_CRITICAS_PREESPECIFICADAS,
    evaluate_agronomic_concordance,
)


class TestFasesCriticasPreespecificadas:
    def test_incluye_floracion_y_llenado_de_grano(self):
        """Vásquez et al. (2024): periodo crítico entre floración y llenado
        de grano, sección 2.4.6."""
        assert FASES_CRITICAS_PREESPECIFICADAS == {"floracion", "llenado_grano"}


class TestEvaluateAgronomicConcordance:
    def test_concordancia_se_sostiene_si_fase_maxima_es_floracion(self):
        resultado = evaluate_agronomic_concordance(
            fase_maxima_contribucion="floracion", horizonte="principal"
        )
        assert resultado.concordancia_sostenida is True

    def test_concordancia_se_sostiene_si_fase_maxima_es_llenado_de_grano(self):
        resultado = evaluate_agronomic_concordance(
            fase_maxima_contribucion="llenado_grano", horizonte="principal"
        )
        assert resultado.concordancia_sostenida is True

    def test_concordancia_se_rechaza_si_fase_maxima_es_otra(self):
        resultado = evaluate_agronomic_concordance(
            fase_maxima_contribucion="emergencia", horizonte="principal"
        )
        assert resultado.concordancia_sostenida is False

    def test_horizonte_temprano_restringe_a_fases_observables_hasta_floracion(self):
        """Sección 3.3.5: en el horizonte temprano, la comparación se
        restringe a las fases observables hasta el cierre de la floración —
        llenado_grano no puede ser la fase máxima observada en ese horizonte
        porque aún no ha ocurrido, así que solo floración cuenta como
        concordante."""
        resultado = evaluate_agronomic_concordance(
            fase_maxima_contribucion="floracion", horizonte="temprano"
        )
        assert resultado.concordancia_sostenida is True

    def test_horizonte_temprano_rechaza_llenado_de_grano(self):
        """llenado_grano no debería aparecer como fase máxima observable en
        el horizonte temprano (fuera de rango); si ocurre por un error de
        construcción del tensor, debe marcarse como discordante, no como
        concordante por pertenecer al conjunto preespecificado general."""
        resultado = evaluate_agronomic_concordance(
            fase_maxima_contribucion="llenado_grano", horizonte="temprano"
        )
        assert resultado.concordancia_sostenida is False

    def test_falla_con_horizonte_desconocido(self):
        with pytest.raises(ValueError, match="horizonte"):
            evaluate_agronomic_concordance(
                fase_maxima_contribucion="floracion", horizonte="inexistente"
            )
