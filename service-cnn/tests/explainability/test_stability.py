"""Pruebas de estabilidad de las explicaciones (HE2a, sección 3.3.3).

Concordancia del ordenamiento de contribución por fase entre campañas
externas: W de Kendall (umbral >= 0.70) e índice de Jaccard de las 3 fases
principales (umbral >= 0.50). Las semillas se agregan por campaña ANTES de
evaluar estabilidad (para no inflar el número de unidades independientes).
"""
import pytest

from src.explainability.stability import (
    StabilityResult,
    evaluate_stability,
    jaccard_index,
    kendalls_w,
)


class TestKendallsW:
    def test_w_es_uno_cuando_el_ordenamiento_es_identico_en_todas_las_campanas(self):
        """3 campañas, 4 fases, mismo ranking exacto en las 3."""
        contribuciones_por_campana = [
            {"emergencia": 1.0, "desarrollo": 2.0, "floracion": 4.0, "madurez": 3.0},
            {"emergencia": 0.5, "desarrollo": 1.5, "floracion": 3.5, "madurez": 2.5},
            {"emergencia": 2.0, "desarrollo": 3.0, "floracion": 8.0, "madurez": 5.0},
        ]
        w = kendalls_w(contribuciones_por_campana)
        assert w == pytest.approx(1.0)

    def test_w_es_bajo_cuando_el_ordenamiento_es_aleatorio_entre_campanas(self):
        contribuciones_por_campana = [
            {"emergencia": 1.0, "desarrollo": 4.0, "floracion": 2.0, "madurez": 3.0},
            {"emergencia": 4.0, "desarrollo": 1.0, "floracion": 3.0, "madurez": 2.0},
            {"emergencia": 2.0, "desarrollo": 3.0, "floracion": 1.0, "madurez": 4.0},
        ]
        w = kendalls_w(contribuciones_por_campana)
        assert w < 0.5

    def test_w_esta_entre_cero_y_uno(self):
        contribuciones_por_campana = [
            {"a": 1.0, "b": 2.0, "c": 3.0},
            {"a": 3.0, "b": 1.0, "c": 2.0},
        ]
        w = kendalls_w(contribuciones_por_campana)
        assert 0.0 <= w <= 1.0

    def test_falla_con_menos_de_dos_campanas(self):
        with pytest.raises(ValueError, match="al menos 2"):
            kendalls_w([{"a": 1.0, "b": 2.0}])


class TestJaccardIndex:
    def test_jaccard_es_uno_si_los_conjuntos_son_identicos(self):
        assert jaccard_index({"floracion", "llenado_grano", "madurez"}, {"floracion", "llenado_grano", "madurez"}) == pytest.approx(1.0)

    def test_jaccard_es_cero_si_no_hay_interseccion(self):
        assert jaccard_index({"a", "b"}, {"c", "d"}) == pytest.approx(0.0)

    def test_jaccard_parcial(self):
        # interseccion={floracion,llenado_grano} (2), union=4 -> J=0.5
        a = {"floracion", "llenado_grano", "madurez"}
        b = {"floracion", "llenado_grano", "emergencia"}
        assert jaccard_index(a, b) == pytest.approx(0.5)


class TestEvaluateStability:
    def test_se_sostiene_cuando_w_y_jaccard_superan_los_umbrales(self):
        contribuciones_por_campana = [
            {"emergencia": 1.0, "desarrollo": 2.0, "floracion": 4.0, "madurez": 3.0},
            {"emergencia": 0.5, "desarrollo": 1.5, "floracion": 3.5, "madurez": 2.5},
            {"emergencia": 2.0, "desarrollo": 3.0, "floracion": 8.0, "madurez": 5.0},
        ]
        resultado = evaluate_stability(
            contribuciones_por_campana, n_fases_principales=3
        )
        assert isinstance(resultado, StabilityResult)
        assert resultado.w_kendall >= 0.70
        assert resultado.jaccard_promedio >= 0.50
        assert resultado.estabilidad_sostenida is True

    def test_no_se_sostiene_cuando_el_ordenamiento_es_inconsistente(self):
        contribuciones_por_campana = [
            {"emergencia": 1.0, "desarrollo": 4.0, "floracion": 2.0, "madurez": 3.0},
            {"emergencia": 4.0, "desarrollo": 1.0, "floracion": 3.0, "madurez": 2.0},
            {"emergencia": 2.0, "desarrollo": 3.0, "floracion": 1.0, "madurez": 4.0},
        ]
        resultado = evaluate_stability(
            contribuciones_por_campana, n_fases_principales=3
        )
        assert resultado.estabilidad_sostenida is False
