"""Pruebas de la agregación de valores SHAP por fase fenológica (sección 4.15).

Unidad de atribución: predictor × paso temporal, agregada por fase fenológica.
Se usa contribución ABSOLUTA (no la suma con signo, que podría cancelarse
entre pasos positivos y negativos dentro de la misma fase) para identificar
qué fase concentra mayor contribución predictiva.
"""
import numpy as np
import pytest

from src.explainability.shap_aggregation import (
    aggregate_shap_by_phase,
    fase_de_maxima_contribucion,
)


class TestAggregateShapByPhase:
    def test_agrega_contribucion_absoluta_por_fase(self):
        # 1 observación, 2 variables, 4 pasos temporales -> 2 fases de 2 pasos c/u
        valores_shap = np.array([[[1.0, -2.0, 3.0, 0.5], [0.0, 1.0, -1.0, 2.0]]])
        mapeo_paso_a_fase = ["emergencia", "emergencia", "floracion", "floracion"]

        resultado = aggregate_shap_by_phase(valores_shap, mapeo_paso_a_fase)

        # emergencia: |1.0| + |-2.0| + |0.0| + |1.0| = 4.0 (variable0: 1+2=3, variable1: 0+1=1 -> total 4)
        assert resultado[0]["emergencia"] == pytest.approx(4.0)
        # floracion: |3.0| + |0.5| + |-1.0| + |2.0| = 6.5
        assert resultado[0]["floracion"] == pytest.approx(6.5)

    def test_devuelve_un_diccionario_por_observacion(self):
        valores_shap = np.zeros((3, 2, 4))
        mapeo_paso_a_fase = ["f1", "f1", "f2", "f2"]
        resultado = aggregate_shap_by_phase(valores_shap, mapeo_paso_a_fase)
        assert len(resultado) == 3

    def test_falla_si_el_mapeo_no_coincide_con_el_numero_de_pasos(self):
        valores_shap = np.zeros((1, 2, 4))
        mapeo_incompleto = ["f1", "f1", "f2"]  # solo 3, se esperan 4
        with pytest.raises(ValueError, match="pasos temporales"):
            aggregate_shap_by_phase(valores_shap, mapeo_incompleto)


class TestFaseDeMaximaContribucion:
    def test_identifica_la_fase_con_mayor_contribucion_absoluta(self):
        contribucion_por_fase = {"emergencia": 4.0, "floracion": 6.5, "madurez": 1.0}
        assert fase_de_maxima_contribucion(contribucion_por_fase) == "floracion"

    def test_falla_si_el_diccionario_esta_vacio(self):
        with pytest.raises(ValueError, match="vac[ií]o"):
            fase_de_maxima_contribucion({})
