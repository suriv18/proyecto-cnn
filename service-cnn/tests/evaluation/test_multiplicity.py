"""Corrección de Holm por multiplicidad (3.3.5).

Los valores de referencia se calcularon a mano con el procedimiento descendente de Holm:
p ordenados de menor a mayor, p_(i) * (m - i + 1), máximo acumulado y tope en 1.
"""
from __future__ import annotations

import pytest

from src.evaluation.multiplicity import holm_adjust, holm_reject


class TestHolmAdjust:
    def test_valores_ajustados_de_un_ejemplo_calculado_a_mano(self):
        ajustados = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})

        assert ajustados["a"] == pytest.approx(0.03)  # 0.01 * 3
        assert ajustados["c"] == pytest.approx(0.06)  # 0.03 * 2
        assert ajustados["b"] == pytest.approx(0.06)  # max(0.06, 0.04 * 1): monotonía

    def test_el_ajustado_nunca_es_menor_que_el_p_original(self):
        p = {"a": 0.2, "b": 0.01, "c": 0.5, "d": 0.04}

        ajustados = holm_adjust(p)

        assert all(ajustados[k] >= p[k] for k in p)

    def test_el_ajustado_se_trunca_en_uno(self):
        assert holm_adjust({"a": 0.6, "b": 0.7})["b"] == 1.0

    def test_con_un_solo_contraste_no_cambia(self):
        assert holm_adjust({"a": 0.03}) == {"a": 0.03}

    def test_conserva_el_orden_y_las_claves_de_entrada(self):
        assert list(holm_adjust({"z": 0.5, "a": 0.01})) == ["z", "a"]

    def test_los_empates_reciben_el_mismo_ajustado(self):
        ajustados = holm_adjust({"a": 0.02, "b": 0.02, "c": 0.9})

        assert ajustados["a"] == ajustados["b"] == pytest.approx(0.06)

    @pytest.mark.parametrize("malo", [-0.1, 1.1, float("nan")])
    def test_rechaza_valores_p_fuera_de_rango(self, malo):
        with pytest.raises(ValueError, match="entre 0 y 1"):
            holm_adjust({"a": malo})

    def test_rechaza_una_familia_vacia(self):
        with pytest.raises(ValueError, match="vacía"):
            holm_adjust({})


class TestHolmReject:
    def test_rechaza_las_hipotesis_cuyo_ajustado_no_supera_alfa(self):
        rechazos = holm_reject({"a": 0.01, "b": 0.04, "c": 0.03}, alfa=0.05)

        assert rechazos == {"a": True, "b": False, "c": False}

    def test_holm_rechaza_mas_que_bonferroni_en_un_caso_conocido(self):
        p = {"a": 0.012, "b": 0.02, "c": 0.9}  # Bonferroni (0.05/3 = 0.0167) solo rechaza "a"

        assert holm_reject(p, alfa=0.05) == {"a": True, "b": True, "c": False}

    def test_el_p_minimo_de_wilcoxon_con_seis_campanas_sobrevive_a_una_familia_de_tres(self):
        assert holm_reject({"a": 1 / 64, "b": 0.9, "c": 0.9}, alfa=0.05)["a"] is True

    def test_el_p_minimo_de_wilcoxon_con_cinco_campanas_no_sobrevive_a_una_familia_de_dos(self):
        assert holm_reject({"a": 1 / 32, "b": 0.9}, alfa=0.05)["a"] is False
