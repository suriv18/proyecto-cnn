"""Valores p de los tres criterios de HE2 y su decisión conjunta con Holm (3.3.5).

Definiciones (propuestas, ver docs/crisp/F4-evaluacion.md):
  * estabilidad: prueba de permutación de las fases dentro de cada campaña; el p es el mayor
    de los de W de Kendall y Jaccard medio de las 3 fases principales (se exigen ambos);
  * fidelidad: prueba de aleatorización exacta con el rango del aumento de RMSE de la fase de
    mayor contribución entre todas las fases ocluidas, sumando rangos entre campañas;
  * concordancia: binomial exacta unilateral con p0 = (fases críticas) / (fases totales).
"""
from __future__ import annotations

import pytest

from src.explainability.he2_pvalues import (
    concordance_pvalue,
    fidelity_pvalue,
    he2_family_decision,
    stability_pvalue,
)

FASES = ["emergencia", "desarrollo_vegetativo", "floracion", "llenado_grano", "madurez"]


def _campana(orden: list[str]) -> dict[str, float]:
    """Contribución decreciente según el orden dado (el primero es el mayor)."""
    return {fase: float(len(orden) - i) for i, fase in enumerate(orden)}


class TestStabilityPvalue:
    def test_campanas_con_el_mismo_orden_dan_un_p_muy_pequeno(self):
        campanas = [_campana(FASES)] * 5

        r = stability_pvalue(campanas, n_fases_principales=3, n_permutaciones=4000, semilla=0)

        assert r.p_w < 0.001 and r.p < 0.01

    def test_ordenes_rotados_sin_acuerdo_dan_un_p_grande(self):
        campanas = [_campana(FASES[i:] + FASES[:i]) for i in range(5)]  # W = 0

        r = stability_pvalue(campanas, n_fases_principales=3, n_permutaciones=2000, semilla=0)

        assert r.p_w > 0.9

    def test_el_p_es_el_mayor_de_los_dos_componentes(self):
        r = stability_pvalue([_campana(FASES)] * 5, n_fases_principales=3, n_permutaciones=1000, semilla=0)

        assert r.p == max(r.p_w, r.p_jaccard)

    def test_es_reproducible_con_la_misma_semilla(self):
        campanas = [_campana(FASES), _campana(FASES[::-1]), _campana(FASES)]

        a = stability_pvalue(campanas, 3, n_permutaciones=500, semilla=7)
        b = stability_pvalue(campanas, 3, n_permutaciones=500, semilla=7)

        assert a == b

    def test_el_p_nunca_es_cero(self):
        r = stability_pvalue([_campana(FASES)] * 5, 3, n_permutaciones=200, semilla=0)

        assert r.p_w >= 1 / 201

    def test_exige_al_menos_dos_campanas(self):
        with pytest.raises(ValueError, match="al menos 2"):
            stability_pvalue([_campana(FASES)], 3, n_permutaciones=100, semilla=0)


def test_los_estadisticos_observados_coinciden_con_evaluate_stability():
    from src.explainability.stability import evaluate_stability

    campanas = [{"a": 3.0, "b": 1.0, "c": 2.0, "d": 5.0, "e": 4.0}, {"a": 2.0, "b": 1.0, "c": 3.0, "d": 5.0, "e": 4.0},
                {"a": 5.0, "b": 4.0, "c": 1.0, "d": 3.0, "e": 2.0}]

    esperado = evaluate_stability(campanas, 3)
    r = stability_pvalue(campanas, 3, n_permutaciones=50, semilla=0)

    assert r.w_kendall == pytest.approx(esperado.w_kendall)
    assert r.jaccard_promedio == pytest.approx(esperado.jaccard_promedio)


class TestFidelityPvalue:
    @staticmethod
    def _aumentos(fase_mayor: str) -> dict[str, float]:
        return {f: (10.0 if f == fase_mayor else 1.0 + i) for i, f in enumerate(FASES)}

    def test_si_la_fase_alta_es_la_mas_danina_en_las_cinco_campanas_p_es_uno_sobre_5_a_la_5(self):
        aumentos = [self._aumentos("floracion")] * 5

        p = fidelity_pvalue(aumentos, ["floracion"] * 5)

        assert p == pytest.approx(1 / 5**5)

    def test_una_campana_en_primer_lugar_da_p_de_un_quinto(self):
        assert fidelity_pvalue([self._aumentos("floracion")], ["floracion"]) == pytest.approx(0.2)

    def test_rangos_intermedios_dan_un_p_cercano_a_la_mitad(self):
        # la fase alta (emergencia, 1.0) queda tercera de cinco en cada campaña
        aumentos = [{"emergencia": 3.0, "desarrollo_vegetativo": 5.0, "floracion": 4.0, "llenado_grano": 2.0, "madurez": 1.0}] * 5

        assert fidelity_pvalue(aumentos, ["emergencia"] * 5) > 0.4

    def test_los_empates_van_en_contra_de_la_fase_alta(self):
        aumentos = [{f: 1.0 for f in FASES}] * 5

        assert fidelity_pvalue(aumentos, ["floracion"] * 5) == pytest.approx(1.0)

    def test_exige_una_fase_alta_por_campana(self):
        with pytest.raises(ValueError, match="una fase alta por campaña"):
            fidelity_pvalue([self._aumentos("floracion")] * 2, ["floracion"])


class TestConcordancePvalue:
    def test_cinco_aciertos_en_el_horizonte_principal(self):
        p = concordance_pvalue(["floracion"] * 5, "principal", n_fases=5)

        assert p == pytest.approx(0.4**5)

    def test_cero_aciertos_dan_p_uno(self):
        assert concordance_pvalue(["madurez"] * 5, "principal", n_fases=5) == pytest.approx(1.0)

    def test_cuatro_de_cinco_en_el_horizonte_principal(self):
        p = concordance_pvalue(["floracion"] * 4 + ["madurez"], "principal", n_fases=5)

        assert p == pytest.approx(5 * 0.4**4 * 0.6 + 0.4**5)

    def test_el_horizonte_temprano_usa_p0_de_un_quinto(self):
        assert concordance_pvalue(["floracion"] * 5, "temprano", n_fases=5) == pytest.approx(0.2**5)

    def test_horizonte_desconocido(self):
        with pytest.raises(ValueError, match="horizonte"):
            concordance_pvalue(["floracion"], "tardio", n_fases=5)


class TestFamiliaSecundariaHe2:
    def test_holm_se_aplica_solo_a_estabilidad_y_fidelidad(self):
        r = he2_family_decision(p_estabilidad=0.01, p_fidelidad=0.03)

        assert set(r.p_ajustados) == {"estabilidad", "fidelidad"}
        assert r.p_ajustados["estabilidad"] == pytest.approx(0.02)  # 0.01 * 2
        assert r.p_ajustados["fidelidad"] == pytest.approx(0.03)  # max(0.02, 0.03 * 1)

    def test_rechaza_los_contrastes_cuyo_ajustado_no_supera_alfa(self):
        r = he2_family_decision(p_estabilidad=0.02, p_fidelidad=0.04)  # ajustados 0.04 y 0.04
        s = he2_family_decision(p_estabilidad=0.03, p_fidelidad=0.04)  # ajustados 0.06 y 0.06

        assert r.rechazos == {"estabilidad": True, "fidelidad": True}
        assert s.rechazos == {"estabilidad": False, "fidelidad": False}

    def test_los_dos_secundarios_se_cumplen_solo_si_ambos_se_rechazan(self):
        assert he2_family_decision(0.0004, 0.0003).secundarios_cumplidos is True
        assert he2_family_decision(0.0004, 0.2).secundarios_cumplidos is False
