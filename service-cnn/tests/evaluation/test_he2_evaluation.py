"""Evaluación conjunta de HE2: estabilidad, fidelidad y concordancia con Holm (3.3.5)."""
from __future__ import annotations

import pytest

from src.evaluation.he2_evaluation import evaluate_he2

FASES = ["emergencia", "desarrollo_vegetativo", "floracion", "llenado_grano", "madurez"]
CAMPANAS = [2021, 2022, 2023, 2024, 2025]


def _contribucion(mayor: str = "floracion") -> dict[str, float]:
    """Misma jerarquía en todas las campañas: la fase `mayor` primero, luego el orden de FASES."""
    resto = [f for f in FASES if f != mayor]
    return {f: float(5 - i) for i, f in enumerate([mayor] + resto)}


def _aumentos(mayor: str = "floracion") -> dict[str, float]:
    return {f: (10.0 if f == mayor else 1.0 + i) for i, f in enumerate(FASES)}


def _todo_concordante(mayor: str = "floracion"):
    return {c: _contribucion(mayor) for c in CAMPANAS}, {c: _aumentos(mayor) for c in CAMPANAS}


def _evaluar(contribuciones, aumentos, **extra):
    base = dict(horizonte="principal", n_fases_principales=3, n_permutaciones=3000, semilla=0)
    base.update(extra)
    return evaluate_he2(contribuciones, aumentos, **base)


class TestHe2Respaldada:
    def test_se_respalda_cuando_los_tres_criterios_se_cumplen_en_las_cinco_campanas(self):
        contribuciones, aumentos = _todo_concordante()

        r = _evaluar(contribuciones, aumentos)

        assert r.he2_respaldada is True
        assert all(r.rechazos.values())
        assert r.campanas == CAMPANAS and r.n_campanas == 5

    def test_la_fase_maxima_se_identifica_por_campana(self):
        contribuciones, aumentos = _todo_concordante()

        r = _evaluar(contribuciones, aumentos)

        assert r.fases_maximas == {c: "floracion" for c in CAMPANAS}

    def test_no_se_respalda_si_la_fase_maxima_cae_fuera_del_referente_agronomico(self):
        contribuciones, aumentos = _todo_concordante(mayor="emergencia")

        r = _evaluar(contribuciones, aumentos)

        assert r.he2_respaldada is False
        assert r.rechazos["concordancia"] is False
        assert r.rechazos["estabilidad"] is True  # estable pero agronómicamente discordante

    def test_no_se_respalda_si_una_sola_campana_discrepa_en_la_concordancia(self):
        contribuciones, aumentos = _todo_concordante()
        contribuciones[2023] = _contribucion("madurez")
        aumentos[2023] = _aumentos("madurez")

        r = _evaluar(contribuciones, aumentos)

        assert r.p_valores["concordancia"] == pytest.approx(5 * 0.4**4 * 0.6 + 0.4**5)
        assert r.he2_respaldada is False

    def test_no_se_respalda_con_ordenes_inestables_aunque_la_fase_maxima_coincida(self):
        contribuciones = {}
        for i, c in enumerate(CAMPANAS):
            resto = [f for f in FASES if f != "floracion"]
            resto = resto[i % 4:] + resto[: i % 4]  # el resto rota; la mayor sigue siendo floración
            contribuciones[c] = {f: float(5 - j) for j, f in enumerate(["floracion"] + resto)}
        aumentos = {c: _aumentos() for c in CAMPANAS}

        r = _evaluar(contribuciones, aumentos)

        assert r.he2_respaldada is False
        assert r.umbrales_de_estabilidad_cumplidos is False


class TestUmbralesDeMagnitud:
    def test_se_informan_w_y_jaccard_junto_a_los_valores_p(self):
        contribuciones, aumentos = _todo_concordante()

        r = _evaluar(contribuciones, aumentos)

        assert r.w_kendall == pytest.approx(1.0)
        assert r.jaccard_promedio == pytest.approx(1.0)
        assert r.umbrales_de_estabilidad_cumplidos is True
        assert set(r.p_valores) == {"estabilidad", "fidelidad", "concordancia"}


class TestValidaciones:
    def test_las_campanas_de_contribucion_y_de_oclusion_deben_coincidir(self):
        contribuciones, aumentos = _todo_concordante()
        del aumentos[2025]

        with pytest.raises(ValueError, match="mismas campañas"):
            _evaluar(contribuciones, aumentos)

    def test_las_fases_deben_ser_las_mismas_en_todas_las_campanas(self):
        contribuciones, aumentos = _todo_concordante()
        contribuciones[2022] = {"otra": 1.0, **{f: 1.0 for f in FASES[:4]}}

        with pytest.raises(ValueError, match="mismas fases"):
            _evaluar(contribuciones, aumentos)

    def test_exige_al_menos_dos_campanas(self):
        with pytest.raises(ValueError, match="al menos 2"):
            _evaluar({2021: _contribucion()}, {2021: _aumentos()})
