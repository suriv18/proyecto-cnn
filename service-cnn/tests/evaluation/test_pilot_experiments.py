"""Pruebas de las reglas de decisión del piloto (tarea T7, experimentos P4, P5, P6, P7, P9).

Las reglas se fijan ANTES de ejecutar (plan del piloto, sección 4): cada función implementa
una regla preespecificada y los empates se resuelven siempre hacia lo más simple.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.fold_builder import build_folds
from src.evaluation.pilot_experiments import (
    baseline_forms,
    border_check,
    decide_baseline_form,
    decide_representation,
    max_epochs_from_best,
    seed_variability,
)


def _rmse(valores: dict[str, list[float]], campanas=(2019, 2020)) -> pd.DataFrame:
    return pd.DataFrame(
        [{"modelo": m, "campana_prueba": c, "rmse": v} for m, vs in valores.items() for c, v in zip(campanas, vs)]
    )


class TestDecideRepresentation:
    def test_elige_la_anomalia_si_reduce_claramente_el_error(self):
        nivel = _rmse({"a": [400.0, 410.0], "b": [380.0, 390.0]})
        anomalia = _rmse({"a": [210.0, 220.0], "b": [205.0, 215.0]})

        r = decide_representation(nivel, anomalia)

        assert r["decision"] == "anomalia"
        assert r["media_anomalia"] < r["media_nivel"]

    def test_elige_el_nivel_si_es_mejor(self):
        r = decide_representation(_rmse({"a": [200.0, 210.0]}), _rmse({"a": [300.0, 310.0]}))

        assert r["decision"] == "nivel"

    def test_un_empate_menor_al_dos_por_ciento_se_resuelve_a_favor_del_nivel(self):
        nivel = _rmse({"a": [200.0, 200.0]})
        anomalia = _rmse({"a": [197.0, 197.0]})  # 1,5 % mejor: dentro del empate

        assert decide_representation(nivel, anomalia, tolerancia_relativa=0.02)["decision"] == "nivel"

    def test_promedia_todos_los_modelos_y_pliegos_sin_favorecer_a_ninguno(self):
        nivel = _rmse({"a": [100.0, 100.0], "b": [300.0, 300.0]})
        anomalia = _rmse({"a": [150.0, 150.0], "b": [150.0, 150.0]})

        r = decide_representation(nivel, anomalia)

        assert r["media_nivel"] == 200.0 and r["media_anomalia"] == 150.0


class TestBaselineForms:
    @pytest.fixture(scope="class")
    def pliego(self):
        rng = np.random.default_rng(0)
        P = ["A", "B", "C"]
        obs = pd.DataFrame([(p, c) for c in range(2016, 2021) for p in P], columns=["provincia_id", "campana_id"])
        # tendencia lineal clara por campaña: la tendencia debería ganar con >= 4 puntos
        obs["rendimiento_kg_ha"] = 1000.0 + 50.0 * (obs.campana_id - 2016) + obs.provincia_id.map({"A": 0, "B": 200, "C": 400})
        X = rng.normal(size=(len(obs), 2, 2))
        Z = rng.normal(size=(len(obs), 1))
        return build_folds(obs, X, Z, 4, range(2016, 2021))[0]  # entrena 2016-2019 (4 puntos), prueba 2020

    def test_devuelve_las_tres_formas(self, pliego):
        t = baseline_forms(pliego)

        assert set(t["forma"]) == {"media", "lineal_min4", "media_ultimas3"}
        assert (t["campana_prueba"] == pliego.campana_prueba).all()

    def test_con_tendencia_lineal_clara_gana_la_lineal_con_cuatro_puntos(self, pliego):
        t = baseline_forms(pliego).set_index("forma")["rmse"]

        assert t["lineal_min4"] < t["media_ultimas3"] < t["media"]
        assert t["lineal_min4"] == pytest.approx(0.0, abs=1e-6)

    def test_la_media_es_la_media_historica_provincial(self, pliego):
        # media de 1000, 1050, 1100, 1150 = 1075; el valor real de 2020 es 1200 (+ nivel provincial)
        t = baseline_forms(pliego).set_index("forma")["rmse"]

        assert t["media"] == pytest.approx(125.0)


class TestDecideBaselineForm:
    def test_elige_la_de_menor_rmse_medio(self):
        t = pd.DataFrame({"forma": ["media", "lineal_min4"] * 2, "campana_prueba": [2019, 2019, 2020, 2020], "rmse": [100.0, 80.0, 100.0, 80.0]})

        assert decide_baseline_form(t)["decision"] == "lineal_min4"

    def test_un_empate_menor_al_dos_por_ciento_se_resuelve_a_favor_de_la_media(self):
        t = pd.DataFrame({"forma": ["media", "lineal_min4"], "campana_prueba": [2019, 2019], "rmse": [100.0, 99.0]})

        assert decide_baseline_form(t, tolerancia_relativa=0.02)["decision"] == "media"


class TestSeedVariability:
    def test_resume_las_semillas_de_los_modelos_estocasticos(self):
        filas = []
        for semilla, e in enumerate([10.0, 12.0, 11.0, 13.0, 9.0]):
            filas += [{"campana_prueba": 2020, "modelo": "xgboost", "semilla": semilla, "y_real": 0.0, "y_pred": e}] * 2
        df = pd.DataFrame(filas)

        r = seed_variability(df).iloc[0]

        assert r["n_semillas"] == 5 and r["mediana"] == 11.0
        assert r["minimo"] == 9.0 and r["maximo"] == 13.0
        assert r["error_estandar_mediana"] == pytest.approx(1.2533 * np.std([10, 12, 11, 13, 9], ddof=1) / np.sqrt(5), rel=1e-3)

    def test_ignora_los_modelos_con_una_sola_semilla(self):
        df = pd.DataFrame([{"campana_prueba": 2020, "modelo": "svr", "semilla": 0, "y_real": 0.0, "y_pred": 1.0}])

        assert seed_variability(df).empty


class TestMaxEpochsFromBest:
    def test_es_uno_coma_cinco_veces_el_percentil_95_redondeado_hacia_arriba(self):
        mejores = list(range(1, 21))  # percentil 95 = 19,05 -> 1,5 x 19,05 = 28,6 -> 29

        assert max_epochs_from_best(mejores) == 29

    def test_exige_al_menos_una_mejor_epoca(self):
        with pytest.raises(ValueError, match="mejor época"):
            max_epochs_from_best([])


class TestBorderCheck:
    ESPACIO = {"filtros": ["int", 5, 15], "dropout": ["float", 0.001, 0.01]}

    def _ensayos(self, filtros, dropouts, valores):
        return pd.DataFrame({"filtros": filtros, "dropout": dropouts, "valor": valores})

    def test_senala_el_borde_cuando_los_mejores_ensayos_se_agrupan_en_un_extremo(self):
        # 10 ensayos; los 2 mejores (20 %) tienen filtros = 15 (borde superior); dropout disperso
        e = self._ensayos([15, 15, 8, 9, 10, 11, 7, 6, 12, 13], [0.002, 0.009, 0.005, 0.004, 0.006, 0.003, 0.007, 0.008, 0.005, 0.006],
                          [1.0, 1.1, 5, 6, 7, 8, 9, 10, 11, 12])

        r = border_check(e, self.ESPACIO).set_index("parametro")

        assert r.loc["filtros", "lado"] == "superior"
        assert r.loc["dropout", "lado"] == "ninguno"

    def test_sin_agrupacion_en_los_bordes_no_senala_nada(self):
        e = self._ensayos(list(range(5, 15)), np.linspace(0.002, 0.009, 10).tolist(), [5, 4, 3, 2, 1, 0, 1, 2, 3, 4])  # los mejores ensayos están en el centro

        r = border_check(e, self.ESPACIO)

        assert (r["lado"] == "ninguno").all()
