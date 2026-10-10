"""Pruebas del ejecutor por pliego (tarea T4 del plan del piloto).

Garantías: presupuesto de ajuste igual para los 5 modelos (4.12.3), la verdad de la
campaña de prueba no interviene en el ajuste, el ajuste de hiperparámetros solo
usa pliegos internos (4.12.2), semillas solo en los modelos estocásticos (4.12.4)
y predicciones siempre reconstruidas en kg/ha (4.11).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.evaluation.experiment_runner import (
    MODELOS,
    benchmark_predictions,
    build_adapter,
    reconstruct_prediction,
    rmse_per_campaign,
    run_fold,
    tune_model,
)
from src.evaluation.fold_builder import build_folds, build_inner_folds

ESPACIOS = {
    "elastic_net": {"alpha": ["float", 0.01, 1.0], "l1_ratio": ["float", 0.1, 0.9]},
    "random_forest": {"n_estimators": ["int", 5, 15], "max_depth": ["int", 2, 4]},
    "xgboost": {"n_estimators": ["int", 5, 15], "max_depth": ["int", 2, 3], "learning_rate": ["float", 0.05, 0.3]},
    "svr": {"C": ["float", 0.1, 10.0], "epsilon": ["float", 0.01, 0.5]},
    "cnn1d": {"filtros_capa1": ["int", 3, 5], "filtros_capa2": ["int", 3, 5], "dropout": ["float", 0.001, 0.01], "tasa_aprendizaje": ["float", 0.001, 0.01]},
}
PROVINCIAS = [f"P{i}" for i in range(6)]
CAMPANAS = (2016, 2017, 2018, 2019, 2020)


def _datos(semilla=0):
    rng = np.random.default_rng(semilla)
    filas = [(p, c) for c in CAMPANAS for p in PROVINCIAS]
    obs = pd.DataFrame(filas, columns=["provincia_id", "campana_id"])
    nivel = {p: 800.0 + 120.0 * i for i, p in enumerate(PROVINCIAS)}
    X = rng.normal(5.0, 2.0, size=(len(obs), 6, 5))
    Z = rng.normal(0.0, 1.0, size=(len(obs), 2))
    señal = X[:, 4, :].mean(axis=1)  # el NDVI explica parte del rendimiento
    obs["rendimiento_kg_ha"] = obs.provincia_id.map(nivel) + 30.0 * señal + rng.normal(0, 15, len(obs))
    return obs, X, Z


@pytest.fixture(scope="module")
def pliego():
    obs, X, Z = _datos()
    return build_folds(obs, X, Z, ventana_inicial=4, campanas_permitidas=CAMPANAS)[0]  # entrena 2016-2019, prueba 2020


@pytest.fixture(scope="module")
def internos(pliego):
    return build_inner_folds(pliego, ventana_interna=2)


class TestReconstruccion:
    def test_nivel_no_cambia_la_prediccion(self, pliego):
        pred = np.ones(len(pliego.y_test))

        np.testing.assert_array_equal(reconstruct_prediction(pred, pliego, "nivel"), pred)

    def test_anomalia_se_suma_a_la_linea_base_provincial(self, pliego):
        pred = np.full(len(pliego.y_test), 25.0)

        np.testing.assert_allclose(reconstruct_prediction(pred, pliego, "anomalia"), 25.0 + pliego.baseline_test)

    def test_representacion_desconocida_falla(self, pliego):
        with pytest.raises(ValueError, match="representación"):
            reconstruct_prediction(np.zeros(len(pliego.y_test)), pliego, "otra")


class TestPresupuestoEquiparado:
    @pytest.mark.parametrize("modelo", MODELOS)
    def test_cada_modelo_ejecuta_exactamente_el_presupuesto(self, modelo, internos):
        resultado = tune_model(build_adapter(modelo, epocas_cnn=3), internos, ESPACIOS[modelo], n_trials=3, representacion="nivel")

        assert resultado.n_trials_ejecutados == 3

    def test_sin_pliegos_internos_falla_con_un_mensaje_claro(self):
        with pytest.raises(ValueError, match="pliegos internos"):
            tune_model(build_adapter("elastic_net"), [], ESPACIOS["elastic_net"], n_trials=2, representacion="nivel")


class TestSinFuga:
    def test_el_ajuste_de_hiperparametros_no_depende_de_la_verdad_de_la_campana_externa(self, pliego):
        a = tune_model(build_adapter("random_forest"), build_inner_folds(pliego, 2), ESPACIOS["random_forest"], 3, "nivel")
        alterado = _datos()[0]
        alterado.loc[alterado.campana_id == 2020, "rendimiento_kg_ha"] = 9e6
        _, X, Z = _datos()
        pliego_b = build_folds(alterado, X, Z, 4, CAMPANAS)[0]
        b = tune_model(build_adapter("random_forest"), build_inner_folds(pliego_b, 2), ESPACIOS["random_forest"], 3, "nivel")

        assert a.mejores_hiperparametros == b.mejores_hiperparametros

    def test_la_prediccion_no_usa_la_verdad_de_la_prueba(self, pliego):
        hp = {"alpha": 0.1, "l1_ratio": 0.5}
        a = build_adapter("elastic_net").ajustar_predecir(hp, pliego, semilla=0, representacion="nivel")
        roto = dataclasses.replace(pliego, y_test=pliego.y_test * 0 + 1e9)  # se rompe la verdad de la prueba
        b = build_adapter("elastic_net").ajustar_predecir(hp, roto, semilla=0, representacion="nivel")

        np.testing.assert_array_equal(a, b)


class TestSemillas:
    def test_los_modelos_deterministas_ignoran_la_semilla(self, pliego):
        hp = {"alpha": 0.1, "l1_ratio": 0.5}
        a = build_adapter("elastic_net").ajustar_predecir(hp, pliego, 0, "nivel")
        b = build_adapter("elastic_net").ajustar_predecir(hp, pliego, 7, "nivel")

        np.testing.assert_array_equal(a, b)

    def test_los_modelos_estocasticos_cambian_con_la_semilla(self, pliego):
        hp = {"n_estimators": 10, "max_depth": 3}
        a = build_adapter("random_forest").ajustar_predecir(hp, pliego, 0, "nivel")
        b = build_adapter("random_forest").ajustar_predecir(hp, pliego, 1, "nivel")

        assert not np.allclose(a, b)

    def test_la_misma_semilla_reproduce_la_cnn(self, pliego):
        hp = {"filtros_capa1": 4, "filtros_capa2": 4, "dropout": 0.005, "tasa_aprendizaje": 0.005}
        a = build_adapter("cnn1d", epocas_cnn=3).ajustar_predecir(hp, pliego, 3, "nivel")
        b = build_adapter("cnn1d", epocas_cnn=3).ajustar_predecir(hp, pliego, 3, "nivel")

        np.testing.assert_allclose(a, b)

    def test_clasificacion_de_modelos_estocasticos(self):
        assert {m for m in MODELOS if build_adapter(m).estocastico} == {"cnn1d", "random_forest", "xgboost"}


class TestPrediccionesEnKgHa:
    def test_las_predicciones_de_cada_modelo_estan_en_la_escala_del_rendimiento(self, pliego):
        hp = {"alpha": 0.01, "l1_ratio": 0.5}

        pred = build_adapter("elastic_net").ajustar_predecir(hp, pliego, 0, "nivel")

        assert pred.shape == pliego.y_test.shape
        assert abs(pred.mean() - pliego.y_test.mean()) < 0.5 * pliego.y_test.std() + 300  # kg/ha, no estandarizado


class TestBenchmarks:
    def test_devuelve_b1_b2_y_b3_para_cada_observacion_de_prueba(self, pliego):
        b = benchmark_predictions(pliego, indice_ndvi_max=4)

        assert set(b) == {"B1", "B2", "B3"}
        assert all(len(v) == len(pliego.y_test) for v in b.values())

    def test_b2_es_la_linea_base_provincial(self, pliego):
        np.testing.assert_array_equal(benchmark_predictions(pliego, 4)["B2"], pliego.baseline_test)

    def test_b1_es_la_media_historica_de_cada_provincia(self, pliego):
        b1 = benchmark_predictions(pliego, 4)["B1"]
        provincia = pliego.provincia_test[0]
        media = pliego.y_train[pliego.provincia_train == provincia].mean()

        assert b1[0] == pytest.approx(media)

    def test_b3_no_usa_la_verdad_de_la_prueba(self, pliego):
        a = benchmark_predictions(pliego, 4)["B3"].copy()
        b = benchmark_predictions(dataclasses.replace(pliego, y_test=pliego.y_test * 0 + 5e8), 4)["B3"]

        np.testing.assert_array_equal(a, b)


class TestRunFold:
    @pytest.fixture(scope="class")
    def resultado(self, pliego):
        return run_fold(
            pliego, ESPACIOS, n_trials=2, n_semillas=2, ventana_interna=2,
            representacion="nivel", indice_ndvi_max=4, epocas_cnn=3,
        )

    def test_incluye_los_cinco_modelos_y_los_tres_benchmarks(self, resultado):
        assert set(resultado.predicciones["modelo"]) == set(MODELOS) | {"B1", "B2", "B3"}

    def test_las_semillas_solo_se_repiten_en_los_modelos_estocasticos(self, resultado):
        semillas = resultado.predicciones.groupby("modelo")["semilla"].nunique()

        assert semillas["xgboost"] == semillas["random_forest"] == semillas["cnn1d"] == 2
        assert semillas["elastic_net"] == semillas["svr"] == semillas["B2"] == 1

    def test_cada_fila_tiene_la_verdad_y_la_prediccion_en_kg_ha(self, resultado, pliego):
        fila = resultado.predicciones.iloc[0]

        assert {"campana_prueba", "modelo", "semilla", "provincia_id", "y_real", "y_pred"} <= set(resultado.predicciones.columns)
        assert fila["campana_prueba"] == pliego.campana_prueba

    def test_registra_el_presupuesto_ejecutado_por_modelo(self, resultado):
        assert set(resultado.ajuste["n_trials"]) == {2}
        assert set(resultado.ajuste["modelo"]) == set(MODELOS)


class TestRmsePorCampana:
    def test_usa_la_mediana_de_las_semillas_dentro_de_cada_campana(self):
        df = pd.DataFrame({
            "campana_prueba": [2020] * 6, "modelo": ["m"] * 6, "semilla": [0, 0, 1, 1, 2, 2],
            "y_real": [0.0] * 6, "y_pred": [1.0, 1.0, 2.0, 2.0, 9.0, 9.0],
        })

        r = rmse_per_campaign(df)

        assert r.loc[0, "rmse"] == 2.0  # mediana de los RMSE 1, 2 y 9

    def test_calcula_el_rmse_por_modelo_y_campana(self):
        df = pd.DataFrame({
            "campana_prueba": [2019, 2019, 2020, 2020], "modelo": ["a"] * 4, "semilla": [0] * 4,
            "y_real": [0.0, 0.0, 0.0, 0.0], "y_pred": [3.0, 3.0, 4.0, 4.0],
        })

        r = rmse_per_campaign(df).set_index("campana_prueba")

        assert r.loc[2019, "rmse"] == 3.0 and r.loc[2020, "rmse"] == 4.0
