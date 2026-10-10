"""Pruebas de la prueba de fuga por permutación (tarea T8, experimento P1 del piloto).

Se permutan las etiquetas de ENTRENAMIENTO y se evalúa en la campaña de prueba: sin
relación entre entradas y etiquetas el R² fuera de muestra debe quedar en torno a 0 o
negativo. Si un modelo sigue prediciendo bien, la verdad de la prueba le está llegando
por otro camino (p. ej. un ajuste o un preprocesamiento hechos con filas de prueba).
Limitación declarada: no detecta una fuga entre entradas que sea coherente con las
etiquetas de entrenamiento; por eso lleva un control positivo.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.evaluation.fold_builder import build_folds
from src.evaluation.leak_test import permuted_training_fold, run_permutation_test

PROVINCIAS = [f"P{i}" for i in range(8)]
CAMPANAS = (2016, 2017, 2018, 2019, 2020)
HP = {
    "elastic_net": {"alpha": 0.05, "l1_ratio": 0.5},
    "random_forest": {"n_estimators": 20, "max_depth": 4},
    "xgboost": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
    "svr": {"C": 1.0, "epsilon": 0.1},
}


@pytest.fixture(scope="module")
def pliego():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame([(p, c) for c in CAMPANAS for p in PROVINCIAS], columns=["provincia_id", "campana_id"])
    X = rng.normal(5.0, 2.0, size=(len(obs), 6, 5))
    Z = rng.normal(0.0, 1.0, size=(len(obs), 2))
    # el rendimiento depende con fuerza de las entradas: hay señal real que la permutación debe destruir
    obs["rendimiento_kg_ha"] = 1000.0 + 120.0 * X[:, 4, :].mean(axis=1) + 80.0 * Z[:, 0] + rng.normal(0, 10, len(obs))
    return build_folds(obs, X, Z, ventana_inicial=4, campanas_permitidas=CAMPANAS)[0]


class TestPermutedTrainingFold:
    def test_conserva_el_conjunto_de_valores_y_no_toca_la_prueba(self, pliego):
        p = permuted_training_fold(pliego, semilla=0)

        np.testing.assert_array_equal(np.sort(p.y_train), np.sort(pliego.y_train))
        assert not np.array_equal(p.y_train, pliego.y_train)
        np.testing.assert_array_equal(p.y_test, pliego.y_test)
        np.testing.assert_array_equal(p.X_train, pliego.X_train)
        np.testing.assert_array_equal(p.X_test, pliego.X_test)

    def test_es_reproducible_con_la_misma_semilla_y_distinta_con_otra(self, pliego):
        a, b, c = (permuted_training_fold(pliego, s).y_train for s in (1, 1, 2))

        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)


class _AdaptadorQueEspia:
    """Control positivo: un 'modelo' que ve la verdad de la prueba (la fuga que se busca detectar)."""

    nombre = "espia"
    estocastico = False

    def ajustar_predecir(self, hp, fold, semilla, representacion):
        return fold.y_test + 1.0


class TestRunPermutationTest:
    def test_sin_permutar_los_modelos_si_aprenden_y_con_permutacion_no(self, pliego):
        # sanidad: el diseño tiene señal real (si no la tuviera, la prueba no probaría nada)
        from src.evaluation.experiment_runner import build_adapter
        from src.evaluation.metrics import r2_fuera_de_muestra

        real = build_adapter("elastic_net").ajustar_predecir(HP["elastic_net"], pliego, 0, "nivel")

        assert r2_fuera_de_muestra(pliego.y_test, real) > 0.5

    def test_los_modelos_honestos_no_superan_el_umbral_con_etiquetas_permutadas(self, pliego):
        tabla = run_permutation_test(pliego, HP, modelos=tuple(HP), n_permutaciones=10, semilla=0, umbral_r2=0.05)

        assert set(tabla["modelo"]) == set(HP)
        assert tabla["pasa"].all()
        assert (tabla["r2_mediano"] <= 0.05).all()

    def test_el_criterio_es_la_mediana_y_no_el_maximo_que_da_falsas_alarmas_con_pocas_filas(self, pliego):
        """Regresión real (2026-10-10): con 8 observaciones de prueba el R² de un modelo con
        etiquetas permutadas oscila entre -2,4 y +0,48; el MÁXIMO de varias permutaciones
        superaba cualquier umbral fijo por puro azar. La mediana bajo la hipótesis nula es <= 0."""
        tabla = run_permutation_test(pliego, HP, ("random_forest", "xgboost"), n_permutaciones=12, semilla=0)

        assert (tabla["r2_maximo"] > 0.05).any()  # el azar sí produce máximos altos...
        assert tabla["pasa"].all()  # ...y aun así ningún modelo honesto es señalado

    def test_detecta_un_modelo_que_ve_la_verdad_de_la_prueba(self, pliego):
        tabla = run_permutation_test(
            pliego, HP, modelos=("espia",), n_permutaciones=2, semilla=0, umbral_r2=0.05,
            adaptadores={"espia": _AdaptadorQueEspia()},
        )

        assert not tabla["pasa"].iloc[0]
        assert tabla["r2_mediano"].iloc[0] > 0.9

    def test_informa_tambien_el_maximo_y_la_lista_de_permutaciones(self, pliego):
        fila = run_permutation_test(pliego, HP, ("elastic_net",), 5, 0, 0.05).iloc[0]

        assert len(fila["r2_por_permutacion"]) == 5
        assert fila["r2_maximo"] >= fila["r2_mediano"]

    def test_no_modifica_el_pliego_original(self, pliego):
        antes = pliego.y_train.copy()

        run_permutation_test(pliego, HP, ("elastic_net",), 2, 0, 0.05)

        np.testing.assert_array_equal(pliego.y_train, antes)
