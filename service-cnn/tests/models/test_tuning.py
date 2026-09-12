"""Pruebas del módulo de tuning con Optuna (sección 4.12.3).

Los 5 modelos ajustables (CNN-1D, Elastic-Net, Random Forest, XGBoost, SVR)
reciben el MISMO número máximo de evaluaciones de hiperparámetros —
"presupuesto de ajuste equiparado". Los espacios de búsqueda se declaran antes
de la evaluación (preespecificación H2) y no se optimizan los benchmarks
(B1, B2, B3).
"""
import numpy as np
import pytest

from src.models.classic_models import MODELOS_DISPONIBLES
from src.models.tuning import (
    PRESUPUESTO_TRIALS_POR_MODELO,
    sugerir_hiperparametro,
    tune_classic_model,
)


@pytest.fixture
def datos_entrenamiento_validacion():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(40, 5))
    y_train = x_train[:, 0] * 3.0 + rng.normal(scale=0.2, size=40)
    x_val = rng.normal(size=(15, 5))
    y_val = x_val[:, 0] * 3.0 + rng.normal(scale=0.2, size=15)
    return x_train, y_train, x_val, y_val


class TestTuneClassicModel:
    def test_devuelve_hiperparametros_dentro_del_espacio_declarado(
        self, datos_entrenamiento_validacion
    ):
        x_train, y_train, x_val, y_val = datos_entrenamiento_validacion
        espacio_busqueda = {
            "alpha": ("float", 0.01, 10.0),
            "l1_ratio": ("float", 0.0, 1.0),
        }
        resultado = tune_classic_model(
            "elastic_net",
            espacio_busqueda,
            x_train,
            y_train,
            x_val,
            y_val,
            n_trials=5,
            semilla=0,
        )
        assert 0.01 <= resultado.mejores_hiperparametros["alpha"] <= 10.0
        assert 0.0 <= resultado.mejores_hiperparametros["l1_ratio"] <= 1.0

    def test_respeta_el_numero_de_trials_solicitado(
        self, datos_entrenamiento_validacion
    ):
        x_train, y_train, x_val, y_val = datos_entrenamiento_validacion
        espacio_busqueda = {"alpha": ("float", 0.01, 10.0)}
        resultado = tune_classic_model(
            "elastic_net",
            espacio_busqueda,
            x_train,
            y_train,
            x_val,
            y_val,
            n_trials=7,
            semilla=0,
        )
        assert resultado.n_trials_ejecutados == 7

    def test_devuelve_la_perdida_de_validacion_del_mejor_trial(
        self, datos_entrenamiento_validacion
    ):
        x_train, y_train, x_val, y_val = datos_entrenamiento_validacion
        espacio_busqueda = {"alpha": ("float", 0.01, 10.0)}
        resultado = tune_classic_model(
            "elastic_net",
            espacio_busqueda,
            x_train,
            y_train,
            x_val,
            y_val,
            n_trials=5,
            semilla=0,
        )
        assert resultado.mejor_rmse_validacion >= 0.0

    def test_espacio_de_tipo_int_genera_enteros(self, datos_entrenamiento_validacion):
        x_train, y_train, x_val, y_val = datos_entrenamiento_validacion
        espacio_busqueda = {
            "n_estimators": ("int", 5, 50),
            "max_depth": ("int", 2, 8),
        }
        resultado = tune_classic_model(
            "random_forest",
            espacio_busqueda,
            x_train,
            y_train,
            x_val,
            y_val,
            n_trials=5,
            semilla=0,
        )
        assert isinstance(resultado.mejores_hiperparametros["n_estimators"], int)
        assert isinstance(resultado.mejores_hiperparametros["max_depth"], int)

    def test_reproducible_con_la_misma_semilla(self, datos_entrenamiento_validacion):
        x_train, y_train, x_val, y_val = datos_entrenamiento_validacion
        espacio_busqueda = {"alpha": ("float", 0.01, 10.0)}
        resultado_a = tune_classic_model(
            "elastic_net", espacio_busqueda, x_train, y_train, x_val, y_val,
            n_trials=5, semilla=7,
        )
        resultado_b = tune_classic_model(
            "elastic_net", espacio_busqueda, x_train, y_train, x_val, y_val,
            n_trials=5, semilla=7,
        )
        assert (
            resultado_a.mejores_hiperparametros == resultado_b.mejores_hiperparametros
        )


class TestPresupuestoEquiparado:
    def test_presupuesto_es_el_mismo_para_los_5_modelos_ajustables(self):
        """Sección 4.12.3: los 5 modelos ajustables reciben el mismo número
        máximo de evaluaciones de hiperparámetros — la CNN-1D se incluye aquí
        aunque su tuning se ejecute con una función distinta (tune_cnn1d)."""
        modelos_ajustables = set(MODELOS_DISPONIBLES) | {"cnn1d"}
        assert set(PRESUPUESTO_TRIALS_POR_MODELO.keys()) == modelos_ajustables
        valores_presupuesto = set(PRESUPUESTO_TRIALS_POR_MODELO.values())
        assert len(valores_presupuesto) == 1  # un único valor para los 5


class TestSugerirHiperparametro:
    def test_falla_con_tipo_no_soportado(self):
        import optuna

        estudio = optuna.create_study()

        def objetivo(trial):
            sugerir_hiperparametro(trial, "x", ("categorico", 0, 1))
            return 0.0

        with pytest.raises(ValueError, match="no soportado"):
            estudio.optimize(objetivo, n_trials=1)
