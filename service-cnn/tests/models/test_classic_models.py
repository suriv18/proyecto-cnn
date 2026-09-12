"""Pruebas de los 4 modelos clásicos comparadores (Tabla 8, sección 4.12.3).

Elastic-Net, Random Forest, XGBoost, SVR: todos reciben entrada tabular (el
tensor V×T ya aplanado por src/features/flatten.py) y comparten una interfaz
uniforme para que el módulo de tuning (Optuna) los trate de forma equiparada.
Random Forest y XGBoost son estocásticos (10 semillas, sección 4.12.4);
Elastic-Net y SVR se ejecutan deterministamente cuando la implementación lo
permite.
"""
import numpy as np
import pytest

from src.models.classic_models import (
    MODELOS_DISPONIBLES,
    build_classic_model,
)


@pytest.fixture
def datos_sinteticos():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 6))
    y = x[:, 0] * 2.0 + x[:, 1] * -1.0 + rng.normal(scale=0.1, size=50)
    return x, y


class TestBuildClassicModel:
    @pytest.mark.parametrize("nombre_modelo", MODELOS_DISPONIBLES)
    def test_todos_los_modelos_declarados_se_pueden_construir(self, nombre_modelo):
        modelo = build_classic_model(nombre_modelo, hiperparametros={}, semilla=0)
        assert modelo is not None

    @pytest.mark.parametrize("nombre_modelo", MODELOS_DISPONIBLES)
    def test_todos_los_modelos_ajustan_y_predicen(
        self, nombre_modelo, datos_sinteticos
    ):
        x, y = datos_sinteticos
        modelo = build_classic_model(nombre_modelo, hiperparametros={}, semilla=0)
        modelo.fit(x, y)
        predicciones = modelo.predict(x)
        assert predicciones.shape == (50,)

    def test_modelo_desconocido_falla_explicitamente(self):
        with pytest.raises(ValueError, match="modelo_inexistente"):
            build_classic_model("modelo_inexistente", hiperparametros={}, semilla=0)

    def test_lista_de_modelos_disponibles_coincide_con_la_tabla_8(self):
        """Tabla 8 de la tesis: Elastic-Net, Random Forest, XGBoost, SVR son
        los 4 comparadores clásicos (la CNN-1D es el artefacto propuesto, no
        un "modelo clásico")."""
        assert set(MODELOS_DISPONIBLES) == {
            "elastic_net",
            "random_forest",
            "xgboost",
            "svr",
        }


class TestReproducibilidadEstocastica:
    def test_random_forest_es_reproducible_con_la_misma_semilla(
        self, datos_sinteticos
    ):
        x, y = datos_sinteticos
        modelo_a = build_classic_model("random_forest", hiperparametros={}, semilla=42)
        modelo_a.fit(x, y)
        modelo_b = build_classic_model("random_forest", hiperparametros={}, semilla=42)
        modelo_b.fit(x, y)
        np.testing.assert_array_equal(modelo_a.predict(x), modelo_b.predict(x))

    def test_random_forest_semillas_distintas_pueden_diferir(self, datos_sinteticos):
        x, y = datos_sinteticos
        modelo_a = build_classic_model("random_forest", hiperparametros={}, semilla=1)
        modelo_a.fit(x, y)
        modelo_b = build_classic_model("random_forest", hiperparametros={}, semilla=2)
        modelo_b.fit(x, y)
        assert not np.array_equal(modelo_a.predict(x), modelo_b.predict(x))

    def test_xgboost_es_reproducible_con_la_misma_semilla(self, datos_sinteticos):
        x, y = datos_sinteticos
        modelo_a = build_classic_model("xgboost", hiperparametros={}, semilla=42)
        modelo_a.fit(x, y)
        modelo_b = build_classic_model("xgboost", hiperparametros={}, semilla=42)
        modelo_b.fit(x, y)
        np.testing.assert_allclose(modelo_a.predict(x), modelo_b.predict(x))

    def test_elastic_net_es_determinista_sin_necesidad_de_semilla(
        self, datos_sinteticos
    ):
        """Sección 4.12.4: Elastic-Net se ejecuta deterministamente."""
        x, y = datos_sinteticos
        modelo_a = build_classic_model("elastic_net", hiperparametros={}, semilla=1)
        modelo_a.fit(x, y)
        modelo_b = build_classic_model("elastic_net", hiperparametros={}, semilla=2)
        modelo_b.fit(x, y)
        np.testing.assert_allclose(modelo_a.predict(x), modelo_b.predict(x))


class TestHiperparametros:
    def test_random_forest_acepta_hiperparametros_configurables(
        self, datos_sinteticos
    ):
        x, y = datos_sinteticos
        modelo = build_classic_model(
            "random_forest",
            hiperparametros={"n_estimators": 10, "max_depth": 3},
            semilla=0,
        )
        modelo.fit(x, y)
        assert modelo.modelo_subyacente.n_estimators == 10
        assert modelo.modelo_subyacente.max_depth == 3

    def test_elastic_net_acepta_alpha_y_l1_ratio(self, datos_sinteticos):
        x, y = datos_sinteticos
        modelo = build_classic_model(
            "elastic_net", hiperparametros={"alpha": 0.5, "l1_ratio": 0.3}, semilla=0
        )
        modelo.fit(x, y)
        assert modelo.modelo_subyacente.alpha == 0.5
        assert modelo.modelo_subyacente.l1_ratio == 0.3
