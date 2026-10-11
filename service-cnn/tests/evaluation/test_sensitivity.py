"""Análisis de sensibilidad sobre las campañas de entrenamiento (4.14), sin tocar campañas externas.

Variantes sobre el mismo conjunto procesado: sin provincias de transición, solo tensores completos
(sin celdas imputadas) y horizonte secundario al cierre de la floración (solo las tres primeras
fases). Además, el error por estrato de participación de quinua.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.sensitivity import (
    VARIANTES,
    apply_variant,
    error_by_stratum,
    run_sensitivity,
    summarize_sensitivity,
)

FASES = ["emergencia", "desarrollo_vegetativo", "floracion", "llenado_grano", "madurez"]
ESPACIOS = {
    "elastic_net": {"alpha": ["float", 0.01, 1.0], "l1_ratio": ["float", 0.1, 0.9]},
    "random_forest": {"n_estimators": ["int", 5, 10], "max_depth": ["int", 2, 3]},
    "xgboost": {"n_estimators": ["int", 5, 10], "max_depth": ["int", 2, 3], "learning_rate": ["float", 0.05, 0.3]},
    "svr": {"C": ["float", 0.1, 10.0], "epsilon": ["float", 0.01, 0.5]},
    "cnn1d": {"filtros_capa1": ["int", 3, 4], "filtros_capa2": ["int", 3, 4], "dropout": ["float", 0.0, 0.1], "tasa_aprendizaje": ["float", 0.001, 0.01]},
}


@pytest.fixture(scope="module")
def datos():
    rng = np.random.default_rng(0)
    provincias = [f"P{i}" for i in range(10)]
    obs = pd.DataFrame([(p, c) for c in range(2016, 2021) for p in provincias], columns=["provincia_id", "campana_id"])
    obs["clasificacion"] = np.where(obs.provincia_id.isin(["P0", "P1"]), "transicion", "altoandina")
    obs["prop_quinua"] = np.where(obs.provincia_id.isin(["P0", "P1", "P2"]), 0.01, np.where(obs.provincia_id.isin(["P3", "P4", "P5", "P6"]), 0.05, 0.2))
    X = rng.normal(5.0, 2.0, size=(len(obs), 6, 5))
    Z = rng.normal(size=(len(obs), 2))
    imputado = np.zeros(X.shape, dtype=bool)
    imputado[::7, 0, 1] = True  # algunas observaciones con una celda imputada
    nivel = {p: 800.0 + 100.0 * i for i, p in enumerate(provincias)}
    obs["rendimiento_kg_ha"] = obs.provincia_id.map(nivel) + 15.0 * X[:, 4, :].mean(axis=1) + 10.0 * (obs.campana_id - 2016) + rng.normal(0, 10, len(obs))
    return obs, X, Z, imputado


class TestApplyVariant:
    def test_base_no_cambia_nada(self, datos):
        obs, X, Z, imp = datos

        o, x, z = apply_variant("base", obs, X, Z, imp)

        assert len(o) == len(obs) and x.shape == X.shape and z.shape == Z.shape

    def test_sin_transicion_quita_esas_provincias_y_mantiene_las_filas_alineadas(self, datos):
        obs, X, Z, imp = datos

        o, x, z = apply_variant("sin_transicion", obs, X, Z, imp)

        assert "transicion" not in set(o["clasificacion"])
        assert len(o) == len(x) == len(z) == (obs["clasificacion"] != "transicion").sum()
        np.testing.assert_array_equal(x[0], X[obs.index[obs["clasificacion"] != "transicion"][0]])

    def test_tensor_completo_quita_las_observaciones_con_celdas_imputadas(self, datos):
        obs, X, Z, imp = datos

        o, x, z = apply_variant("tensor_completo", obs, X, Z, imp)

        assert len(o) == (~imp.any(axis=(1, 2))).sum() < len(obs)

    def test_horizonte_temprano_conserva_solo_las_tres_primeras_fases(self, datos):
        obs, X, Z, imp = datos

        o, x, z = apply_variant("horizonte_temprano", obs, X, Z, imp)

        assert x.shape == (len(obs), 6, 3)
        np.testing.assert_array_equal(x, X[:, :, :3])

    def test_variante_desconocida(self, datos):
        obs, X, Z, imp = datos

        with pytest.raises(ValueError, match="variante"):
            apply_variant("inventada", obs, X, Z, imp)

    def test_las_variantes_declaradas(self):
        assert set(VARIANTES) == {"base", "sin_transicion", "tensor_completo", "horizonte_temprano"}


class TestRunSensitivity:
    @pytest.fixture(scope="module")
    def tabla(self, datos):
        obs, X, Z, imp = datos
        o, x, z = apply_variant("base", obs, X, Z, imp)
        return run_sensitivity(o, x, z, ESPACIOS, permitidas=range(2016, 2021), ventana_piloto=3, ventana_interna=2,
                               n_trials=2, n_semillas=2, epocas_cnn=3, indice_ndvi_max=4, semilla=0)

    def test_devuelve_el_rmse_por_modelo_y_campana_de_prueba(self, tabla):
        assert {"modelo", "campana_prueba", "rmse"} <= set(tabla.columns)
        assert set(tabla["campana_prueba"]) == {2019, 2020}
        assert {"cnn1d", "xgboost", "B2", "B4"} <= set(tabla["modelo"])

    def test_no_usa_campanas_fuera_de_las_permitidas(self, datos):
        obs, X, Z, imp = datos
        o, x, z = apply_variant("base", obs, X, Z, imp)

        with pytest.raises(Exception, match="2020|campañ"):
            run_sensitivity(o, x, z, ESPACIOS, permitidas=range(2016, 2020), ventana_piloto=3, ventana_interna=2,
                            n_trials=2, n_semillas=2, epocas_cnn=3, indice_ndvi_max=4, semilla=0)


class TestResumen:
    def test_compara_cada_variante_con_b2_y_b4(self):
        base = pd.DataFrame({"modelo": ["cnn1d", "B2", "B4"], "campana_prueba": [2020] * 3, "rmse": [200.0, 190.0, 170.0]})
        otra = pd.DataFrame({"modelo": ["cnn1d", "B2", "B4"], "campana_prueba": [2020] * 3, "rmse": [180.0, 190.0, 170.0]})

        r = summarize_sensitivity({"base": base, "sin_transicion": otra})

        fila = r[(r.variante == "sin_transicion") & (r.modelo == "cnn1d")].iloc[0]
        assert fila["q_frente_a_b2"] == pytest.approx(180 / 190 - 1)
        assert fila["q_frente_a_b4"] == pytest.approx(180 / 170 - 1)
        assert fila["cambio_vs_base_pct"] == pytest.approx((180 / 200 - 1) * 100)


class TestErrorPorEstrato:
    def test_agrupa_por_terciles_de_participacion_de_quinua(self, datos):
        obs, *_ = datos
        pred = obs[["provincia_id", "campana_id"]].rename(columns={"campana_id": "campana_prueba"}).assign(modelo="cnn1d", semilla=0, y_real=100.0, y_pred=100.0 + 5.0)

        r = error_by_stratum(pred, obs, columna="prop_quinua", n_estratos=3)

        assert set(r["estrato"]) == {"bajo", "medio", "alto"}
        assert r["rmse"].tolist() == pytest.approx([5.0, 5.0, 5.0])
        assert r["n"].sum() == len(obs)
