"""Pruebas del aplanado del tensor V×T a vector de features tabular.

Referencia: sección 4.10.3 (estructura del tensor) y Tabla 8 (los modelos
clásicos —Elastic-Net, Random Forest, XGBoost, SVR— no procesan la estructura
secuencial nativamente, a diferencia de la CNN-1D; deben recibir el MISMO
conjunto de información, aplanado a un vector, para que la comparación sea
metodológicamente válida, sección 4.3, "mismas condiciones").
"""
import numpy as np
import pytest

from src.features.flatten import flatten_tensor, flattened_feature_names


class TestFlattenTensor:
    def test_aplana_tensor_v_por_t_a_vector_de_longitud_v_veces_t(self):
        tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # V=2, T=3
        covariables = np.array([100.0, 0.5])
        vector = flatten_tensor(tensor, covariables)
        assert vector.shape == (2 * 3 + 2,)

    def test_incluye_las_covariables_al_final(self):
        tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
        covariables = np.array([99.0, 0.8])
        vector = flatten_tensor(tensor, covariables)
        assert list(vector[-2:]) == [99.0, 0.8]

    def test_orden_de_aplanado_es_por_variable_luego_por_paso_temporal(self):
        tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        covariables = np.array([])
        vector = flatten_tensor(tensor, covariables)
        # fila 0 (variable 0) completa, luego fila 1 (variable 1) completa
        assert list(vector) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_preserva_nan_del_tensor(self):
        tensor = np.array([[1.0, np.nan], [3.0, 4.0]])
        covariables = np.array([])
        vector = flatten_tensor(tensor, covariables)
        assert np.isnan(vector[1])


class TestFlattenedFeatureNames:
    def test_genera_un_nombre_por_celda_variable_fase_mas_covariables(self):
        nombres = flattened_feature_names(
            orden_variables=["precipitacion", "ndvi"],
            orden_fases=["emergencia", "floracion"],
            nombres_covariables=["altitud_media_agricola", "prop_superficie_quinua"],
        )
        assert nombres == [
            "precipitacion__emergencia",
            "precipitacion__floracion",
            "ndvi__emergencia",
            "ndvi__floracion",
            "altitud_media_agricola",
            "prop_superficie_quinua",
        ]

    def test_longitud_coincide_con_flatten_tensor(self):
        tensor = np.zeros((3, 4))
        covariables = np.zeros(2)
        vector = flatten_tensor(tensor, covariables)
        nombres = flattened_feature_names(
            orden_variables=["v1", "v2", "v3"],
            orden_fases=["f1", "f2", "f3", "f4"],
            nombres_covariables=["c1", "c2"],
        )
        assert len(nombres) == len(vector)
