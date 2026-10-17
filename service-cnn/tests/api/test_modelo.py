"""Predicción sobre un tensor editado con la CNN-1D serializada (POST /predicciones/inferir con `tensor`)."""
from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.models.inferencia import ModeloCNN1D
from tests.api.conftest import construir_raiz
from tests.api.test_contrato import _validar
from tests.helpers_modelo import K, T, V, crear_artefacto

RUTA = "/api/v1/predicciones/inferir"
TENSOR = {"valores": [[10.0] * T] * V, "altitud_media_m": 3800.0, "prop_quinua": 0.1}


@pytest.fixture(scope="module")
def raiz_con_modelo(tmp_path_factory) -> Path:
    raiz = construir_raiz(tmp_path_factory.mktemp("con_modelo"))
    crear_artefacto(raiz / "models" / "cnn1d_piloto", vinculante=False)
    return raiz


@pytest.fixture()
def cliente_modelo(raiz_con_modelo) -> TestClient:
    return TestClient(create_app(raiz_con_modelo))


def _pedir(cliente, **extra):
    return cliente.post(RUTA, json={"provincia_id": "AAA", "campana": 2020, "tensor": TENSOR, **extra})


def test_el_rendimiento_es_b2_mas_la_mediana_de_las_anomalias(cliente_modelo, raiz_con_modelo, contrato):
    r = _pedir(cliente_modelo)
    _validar(contrato, contrato["paths"]["/predicciones/inferir"]["post"], r)
    assert r.status_code == 200
    modelo = ModeloCNN1D.cargar(raiz_con_modelo / "models" / "cnn1d_piloto")
    anomalias = modelo.predecir_anomalia(np.array([TENSOR["valores"]]), np.array([[3800.0, 0.1]]))[:, 0]
    cuerpo = r.json()
    assert cuerpo["y_b2"] == pytest.approx(940.0)  # 900 + 10 por año desde 2016
    assert cuerpo["anomalia"] == pytest.approx(statistics.median(anomalias))
    assert cuerpo["y_pred"] == pytest.approx(940.0 + statistics.median(anomalias))


def test_no_inventa_el_observado_y_declara_el_estado_del_modelo(cliente_modelo):
    cuerpo = _pedir(cliente_modelo).json()
    assert cuerpo["y_real"] is None
    assert cuerpo["modelo_version"] == "cnn1d-abababab"
    assert cuerpo["vinculante"] is False  # viene de la tarjeta del modelo, no de una constante


def test_un_tensor_distinto_da_otra_prediccion(cliente_modelo):
    otro = {**TENSOR, "valores": [[30.0] * T] * V}
    a = _pedir(cliente_modelo).json()["y_pred"]
    b = cliente_modelo.post(RUTA, json={"provincia_id": "AAA", "campana": 2020, "tensor": otro}).json()["y_pred"]
    assert a != pytest.approx(b)


def test_la_misma_solicitud_da_el_mismo_resultado(cliente_modelo):
    assert _pedir(cliente_modelo).json() == _pedir(cliente_modelo).json()


@pytest.mark.parametrize(
    "cuerpo, estado, titulo",
    [
        ({"campana": 2020, "tensor": TENSOR}, 422, "Solicitud inválida"),
        ({"provincia_id": "AAA", "tensor": TENSOR}, 422, "Solicitud inválida"),
        ({"provincia_id": "AAA", "campana": 2022, "tensor": TENSOR}, 422, "Campaña no permitida"),
        ({"provincia_id": "NO EXISTE", "campana": 2020, "tensor": TENSOR}, 404, "Sin línea base"),
        ({"provincia_id": "AAA", "campana": 2020, "tensor": {**TENSOR, "valores": [[1.0] * (T + 1)] * V}}, 422, "Solicitud inválida"),
        ({"provincia_id": "AAA", "campana": 2020, "tensor": {**TENSOR, "valores": [[1.0] * T] * (V - 1)}}, 422, None),
    ],
)
def test_los_casos_invalidos_se_rechazan_sin_predecir(cliente_modelo, contrato, cuerpo, estado, titulo):
    r = cliente_modelo.post(RUTA, json=cuerpo)
    assert r.status_code == estado
    if titulo:
        assert r.json()["title"] == titulo
    _validar(contrato, contrato["paths"]["/predicciones/inferir"]["post"], r)


def test_sin_modelo_serializado_el_tensor_responde_501_con_instrucciones(cliente):
    r = cliente.post(RUTA, json={"provincia_id": "AAA", "campana": 2020, "tensor": TENSOR})
    assert r.status_code == 501
    assert "export_modelo.py" in r.json()["detail"]


def test_la_observacion_del_piloto_sigue_usando_las_predicciones_guardadas(cliente_modelo):
    r = cliente_modelo.post(RUTA, json={"provincia_id": "AAA", "campana": 2019}).json()
    assert r["y_pred"] == pytest.approx(1000.0)  # mediana de las semillas del piloto, no del modelo servido
    assert r["y_real"] == pytest.approx(1000.0)


def test_las_covariables_entran_en_el_orden_de_la_tarjeta(cliente_modelo):
    base = _pedir(cliente_modelo).json()["y_pred"]
    cambiada = cliente_modelo.post(
        RUTA, json={"provincia_id": "AAA", "campana": 2020, "tensor": {**TENSOR, "altitud_media_m": 4500.0}}
    ).json()["y_pred"]
    assert cambiada != pytest.approx(base)
    assert K == 2


def test_con_tensor_devuelve_la_matriz_shap_de_variables_por_fases(cliente_modelo, raiz_con_modelo, contrato):
    r = _pedir(cliente_modelo)
    _validar(contrato, contrato["paths"]["/predicciones/inferir"]["post"], r)
    shap = np.array(r.json()["shap"])
    assert shap.shape == (V, T)
    modelo = ModeloCNN1D.cargar(raiz_con_modelo / "models" / "cnn1d_piloto")
    esperado = modelo.explicar(np.array([TENSOR["valores"]]), np.array([[3800.0, 0.1]]))[0]
    np.testing.assert_allclose(shap, esperado)


def test_sin_fondo_de_referencia_el_shap_es_null_pero_la_prediccion_se_entrega(tmp_path):
    raiz = construir_raiz(tmp_path)
    crear_artefacto(raiz / "models" / "cnn1d_piloto", con_fondo=False)
    r = _pedir(TestClient(create_app(raiz)))
    assert r.status_code == 200
    assert r.json()["shap"] is None
    assert r.json()["y_pred"] > 0


def test_en_modo_observacion_el_shap_sigue_siendo_null(cliente_modelo):
    r = cliente_modelo.post(RUTA, json={"provincia_id": "AAA", "campana": 2019}).json()
    assert r["shap"] is None
