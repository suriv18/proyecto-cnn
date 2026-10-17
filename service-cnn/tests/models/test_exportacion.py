"""Serialización de la CNN-1D: el modelo guardado y leído con NumPy reproduce al de PyTorch."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.evaluation.experiment_runner import build_adapter  # noqa: E402
from src.evaluation.fold_builder import build_folds  # noqa: E402
from src.models.exportacion import exportar_cnn1d, huella_de_pesos  # noqa: E402
from src.models.inferencia import ModeloCNN1D  # noqa: E402

HP = {"filtros_capa1": 3, "filtros_capa2": 4, "dropout": 0.1, "tasa_aprendizaje": 0.01}
CAMPANAS = [2016, 2017, 2018, 2019, 2020]
PROVINCIAS = [f"P{i}" for i in range(8)]
CONTEXTO = {
    "orden_variables": ["v"] * 6, "orden_fases": ["f"] * 5, "covariables": ["altitud_media_m", "prop_quinua"],
    "campanas_permitidas": CAMPANAS, "sha256_tensores": "0" * 64, "modo": "tecnico", "vinculante": False,
}


@pytest.fixture(scope="module")
def pliego():
    rng = np.random.default_rng(0)
    filas = [(p, c) for c in CAMPANAS for p in PROVINCIAS]
    obs = pd.DataFrame(filas, columns=["provincia_id", "campana_id"])
    obs["rendimiento_kg_ha"] = 1000 + 20 * (obs["campana_id"] - 2016) + rng.normal(0, 60, len(obs))
    X = rng.normal(10, 3, (len(obs), 6, 5))
    X[rng.random(X.shape) < 0.03] = np.nan
    Z = np.column_stack([rng.normal(3800, 300, len(obs)), rng.uniform(0, 0.3, len(obs))])
    return build_folds(obs, X, Z, 3, CAMPANAS)[-1]


@pytest.fixture(scope="module")
def exportado(pliego, tmp_path_factory):
    destino = tmp_path_factory.mktemp("modelo")
    tarjeta = exportar_cnn1d(pliego, HP, "anomalia", epocas=4, n_semillas=2, destino=destino, contexto=CONTEXTO)
    return destino, tarjeta


def _z_crudo(pliego):
    p = pliego.preprocesador
    return pliego.Z_test * p.z_desviacion + p.z_media


def test_numpy_reproduce_a_pytorch_semilla_por_semilla(pliego, exportado):
    destino, _ = exportado
    modelo = ModeloCNN1D.cargar(destino)
    propio = modelo.predecir_anomalia(pliego.X_test_crudo, _z_crudo(pliego))
    for semilla in range(2):
        torch_modelo, media, desviacion = build_adapter("cnn1d", 4).entrenar(HP, pliego, semilla, "anomalia")
        with torch.no_grad():
            salida = torch_modelo(
                torch.tensor(pliego.X_test, dtype=torch.float32), torch.tensor(pliego.Z_test, dtype=torch.float32)
            ).squeeze(1).numpy()
        np.testing.assert_allclose(propio[semilla], salida * desviacion + media, rtol=1e-4, atol=1e-3)


def test_la_linea_base_guardada_reproduce_la_del_pliego(pliego, exportado):
    modelo = ModeloCNN1D.cargar(exportado[0])
    for provincia, campana, esperado in zip(pliego.provincia_test, pliego.campana_test, pliego.baseline_test):
        assert modelo.linea_base.predecir(str(provincia), int(campana)) == pytest.approx(esperado)


def test_la_tarjeta_documenta_la_procedencia_y_no_afirma_mas_de_lo_que_es(exportado):
    destino, tarjeta = exportado
    assert tarjeta["campanas_entrenamiento"] == [2016, 2017, 2018, 2019]
    assert tarjeta["representacion"] == "anomalia"
    assert tarjeta["semillas"] == [0, 1]
    assert tarjeta["vinculante"] is False
    assert "no es el modelo definitivo" in tarjeta["advertencia"].lower()
    assert json.loads((destino / "modelo.json").read_text(encoding="utf-8"))["huella_pesos"] == tarjeta["huella_pesos"]


def test_el_entrenamiento_es_reproducible_y_la_huella_lo_refleja(pliego, exportado, tmp_path):
    segunda = exportar_cnn1d(pliego, HP, "anomalia", epocas=4, n_semillas=2, destino=tmp_path, contexto=CONTEXTO)
    assert segunda["huella_pesos"] == exportado[1]["huella_pesos"]


def test_la_huella_cambia_si_cambia_un_peso():
    a = {"s0.w": np.array([1.0, 2.0])}
    b = {"s0.w": np.array([1.0, 2.0000001])}
    assert huella_de_pesos(a) == huella_de_pesos({"s0.w": np.array([1.0, 2.0])})
    assert huella_de_pesos(a) != huella_de_pesos(b)


def test_no_se_sirve_la_representacion_en_nivel(pliego, tmp_path):
    with pytest.raises(ValueError, match="anomalía"):
        exportar_cnn1d(pliego, HP, "nivel", epocas=2, n_semillas=1, destino=tmp_path, contexto=CONTEXTO)


def test_los_archivos_no_usan_pickle(exportado):
    destino, _ = exportado
    with np.load(destino / "pesos.npz", allow_pickle=False) as p:  # falla si hubiera objetos serializados
        assert len(p.files) > 0
    with np.load(destino / "preprocesador.npz", allow_pickle=False) as p:
        assert set(p.files) == {"mediana", "media", "desviacion", "z_media", "z_desviacion"}
