"""El modelo serializado y servido reproduce las predicciones del piloto. Requiere reports/, data/ y models/."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.api.conftest import RAIZ_REAL

MODELO = RAIZ_REAL / "models" / "cnn1d_piloto" / "modelo.json"
PROCESADO = RAIZ_REAL / "data" / "processed"


@pytest.mark.skipif(not MODELO.exists(), reason="No hay modelo serializado (scripts/export_modelo.py).")
def test_la_api_con_tensor_reproduce_la_mediana_del_piloto_en_2020(cliente_real):
    pytest.importorskip("pyarrow")
    if not (PROCESADO / "observaciones.parquet").exists():
        pytest.skip("No hay data/processed.")
    obs = pd.read_parquet(PROCESADO / "observaciones.parquet")
    datos = np.load(PROCESADO / "tensores.npz")
    X, Z = datos["X"], datos["Z"]
    piloto = pd.read_csv(RAIZ_REAL / "reports" / "piloto_oficial" / "predicciones.csv")
    cnn = piloto[(piloto["modelo"] == "cnn1d") & (piloto["campana_prueba"] == 2020)]

    candidatas = [i for i in obs.index[obs["campana_id"] == 2020] if not np.isnan(X[i]).any() and obs.loc[i, "provincia_id"] in set(cnn["provincia_id"])]
    assert len(candidatas) >= 10
    for i in candidatas[:15]:
        provincia = obs.loc[i, "provincia_id"]
        r = cliente_real.post(
            "/api/v1/predicciones/inferir",
            json={"provincia_id": provincia, "campana": 2020,
                  "tensor": {"valores": X[i].tolist(), "altitud_media_m": float(Z[i][0]), "prop_quinua": float(Z[i][1])}},
        )
        assert r.status_code == 200, r.text
        esperado = cnn[cnn["provincia_id"] == provincia]["y_pred"].median()
        assert r.json()["y_pred"] == pytest.approx(esperado, abs=1e-3), provincia


@pytest.mark.skipif(not MODELO.exists(), reason="No hay modelo serializado (scripts/export_modelo.py).")
def test_el_shap_en_numpy_reproduce_la_contribucion_por_fase_publicada_por_el_piloto():
    """P8 del piloto oficial: semilla 0, 30 primeras observaciones de prueba y fondo de 100 filas."""
    pytest.importorskip("pyarrow")
    import json

    from src.evaluation.fold_builder import build_folds, load_pilot_dataset
    from src.explainability.shap_aggregation import aggregate_shap_by_phase
    from src.explainability.shap_numpy import deeplift_semilla
    from src.models.inferencia import ModeloCNN1D

    if not (PROCESADO / "observaciones.parquet").exists():
        pytest.skip("No hay data/processed.")
    resultados = json.loads((RAIZ_REAL / "reports" / "piloto_oficial" / "resultados.json").read_text(encoding="utf-8"))
    permitidas = resultados["p0"]["campanas_permitidas"]
    obs, X, Z, _ = load_pilot_dataset(PROCESADO, permitidas)
    pliego = build_folds(obs, X, Z, 3, permitidas)[-1]
    fases = json.loads((PROCESADO / "metadata.json").read_text(encoding="utf-8"))["orden_fases"]

    modelo = ModeloCNN1D.cargar(MODELO.parent)
    n = min(30, len(pliego.y_test))
    r = deeplift_semilla(modelo.pesos[0], pliego.X_test[:n], pliego.Z_test[:n], modelo.fondo_x, modelo.fondo_z)
    por_observacion = aggregate_shap_by_phase(r.tensor, fases)
    propio = {f: float(np.mean([o[f] for o in por_observacion])) for f in fases}

    publicado = resultados["p8"]["contribucion_por_fase"]
    for fase in fases:
        assert propio[fase] == pytest.approx(publicado[fase], abs=1e-6), fase
    assert max(propio, key=propio.get) == resultados["p8"]["fase_maxima"]
    reconstruida = r.base + r.tensor.sum(axis=(1, 2)) + r.covariables.sum(axis=1)
    np.testing.assert_allclose(reconstruida, r.salida, atol=1e-9)
