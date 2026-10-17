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
