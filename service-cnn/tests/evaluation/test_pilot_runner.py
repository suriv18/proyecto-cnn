"""Pruebas del orquestador del piloto (tarea T7).

Sobre un conjunto sintético pequeño verifican las garantías del piloto: las campañas
externas se descartan y se registran, el modo oficial exige el gate de F2, el modo técnico
se marca no vinculante, los 5 modelos reciben el mismo presupuesto y se ejecutan P0-P10.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.pilot_runner import (
    PilotConfig,
    PilotGateError,
    render_pilot_report,
    run_pilot,
    save_pilot_results,
)

ESPACIOS = {
    "elastic_net": {"alpha": ["float", 0.01, 1.0], "l1_ratio": ["float", 0.1, 0.9]},
    "random_forest": {"n_estimators": ["int", 5, 15], "max_depth": ["int", 2, 4]},
    "xgboost": {"n_estimators": ["int", 5, 15], "max_depth": ["int", 2, 3], "learning_rate": ["float", 0.05, 0.3]},
    "svr": {"C": ["float", 0.1, 10.0], "epsilon": ["float", 0.01, 0.5]},
    "cnn1d": {"filtros_capa1": ["int", 3, 5], "filtros_capa2": ["int", 3, 5], "dropout": ["float", 0.001, 0.01], "tasa_aprendizaje": ["float", 0.001, 0.01]},
}
FASES = ["emergencia", "desarrollo_vegetativo", "floracion", "llenado_grano", "madurez"]


@pytest.fixture(scope="module")
def procesado(tmp_path_factory):
    d = tmp_path_factory.mktemp("procesado")
    rng = np.random.default_rng(0)
    provincias = [f"P{i}" for i in range(8)]
    obs = pd.DataFrame([(p, c) for c in range(2016, 2022) for p in provincias], columns=["provincia_id", "campana_id"])
    X = rng.normal(5.0, 2.0, size=(len(obs), 6, 5))
    Z = rng.normal(0.0, 1.0, size=(len(obs), 2))
    nivel = {p: 800.0 + 120.0 * i for i, p in enumerate(provincias)}
    obs["rendimiento_kg_ha"] = obs.provincia_id.map(nivel) + 20.0 * X[:, 4, :].mean(axis=1) + rng.normal(0, 15, len(obs))
    obs.to_parquet(d / "observaciones.parquet", index=False)
    np.savez_compressed(d / "tensores.npz", X=X, Z=Z, imputado=np.zeros_like(X, dtype=bool))
    (d / "metadata.json").write_text(json.dumps({"orden_fases": FASES}), encoding="utf-8")
    return d


def _cfg(procesado, tmp_path, **extra):
    base = dict(
        directorio_procesado=procesado, directorio_salida=tmp_path, espacios=ESPACIOS,
        ventana_confirmatoria=4, ventana_piloto=3, n_trials=2, n_semillas=2, epocas_cnn=3,
        n_permutaciones=3, n_explicadas=4, indice_ndvi_max=5,
    )
    base.update(extra)
    return PilotConfig(**base)


@pytest.fixture(scope="module")
def resultados(procesado, tmp_path_factory):
    return run_pilot(_cfg(procesado, tmp_path_factory.mktemp("salida")))


class TestGuardaDeCampanas:
    def test_descarta_las_campanas_externas_y_las_registra(self, resultados):
        assert resultados.p0["campanas_permitidas"] == [2016, 2017, 2018, 2019]
        assert resultados.p0["campanas_descartadas"] == [2020, 2021]

    def test_los_pliegos_del_piloto_no_incluyen_campanas_externas(self, resultados):
        assert resultados.p0["campanas_de_prueba"] == [2019]
        assert set(resultados.predicciones["campana_prueba"]) == {2019}


class TestModos:
    def test_el_modo_oficial_exige_el_gate_de_f2(self, procesado, tmp_path):
        with pytest.raises(PilotGateError, match="gate"):
            run_pilot(_cfg(procesado, tmp_path, modo="oficial", gate_f2_aprobado=False))

    def test_el_modo_tecnico_se_marca_no_vinculante(self, resultados):
        assert resultados.p0["modo"] == "tecnico"
        assert resultados.p0["vinculante"] is False


class TestExperimentos:
    def test_p1_prueba_de_fuga_para_los_cinco_modelos(self, resultados):
        assert set(resultados.p1["modelo"]) == {"cnn1d", "elastic_net", "random_forest", "xgboost", "svr"}
        assert {"r2_mediano", "pasa"} <= set(resultados.p1.columns)

    def test_p2_referencias_b1_b2_b3(self, resultados):
        assert set(resultados.p2["modelo"]) == {"B1", "B2", "B3"}

    def test_p3_el_presupuesto_es_igual_para_los_cinco_modelos(self, resultados):
        assert set(resultados.p3["n_trials"]) == {2}
        assert len(resultados.p3) == 5 * 2  # 5 modelos x 2 representaciones

    def test_p4_decide_la_representacion(self, resultados):
        assert resultados.p4["decision"] in {"nivel", "anomalia"}

    def test_p5_decide_la_forma_de_b2(self, resultados):
        assert resultados.p5["decision"] in {"media", "lineal_min4", "media_ultimas3"}

    def test_p6_variabilidad_por_semilla_de_los_estocasticos(self, resultados):
        assert set(resultados.p6["modelo"]) == {"cnn1d", "random_forest", "xgboost"}

    def test_p7_compara_las_dos_particiones_y_fija_las_epocas_maximas(self, resultados):
        assert set(resultados.p7["tabla"]["particion"]) == {"temporal", "aleatoria"}
        assert isinstance(resultados.p7["epocas_maximas"], int) and resultados.p7["epocas_maximas"] >= 1

    def test_p8_shap_informa_la_variante_y_la_precision_local(self, resultados):
        assert resultados.p8["variante"] in {"deep_explainer", "kernel_explainer"}
        assert resultados.p8["error_deep_explainer"] >= 0.0

    def test_p9_revisa_los_bordes_de_los_espacios_de_busqueda(self, resultados):
        assert {"modelo", "parametro", "lado"} <= set(resultados.p9.columns)

    def test_p10_curva_de_aprendizaje_con_mas_campanas_de_entrenamiento(self, resultados):
        assert list(resultados.p10["n_campanas_entrenamiento"]) == [1, 2, 3]


class TestSalidas:
    def test_guarda_los_resultados_en_disco(self, resultados, tmp_path):
        save_pilot_results(resultados, tmp_path)

        assert (tmp_path / "resultados.json").exists()
        assert (tmp_path / "p1_prueba_de_fuga.csv").exists()
        assert (tmp_path / "informe_piloto.md").exists()

    def test_el_informe_incluye_el_aviso_y_las_decisiones(self, resultados):
        texto = render_pilot_report(resultados)

        assert "NO VINCULANTE" in texto
        assert "Representación" in texto and "Forma de B2" in texto
        assert "2020, 2021" in texto  # campañas descartadas
