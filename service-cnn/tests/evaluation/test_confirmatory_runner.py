"""Pruebas del orquestador de la evaluación confirmatoria.

Sobre un conjunto sintético pequeño verifican las garantías del procedimiento: el ensayo nunca carga las campañas
externas, las compuertas de E5 y de los datos se exigen, la corrida oficial no se puede cambiar ni repetir, B* se
elige sin campañas externas y la decisión de la hipótesis general es la conjunción de HE1 y HE2.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation.confirmatory_runner import (
    ARCHIVO_DE_EJECUCION,
    ConfirmatoryConfig,
    ConfirmatoryGateError,
    comprobar_puertas,
    marcar_ejecucion,
    parametros_de_e5,
    run_confirmatory,
    save_confirmatory_results,
)
from src.evaluation.preregistration import load_preregistration

E5_REAL = Path("configs/preespecificacion.yaml")
ESPACIOS = {
    "elastic_net": {"alpha": ["float", 0.01, 1.0], "l1_ratio": ["float", 0.1, 0.9]},
    "random_forest": {"n_estimators": ["int", 5, 15], "max_depth": ["int", 2, 4]},
    "xgboost": {"n_estimators": ["int", 5, 15], "max_depth": ["int", 2, 3], "learning_rate": ["float", 0.05, 0.3]},
    "svr": {"C": ["float", 0.1, 10.0], "epsilon": ["float", 0.01, 0.5]},
    "cnn1d": {"filtros_capa1": ["int", 3, 5], "filtros_capa2": ["int", 3, 5], "dropout": ["float", 0.001, 0.01], "tasa_aprendizaje": ["float", 0.001, 0.01]},
}
FASES = ["emergencia", "desarrollo_vegetativo", "floracion", "llenado_grano", "madurez"]
VARIABLES = ["precipitacion", "temperatura_maxima", "temperatura_minima", "radiacion_solar", "ndvi_mean", "ndvi_max"]


def _crear_procesado(directorio: Path, exploratorio: bool = True) -> Path:
    rng = np.random.default_rng(0)
    provincias = [f"P{i}" for i in range(8)]
    obs = pd.DataFrame([(p, c) for c in range(2016, 2023) for p in provincias], columns=["provincia_id", "campana_id"])
    X = rng.normal(5.0, 2.0, size=(len(obs), 6, 5))
    Z = rng.normal(0.0, 1.0, size=(len(obs), 2))
    nivel = {p: 800.0 + 120.0 * i for i, p in enumerate(provincias)}
    obs["rendimiento_kg_ha"] = obs.provincia_id.map(nivel) + 20.0 * X[:, 4, :].mean(axis=1) + rng.normal(0, 15, len(obs))
    obs.to_parquet(directorio / "observaciones.parquet", index=False)
    np.savez_compressed(directorio / "tensores.npz", X=X, Z=Z, imputado=np.zeros_like(X, dtype=bool))
    (directorio / "metadata.json").write_text(
        json.dumps({"orden_fases": FASES, "orden_variables": VARIABLES, "exploratorio": exploratorio}), encoding="utf-8"
    )
    return directorio


def _e5_para(procesado: Path, destino: Path, **cambios_meta) -> Path:
    """Copia de E5 con la huella de los datos sintéticos; `cambios_meta` altera los metadatos."""
    prereg = load_preregistration(E5_REAL)
    prereg["metadatos"]["sha256_conjunto_de_datos"] = hashlib.sha256((procesado / "tensores.npz").read_bytes()).hexdigest()
    prereg["metadatos"].update(cambios_meta)
    destino.write_text(yaml.safe_dump(prereg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destino


@pytest.fixture(scope="module")
def procesado(tmp_path_factory) -> Path:
    return _crear_procesado(tmp_path_factory.mktemp("procesado"))


@pytest.fixture(scope="module")
def e5(procesado, tmp_path_factory) -> Path:
    return _e5_para(procesado, tmp_path_factory.mktemp("e5") / "e5.yaml")


def _cfg(procesado, e5, salida, **extra) -> ConfirmatoryConfig:
    base = dict(
        directorio_procesado=procesado, directorio_salida=salida, espacios=ESPACIOS, ruta_preespecificacion=e5,
        ventana_inicial=3, n_trials=2, n_semillas=2, epocas_cnn=3, n_permutaciones=50,
    )
    base.update(extra)
    return ConfirmatoryConfig(**base)


@pytest.fixture(scope="module")
def ensayo(procesado, e5, tmp_path_factory):
    return run_confirmatory(_cfg(procesado, e5, tmp_path_factory.mktemp("salida")))


class TestEnsayoNoAbreLasCampanasExternas:
    def test_solo_carga_las_campanas_de_entrenamiento_de_e5(self, ensayo):
        assert ensayo.p0["campanas_cargadas"] == [2016, 2017, 2018, 2019, 2020]
        assert ensayo.p0["campanas_descartadas_al_cargar"] == [2021, 2022]

    def test_hace_pasar_por_externas_las_ultimas_campanas_permitidas(self, ensayo):
        assert ensayo.p0["campanas_externas_evaluadas"] == [2019, 2020]
        assert set(ensayo.predicciones["campana_prueba"]) == {2019, 2020}

    def test_se_marca_no_vinculante(self, ensayo):
        assert ensayo.p0["modo"] == "ensayo" and ensayo.p0["vinculante"] is False


class TestProcedimiento:
    def test_toma_los_parametros_de_e5(self, ensayo):
        reales = parametros_de_e5(load_preregistration(E5_REAL))
        usados = ensayo.p0["parametros"]

        assert usados["representacion"] == reales["representacion"] == "anomalia"
        assert usados["margen"] == reales["margen"] == 0.05
        assert usados["ventana_interna"] == reales["ventana_interna"] == 2

    def test_b_star_sale_de_la_validacion_interna_del_pliego_inicial(self, ensayo):
        assert ensayo.b_star in ("B2", "B4")
        assert set(ensayo.rmse_validacion_interna_b) == {"B2", "B4"}
        assert ensayo.b_star == min(ensayo.rmse_validacion_interna_b, key=lambda b: (ensayo.rmse_validacion_interna_b[b], b != "B2"))

    def test_evalua_los_cinco_modelos_y_las_cuatro_lineas_base(self, ensayo):
        modelos = set(ensayo.rmse_por_campana["modelo"])

        assert modelos == {"cnn1d", "elastic_net", "random_forest", "xgboost", "svr", "B1", "B2", "B3", "B4"}

    def test_explica_todas_las_observaciones_externas_con_el_mismo_modelo_evaluado(self, ensayo):
        assert set(ensayo.contribuciones) == set(ensayo.aumentos_rmse) == {2019, 2020}
        assert all(set(c) == set(FASES) for c in ensayo.contribuciones.values())
        diferencias = ensayo.verificaciones["max_diferencia_prediccion_cnn_reentrenada_kg_ha"]
        assert all(d < 1e-2 for d in diferencias.values()), diferencias

    def test_la_hipotesis_general_es_la_conjuncion_de_he1_y_he2(self, ensayo):
        assert ensayo.hipotesis_general_respaldada == (ensayo.he1.he1_respaldada and ensayo.he2.he2_respaldada)

    def test_he1_usa_las_campanas_externas_del_ensayo(self, ensayo):
        assert ensayo.he1.campanas == [2019, 2020] and ensayo.he1.b_star == ensayo.b_star

    def test_guarda_los_resultados_y_un_informe_que_dice_que_es_un_ensayo(self, ensayo, tmp_path):
        save_confirmatory_results(ensayo, tmp_path)

        for nombre in ("predicciones.csv", "rmse_por_campana.csv", "ajuste.csv", "resultados.json", "informe_confirmatorio.md"):
            assert (tmp_path / nombre).exists(), nombre
        assert "ENSAYO, NO VINCULANTE" in (tmp_path / "informe_confirmatorio.md").read_text(encoding="utf-8")
        resumen = json.loads((tmp_path / "resultados.json").read_text(encoding="utf-8"))
        assert resumen["p0"]["vinculante"] is False and resumen["he1"]["campanas"] == [2019, 2020]


class TestMarcaDeEjecucion:
    def test_al_completarse_conserva_la_hora_de_inicio(self, tmp_path):
        marcar_ejecucion(tmp_path, "iniciada", "2026-10-17T22:30:12+00:00")
        al_iniciar = json.loads((tmp_path / ARCHIVO_DE_EJECUCION).read_text(encoding="utf-8"))

        marcar_ejecucion(tmp_path, "completada", "2026-10-17T22:30:12+00:00", "2026-10-17T23:07:23+00:00")
        al_terminar = json.loads((tmp_path / ARCHIVO_DE_EJECUCION).read_text(encoding="utf-8"))

        assert al_iniciar == {"inicio": "2026-10-17T22:30:12+00:00", "estado": "iniciada"}
        assert al_terminar == {"inicio": "2026-10-17T22:30:12+00:00", "estado": "completada", "fin": "2026-10-17T23:07:23+00:00"}

    def test_su_existencia_impide_repetir_la_corrida_oficial(self, tmp_path):
        marcar_ejecucion(tmp_path, "iniciada", "2026-10-17T22:30:12+00:00")

        assert (tmp_path / ARCHIVO_DE_EJECUCION).exists()


class TestPuertasDeE5YDeLosDatos:
    def test_exige_e5_congelada(self, procesado, tmp_path):
        e5_borrador = _e5_para(procesado, tmp_path / "e5.yaml", estado_documento="BORRADOR")

        with pytest.raises(ConfirmatoryGateError, match="no está congelada"):
            comprobar_puertas(_cfg(procesado, e5_borrador, tmp_path / "s"))

    def test_exige_que_la_huella_de_los_datos_coincida_con_la_de_e5(self, procesado, tmp_path):
        e5_otra = _e5_para(procesado, tmp_path / "e5.yaml", sha256_conjunto_de_datos="0" * 64)

        with pytest.raises(ConfirmatoryGateError, match="huella"):
            comprobar_puertas(_cfg(procesado, e5_otra, tmp_path / "s"))

    def test_exige_que_e5_coincida_con_el_codigo(self, procesado, tmp_path):
        prereg = load_preregistration(E5_REAL)
        prereg["modelos"]["presupuesto_de_trials_por_modelo"]["valor"] = 31
        prereg["metadatos"]["sha256_conjunto_de_datos"] = hashlib.sha256((procesado / "tensores.npz").read_bytes()).hexdigest()
        ruta = tmp_path / "e5.yaml"
        ruta.write_text(yaml.safe_dump(prereg, allow_unicode=True, sort_keys=False), encoding="utf-8")

        with pytest.raises(ConfirmatoryGateError, match="no coincide con el código"):
            comprobar_puertas(_cfg(procesado, ruta, tmp_path / "s"))


class TestModoOficial:
    @pytest.fixture
    def definitivo(self, tmp_path):
        return _crear_procesado(tmp_path, exploratorio=False)

    def _oficial(self, procesado, e5, salida, **extra) -> ConfirmatoryConfig:
        return ConfirmatoryConfig(
            directorio_procesado=procesado, directorio_salida=salida, espacios=ESPACIOS, ruta_preespecificacion=e5,
            modo="oficial", **extra,
        )

    def test_las_puertas_pasan_con_todo_en_regla(self, definitivo, tmp_path):
        e5 = _e5_para(definitivo, tmp_path / "e5.yaml")

        _, parametros = comprobar_puertas(self._oficial(definitivo, e5, tmp_path / "s", confirmo_abrir_campanas_externas=True))

        assert parametros["ventana_inicial"] == 5 and parametros["n_trials"] == 30 and parametros["n_semillas"] == 10

    def test_exige_confirmar_la_apertura_de_las_campanas_externas(self, definitivo, tmp_path):
        e5 = _e5_para(definitivo, tmp_path / "e5.yaml")

        with pytest.raises(ConfirmatoryGateError, match="una sola vez"):
            comprobar_puertas(self._oficial(definitivo, e5, tmp_path / "s"))

    def test_rechaza_un_conjunto_exploratorio(self, procesado, tmp_path):
        e5 = _e5_para(procesado, tmp_path / "e5.yaml")

        with pytest.raises(ConfirmatoryGateError, match="exploratorio"):
            comprobar_puertas(self._oficial(procesado, e5, tmp_path / "s", confirmo_abrir_campanas_externas=True))

    @pytest.mark.parametrize("campo,valor", [("ventana_inicial", 4), ("n_trials", 5), ("n_semillas", 3), ("epocas_cnn", 10)])
    def test_los_parametros_no_se_pueden_cambiar(self, definitivo, tmp_path, campo, valor):
        e5 = _e5_para(definitivo, tmp_path / "e5.yaml")
        cfg = self._oficial(definitivo, e5, tmp_path / "s", confirmo_abrir_campanas_externas=True, **{campo: valor})

        with pytest.raises(ConfirmatoryGateError, match="no se pueden cambiar"):
            comprobar_puertas(cfg)

    def test_no_se_repite_si_ya_hubo_una_ejecucion(self, definitivo, tmp_path):
        e5 = _e5_para(definitivo, tmp_path / "e5.yaml")
        salida = tmp_path / "s"
        salida.mkdir()
        (salida / ARCHIVO_DE_EJECUCION).write_text("{}", encoding="utf-8")

        with pytest.raises(ConfirmatoryGateError, match="no se repite"):
            comprobar_puertas(self._oficial(definitivo, e5, salida, confirmo_abrir_campanas_externas=True))

    def test_run_confirmatory_no_toca_nada_si_una_puerta_falla(self, procesado, tmp_path):
        e5 = _e5_para(procesado, tmp_path / "e5.yaml")
        salida = tmp_path / "s"

        with pytest.raises(ConfirmatoryGateError):
            run_confirmatory(self._oficial(procesado, e5, salida, confirmo_abrir_campanas_externas=True))

        assert not salida.exists()
