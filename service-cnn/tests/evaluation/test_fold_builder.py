"""Pruebas del constructor de pliegos (tarea T2 del plan del piloto).

Garantías: (1) el piloto solo ve campañas permitidas; (2) origen móvil, también
dentro del entrenamiento; (3) todo el preprocesamiento (imputación, estandarización,
línea base provincial) se ajusta solo con el entrenamiento del pliego (4.12.2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.fold_builder import (
    build_folds,
    build_inner_folds,
    load_pilot_dataset,
    restrict_to_campaigns,
)
from src.evaluation.pilot_guard import CampanaExternaError

PROVINCIAS = ["A", "B", "C"]


def _datos(campanas=(2016, 2017, 2018, 2019, 2020), semilla=0):
    """3 provincias × campañas; V=2 variables, T=3 fases, K=1 covariable."""
    rng = np.random.default_rng(semilla)
    filas = [(p, c) for c in campanas for p in PROVINCIAS]
    obs = pd.DataFrame(filas, columns=["provincia_id", "campana_id"])
    obs["rendimiento_kg_ha"] = 1000.0 + 100.0 * (obs.campana_id - campanas[0]) + rng.normal(0, 20, len(obs))
    X = rng.normal(5.0, 2.0, size=(len(obs), 2, 3))
    Z = rng.normal(0.0, 1.0, size=(len(obs), 1))
    return obs, X, Z


class TestRestrictToCampaigns:
    def test_descarta_las_campanas_no_permitidas_y_las_informa(self):
        obs, X, Z = _datos(campanas=(2016, 2017, 2018, 2021))

        obs_r, X_r, Z_r, descartadas = restrict_to_campaigns(obs, X, Z, permitidas=[2016, 2017, 2018])

        assert descartadas == [2021]
        assert set(obs_r.campana_id) == {2016, 2017, 2018}
        assert X_r.shape[0] == Z_r.shape[0] == len(obs_r) == 9

    def test_las_filas_conservadas_siguen_alineadas_con_los_tensores(self):
        obs, X, Z = _datos(campanas=(2016, 2017, 2021))

        obs_r, X_r, Z_r, _ = restrict_to_campaigns(obs, X, Z, permitidas=[2016, 2017])

        np.testing.assert_array_equal(X_r, X[obs.campana_id.isin([2016, 2017]).to_numpy()])

    def test_si_no_hay_nada_que_descartar_no_informa_campanas(self):
        obs, X, Z = _datos(campanas=(2016, 2017))

        *_, descartadas = restrict_to_campaigns(obs, X, Z, permitidas=[2016, 2017, 2018])

        assert descartadas == []


class TestBuildFolds:
    def test_hay_un_pliego_por_campana_posterior_a_la_ventana(self):
        obs, X, Z = _datos()

        pliegos = build_folds(obs, X, Z, ventana_inicial=3, campanas_permitidas=range(2016, 2021))

        assert [p.campana_prueba for p in pliegos] == [2019, 2020]

    def test_el_entrenamiento_es_siempre_anterior_a_la_prueba(self):
        obs, X, Z = _datos()

        for p in build_folds(obs, X, Z, 3, range(2016, 2021)):
            assert max(p.campanas_entrenamiento) < p.campana_prueba
            assert (p.campana_train < p.campana_prueba).all()
            assert (p.campana_test == p.campana_prueba).all()

    def test_rechaza_datos_con_campanas_fuera_de_las_permitidas(self):
        obs, X, Z = _datos(campanas=(2016, 2017, 2018, 2019, 2020, 2021))

        with pytest.raises(CampanaExternaError, match="2021"):
            build_folds(obs, X, Z, 3, campanas_permitidas=range(2016, 2021))

    def test_los_tamanos_son_coherentes(self):
        obs, X, Z = _datos()

        p = build_folds(obs, X, Z, 3, range(2016, 2021))[0]

        assert p.X_train.shape == (9, 2, 3) and p.X_test.shape == (3, 2, 3)
        assert p.Z_train.shape == (9, 1) and p.y_train.shape == (9,)


class TestPreprocesamientoSinFuga:
    def test_el_entrenamiento_queda_estandarizado_con_sus_propias_estadisticas(self):
        obs, X, Z = _datos()

        p = build_folds(obs, X, Z, 3, range(2016, 2021))[0]

        np.testing.assert_allclose(p.X_train.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(p.X_train.std(axis=0), 1.0, atol=1e-9)

    def test_alterar_la_prueba_no_cambia_el_entrenamiento_ni_las_estadisticas(self):
        obs, X, Z = _datos()
        alterado = X.copy()
        alterado[(obs.campana_id == 2019).to_numpy()] += 1e6  # la campaña de prueba del primer pliego

        a = build_folds(obs, X, Z, 3, range(2016, 2021))[0]
        b = build_folds(obs, alterado, Z, 3, range(2016, 2021))[0]

        np.testing.assert_array_equal(a.X_train, b.X_train)
        np.testing.assert_array_equal(a.preprocesador.media, b.preprocesador.media)

    def test_las_celdas_vacias_de_la_prueba_se_imputan_con_la_mediana_del_entrenamiento(self):
        obs, X, Z = _datos()
        X = X.copy()
        fila_prueba = np.where((obs.campana_id == 2019).to_numpy())[0][0]
        X[fila_prueba, 0, 1] = np.nan
        entrenamiento = (obs.campana_id < 2019).to_numpy()
        mediana_entrenamiento = np.nanmedian(X[entrenamiento][:, 0, 1])

        p = build_folds(obs, X, Z, 3, range(2016, 2021))[0]

        esperado = (mediana_entrenamiento - p.preprocesador.media[0, 1]) / p.preprocesador.desviacion[0, 1]
        assert p.X_test[0, 0, 1] == pytest.approx(esperado)
        assert not np.isnan(p.X_test).any()

    def test_una_celda_constante_no_produce_division_por_cero(self):
        obs, X, Z = _datos()
        X = X.copy()
        X[:, 1, 2] = 7.0

        p = build_folds(obs, X, Z, 3, range(2016, 2021))[0]

        assert np.isfinite(p.X_train).all() and np.isfinite(p.X_test).all()


class TestLineaBaseProvincial:
    def test_la_linea_base_se_ajusta_solo_con_el_entrenamiento(self):
        obs, X, Z = _datos()
        alterado = obs.copy()
        alterado.loc[alterado.campana_id == 2019, "rendimiento_kg_ha"] = 99999.0  # verdad de la prueba

        a = build_folds(obs, X, Z, 3, range(2016, 2021))[0]
        b = build_folds(alterado, X, Z, 3, range(2016, 2021))[0]

        np.testing.assert_array_equal(a.baseline_test, b.baseline_test)

    def test_la_anomalia_es_el_rendimiento_menos_la_linea_base(self):
        obs, X, Z = _datos()

        p = build_folds(obs, X, Z, 3, range(2016, 2021))[0]

        np.testing.assert_allclose(p.y_anomalia_train, p.y_train - p.baseline_train)
        np.testing.assert_allclose(p.y_anomalia_test, p.y_test - p.baseline_test)

    def test_una_provincia_sin_historia_usa_la_media_de_entrenamiento_y_se_cuenta(self):
        obs, X, Z = _datos()
        nueva = pd.DataFrame([("Z", 2019, 1500.0)], columns=["provincia_id", "campana_id", "rendimiento_kg_ha"])
        obs = pd.concat([obs, nueva], ignore_index=True)
        X = np.concatenate([X, X[:1]])
        Z = np.concatenate([Z, Z[:1]])

        p = build_folds(obs, X, Z, 3, range(2016, 2021))[0]

        assert p.n_sin_historia_test == 1
        idx = int(np.where(p.provincia_test == "Z")[0][0])
        assert p.baseline_test[idx] == pytest.approx(p.y_train.mean())


class TestPliegosInternos:
    def test_validacion_interna_de_origen_movil_dentro_del_entrenamiento(self):
        obs, X, Z = _datos()
        externo = build_folds(obs, X, Z, 3, range(2016, 2021))[1]  # entrena 2016-2019

        internos = build_inner_folds(externo, ventana_interna=2)

        assert [i.campana_prueba for i in internos] == [2018, 2019]
        for i in internos:
            assert max(i.campanas_entrenamiento) < i.campana_prueba
            assert i.campana_prueba < externo.campana_prueba

    def test_los_pliegos_internos_no_contienen_la_campana_externa(self):
        obs, X, Z = _datos()
        externo = build_folds(obs, X, Z, 3, range(2016, 2021))[1]

        for i in build_inner_folds(externo, 2):
            assert externo.campana_prueba not in i.campanas_entrenamiento + [i.campana_prueba]

    def test_cada_pliego_interno_reajusta_su_propio_preprocesamiento(self):
        obs, X, Z = _datos()
        externo = build_folds(obs, X, Z, 3, range(2016, 2021))[1]

        interno = build_inner_folds(externo, 2)[0]

        np.testing.assert_allclose(interno.X_train.mean(axis=0), 0.0, atol=1e-9)
        assert not np.allclose(interno.preprocesador.media, externo.preprocesador.media)

    def test_sin_campanas_suficientes_no_hay_pliegos_internos(self):
        obs, X, Z = _datos(campanas=(2016, 2017, 2018))
        externo = build_folds(obs, X, Z, 2, range(2016, 2019))[0]  # entrena solo 2016-2017

        assert build_inner_folds(externo, ventana_interna=2) == []


class TestLoadPilotDataset:
    """El cargador descarta las campañas externas al leer y solo informa sus ids."""

    @staticmethod
    def _escribir(directorio, campanas):
        obs, X, Z = _datos(campanas=campanas)
        obs.to_parquet(directorio / "observaciones.parquet", index=False)
        np.savez_compressed(directorio / "tensores.npz", X=X, Z=Z, imputado=np.zeros_like(X, dtype=bool))
        return obs, X, Z

    def test_devuelve_solo_las_campanas_permitidas_alineadas(self, tmp_path):
        self._escribir(tmp_path, (2016, 2017, 2018, 2021))

        obs, X, Z, descartadas = load_pilot_dataset(tmp_path, campanas_permitidas=[2016, 2017, 2018])

        assert descartadas == [2021]
        assert set(obs.campana_id) == {2016, 2017, 2018}
        assert X.shape[0] == Z.shape[0] == len(obs)

    def test_los_valores_conservados_son_los_del_archivo(self, tmp_path):
        obs0, X0, _ = self._escribir(tmp_path, (2016, 2017, 2021))

        _, X, _, _ = load_pilot_dataset(tmp_path, [2016, 2017])

        np.testing.assert_array_equal(X, X0[obs0.campana_id.isin([2016, 2017]).to_numpy()])

    def test_falla_con_un_mensaje_claro_si_el_conjunto_no_tiene_covariables(self, tmp_path):
        obs, X, _ = _datos()
        obs.to_parquet(tmp_path / "observaciones.parquet", index=False)
        np.savez_compressed(tmp_path / "tensores.npz", X=X, Z=np.zeros((len(obs), 0)))

        with pytest.raises(ValueError, match="covariables"):
            load_pilot_dataset(tmp_path, range(2016, 2021))
