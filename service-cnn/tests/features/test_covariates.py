"""Pruebas de las covariables del tensor (sección 4.10.3).

- Estática: altitud media de la superficie agrícola de la provincia.
- Dinámica: proporción de la superficie agrícola sembrada con quinua en la campaña.
Ambas se concatenan después del bloque convolucional (no forman parte de V×T).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.covariates import build_covariates


def _obs():
    return pd.DataFrame(
        [
            dict(provincia_id="AZANGARO", campana_id=2021),
            dict(provincia_id="AZANGARO", campana_id=2022),
            dict(provincia_id="CARABAYA", campana_id=2021),
        ]
    )


def _siembra():
    return pd.DataFrame(
        [
            dict(provincia_id="AZANGARO", campana_id=2021, mes=10, superficie_ha=3000.0),
            dict(provincia_id="AZANGARO", campana_id=2021, mes=11, superficie_ha=1000.0),
            dict(provincia_id="AZANGARO", campana_id=2022, mes=10, superficie_ha=2000.0),
            dict(provincia_id="CARABAYA", campana_id=2021, mes=10, superficie_ha=100.0),
        ]
    )


class TestBuildCovariates:
    def kw(self, **extra):
        base = dict(
            observaciones=_obs(),
            siembra_mensual=_siembra(),
            altitud_media_por_provincia={"AZANGARO": 3850.0, "CARABAYA": 3700.0},
            area_cultivo_ha_por_provincia={"AZANGARO": 80000.0, "CARABAYA": 800.0},
        )
        base.update(extra)
        return base

    def test_altitud_media_es_la_de_la_provincia(self):
        c = build_covariates(**self.kw())

        assert list(c["altitud_media_m"]) == [3850.0, 3850.0, 3700.0]

    def test_proporcion_es_superficie_sembrada_de_la_campana_sobre_area_de_cultivo(self):
        c = build_covariates(**self.kw())

        assert c["prop_quinua"].iloc[0] == pytest.approx(4000.0 / 80000.0)  # suma los meses de la campaña
        assert c["prop_quinua"].iloc[1] == pytest.approx(2000.0 / 80000.0)
        assert c["prop_quinua"].iloc[2] == pytest.approx(100.0 / 800.0)

    def test_conserva_el_orden_y_el_indice_de_las_observaciones(self):
        obs = _obs().iloc[::-1].reset_index(drop=True)

        c = build_covariates(**self.kw(observaciones=obs))

        assert c["prop_quinua"].iloc[2] == pytest.approx(4000.0 / 80000.0)  # AZANGARO 2021 quedó al final

    def test_provincia_sin_dato_de_area_o_altitud_da_nan_y_no_falla(self):
        c = build_covariates(**self.kw(area_cultivo_ha_por_provincia={"AZANGARO": 80000.0},
                                       altitud_media_por_provincia={"AZANGARO": 3850.0}))

        assert np.isnan(c["prop_quinua"].iloc[2])
        assert np.isnan(c["altitud_media_m"].iloc[2])

    def test_no_recorta_las_proporciones_mayores_que_uno(self):
        """Una proporción > 1 indica una inconsistencia entre fuentes (SISAGRI frente
        a WorldCover): debe quedar visible, no ocultarse recortándola."""
        c = build_covariates(**self.kw(area_cultivo_ha_por_provincia={"AZANGARO": 1000.0, "CARABAYA": 800.0}))

        assert c["prop_quinua"].iloc[0] == pytest.approx(4.0)
