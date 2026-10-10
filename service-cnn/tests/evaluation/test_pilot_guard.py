"""Pruebas de la guarda de campañas del piloto (tarea T1 del plan del piloto).

El ensayo piloto solo puede usar las primeras `ventana_inicial` campañas: son las
de entrenamiento en CUALQUIER esquema confirmatorio con esa ventana. Una campaña
posterior será externa y no puede influir en ninguna decisión de diseño (4.12.1).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.pilot_guard import (
    CampanaExternaError,
    assert_only_allowed_campaigns,
    pilot_campaigns,
)


class TestPilotCampaigns:
    def test_son_las_primeras_campanas_segun_la_ventana(self):
        assert pilot_campaigns(primera_campana=2016, ventana_inicial=5) == [2016, 2017, 2018, 2019, 2020]

    def test_una_ventana_menor_permite_menos_campanas(self):
        assert pilot_campaigns(2016, 4) == [2016, 2017, 2018, 2019]

    def test_exige_una_ventana_positiva(self):
        with pytest.raises(ValueError, match="ventana"):
            pilot_campaigns(2016, 0)


class TestAssertOnlyAllowedCampaigns:
    def test_acepta_un_subconjunto_de_las_permitidas(self):
        assert_only_allowed_campaigns([2016, 2018], permitidas=[2016, 2017, 2018])

    def test_rechaza_una_campana_externa_y_la_nombra(self):
        with pytest.raises(CampanaExternaError, match="2021"):
            assert_only_allowed_campaigns([2018, 2021], permitidas=[2016, 2017, 2018, 2019, 2020])

    def test_nombra_todas_las_campanas_externas_ordenadas(self):
        with pytest.raises(CampanaExternaError, match=r"\[2022, 2025\]"):
            assert_only_allowed_campaigns([2025, 2016, 2022], permitidas=[2016, 2017])

    def test_acepta_enteros_de_numpy(self):
        assert_only_allowed_campaigns(np.array([2016, 2017]), permitidas=[2016, 2017, 2018])

    def test_el_contexto_aparece_en_el_mensaje(self):
        with pytest.raises(CampanaExternaError, match="constructor de pliegos"):
            assert_only_allowed_campaigns([2030], permitidas=[2016], contexto="constructor de pliegos")

    def test_una_lista_vacia_es_valida(self):
        assert_only_allowed_campaigns([], permitidas=[2016])
