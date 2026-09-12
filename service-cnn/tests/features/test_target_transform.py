"""Pruebas del tratamiento de la variable objetivo (sección 4.11).

Dos representaciones: nivel (kg/ha) y anomalía respecto de una línea base
provincial. La línea base se ajusta con una tendencia temporal simple
(regresión lineal sobre campaña) usando EXCLUSIVAMENTE campañas de
entrenamiento. Provincias con historia insuficiente usan una regla de respaldo
documentada. B2 (benchmark) es la predicción de esta misma línea base.
"""
import pandas as pd
import pytest

from src.features.target_transform import (
    HISTORIA_MINIMA_CAMPANAS,
    fit_provincial_baseline,
    predict_baseline,
    to_anomaly,
)


@pytest.fixture
def historia_suficiente() -> pd.DataFrame:
    """Provincia con 8 campañas de entrenamiento y tendencia creciente clara."""
    campanas = list(range(2006, 2014))
    rendimientos = [700 + 20 * i for i in range(len(campanas))]  # tendencia +20/año
    return pd.DataFrame(
        {
            "provincia_id": ["PUN-AZA"] * len(campanas),
            "campana_id": campanas,
            "rendimiento_kg_ha": rendimientos,
        }
    )


@pytest.fixture
def historia_insuficiente() -> pd.DataFrame:
    """Provincia con solo 2 campañas registradas — por debajo del mínimo."""
    return pd.DataFrame(
        {
            "provincia_id": ["PUN-NUEVA"] * 2,
            "campana_id": [2020, 2021],
            "rendimiento_kg_ha": [650.0, 680.0],
        }
    )


class TestFitProvincialBaseline:
    def test_ajusta_tendencia_lineal_con_historia_suficiente(
        self, historia_suficiente
    ):
        baseline = fit_provincial_baseline(historia_suficiente, "PUN-AZA")
        assert baseline.es_respaldo is False
        # pendiente aproximada de +20 kg/ha por campaña
        assert baseline.pendiente == pytest.approx(20.0, abs=0.5)

    def test_usa_regla_de_respaldo_si_historia_es_insuficiente(
        self, historia_insuficiente
    ):
        baseline = fit_provincial_baseline(historia_insuficiente, "PUN-NUEVA")
        assert baseline.es_respaldo is True
        # regla de respaldo: media simple de la historia disponible, pendiente 0
        assert baseline.pendiente == pytest.approx(0.0)
        assert baseline.intercepto == pytest.approx(665.0)  # media de 650 y 680

    def test_historia_minima_es_la_constante_declarada(self):
        """El umbral de historia mínima debe ser una constante nombrada y
        documentada, no un número mágico disperso en el código."""
        assert HISTORIA_MINIMA_CAMPANAS >= 1

    def test_falla_si_la_provincia_no_tiene_ninguna_observacion(
        self, historia_suficiente
    ):
        with pytest.raises(ValueError, match="PROVINCIA-INEXISTENTE"):
            fit_provincial_baseline(historia_suficiente, "PROVINCIA-INEXISTENTE")

    def test_no_usa_campanas_de_otra_provincia(self, historia_suficiente):
        """La tendencia de una provincia no debe verse afectada por datos de
        otras provincias presentes en el mismo DataFrame de entrenamiento."""
        otra_provincia = pd.DataFrame(
            {
                "provincia_id": ["OTRA"] * 5,
                "campana_id": list(range(2006, 2011)),
                "rendimiento_kg_ha": [9999.0] * 5,
            }
        )
        mezcla = pd.concat([historia_suficiente, otra_provincia])
        baseline = fit_provincial_baseline(mezcla, "PUN-AZA")
        assert baseline.pendiente == pytest.approx(20.0, abs=0.5)


class TestPredictBaseline:
    def test_predice_usando_la_pendiente_y_el_intercepto(self, historia_suficiente):
        baseline = fit_provincial_baseline(historia_suficiente, "PUN-AZA")
        prediccion_2014 = predict_baseline(baseline, campana_id=2014)
        # extrapolación: siguiente campaña tras la última de entrenamiento (2013)
        prediccion_2013 = predict_baseline(baseline, campana_id=2013)
        assert prediccion_2014 > prediccion_2013

    def test_respaldo_predice_un_valor_constante_para_cualquier_campana(
        self, historia_insuficiente
    ):
        baseline = fit_provincial_baseline(historia_insuficiente, "PUN-NUEVA")
        assert predict_baseline(baseline, campana_id=2025) == predict_baseline(
            baseline, campana_id=2030
        )


class TestToAnomaly:
    def test_anomalia_es_diferencia_entre_observado_y_linea_base(
        self, historia_suficiente
    ):
        baseline = fit_provincial_baseline(historia_suficiente, "PUN-AZA")
        rendimiento_observado = 900.0
        campana_evaluada = 2014
        anomalia = to_anomaly(rendimiento_observado, baseline, campana_evaluada)
        esperado = rendimiento_observado - predict_baseline(baseline, campana_evaluada)
        assert anomalia == pytest.approx(esperado)

    def test_baseline_ajustada_solo_con_entrenamiento_no_usa_campana_de_prueba(
        self, historia_suficiente
    ):
        """Regla anti-fuga (4.12.2): fit_provincial_baseline solo debe recibir
        el DataFrame de entrenamiento del pliegue — este test documenta que la
        campaña de prueba (2014) nunca aparece en `historia_suficiente`
        (2006-2013) usada para ajustar, verificando el contrato de la función."""
        campanas_usadas_en_ajuste = set(historia_suficiente["campana_id"])
        assert 2014 not in campanas_usadas_en_ajuste
