"""Pruebas de la máscara de superficie agrícola (sección 4.10.1).

La máscara combina: capa de cobertura de suelo + modelo digital de elevación
(MDE) + criterios fenométricos de la serie NDVI. Los umbrales (altitud,
pendiente, amplitud NDVI) deben calibrarse solo con campañas de entrenamiento o
fuentes externas independientes del resultado de prueba (misma disciplina
anti-fuga que el resto del pipeline).

La máscara reduce contaminación por coberturas no agrícolas, pero NO distingue
quinua de otros cultivos — esa limitación se controla aparte con la proporción
de superficie sembrada con quinua (covariable dinámica).
"""
import pandas as pd
import pytest

from src.preprocessing.mask import (
    MaskThresholds,
    apply_agricultural_mask,
    calibrate_ndvi_amplitude_threshold,
)


@pytest.fixture
def umbrales() -> MaskThresholds:
    return MaskThresholds(
        altitud_min_m=2500,
        altitud_max_m=4200,
        pendiente_max_grados=25.0,
        amplitud_ndvi_min=0.15,
    )


@pytest.fixture
def celdas_candidatas() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "celda_id": "c1",
                "cobertura_suelo": "cultivo",
                "altitud_m": 3800,
                "pendiente_grados": 10.0,
                "amplitud_ndvi": 0.30,
            },
            {  # altitud fuera de rango (demasiado baja, ceja de selva/costa)
                "celda_id": "c2",
                "cobertura_suelo": "cultivo",
                "altitud_m": 1500,
                "pendiente_grados": 10.0,
                "amplitud_ndvi": 0.30,
            },
            {  # pendiente excesiva (terreno no cultivable de forma extensiva)
                "celda_id": "c3",
                "cobertura_suelo": "cultivo",
                "altitud_m": 3800,
                "pendiente_grados": 40.0,
                "amplitud_ndvi": 0.30,
            },
            {  # amplitud NDVI insuficiente (sin ciclo estacional marcado)
                "celda_id": "c4",
                "cobertura_suelo": "cultivo",
                "altitud_m": 3800,
                "pendiente_grados": 10.0,
                "amplitud_ndvi": 0.05,
            },
            {  # cobertura de suelo no agrícola (bosque, urbano, cuerpo de agua)
                "celda_id": "c5",
                "cobertura_suelo": "bosque",
                "altitud_m": 3800,
                "pendiente_grados": 10.0,
                "amplitud_ndvi": 0.30,
            },
        ]
    )


class TestApplyAgriculturalMask:
    def test_celda_que_cumple_todos_los_criterios_es_elegible(
        self, celdas_candidatas, umbrales
    ):
        resultado = apply_agricultural_mask(celdas_candidatas, umbrales)
        fila = resultado.set_index("celda_id").loc["c1"]
        assert fila["es_agricola"] == True  # noqa: E712

    def test_excluye_por_altitud_fuera_de_rango(self, celdas_candidatas, umbrales):
        resultado = apply_agricultural_mask(celdas_candidatas, umbrales)
        fila = resultado.set_index("celda_id").loc["c2"]
        assert fila["es_agricola"] == False  # noqa: E712
        assert "altitud" in fila["motivo_exclusion"]

    def test_excluye_por_pendiente_excesiva(self, celdas_candidatas, umbrales):
        resultado = apply_agricultural_mask(celdas_candidatas, umbrales)
        fila = resultado.set_index("celda_id").loc["c3"]
        assert fila["es_agricola"] == False  # noqa: E712
        assert "pendiente" in fila["motivo_exclusion"]

    def test_excluye_por_amplitud_ndvi_insuficiente(self, celdas_candidatas, umbrales):
        resultado = apply_agricultural_mask(celdas_candidatas, umbrales)
        fila = resultado.set_index("celda_id").loc["c4"]
        assert fila["es_agricola"] == False  # noqa: E712
        assert "amplitud_ndvi" in fila["motivo_exclusion"]

    def test_excluye_por_cobertura_de_suelo_no_agricola(
        self, celdas_candidatas, umbrales
    ):
        resultado = apply_agricultural_mask(celdas_candidatas, umbrales)
        fila = resultado.set_index("celda_id").loc["c5"]
        assert fila["es_agricola"] == False  # noqa: E712
        assert "cobertura_suelo" in fila["motivo_exclusion"]

    def test_celda_elegible_no_tiene_motivo_de_exclusion(
        self, celdas_candidatas, umbrales
    ):
        resultado = apply_agricultural_mask(celdas_candidatas, umbrales)
        fila = resultado.set_index("celda_id").loc["c1"]
        assert fila["motivo_exclusion"] == ""

    def test_registra_todos_los_motivos_si_una_celda_falla_varios_criterios(
        self, umbrales
    ):
        """La máscara no distingue quinua de otros cultivos (limitación
        declarada en 4.10.1), pero sí debe ser transparente sobre TODAS las
        razones de exclusión de una celda, no solo la primera encontrada —
        necesario para el análisis de sensibilidad por umbral."""
        celda_multiple_falla = pd.DataFrame(
            [
                {
                    "celda_id": "c6",
                    "cobertura_suelo": "cultivo",
                    "altitud_m": 1000,  # falla altitud
                    "pendiente_grados": 45.0,  # falla pendiente
                    "amplitud_ndvi": 0.30,
                }
            ]
        )
        resultado = apply_agricultural_mask(celda_multiple_falla, umbrales)
        motivo = resultado.iloc[0]["motivo_exclusion"]
        assert "altitud" in motivo
        assert "pendiente" in motivo


class TestCalibrateNdviAmplitudeThreshold:
    def test_calibra_solo_con_campanas_de_entrenamiento(self):
        """Sección 4.10.1: los umbrales se calibran únicamente con campañas de
        entrenamiento o fuentes externas independientes del resultado de
        prueba — esta función no debe aceptar ni usar información de prueba."""
        serie_ndvi_entrenamiento = pd.DataFrame(
            [
                {"celda_id": "c1", "campana_id": 2006, "ndvi_max": 0.6, "ndvi_min": 0.2},
                {"celda_id": "c1", "campana_id": 2007, "ndvi_max": 0.5, "ndvi_min": 0.1},
                {"celda_id": "c2", "campana_id": 2006, "ndvi_max": 0.9, "ndvi_min": 0.8},
            ]
        )
        umbral = calibrate_ndvi_amplitude_threshold(
            serie_ndvi_entrenamiento, percentil=10
        )
        amplitudes = serie_ndvi_entrenamiento["ndvi_max"] - serie_ndvi_entrenamiento["ndvi_min"]
        assert umbral == pytest.approx(amplitudes.quantile(0.10))

    def test_umbral_es_positivo_y_finito(self):
        serie_ndvi_entrenamiento = pd.DataFrame(
            [
                {"celda_id": "c1", "campana_id": 2006, "ndvi_max": 0.6, "ndvi_min": 0.2},
                {"celda_id": "c2", "campana_id": 2006, "ndvi_max": 0.9, "ndvi_min": 0.8},
            ]
        )
        umbral = calibrate_ndvi_amplitude_threshold(
            serie_ndvi_entrenamiento, percentil=10
        )
        assert umbral > 0
        assert umbral < float("inf")
