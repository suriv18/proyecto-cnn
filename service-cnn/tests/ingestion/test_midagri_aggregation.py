"""Pruebas de agregación de registros mensuales de MIDAGRI a celdas provincia-campaña.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.5.6 (N4:
celdas con producción mayor que cero). El archivo real de MIDAGRI reporta un
registro por mes; una celda provincia-campaña agrega los 12 meses de esa
campaña (según la derivación de campana_id, sección 4.4).
"""
import pandas as pd
import pytest

from src.ingestion.midagri_aggregation import aggregate_to_provincia_campana


class TestAggregateToProvinciaCampana:
    def test_suma_produccion_de_varios_meses_de_la_misma_campana(self):
        """Dos registros mensuales de la misma provincia-campaña deben sumarse
        en una sola celda."""
        registros_mensuales = pd.DataFrame(
            {
                "provincia": ["CAYLLOMA", "CAYLLOMA"],
                "campana_id": [2026, 2026],
                "produccion_ton": [42.5, 15.2],
                "superficie_cosechada_ha": [10.0, 4.0],
            }
        )
        resultado = aggregate_to_provincia_campana(registros_mensuales)

        assert len(resultado) == 1
        fila = resultado.iloc[0]
        assert fila["provincia"] == "CAYLLOMA"
        assert fila["campana_id"] == 2026
        assert fila["produccion_ton"] == pytest.approx(57.7)
        assert fila["superficie_cosechada_ha"] == pytest.approx(14.0)

    def test_no_mezcla_campanas_distintas_de_la_misma_provincia(self):
        registros_mensuales = pd.DataFrame(
            {
                "provincia": ["CAYLLOMA", "CAYLLOMA"],
                "campana_id": [2025, 2026],
                "produccion_ton": [100.0, 200.0],
                "superficie_cosechada_ha": [20.0, 30.0],
            }
        )
        resultado = aggregate_to_provincia_campana(registros_mensuales)

        assert len(resultado) == 2
        assert set(resultado["campana_id"]) == {2025, 2026}

    def test_no_mezcla_provincias_distintas_de_la_misma_campana(self):
        registros_mensuales = pd.DataFrame(
            {
                "provincia": ["CAYLLOMA", "PUNO"],
                "campana_id": [2026, 2026],
                "produccion_ton": [100.0, 200.0],
                "superficie_cosechada_ha": [20.0, 30.0],
            }
        )
        resultado = aggregate_to_provincia_campana(registros_mensuales)

        assert len(resultado) == 2
        assert set(resultado["provincia"]) == {"CAYLLOMA", "PUNO"}

    def test_devuelve_dataframe_vacio_si_la_entrada_esta_vacia(self):
        vacio = pd.DataFrame(
            columns=["provincia", "campana_id", "produccion_ton", "superficie_cosechada_ha"]
        )
        resultado = aggregate_to_provincia_campana(vacio)
        assert len(resultado) == 0
