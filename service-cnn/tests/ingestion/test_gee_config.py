"""Pruebas de la configuración de colecciones de Google Earth Engine (Tabla 7).

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.9, Tabla 7.
"""
import pytest

from src.ingestion.gee_config import (
    AggregationRule,
    GeeCollectionSpec,
    get_collection_spec,
)


class TestGetCollectionSpec:
    def test_chirps_precipitacion(self):
        spec = get_collection_spec("precipitacion")
        assert spec.collection_id == "UCSB-CHG/CHIRPS/DAILY"
        assert spec.band == "precipitation"
        assert spec.spatial_resolution_m == 5566  # ~0.05 grados
        assert spec.temporal_resolution == "diaria"
        assert spec.aggregation == AggregationRule.SUM
        assert spec.factor_escala == 1.0
        assert spec.offset_aditivo == 0.0

    def test_era5_land_temperatura_maxima(self):
        # ECMWF/ERA5_LAND/HOURLY no tiene banda temperature_2m_max (solo
        # temperature_2m horario); la variante DAILY_AGGR sí la agrega
        # nativamente por día — verificado contra la API real (cnn-sentinel).
        # ERA5-Land reporta temperatura en Kelvin: offset_aditivo la lleva a
        # Celsius (valor_real = crudo * factor_escala + offset_aditivo).
        spec = get_collection_spec("temperatura_maxima")
        assert spec.collection_id == "ECMWF/ERA5_LAND/DAILY_AGGR"
        assert spec.band == "temperature_2m_max"
        assert spec.aggregation == AggregationRule.MAX
        assert spec.factor_escala == 1.0
        assert spec.offset_aditivo == -273.15

    def test_era5_land_temperatura_minima(self):
        spec = get_collection_spec("temperatura_minima")
        assert spec.collection_id == "ECMWF/ERA5_LAND/DAILY_AGGR"
        assert spec.band == "temperature_2m_min"
        assert spec.aggregation == AggregationRule.MIN
        assert spec.offset_aditivo == -273.15

    def test_era5_land_radiacion(self):
        spec = get_collection_spec("radiacion_solar")
        assert spec.collection_id == "ECMWF/ERA5_LAND/DAILY_AGGR"
        assert spec.band == "surface_solar_radiation_downwards_sum"
        assert spec.aggregation == AggregationRule.SUM
        assert spec.factor_escala == 1.0
        assert spec.offset_aditivo == 0.0

    def test_modis_ndvi(self):
        spec = get_collection_spec("ndvi")
        assert spec.collection_id == "MODIS/061/MOD13Q1"
        assert spec.spatial_resolution_m == 250
        assert spec.temporal_resolution == "16 dias"
        # Tabla 7: "media y máximo por fase sobre máscara agrícola"
        assert spec.aggregation == AggregationRule.MEAN
        # MOD13Q1 codifica NDVI como entero [-2000, 10000]; factor de escala
        # oficial del catálogo MODIS/USGS: valor_real = crudo * 0.0001.
        assert spec.factor_escala == 0.0001
        assert spec.offset_aditivo == 0.0

    def test_variable_desconocida_lanza_error_explicito(self):
        with pytest.raises(KeyError, match="variable_inexistente"):
            get_collection_spec("variable_inexistente")

    def test_spec_es_inmutable(self):
        spec = get_collection_spec("precipitacion")
        with pytest.raises(Exception):
            spec.collection_id = "otra-cosa"  # type: ignore[misc]
