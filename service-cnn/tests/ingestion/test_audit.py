"""Pruebas de la auditoría de cobertura provincia-campaña (Actividad 1 / Hito H1).

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, secciones 4.5-4.5.7 y Tabla 5.
"""
import pandas as pd
import pytest

from src.ingestion.audit import (
    ClasificacionTerritorial,
    build_coverage_cascade,
    classify_provinces,
)


@pytest.fixture
def provincias_ocho_departamentos() -> pd.DataFrame:
    """3 provincias de muestra cubriendo los 3 tipos de clasificación territorial (Tabla 4)."""
    return pd.DataFrame(
        [
            {"provincia_id": "PUN-AZA", "departamento": "Puno", "clasificacion": "altoandina"},
            {"provincia_id": "JUN-SAT", "departamento": "Junín", "clasificacion": "selva"},
            {"provincia_id": "ARE-CAM", "departamento": "Arequipa", "clasificacion": "costa_riego"},
        ]
    )


@pytest.fixture
def produccion_documentada() -> pd.DataFrame:
    """Registro de si cada provincia altoandina tiene producción documentada (Tabla 3)."""
    return pd.DataFrame(
        [
            {"provincia_id": "PUN-AZA", "produccion_documentada": True},
        ]
    )


class TestClassifyProvinces:
    def test_marca_selva_como_excluida(self, provincias_ocho_departamentos):
        resultado = classify_provinces(provincias_ocho_departamentos)
        fila = resultado.set_index("provincia_id").loc["JUN-SAT"]
        assert fila["clasificacion_final"] == ClasificacionTerritorial.SELVA

    def test_marca_costa_bajo_riego_como_excluida(self, provincias_ocho_departamentos):
        resultado = classify_provinces(provincias_ocho_departamentos)
        fila = resultado.set_index("provincia_id").loc["ARE-CAM"]
        assert fila["clasificacion_final"] == ClasificacionTerritorial.COSTA_RIEGO

    def test_marca_altoandina_como_elegible(self, provincias_ocho_departamentos):
        resultado = classify_provinces(provincias_ocho_departamentos)
        fila = resultado.set_index("provincia_id").loc["PUN-AZA"]
        assert fila["clasificacion_final"] == ClasificacionTerritorial.ALTOANDINA


class TestCoverageCascade:
    """Reproduce la cascada N0-N5 de la Tabla 5 con un universo de prueba pequeño."""

    def test_n0_es_provincias_totales_por_campanas(
        self, provincias_ocho_departamentos
    ):
        n_campanas = 19
        cascada = build_coverage_cascade(
            provincias=provincias_ocho_departamentos,
            n_campanas=n_campanas,
            produccion_documentada=None,
            celdas_con_produccion=None,
            celdas_con_calidad=None,
        )
        assert cascada["N0"] == len(provincias_ocho_departamentos) * n_campanas

    def test_n1_excluye_selva(self, provincias_ocho_departamentos):
        n_campanas = 19
        cascada = build_coverage_cascade(
            provincias=provincias_ocho_departamentos,
            n_campanas=n_campanas,
            produccion_documentada=None,
            celdas_con_produccion=None,
            celdas_con_calidad=None,
        )
        # De 3 provincias, 1 es selva -> N1 = 2 * 19
        assert cascada["N1"] == 2 * n_campanas

    def test_n2_excluye_ademas_costa_bajo_riego(self, provincias_ocho_departamentos):
        n_campanas = 19
        cascada = build_coverage_cascade(
            provincias=provincias_ocho_departamentos,
            n_campanas=n_campanas,
            produccion_documentada=None,
            celdas_con_produccion=None,
            celdas_con_calidad=None,
        )
        # De 3 provincias, 1 selva + 1 costa_riego -> N2 = 1 * 19
        assert cascada["N2"] == 1 * n_campanas

    def test_n3_requiere_produccion_documentada(
        self, provincias_ocho_departamentos, produccion_documentada
    ):
        n_campanas = 19
        cascada = build_coverage_cascade(
            provincias=provincias_ocho_departamentos,
            n_campanas=n_campanas,
            produccion_documentada=produccion_documentada,
            celdas_con_produccion=None,
            celdas_con_calidad=None,
        )
        assert cascada["N3"] == 1 * n_campanas

    def test_n4_y_n5_son_none_si_no_hay_datos_reales(
        self, provincias_ocho_departamentos, produccion_documentada
    ):
        """N4/N5 no deben estimarse con tasas supuestas (sección 4.5.6): si no hay
        datos reales de celdas, deben quedar explícitamente pendientes, nunca en cero
        ni con un valor inventado."""
        cascada = build_coverage_cascade(
            provincias=provincias_ocho_departamentos,
            n_campanas=19,
            produccion_documentada=produccion_documentada,
            celdas_con_produccion=None,
            celdas_con_calidad=None,
        )
        assert cascada["N4"] is None
        assert cascada["N5"] is None

    def test_n4_cuenta_celdas_con_produccion_mayor_a_cero(
        self, provincias_ocho_departamentos, produccion_documentada
    ):
        celdas_con_produccion = pd.DataFrame(
            [
                {"provincia_id": "PUN-AZA", "campana_id": 2006, "produccion_ton": 100.0},
                {"provincia_id": "PUN-AZA", "campana_id": 2007, "produccion_ton": 0.0},
            ]
        )
        cascada = build_coverage_cascade(
            provincias=provincias_ocho_departamentos,
            n_campanas=19,
            produccion_documentada=produccion_documentada,
            celdas_con_produccion=celdas_con_produccion,
            celdas_con_calidad=None,
        )
        assert cascada["N4"] == 1  # solo la celda con producción > 0

    def test_monotonia_no_creciente_de_la_cascada(
        self, provincias_ocho_departamentos, produccion_documentada
    ):
        """Cada nivel de la cascada debe ser <= al anterior (son exclusiones sucesivas)."""
        celdas_con_produccion = pd.DataFrame(
            [{"provincia_id": "PUN-AZA", "campana_id": 2006, "produccion_ton": 100.0}]
        )
        celdas_con_calidad = pd.DataFrame(
            [{"provincia_id": "PUN-AZA", "campana_id": 2006, "cumple_calidad": True}]
        )
        cascada = build_coverage_cascade(
            provincias=provincias_ocho_departamentos,
            n_campanas=19,
            produccion_documentada=produccion_documentada,
            celdas_con_produccion=celdas_con_produccion,
            celdas_con_calidad=celdas_con_calidad,
        )
        niveles = ["N0", "N1", "N2", "N3", "N4", "N5"]
        valores = [cascada[n] for n in niveles]
        for anterior, siguiente in zip(valores, valores[1:]):
            assert siguiente <= anterior
