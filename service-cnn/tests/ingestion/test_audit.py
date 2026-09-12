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
    """4 provincias de muestra cubriendo los 4 tipos de clasificación territorial (Tabla 4)."""
    return pd.DataFrame(
        [
            {"provincia_id": "PUN-AZA", "departamento": "Puno", "clasificacion": "altoandina"},
            {"provincia_id": "JUN-SAT", "departamento": "Junín", "clasificacion": "selva"},
            {"provincia_id": "ARE-CAM", "departamento": "Arequipa", "clasificacion": "costa_riego"},
            {
                "provincia_id": "LAL-GCH",
                "departamento": "La Libertad",
                "clasificacion": "pendiente_verificacion",
            },
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
        # De 4 provincias, 1 es selva -> N1 = 3 * 19
        assert cascada["N1"] == 3 * n_campanas

    def test_n2_excluye_ademas_costa_bajo_riego(self, provincias_ocho_departamentos):
        n_campanas = 19
        cascada = build_coverage_cascade(
            provincias=provincias_ocho_departamentos,
            n_campanas=n_campanas,
            produccion_documentada=None,
            celdas_con_produccion=None,
            celdas_con_calidad=None,
        )
        # De 4 provincias: 1 selva + 1 costa_riego + 1 pendiente_verificacion
        # excluidas -> N2 = 1 * 19 (solo la altoandina)
        assert cascada["N2"] == 1 * n_campanas

    def test_n2_excluye_pendiente_verificacion(self, provincias_ocho_departamentos):
        """Sección 4.5.3: una provincia marcada como 'requiere verificación' y
        NO contada dentro de las 'preliminares de sierra' de su departamento
        (ej. Gran Chimú en La Libertad, Tabla 3) debe excluirse de N2 al
        igual que selva/costa — a diferencia de 'transicion' (Carabaya,
        Sandia, La Mar, Huanta), que sí cuenta dentro del total preliminar de
        su departamento y por tanto permanece en N2."""
        n_campanas = 19
        cascada = build_coverage_cascade(
            provincias=provincias_ocho_departamentos,
            n_campanas=n_campanas,
            produccion_documentada=None,
            celdas_con_produccion=None,
            celdas_con_calidad=None,
        )
        assert cascada["N2"] == 1 * n_campanas  # LAL-GCH no debe contarse

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

    def test_n4_excluye_celdas_de_provincias_no_elegibles_en_n3(
        self, provincias_ocho_departamentos, produccion_documentada
    ):
        """N4 debe ser <= N3 (sección 4.5.6, cascada de exclusiones sucesivas):
        una celda con producción > 0 en una provincia que ya fue excluida en
        N1/N2/N3 (selva, costa, pendiente_verificacion, o sin producción
        documentada) NO debe contarse en N4, aunque el archivo de origen
        registre producción ahí — es un caso real detectado al ejecutar la
        auditoría contra el archivo completo de MIDAGRI (N4 salió mayor que
        N3 antes de esta corrección)."""
        celdas_con_produccion = pd.DataFrame(
            [
                # PUN-AZA: elegible (altoandina, con producción documentada)
                {"provincia_id": "PUN-AZA", "campana_id": 2006, "produccion_ton": 100.0},
                # ARE-CAM: NO elegible (costa_riego, excluida desde N1/N2)
                {"provincia_id": "ARE-CAM", "campana_id": 2006, "produccion_ton": 50.0},
                # LAL-GCH: NO elegible (pendiente_verificacion, excluida desde N2)
                {"provincia_id": "LAL-GCH", "campana_id": 2006, "produccion_ton": 30.0},
            ]
        )
        cascada = build_coverage_cascade(
            provincias=provincias_ocho_departamentos,
            n_campanas=19,
            produccion_documentada=produccion_documentada,
            celdas_con_produccion=celdas_con_produccion,
            celdas_con_calidad=None,
        )
        assert cascada["N4"] == 1  # solo PUN-AZA, no ARE-CAM ni LAL-GCH
        assert cascada["N4"] <= cascada["N3"]

    def test_n4_excluye_celdas_fuera_del_rango_de_campanas_delimitado(
        self, provincias_ocho_departamentos, produccion_documentada
    ):
        """N4 debe respetar el periodo delimitado (sección 1.5/4.5): una
        celda con producción > 0 en una campaña FUERA del rango vigente (ej.
        campañas excluidas por la enmienda de periodo, ver
        data/manifest/midagri_sisagri.yaml) no debe contarse en N4, aunque el
        archivo de origen la registre — bug real detectado al ejecutar la
        auditoría contra el archivo completo de MIDAGRI (campañas 2015 y
        2026 aparecían en los datos reales pese a estar excluidas por la
        enmienda a 2016-2025)."""
        celdas_con_produccion = pd.DataFrame(
            [
                {"provincia_id": "PUN-AZA", "campana_id": 2016, "produccion_ton": 100.0},
                {"provincia_id": "PUN-AZA", "campana_id": 2015, "produccion_ton": 50.0},
                {"provincia_id": "PUN-AZA", "campana_id": 2026, "produccion_ton": 30.0},
            ]
        )
        cascada = build_coverage_cascade(
            provincias=provincias_ocho_departamentos,
            n_campanas=19,
            produccion_documentada=produccion_documentada,
            celdas_con_produccion=celdas_con_produccion,
            celdas_con_calidad=None,
            campanas_validas=set(range(2016, 2026)),
        )
        assert cascada["N4"] == 1  # solo la campaña 2016, dentro del rango

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
