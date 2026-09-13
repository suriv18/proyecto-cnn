"""Pruebas del orquestador de alineación fenológica para todas las
provincia-campañas del conjunto de datos.

Referencia: sección 4.10.2. Conecta `preprocessing.phenology.
estimate_sowing_date` + `build_phase_windows` (ya probados de forma aislada)
con las fuentes de datos reales: `midagri_siembra_mensual` (nivel primario) y
`data/raw/gee_series_crudo.parquet` (NDVI, nivel secundario) — ver
`gee_extraction_pipeline`. El nivel de respaldo (fecha modal departamental)
se recibe ya calculado por el llamador: depende de qué campañas son
"entrenamiento" en cada pliegue de validación (`evaluation.splits`), lógica
que este módulo no debe conocer para mantener la separación de
responsabilidades ya establecida en el proyecto.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.preprocessing.phenology import FuenteAlineacion
from src.preprocessing.phenology_pipeline import (
    build_all_phase_windows,
    estimate_departmental_modal_dates_exploratory,
    load_fenologia_bbch,
)


class TestBuildAllPhaseWindows:
    @pytest.fixture
    def siembra_mensual(self):
        return pd.DataFrame(
            [
                {"provincia_id": "AZANGARO", "campana_id": 2021, "mes": 9, "superficie_ha": 50.0},
                {"provincia_id": "AZANGARO", "campana_id": 2021, "mes": 10, "superficie_ha": 300.0},
            ]
        )

    @pytest.fixture
    def serie_ndvi_cruda(self):
        # Formato de gee_series_crudo.parquet: provincia_id, fecha, valor,
        # variable, campana_id
        return pd.DataFrame(
            [
                {
                    "provincia_id": "LAMPA",
                    "fecha": "2021-09-01",
                    "valor": 0.1,
                    "variable": "ndvi",
                    "campana_id": 2021,
                },
                {
                    "provincia_id": "LAMPA",
                    "fecha": "2021-10-03",
                    "valor": 0.35,
                    "variable": "ndvi",
                    "campana_id": 2021,
                },
            ]
        )

    @pytest.fixture
    def duracion_dias_por_fase(self):
        return {
            "emergencia": 15,
            "desarrollo_vegetativo": 45,
            "floracion": 20,
            "llenado_grano": 30,
            "madurez": 20,
        }

    def test_usa_nivel_primario_cuando_hay_siembra_mensual(
        self, siembra_mensual, serie_ndvi_cruda, duracion_dias_por_fase
    ):
        resultado = build_all_phase_windows(
            provincia_campanas=[("AZANGARO", 2021)],
            siembra_mensual=siembra_mensual,
            serie_ndvi_cruda=serie_ndvi_cruda,
            fechas_modales_departamentales={},
            duracion_dias_por_fase=duracion_dias_por_fase,
        )

        assert len(resultado) == 1
        entrada = resultado[0]
        assert entrada["provincia_id"] == "AZANGARO"
        assert entrada["campana_id"] == 2021
        assert entrada["fuente"] == FuenteAlineacion.PRIMARIO
        # octubre pertenece a la segunda mitad del año agrícola (sección 4.4):
        # campaña 2021 (año de cosecha) -> octubre 2020 (año de siembra real)
        assert entrada["fecha_siembra"] == date(2020, 10, 16)
        assert len(entrada["ventanas"]) == 5

    def test_usa_nivel_secundario_via_ndvi_cuando_no_hay_siembra_mensual(
        self, serie_ndvi_cruda, duracion_dias_por_fase
    ):
        resultado = build_all_phase_windows(
            provincia_campanas=[("LAMPA", 2021)],
            siembra_mensual=pd.DataFrame(
                columns=["provincia_id", "campana_id", "mes", "superficie_ha"]
            ),
            serie_ndvi_cruda=serie_ndvi_cruda,
            fechas_modales_departamentales={},
            duracion_dias_por_fase=duracion_dias_por_fase,
            umbral_ascenso_ndvi=0.3,
        )

        assert len(resultado) == 1
        entrada = resultado[0]
        assert entrada["fuente"] == FuenteAlineacion.SECUNDARIO
        assert entrada["fecha_siembra"] == date(2021, 10, 3)

    def test_usa_respaldo_departamental_cuando_no_hay_primario_ni_secundario(
        self, duracion_dias_por_fase
    ):
        resultado = build_all_phase_windows(
            provincia_campanas=[("CHUCUITO", 2021)],
            siembra_mensual=pd.DataFrame(
                columns=["provincia_id", "campana_id", "mes", "superficie_ha"]
            ),
            serie_ndvi_cruda=pd.DataFrame(
                columns=["provincia_id", "fecha", "valor", "variable", "campana_id"]
            ),
            fechas_modales_departamentales={"PUNO": (10, 15)},
            duracion_dias_por_fase=duracion_dias_por_fase,
            provincia_a_departamento={"CHUCUITO": "PUNO"},
        )

        assert len(resultado) == 1
        entrada = resultado[0]
        assert entrada["fuente"] == FuenteAlineacion.RESPALDO
        # octubre pertenece a la segunda mitad del año agrícola (sección 4.4):
        # campaña 2021 -> octubre 2020 (año de siembra, no de cosecha)
        assert entrada["fecha_siembra"] == date(2020, 10, 15)
        assert entrada["es_imputado"] is True

    def test_respaldo_usa_el_anio_de_la_campana_en_curso_no_uno_fijo(
        self, duracion_dias_por_fase
    ):
        """Bug real detectado: la fecha de respaldo departamental debe
        combinarse con el año de CADA campaña que cae al nivel de respaldo,
        nunca con un año fijo — de lo contrario una campaña 2017 podría
        terminar con una fecha de siembra en, por ejemplo, 2026 (el año
        máximo histórico del departamento). Octubre pertenece a la segunda
        mitad del año agrícola (sección 4.4): campaña 2017 -> octubre 2016;
        campaña 2023 -> octubre 2022."""
        resultado = build_all_phase_windows(
            provincia_campanas=[("CHUCUITO", 2017), ("CHUCUITO", 2023)],
            siembra_mensual=pd.DataFrame(
                columns=["provincia_id", "campana_id", "mes", "superficie_ha"]
            ),
            serie_ndvi_cruda=pd.DataFrame(
                columns=["provincia_id", "fecha", "valor", "variable", "campana_id"]
            ),
            fechas_modales_departamentales={"PUNO": (10, 15)},
            duracion_dias_por_fase=duracion_dias_por_fase,
            provincia_a_departamento={"CHUCUITO": "PUNO"},
        )

        fechas = {r["campana_id"]: r["fecha_siembra"] for r in resultado}
        assert fechas[2017] == date(2016, 10, 15)
        assert fechas[2023] == date(2022, 10, 15)

    def test_procesa_varias_provincia_campanas_independientemente(
        self, siembra_mensual, serie_ndvi_cruda, duracion_dias_por_fase
    ):
        resultado = build_all_phase_windows(
            provincia_campanas=[("AZANGARO", 2021), ("LAMPA", 2021)],
            siembra_mensual=siembra_mensual,
            serie_ndvi_cruda=serie_ndvi_cruda,
            fechas_modales_departamentales={},
            duracion_dias_por_fase=duracion_dias_por_fase,
        )

        assert len(resultado) == 2
        provincias_resultado = {r["provincia_id"] for r in resultado}
        assert provincias_resultado == {"AZANGARO", "LAMPA"}

    def test_registra_fallo_explicito_si_ningun_nivel_esta_disponible(
        self, duracion_dias_por_fase
    ):
        """Sin dato primario, secundario ni de respaldo, no se debe inventar
        una fecha: se registra como fallo explícito (no se propaga la
        excepción para no descartar el resto del lote — sección 4.5.2)."""
        resultado = build_all_phase_windows(
            provincia_campanas=[("CHUCUITO", 2021)],
            siembra_mensual=pd.DataFrame(
                columns=["provincia_id", "campana_id", "mes", "superficie_ha"]
            ),
            serie_ndvi_cruda=pd.DataFrame(
                columns=["provincia_id", "fecha", "valor", "variable", "campana_id"]
            ),
            fechas_modales_departamentales={},
            duracion_dias_por_fase=duracion_dias_por_fase,
        )

        assert len(resultado) == 1
        assert resultado[0]["error"] is not None
        assert resultado[0]["provincia_id"] == "CHUCUITO"

    def test_aplica_fecha_de_corte_a_las_ventanas(
        self, siembra_mensual, serie_ndvi_cruda, duracion_dias_por_fase
    ):
        # Siembra estimada: 2020-10-16 (octubre pertenece al año de siembra
        # real, anterior al año de cosecha 2021 que identifica la campaña).
        resultado = build_all_phase_windows(
            provincia_campanas=[("AZANGARO", 2021)],
            siembra_mensual=siembra_mensual,
            serie_ndvi_cruda=serie_ndvi_cruda,
            fechas_modales_departamentales={},
            duracion_dias_por_fase=duracion_dias_por_fase,
            fecha_corte=date(2020, 11, 10),
        )
        entrada = resultado[0]
        assert entrada["ventanas"][-1].fin == date(2020, 11, 10)


class TestLoadFenologiaBbch:
    def test_carga_las_5_fases_en_orden_desde_configs(self, tmp_path):
        ruta = tmp_path / "hyperparams.yaml"
        ruta.write_text(
            "fenologia_bbch:\n"
            "  emergencia: 3\n"
            "  desarrollo_vegetativo: 77\n"
            "  floracion: 17\n"
            "  llenado_grano: 44\n"
            "  madurez: 49\n",
            encoding="utf-8",
        )

        resultado = load_fenologia_bbch(ruta)

        assert list(resultado.keys()) == [
            "emergencia",
            "desarrollo_vegetativo",
            "floracion",
            "llenado_grano",
            "madurez",
        ]
        assert resultado["emergencia"] == 3
        assert sum(resultado.values()) == 190

    def test_carga_desde_el_archivo_real_del_proyecto(self):
        """configs/hyperparams.yaml debe tener la sección fenologia_bbch
        (calibrada con INIA Perú, Pérez Ávila 2005) ya en el repositorio."""
        resultado = load_fenologia_bbch("configs/hyperparams.yaml")
        assert set(resultado.keys()) == {
            "emergencia",
            "desarrollo_vegetativo",
            "floracion",
            "llenado_grano",
            "madurez",
        }


class TestEstimateDepartmentalModalDatesExploratory:
    """Aproximación EXPLORATORIA (no válida para H2/evaluación definitiva —
    violaría la regla anti-fuga temporal si mezcla campañas de prueba en el
    cálculo de respaldo de esas mismas campañas) usada solo mientras el
    esquema de validación real (evaluation.splits) no está conectado al
    pipeline de alineación fenológica."""

    def test_calcula_fecha_modal_por_departamento_desde_siembra_mensual(self):
        siembra_mensual = pd.DataFrame(
            [
                {"provincia_id": "AZANGARO", "campana_id": 2020, "mes": 10, "superficie_ha": 300.0},
                {"provincia_id": "LAMPA", "campana_id": 2021, "mes": 10, "superficie_ha": 200.0},
                {"provincia_id": "LAMPA", "campana_id": 2022, "mes": 11, "superficie_ha": 50.0},
            ]
        )
        provincia_a_departamento = {
            "AZANGARO": "PUNO",
            "LAMPA": "PUNO",
        }

        resultado = estimate_departmental_modal_dates_exploratory(
            siembra_mensual=siembra_mensual,
            provincia_a_departamento=provincia_a_departamento,
        )

        # mes modal de PUNO: octubre aparece 2 veces (300+200 ha), noviembre 1 vez (50 ha)
        # Se devuelve (mes, dia) SIN año fijo — el año lo aporta la campaña
        # real que se está alineando (build_all_phase_windows), nunca un año
        # histórico arbitrario del departamento (bug real detectado: usar el
        # año máximo histórico producía fechas de respaldo con el año
        # equivocado para campañas más antiguas, ej. "CAMANA 2017" con
        # fecha de respaldo en 2026).
        mes, dia = resultado["PUNO"]
        assert mes == 10
        assert dia == 16  # punto medio de octubre (31 días)

    def test_departamento_sin_ningun_dato_no_aparece_en_el_resultado(self):
        siembra_mensual = pd.DataFrame(
            [{"provincia_id": "AZANGARO", "campana_id": 2020, "mes": 10, "superficie_ha": 300.0}]
        )
        resultado = estimate_departmental_modal_dates_exploratory(
            siembra_mensual=siembra_mensual,
            provincia_a_departamento={"AZANGARO": "PUNO", "ABANCAY": "APURIMAC"},
        )
        assert "APURIMAC" not in resultado
        assert "PUNO" in resultado
