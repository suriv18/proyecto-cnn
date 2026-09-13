"""Pruebas de alineación fenológica (sección 4.10.2).

Jerarquía de 3 niveles: fecha modal de siembra (primario) -> fenometría NDVI
(secundario) -> fecha modal departamental de respaldo (nivel de respaldo).
A partir del inicio estimado, las fases siguen la escala BBCH de la quinua
(Sosa-Zuniga et al., 2017): emergencia, desarrollo_vegetativo, floracion,
llenado_grano, madurez.
"""
from datetime import date

import pandas as pd
import pytest

from src.preprocessing.phenology import (
    FuenteAlineacion,
    build_phase_windows,
    estimate_sowing_date,
)
from src.preprocessing.campaign_calendar import derive_campana_from_month


class TestEstimateSowingDateNivelPrimario:
    def test_usa_el_punto_medio_del_mes_de_mayor_superficie_sembrada(self):
        """Nivel primario: fecha modal de siembra con superficie sembrada
        mensual por provincia-campaña, punto medio del mes de mayor
        superficie. Octubre (mes >= 7) pertenece al año calendario anterior
        al año de cosecha que identifica la campaña (sección 4.4, misma
        convención que campaign_calendar.derive_campana_from_month aplicada
        en sentido inverso) — bug real detectado: la implementación original
        usaba `anio = campana_id` directamente, produciendo fechas de
        siembra con el año de cosecha en vez del año de siembra real."""
        siembra_mensual = pd.DataFrame(
            [
                {"provincia_id": "PUN-AZA", "campana_id": 2021, "mes": 9, "superficie_ha": 50.0},
                {"provincia_id": "PUN-AZA", "campana_id": 2021, "mes": 10, "superficie_ha": 300.0},
                {"provincia_id": "PUN-AZA", "campana_id": 2021, "mes": 11, "superficie_ha": 80.0},
            ]
        )
        resultado = estimate_sowing_date(
            provincia_id="PUN-AZA",
            campana_id=2021,
            siembra_mensual=siembra_mensual,
            serie_ndvi=None,
            fecha_modal_departamental=None,
        )
        # Octubre tiene 31 días -> punto medio es el día 16; año 2020 (año de
        # siembra real, no el año de cosecha 2021 que identifica la campaña)
        assert resultado.fecha == date(2020, 10, 16)
        assert resultado.fuente == FuenteAlineacion.PRIMARIO

    def test_mes_de_la_primera_mitad_del_anio_agricola_usa_el_anio_de_la_campana(self):
        """Un mes de siembra tardía (ene-jun, ej. campañas con siembra
        residual de verano) pertenece al mismo año calendario que
        `campana_id` — a diferencia de jul-dic, que pertenece al año
        anterior (ver test de arriba)."""
        siembra_mensual = pd.DataFrame(
            [{"provincia_id": "PUN-AZA", "campana_id": 2021, "mes": 3, "superficie_ha": 100.0}]
        )
        resultado = estimate_sowing_date(
            provincia_id="PUN-AZA",
            campana_id=2021,
            siembra_mensual=siembra_mensual,
            serie_ndvi=None,
            fecha_modal_departamental=None,
        )
        assert resultado.fecha.year == 2021
        assert resultado.fecha.month == 3

    def test_ignora_meses_sin_datos_de_la_provincia_campana_correcta(self):
        siembra_mensual = pd.DataFrame(
            [
                {"provincia_id": "PUN-AZA", "campana_id": 2021, "mes": 9, "superficie_ha": 300.0},
                {"provincia_id": "OTRA-PROV", "campana_id": 2021, "mes": 10, "superficie_ha": 999.0},
                {"provincia_id": "PUN-AZA", "campana_id": 2022, "mes": 10, "superficie_ha": 999.0},
            ]
        )
        resultado = estimate_sowing_date(
            provincia_id="PUN-AZA",
            campana_id=2021,
            siembra_mensual=siembra_mensual,
            serie_ndvi=None,
            fecha_modal_departamental=None,
        )
        assert resultado.fecha.month == 9
        assert resultado.fuente == FuenteAlineacion.PRIMARIO


class TestEstimateSowingDateNivelSecundario:
    def test_usa_fenometria_ndvi_si_no_hay_serie_mensual(self):
        """Nivel secundario: si la serie mensual no está disponible, se usa el
        inicio de estación estimado por fenometría NDVI."""
        serie_ndvi = pd.DataFrame(
            [
                {"fecha": date(2021, 9, 1), "ndvi": 0.15},
                {"fecha": date(2021, 9, 17), "ndvi": 0.16},
                {"fecha": date(2021, 10, 3), "ndvi": 0.35},  # inicio de ascenso
                {"fecha": date(2021, 10, 19), "ndvi": 0.50},
            ]
        )
        resultado = estimate_sowing_date(
            provincia_id="PUN-AZA",
            campana_id=2021,
            siembra_mensual=None,
            serie_ndvi=serie_ndvi,
            fecha_modal_departamental=None,
            umbral_ascenso_ndvi=0.3,
        )
        assert resultado.fecha == date(2021, 10, 3)
        assert resultado.fuente == FuenteAlineacion.SECUNDARIO

    def test_cae_a_respaldo_si_ndvi_nunca_cruza_el_umbral_de_ascenso(self):
        """Si la serie NDVI nunca alcanza el umbral de ascenso (ej. cobertura
        de nubes total o cultivo no detectado), el nivel secundario debe
        considerarse no disponible y caer al nivel de respaldo, no fallar ni
        devolver una fecha arbitraria."""
        serie_ndvi_sin_ascenso = pd.DataFrame(
            [
                {"fecha": date(2021, 9, 1), "ndvi": 0.10},
                {"fecha": date(2021, 9, 17), "ndvi": 0.12},
            ]
        )
        resultado = estimate_sowing_date(
            provincia_id="PUN-AZA",
            campana_id=2021,
            siembra_mensual=None,
            serie_ndvi=serie_ndvi_sin_ascenso,
            fecha_modal_departamental=date(2021, 10, 1),
            umbral_ascenso_ndvi=0.3,
        )
        assert resultado.fuente == FuenteAlineacion.RESPALDO
        assert resultado.fecha == date(2021, 10, 1)

    def test_usa_fenometria_si_la_provincia_no_tiene_registro_en_siembra_mensual(self):
        """Si la tabla de siembra mensual existe pero no tiene filas para esta
        provincia-campaña específica, debe caer también al nivel secundario."""
        siembra_mensual_vacia_para_esta_provincia = pd.DataFrame(
            [{"provincia_id": "OTRA", "campana_id": 2021, "mes": 9, "superficie_ha": 10.0}]
        )
        serie_ndvi = pd.DataFrame(
            [
                {"fecha": date(2021, 9, 1), "ndvi": 0.1},
                {"fecha": date(2021, 9, 17), "ndvi": 0.4},
            ]
        )
        resultado = estimate_sowing_date(
            provincia_id="PUN-AZA",
            campana_id=2021,
            siembra_mensual=siembra_mensual_vacia_para_esta_provincia,
            serie_ndvi=serie_ndvi,
            fecha_modal_departamental=None,
            umbral_ascenso_ndvi=0.3,
        )
        assert resultado.fuente == FuenteAlineacion.SECUNDARIO


class TestEstimateSowingDateNivelDeRespaldo:
    def test_usa_fecha_modal_departamental_si_no_hay_mensual_ni_ndvi(self):
        """Nivel de respaldo: fecha modal departamental estimada con campañas de
        entrenamiento, con indicador de imputación explícito."""
        resultado = estimate_sowing_date(
            provincia_id="PUN-AZA",
            campana_id=2021,
            siembra_mensual=None,
            serie_ndvi=None,
            fecha_modal_departamental=date(2021, 10, 1),
        )
        assert resultado.fecha == date(2021, 10, 1)
        assert resultado.fuente == FuenteAlineacion.RESPALDO
        assert resultado.es_imputado is True

    def test_niveles_primario_y_secundario_no_estan_marcados_como_imputados(self):
        siembra_mensual = pd.DataFrame(
            [{"provincia_id": "PUN-AZA", "campana_id": 2021, "mes": 9, "superficie_ha": 300.0}]
        )
        resultado = estimate_sowing_date(
            provincia_id="PUN-AZA",
            campana_id=2021,
            siembra_mensual=siembra_mensual,
            serie_ndvi=None,
            fecha_modal_departamental=None,
        )
        assert resultado.es_imputado is False

    def test_falla_explicitamente_si_ningun_nivel_tiene_datos(self):
        """Sin fecha modal departamental de respaldo tampoco disponible, no hay
        forma de estimar la siembra: debe fallar explícitamente, nunca asumir
        una fecha arbitraria (disciplina de la sección 4.5.2)."""
        with pytest.raises(ValueError, match="ning[uú]n nivel"):
            estimate_sowing_date(
                provincia_id="PUN-AZA",
                campana_id=2021,
                siembra_mensual=None,
                serie_ndvi=None,
                fecha_modal_departamental=None,
            )


class TestBuildPhaseWindows:
    def test_construye_las_cinco_fases_bbch_en_orden(self):
        """Escala BBCH de la quinua (Sosa-Zuniga et al., 2017): emergencia,
        desarrollo_vegetativo, floracion, llenado_grano, madurez."""
        ventanas = build_phase_windows(
            fecha_siembra=date(2021, 10, 1),
            duracion_dias_por_fase={
                "emergencia": 15,
                "desarrollo_vegetativo": 45,
                "floracion": 20,
                "llenado_grano": 30,
                "madurez": 20,
            },
        )
        fases = [v.fase for v in ventanas]
        assert fases == [
            "emergencia",
            "desarrollo_vegetativo",
            "floracion",
            "llenado_grano",
            "madurez",
        ]

    def test_las_ventanas_son_contiguas_sin_solapamiento(self):
        ventanas = build_phase_windows(
            fecha_siembra=date(2021, 10, 1),
            duracion_dias_por_fase={
                "emergencia": 15,
                "desarrollo_vegetativo": 45,
                "floracion": 20,
                "llenado_grano": 30,
                "madurez": 20,
            },
        )
        from datetime import timedelta

        for anterior, siguiente in zip(ventanas, ventanas[1:]):
            assert siguiente.inicio == anterior.fin + timedelta(days=1)

    def test_primera_fase_inicia_en_la_fecha_de_siembra(self):
        ventanas = build_phase_windows(
            fecha_siembra=date(2021, 10, 1),
            duracion_dias_por_fase={"emergencia": 15},
        )
        assert ventanas[0].inicio == date(2021, 10, 1)

    def test_respeta_el_horizonte_de_corte_truncando_la_ultima_fase(self):
        """Sección 4.10.4: no se incluye información posterior al punto de
        corte del horizonte de pronóstico."""
        ventanas = build_phase_windows(
            fecha_siembra=date(2021, 10, 1),
            duracion_dias_por_fase={
                "emergencia": 15,
                "desarrollo_vegetativo": 45,
                "floracion": 20,
            },
            fecha_corte=date(2021, 11, 10),
        )
        # emergencia: 10/01-10/15, desarrollo: 10/16-11/29 (truncada a 11/10),
        # floracion: excluida por completo (fuera de horizonte)
        assert len(ventanas) == 2
        assert ventanas[-1].fase == "desarrollo_vegetativo"
        assert ventanas[-1].fin == date(2021, 11, 10)

    def test_excluye_por_completo_una_fase_que_empieza_despues_del_corte(self):
        """Distingue el truncamiento (fase que cruza el corte) de la exclusión
        total (fase que empieza por completo después del corte, sección
        4.10.4) — con dos fases tras la truncada, la segunda no debe
        aparecer en absoluto en el resultado."""
        ventanas = build_phase_windows(
            fecha_siembra=date(2021, 10, 1),
            duracion_dias_por_fase={
                "emergencia": 15,
                "desarrollo_vegetativo": 45,
                "floracion": 20,
                "llenado_grano": 30,
            },
            fecha_corte=date(2021, 11, 10),
        )
        fases_presentes = [v.fase for v in ventanas]
        assert "floracion" not in fases_presentes
        assert "llenado_grano" not in fases_presentes
