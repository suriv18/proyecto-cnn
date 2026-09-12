"""Pruebas de lectura del archivo real de MIDAGRI (SISAGRI.xlsx).

Referencia: configs/column_mapping.yaml — el archivo real SÍ tiene fila de
cabecera con nombres reales (AÑO, MES, DEPARTAMENTO, PROVINCIA, DISTRITO,
PRODUCTO, SIEMBRA, COSECHA, PRODUCCION, ...), confirmados contra el XML
crudo del archivo. Reporta MES calendario, no campaña agrícola directamente
— estas funciones derivan campana_id (sección 4.4) y adaptan las columnas al
esquema que espera `load_midagri_production`.
"""
import pandas as pd
import pytest

from src.ingestion.midagri_headerless import (
    SisagriHeaderlessConfig,
    derive_campana_column,
    load_sisagri_headerless,
    load_sisagri_headerless_config,
    select_and_rename_columns,
)


class TestSelectAndRenameColumns:
    def test_selecciona_y_renombra_columnas_por_nombre(self):
        crudo = pd.DataFrame(
            {
                "AÑO": [2025],
                "MES": [12],
                "DEPARTAMENTO": ["APURIMAC"],
                "PROVINCIA": ["CHINCHEROS"],
                "PRODUCTO": ["QUINUA"],
                "SIEMBRA": [0.0],
                "COSECHA": [360.0],
                "PRODUCCION": [1530.0],
                "COD_UBIGEO": ["030609"],  # columna no mapeada, debe descartarse
            }
        )
        mapeo = {
            "AÑO": "col_anio",
            "MES": "col_mes",
            "DEPARTAMENTO": "departamento",
            "PROVINCIA": "provincia",
            "PRODUCTO": "cultivo",
            "SIEMBRA": "superficie_sembrada_ha",
            "COSECHA": "superficie_cosechada_ha",
            "PRODUCCION": "produccion_ton",
        }
        resultado = select_and_rename_columns(crudo, mapeo)
        assert set(resultado.columns) == set(mapeo.values())
        assert resultado.iloc[0]["cultivo"] == "QUINUA"

    def test_falla_si_una_columna_del_mapeo_no_existe(self):
        crudo = pd.DataFrame({"A": [1], "B": [2]})
        with pytest.raises(ValueError, match="columna"):
            select_and_rename_columns(crudo, {"A": "col_a", "C": "col_c"})


class TestDeriveCampanaColumn:
    def test_agrega_columna_campana_id_desde_anio_y_mes(self):
        datos = pd.DataFrame({"col_anio": [2025, 2026], "col_mes": [12, 1]})
        resultado = derive_campana_column(
            datos, columna_anio="col_anio", columna_mes="col_mes", nombre_salida="campana_id"
        )
        assert list(resultado["campana_id"]) == [2026, 2026]

    def test_no_modifica_las_columnas_originales(self):
        datos = pd.DataFrame({"col_anio": [2025], "col_mes": [7]})
        resultado = derive_campana_column(
            datos, columna_anio="col_anio", columna_mes="col_mes", nombre_salida="campana_id"
        )
        assert "col_anio" in resultado.columns
        assert "col_mes" in resultado.columns


@pytest.fixture
def crudo_sisagri() -> pd.DataFrame:
    """Simula la forma real de SISAGRI.xlsx leído con header=0: columnas con
    nombre real, valores tomados de filas verificadas del archivo real."""
    return pd.DataFrame(
        {
            "AÑO": [2025, 2026, 2025],
            "MES": [12, 1, 12],
            "COD_UBIGEO": ["040520", "040520", "030609"],
            "DEPARTAMENTO": ["AREQUIPA", "AREQUIPA", "APURIMAC"],
            "PROVINCIA": ["CAYLLOMA", "CAYLLOMA", "CHINCHEROS"],
            "DISTRITO": ["MAJES", "MAJES", "ROCCHACC"],
            "COD_PRODUCTO": ["14010090000", "14010090000", "14060090000"],
            "PRODUCTO": ["QUINUA", "QUINUA", "PAPA"],
            "SIEMBRA": [0.0, 20.0, 100.0],
            "COSECHA": [360.0, 320.0, 50.0],
            "PRODUCCION": [1530.0, 1273.6, 0.0],
            "VERDE_ACTUAL": [490.0, 190.0, 0.0],
            "PRECIO_CHACRA": [6.2, 5.5, 0.0],
        }
    )


class TestLoadSisagriHeaderlessConfig:
    def test_carga_configuracion_desde_yaml(self, tmp_path):
        yaml_path = tmp_path / "column_mapping.yaml"
        yaml_path.write_text(
            """
sisagri_headerless:
  anio: "AÑO"
  mes: "MES"
  departamento: "DEPARTAMENTO"
  provincia: "PROVINCIA"
  distrito: "DISTRITO"
  cultivo: "PRODUCTO"
  superficie_sembrada_ha: "SIEMBRA"
  superficie_cosechada_ha: "COSECHA"
  produccion_ton: "PRODUCCION"
""",
            encoding="utf-8",
        )
        config = load_sisagri_headerless_config(yaml_path)
        assert isinstance(config, SisagriHeaderlessConfig)
        assert config.cultivo == "PRODUCTO"
        assert config.produccion_ton == "PRODUCCION"


class TestLoadSisagriHeaderless:
    def test_filtra_quinua_y_deriva_campana_de_punta_a_punta(self, crudo_sisagri):
        config = SisagriHeaderlessConfig(
            anio="AÑO", mes="MES", departamento="DEPARTAMENTO", provincia="PROVINCIA",
            distrito="DISTRITO", cultivo="PRODUCTO", superficie_sembrada_ha="SIEMBRA",
            superficie_cosechada_ha="COSECHA", produccion_ton="PRODUCCION",
        )
        resultado = load_sisagri_headerless(crudo_sisagri, config)

        assert len(resultado) == 2  # las 2 filas de QUINUA, PAPA excluida
        assert set(resultado["provincia"]) == {"CAYLLOMA"}
        # dic-2025 (mes 12) -> campaña 2026; ene-2026 (mes 1) -> campaña 2026
        assert list(resultado["campana_id"]) == [2026, 2026]

    def test_rendimiento_se_reconstruye_desde_produccion_y_cosecha(self, crudo_sisagri):
        config = SisagriHeaderlessConfig(
            anio="AÑO", mes="MES", departamento="DEPARTAMENTO", provincia="PROVINCIA",
            distrito="DISTRITO", cultivo="PRODUCTO", superficie_sembrada_ha="SIEMBRA",
            superficie_cosechada_ha="COSECHA", produccion_ton="PRODUCCION",
        )
        resultado = load_sisagri_headerless(crudo_sisagri, config)
        fila = resultado[resultado["campana_id"] == 2026].iloc[0]
        assert fila["rendimiento_kg_ha"] == pytest.approx(1530.0 * 1000.0 / 360.0)
        assert fila["rendimiento_reconstruido"] == True  # noqa: E712
