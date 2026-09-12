"""Pruebas del cargador configurable de datos MIDAGRI/SIEA (sección 4.9, Tabla 7).

El esquema REAL del archivo público de datosabiertos.gob.pe no pudo verificarse
programáticamente (login/JS bloquean el acceso automatizado). Por eso el cargador
NO asume nombres de columna fijos: recibe un mapeo configurable que se ajusta una
vez que el tesista descarga y confirma manualmente el archivo real durante H1.
"""
import pandas as pd
import pytest

from src.ingestion.midagri_loader import (
    ColumnMappingError,
    MidagriColumnMapping,
    load_column_mapping,
    load_midagri_production,
)


@pytest.fixture
def mapeo_columnas() -> MidagriColumnMapping:
    """Ejemplo de mapeo: nombres de columna reales -> esquema interno estándar."""
    return MidagriColumnMapping(
        departamento="DEPARTAMENTO",
        provincia="PROVINCIA",
        campana_id="ANIO_COSECHA",
        cultivo="CULTIVO",
        superficie_sembrada_ha="SUPERFICIE_SEMBRADA",
        superficie_cosechada_ha="SUPERFICIE_COSECHADA",
        produccion_ton="PRODUCCION",
        rendimiento_kg_ha="RENDIMIENTO",
    )


@pytest.fixture
def archivo_crudo_simulado() -> pd.DataFrame:
    """Simula la forma esperada de un extracto crudo de MIDAGRI con columnas
    arbitrarias (mayúsculas, en español, sin normalizar) y múltiples cultivos."""
    return pd.DataFrame(
        [
            {
                "DEPARTAMENTO": "Puno",
                "PROVINCIA": "Azángaro",
                "ANIO_COSECHA": 2020,
                "CULTIVO": "Quinua",
                "SUPERFICIE_SEMBRADA": 500.0,
                "SUPERFICIE_COSECHADA": 480.0,
                "PRODUCCION": 350.0,
                "RENDIMIENTO": 729.0,
            },
            {
                "DEPARTAMENTO": "Puno",
                "PROVINCIA": "Azángaro",
                "ANIO_COSECHA": 2020,
                "CULTIVO": "Papa",
                "SUPERFICIE_SEMBRADA": 1000.0,
                "SUPERFICIE_COSECHADA": 950.0,
                "PRODUCCION": 8000.0,
                "RENDIMIENTO": 8421.0,
            },
            {
                "DEPARTAMENTO": "Ayacucho",
                "PROVINCIA": "Huamanga",
                "ANIO_COSECHA": 2020,
                "CULTIVO": "QUINUA (grano seco)",
                "SUPERFICIE_SEMBRADA": 200.0,
                "SUPERFICIE_COSECHADA": 190.0,
                "PRODUCCION": 120.0,
                "RENDIMIENTO": 631.0,
            },
        ]
    )


class TestLoadMidagriProduction:
    def test_filtra_solo_registros_de_quinua_con_variantes_de_nombre(
        self, archivo_crudo_simulado, mapeo_columnas
    ):
        resultado = load_midagri_production(archivo_crudo_simulado, mapeo_columnas)
        assert len(resultado) == 2
        assert set(resultado["provincia"]) == {"Azángaro", "Huamanga"}

    def test_normaliza_nombres_de_columna_al_esquema_interno(
        self, archivo_crudo_simulado, mapeo_columnas
    ):
        resultado = load_midagri_production(archivo_crudo_simulado, mapeo_columnas)
        columnas_esperadas = {
            "departamento",
            "provincia",
            "campana_id",
            "cultivo",
            "superficie_sembrada_ha",
            "superficie_cosechada_ha",
            "produccion_ton",
            "rendimiento_kg_ha",
        }
        assert columnas_esperadas.issubset(set(resultado.columns))

    def test_preserva_valores_numericos_correctos(
        self, archivo_crudo_simulado, mapeo_columnas
    ):
        resultado = load_midagri_production(archivo_crudo_simulado, mapeo_columnas)
        fila = resultado[resultado["provincia"] == "Azángaro"].iloc[0]
        assert fila["produccion_ton"] == 350.0
        assert fila["rendimiento_kg_ha"] == 729.0

    def test_falla_explicitamente_si_falta_una_columna_mapeada(
        self, archivo_crudo_simulado, mapeo_columnas
    ):
        """El diseño anti-fuga y de trazabilidad de la tesis exige nunca continuar
        silenciosamente con datos incompletos (sección 4.5.2, 4.7): si el mapeo
        configurado no coincide con el archivo real, debe fallar con un mensaje
        claro, no producir columnas vacías o NaN silenciosos."""
        archivo_incompleto = archivo_crudo_simulado.drop(columns=["RENDIMIENTO"])
        with pytest.raises(ColumnMappingError, match="RENDIMIENTO"):
            load_midagri_production(archivo_incompleto, mapeo_columnas)

    def test_reconstruye_rendimiento_si_falta_pero_hay_produccion_y_superficie(
        self, mapeo_columnas
    ):
        """Sección 4.5.2: si el rendimiento no está publicado, se reconstruye como
        producción / superficie cosechada, siempre que ambas sean consistentes."""
        sin_rendimiento_explicito = pd.DataFrame(
            [
                {
                    "DEPARTAMENTO": "Puno",
                    "PROVINCIA": "Azángaro",
                    "ANIO_COSECHA": 2020,
                    "CULTIVO": "Quinua",
                    "SUPERFICIE_SEMBRADA": 500.0,
                    "SUPERFICIE_COSECHADA": 500.0,
                    "PRODUCCION": 400.0,
                }
            ]
        )
        mapeo_sin_rendimiento = MidagriColumnMapping(
            departamento="DEPARTAMENTO",
            provincia="PROVINCIA",
            campana_id="ANIO_COSECHA",
            cultivo="CULTIVO",
            superficie_sembrada_ha="SUPERFICIE_SEMBRADA",
            superficie_cosechada_ha="SUPERFICIE_COSECHADA",
            produccion_ton="PRODUCCION",
            rendimiento_kg_ha=None,
        )
        resultado = load_midagri_production(
            sin_rendimiento_explicito, mapeo_sin_rendimiento
        )
        # 400 ton / 500 ha = 0.8 ton/ha = 800 kg/ha
        assert resultado.iloc[0]["rendimiento_kg_ha"] == pytest.approx(800.0)
        assert resultado.iloc[0]["rendimiento_reconstruido"] == True  # noqa: E712

    def test_excluye_celdas_con_superficie_cosechada_cero(self, mapeo_columnas):
        """Sección 4.5.1: contenido válido requiere producción > 0 y superficie
        cosechada positiva; de lo contrario no se puede calcular rendimiento."""
        con_superficie_cero = pd.DataFrame(
            [
                {
                    "DEPARTAMENTO": "Puno",
                    "PROVINCIA": "Azángaro",
                    "ANIO_COSECHA": 2020,
                    "CULTIVO": "Quinua",
                    "SUPERFICIE_SEMBRADA": 500.0,
                    "SUPERFICIE_COSECHADA": 0.0,
                    "PRODUCCION": 0.0,
                    "RENDIMIENTO": 0.0,
                }
            ]
        )
        resultado = load_midagri_production(con_superficie_cero, mapeo_columnas)
        assert len(resultado) == 0


class TestLoadColumnMapping:
    def test_carga_mapeo_desde_yaml(self, tmp_path):
        yaml_path = tmp_path / "column_mapping.yaml"
        yaml_path.write_text(
            """
midagri_produccion:
  departamento: "DEP"
  provincia: "PROV"
  campana_id: "ANIO"
  cultivo: "CULT"
  superficie_sembrada_ha: "SS"
  superficie_cosechada_ha: "SC"
  produccion_ton: "PROD"
  rendimiento_kg_ha: "REND"
""",
            encoding="utf-8",
        )
        mapeo = load_column_mapping(yaml_path, seccion="midagri_produccion")
        assert isinstance(mapeo, MidagriColumnMapping)
        assert mapeo.departamento == "DEP"
        assert mapeo.rendimiento_kg_ha == "REND"

    def test_carga_mapeo_con_rendimiento_nulo(self, tmp_path):
        yaml_path = tmp_path / "column_mapping.yaml"
        yaml_path.write_text(
            """
midagri_produccion:
  departamento: "DEP"
  provincia: "PROV"
  campana_id: "ANIO"
  cultivo: "CULT"
  superficie_sembrada_ha: "SS"
  superficie_cosechada_ha: "SC"
  produccion_ton: "PROD"
  rendimiento_kg_ha: ~
""",
            encoding="utf-8",
        )
        mapeo = load_column_mapping(yaml_path, seccion="midagri_produccion")
        assert mapeo.rendimiento_kg_ha is None
