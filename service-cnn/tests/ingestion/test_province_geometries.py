"""Pruebas de normalización de nombres para el mapeo provincia -> geometría GAUL.

Referencia: docs/02-arquitectura-tecnica.md §2. Las geometrías reales de las
80 provincias se obtienen del dataset FAO/GAUL/2015/level2 en Google Earth
Engine, cuyos nombres (Título Case, con tildes) difieren en formato de los
usados en configs/provincias.csv (MAYÚSCULAS sin tildes, convención de
MIDAGRI) — verificado que ambas fuentes listan exactamente las mismas 80
provincias para los 8 departamentos delimitados.
"""
import pytest

from src.ingestion.province_geometries import (
    DEPARTAMENTOS_GAUL,
    departamentos_a_nombres_gaul,
    normalize_province_name,
)


class TestNormalizeProvinceName:
    def test_normaliza_mayusculas_a_forma_comparable(self):
        assert normalize_province_name("AZANGARO") == normalize_province_name("Azángaro")

    def test_normaliza_tildes(self):
        assert normalize_province_name("Apurímac") == normalize_province_name("APURIMAC")

    def test_normaliza_espacios_multiples(self):
        assert normalize_province_name("La  Union") == normalize_province_name("LA UNION")

    def test_provincias_distintas_no_son_iguales(self):
        assert normalize_province_name("PUNO") != normalize_province_name("Junín")

    def test_casos_reales_del_conjunto_de_8_departamentos(self):
        """Verificado contra el dataset real GAUL (docs/02-arquitectura-
        tecnica.md): nombres que difieren en capitalización y tildes deben
        normalizar igual."""
        pares = [
            ("SANCHEZ CARRION", "Sánchez Carrión"),
            ("VILCAS HUAMAN", "Vilcas Huamán"),
            ("PAUCAR DEL SARA SARA", "Paucar del Sara Sara"),
            ("SAN ANTONIO DE PUTINA", "San Antonio de Putina"),
        ]
        for nombre_midagri, nombre_gaul in pares:
            assert normalize_province_name(nombre_midagri) == normalize_province_name(
                nombre_gaul
            )


class TestDepartamentosAGaul:
    def test_traduce_departamentos_con_tilde_verificados_contra_gaul(self):
        """APURIMAC y JUNIN (convención MIDAGRI, configs/provincias.csv) deben
        traducirse a los nombres reales con tilde de FAO/GAUL/2015/level2
        ('Apurímac', 'Junín'), verificados contra la API real (cnn-sentinel)."""
        resultado = departamentos_a_nombres_gaul(["APURIMAC", "JUNIN", "PUNO"])
        assert resultado == ["Apurímac", "Junín", "Puno"]

    def test_departamento_sin_traduccion_conocida_lanza_error_explicito(self):
        with pytest.raises(KeyError, match="LIMA"):
            departamentos_a_nombres_gaul(["LIMA"])

    def test_los_8_departamentos_de_la_tesis_estan_en_el_catalogo(self):
        """Los 8 departamentos delimitados por la tesis (docs/Proyecto_Tesis_
        Maestria_UNMSM_WMSR.docx sección 1.5) deben tener traducción GAUL
        conocida, para no descubrir un departamento faltante recién al
        ejecutar la extracción real contra la API."""
        departamentos_tesis = [
            "APURIMAC",
            "AREQUIPA",
            "AYACUCHO",
            "CUSCO",
            "HUANCAVELICA",
            "JUNIN",
            "LA LIBERTAD",
            "PUNO",
        ]
        resultado = departamentos_a_nombres_gaul(departamentos_tesis)
        assert len(resultado) == 8
        assert all(nombre in DEPARTAMENTOS_GAUL.values() for nombre in resultado)
