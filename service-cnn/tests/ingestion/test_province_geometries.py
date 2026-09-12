"""Pruebas de normalización de nombres para el mapeo provincia -> geometría GAUL.

Referencia: docs/02-arquitectura-tecnica.md §2. Las geometrías reales de las
80 provincias se obtienen del dataset FAO/GAUL/2015/level2 en Google Earth
Engine, cuyos nombres (Título Case, con tildes) difieren en formato de los
usados en configs/provincias.csv (MAYÚSCULAS sin tildes, convención de
MIDAGRI) — verificado que ambas fuentes listan exactamente las mismas 80
provincias para los 8 departamentos delimitados.
"""
import pytest

from src.ingestion.province_geometries import normalize_province_name


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
