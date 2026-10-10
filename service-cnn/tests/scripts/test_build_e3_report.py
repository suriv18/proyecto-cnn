"""Pruebas de las utilidades del generador del informe E3.

Regresión real (2026-10-10): la lista de «asociaciones más fuertes» se ordenaba
fila por fila (`np.argsort` sin `axis=None`) y mostraba correlaciones de 0,2 como
las mayores cuando la figura mostraba 0,45.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_RUTA = Path(__file__).resolve().parents[2] / "scripts" / "build_e3_report.py"
_spec = importlib.util.spec_from_file_location("build_e3_report", _RUTA)
e3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e3)


class TestMayoresPorMagnitud:
    def test_ordena_sobre_toda_la_matriz_y_no_fila_por_fila(self):
        m = np.array([[0.19, 0.22, -0.24],
                      [0.45, 0.10, 0.05]])  # el mayor está en la fila 2

        mayores = e3.mayores_por_magnitud(m, k=2)

        assert mayores[0] == (1, 0, 0.45)
        assert mayores[1] == (0, 2, -0.24)

    def test_usa_el_valor_absoluto_y_conserva_el_signo(self):
        m = np.array([[0.3, -0.5], [0.1, 0.2]])

        assert e3.mayores_por_magnitud(m, k=1) == [(0, 1, -0.5)]

    def test_ignora_los_nan(self):
        m = np.array([[np.nan, 0.2], [0.1, np.nan]])

        assert e3.mayores_por_magnitud(m, k=1) == [(0, 1, 0.2)]


class TestFormato:
    def test_usa_coma_decimal_y_espacio_de_miles(self):
        assert e3.fmt(1234.5, 1) == "1 234,5"
        assert e3.fmt(0.323, 3) == "0,323"

    def test_valor_ausente_se_muestra_como_guion(self):
        assert e3.fmt(float("nan")) == "—"

    def test_tabla_markdown(self):
        tabla = e3.md_tabla(pd.DataFrame({"a": [1], "b": ["x"]}))

        assert tabla.splitlines() == ["| a | b |", "|---|---|", "| 1 | x |"]

    def test_nombre_de_departamento_con_tilde(self):
        assert e3.nombre_dep("JUNIN") == "Junín"
        assert e3.nombre_dep("APURIMAC") == "Apurímac"
