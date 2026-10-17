"""Guarda de CI del registro E5 congelado: solo necesita `pyyaml`, sin `torch` ni `xgboost`.

La coherencia con el código y la huella de los datos se prueban en `test_preregistration.py`, que necesita el
entorno completo y no corre en el CI del backend. Aquí se comprueba lo que se puede verificar con el YAML solo:
que el registro siga congelado, completo y sin decisiones pendientes. Un cambio posterior al congelamiento exige
una enmienda fechada; si alguien edita el registro a mano, esta prueba lo hace visible.
"""
from __future__ import annotations

import re

import pytest

from src.evaluation.preregistration import assert_can_freeze, load_preregistration

RUTA = "configs/preespecificacion.yaml"


@pytest.fixture(scope="module")
def prereg() -> dict:
    return load_preregistration(RUTA)


def test_el_registro_esta_congelado(prereg):
    meta = prereg["metadatos"]

    assert meta["estado_documento"] == "CONGELADO"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(meta["fecha_congelamiento"]))
    assert str(meta["fecha_congelamiento"]) >= str(meta["fecha_borrador"])


def test_el_congelamiento_registra_el_commit_y_la_huella_de_los_datos(prereg):
    meta = prereg["metadatos"]

    assert re.fullmatch(r"[0-9a-f]{40}", meta["commit_congelamiento"])
    assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256_conjunto_de_datos"])


def test_no_quedan_decisiones_pendientes_ni_valores_sin_definir(prereg):
    assert_can_freeze(prereg)
