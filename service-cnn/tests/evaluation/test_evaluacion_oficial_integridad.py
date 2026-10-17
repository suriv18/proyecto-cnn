"""Integridad del registro de la evaluación confirmatoria oficial (solo biblioteca estándar y pytest).

La carpeta `docs/crisp/evaluacion-confirmatoria-oficial` es el registro de la única ejecución oficial: sus
archivos no se editan. Si alguien los altera, esta prueba falla y obliga a explicarlo en una nota aparte.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CARPETA = Path("docs/crisp/evaluacion-confirmatoria-oficial")
ARCHIVOS = {
    "ajuste.csv", "ejecucion_confirmatoria.json", "informe_confirmatorio.md", "oclusion_aumento_rmse_por_fase.csv",
    "predicciones.csv", "resultados.json", "rmse_por_campana.csv", "shap_contribucion_por_fase.csv",
}


def _manifiesto() -> dict[str, str]:
    filas = [linea.split("  ", 1) for linea in (CARPETA / "MANIFIESTO.sha256").read_text(encoding="utf-8").splitlines() if linea]
    return {nombre: huella for huella, nombre in filas}


def test_el_manifiesto_cubre_todos_los_archivos_del_registro():
    assert set(_manifiesto()) == ARCHIVOS


def test_cada_archivo_coincide_con_su_huella():
    for nombre, esperada in _manifiesto().items():
        assert hashlib.sha256((CARPETA / nombre).read_bytes()).hexdigest() == esperada, f"{nombre} fue modificado"


def test_el_registro_es_una_ejecucion_oficial_con_los_parametros_de_e5():
    resultados = json.loads((CARPETA / "resultados.json").read_text(encoding="utf-8"))
    p0 = resultados["p0"]

    assert p0["modo"] == "oficial" and p0["vinculante"] is True
    assert p0["campanas_externas_evaluadas"] == [2021, 2022, 2023, 2024, 2025]
    assert p0["parametros"]["ventana_inicial"] == 5 and p0["parametros"]["n_trials"] == 30
    assert p0["e5"]["estado_documento"] == "CONGELADO"
