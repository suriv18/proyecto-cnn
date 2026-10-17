"""Datos sintéticos mínimos para probar el servicio sin depender de `reports/` (que git ignora).

El conjunto incluye a propósito filas de campañas externas (2021) en cada artefacto: la API debe
descartarlas. Los valores son inventados y solo sirven para probar forma y reglas.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from src.api.main import create_app

REPO = Path(__file__).resolve().parents[3]
OPENAPI = REPO / "app-quinuaprod" / "api" / "openapi.yaml"
FIXTURES = REPO / "app-quinuaprod" / "src" / "testing" / "mocks" / "fixtures"
RAIZ_REAL = REPO / "service-cnn"

VARIABLES = ["precipitacion", "temperatura_maxima", "temperatura_minima", "radiacion_solar", "ndvi_mean", "ndvi_max"]
FASES = ["emergencia", "desarrollo_vegetativo", "floracion", "llenado_grano", "madurez"]
POR_FASE = dict(zip(FASES, [0.10, 0.12, 0.15, 0.14, 0.09]))


def _escribir(ruta: Path, contenido: str) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(contenido, encoding="utf-8")


def _json(ruta: Path, datos: object) -> None:
    _escribir(ruta, json.dumps(datos))


def construir_raiz(raiz: Path) -> Path:
    piloto = raiz / "reports" / "piloto_oficial"
    _json(
        piloto / "resultados.json",
        {
            "p0": {
                "modo": "oficial",
                "vinculante": True,
                "campanas_permitidas": [2016, 2017, 2018, 2019, 2020],
                "campanas_descartadas": [2021, 2022, 2023, 2024, 2025],
                "campanas_de_prueba": [2019, 2020],
                "n_observaciones": 10,
                "sha256_tensores": "a" * 64,
            },
            "p4": {"representacion_elegida": "anomalia"},
            "p7": {"hiperparametros_cnn": {"filtros_capa1": 4, "filtros_capa2": 4, "dropout": 0.1, "tasa_aprendizaje": 0.01}},
            "p8": {"contribucion_por_fase": POR_FASE, "fase_maxima": "floracion", "n_explicadas": 3},
            "p11": {
                "x1": {"campo_receptivo_pasos": 5, "predicciones_validas": 1.0, "ejecuciones_reproducibles": 1.0, "semillas_verificadas": [0, 1]},
                "x2": {
                    "fraccion_explicada": 0.5,
                    "n_explicadas": 3,
                    "n_predicciones": 6,
                    "importancia_por_predictor": {v: 0.1 for v in VARIABLES},
                    "orden_de_predictores": VARIABLES,
                },
            },
        },
    )
    _escribir(
        piloto / "predicciones.csv",
        "campana_prueba,modelo,semilla,provincia_id,y_real,y_pred\n"
        "2019,cnn1d,0,AAA,1000.0,900.0\n2019,cnn1d,1,AAA,1000.0,1100.0\n2019,cnn1d,2,AAA,1000.0,1000.0\n"
        "2019,B2,0,AAA,1000.0,950.0\n2019,xgboost,0,AAA,1000.0,980.0\n2019,B2,0,BBB,2000.0,1900.0\n"
        "2021,cnn1d,0,AAA,1500.0,1400.0\n2021,B2,0,AAA,1500.0,1450.0\n",
    )
    _escribir(
        piloto / "p11_indicadores_y1.csv",
        "modelo,campana_prueba,rmse,mae,rrmse,r2\ncnn1d,2019,200.0,150.0,15.0,0.8\nB2,2019,210.0,160.0,16.0,0.79\n"
        "cnn1d,2021,250.0,190.0,18.0,0.7\n",
    )
    _escribir(
        piloto / "p6_variabilidad_por_semilla.csv",
        "modelo,campana_prueba,n_semillas,mediana,q1,q3,minimo,maximo,error_estandar_mediana\n"
        "cnn1d,2019,3,200.0,198.0,202.0,195.0,205.0,1.5\ncnn1d,2021,3,260.0,255.0,265.0,250.0,270.0,2.0\n",
    )
    _json(
        raiz / "reports" / "he2_tecnico" / "resultados.json",
        {
            "vinculante": False,
            "campanas_evaluadas": [2018, 2019, 2021],
            "contribuciones": {"2018": POR_FASE, "2019": POR_FASE, "2021": POR_FASE},
            "aumentos_rmse": {"2018": POR_FASE, "2019": POR_FASE, "2021": POR_FASE},
            "w_kendall": 0.8,
            "jaccard_promedio": 0.5,
            "he2_respaldada": False,
        },
    )
    _escribir(
        raiz / "reports" / "sensibilidades" / "sensibilidades_rmse.csv",
        "variante,modelo,campana_prueba,rmse,q_frente_a_b2,q_frente_a_b4,cambio_vs_base_pct\n"
        "base,cnn1d,2019,200.0,-0.01,0.007,0.0\nsin_transicion,cnn1d,2019,205.0,0.0,0.01,2.5\n"
        "base,B2,2019,202.0,0.0,0.02,0.0\nbase,cnn1d,2021,260.0,0.1,0.1,0.0\n",
    )
    _escribir(
        raiz / "reports" / "sensibilidades" / "error_por_estrato_de_quinua.csv",
        "modelo,estrato,rmse,n\ncnn1d,bajo,220.0,10\ncnn1d,medio,250.0,12\ncnn1d,alto,140.0,9\n",
    )
    _json(raiz / "reports" / "E1_matriz_cobertura_enmendado_2016_2025.json", {"N0": 10, "N1": 9, "N2": 8, "N3": 7, "N4": 6, "N5": None})
    _escribir(
        raiz / "data" / "processed" / "exclusiones.csv",
        "provincia_id,campana_id,motivo,detalle\nAAA,2016,tensor_incompleto,x\nBBB,2021,tensor_incompleto,y\n"
        "BBB,2017,fuera_del_ambito_altoandino,clasificacion=selva\n",
    )
    _json(raiz / "data" / "processed" / "metadata.json", {"orden_variables": VARIABLES, "orden_fases": FASES})
    _escribir(
        raiz / "configs" / "provincias.csv",
        "provincia_id,departamento,clasificacion\nAAA,PUNO,altoandina\nBBB,JUNIN,selva\n",
    )
    _escribir(
        raiz / "configs" / "preespecificacion.yaml",
        yaml.safe_dump(
            {
                "metadatos": {"estado_documento": "BORRADOR", "version": "0.1", "fecha_borrador": "2026-10-10", "fecha_congelamiento": None},
                "hipotesis": {"alfa": {"valor": 0.05, "estado": "fijado_por_tesis", "fuente": "3.3"}},
            }
        ),
    )
    return raiz


@pytest.fixture(scope="session")
def raiz_sintetica(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return construir_raiz(tmp_path_factory.mktemp("quinuaprod"))


@pytest.fixture()
def cliente(raiz_sintetica: Path) -> TestClient:
    return TestClient(create_app(raiz_sintetica))


@pytest.fixture(scope="session")
def contrato() -> dict:
    if not OPENAPI.exists():
        pytest.skip("No está app-quinuaprod/api/openapi.yaml (el servicio se probó fuera del monorepo).")
    return yaml.safe_load(OPENAPI.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def cliente_real() -> TestClient:
    if not (RAIZ_REAL / "reports" / "piloto_oficial" / "resultados.json").exists():
        pytest.skip("No hay reports/ del piloto en este equipo (están ignorados por git).")
    return TestClient(create_app(RAIZ_REAL))
