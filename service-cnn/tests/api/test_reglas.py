"""Reglas de honestidad y de servicio que la API debe cumplir por sí misma."""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app

PREFIJO = "/api/v1"
EXTERNA = re.compile(r"\b202[1-5]\b")
GETS = [
    "/meta",
    "/provincias",
    "/cobertura/embudo",
    "/cobertura/exclusiones",
    "/predicciones?limit=5000",
    "/indicadores/y1",
    "/indicadores/x1-x2",
    "/indicadores/variabilidad-semillas",
    "/explicabilidad",
    "/sensibilidades",
    "/sensibilidades/estratos",
    "/preespecificacion",
]


@pytest.mark.parametrize("ruta", GETS)
def test_ninguna_respuesta_expone_campanas_externas(cliente, ruta):
    """RNF1: las filas 2021 del conjunto sintético no deben salir por ningún endpoint."""
    r = cliente.get(PREFIJO + ruta)
    assert r.status_code == 200
    assert not EXTERNA.search(r.text), f"{ruta} expone una campaña externa"


def test_las_campanas_permitidas_salen_del_piloto_y_no_de_la_interfaz(cliente):
    meta = cliente.get(PREFIJO + "/meta").json()
    assert meta["campanas_permitidas"] == [2016, 2017, 2018, 2019, 2020]
    assert meta["campanas_de_prueba"] == [2019, 2020]


def test_las_exclusiones_de_campanas_externas_se_descartan(cliente):
    filas = cliente.get(PREFIJO + "/cobertura/exclusiones").json()
    assert {f["campana"] for f in filas} == {2016, 2017}


def test_las_predicciones_de_campanas_externas_se_descartan(cliente):
    cuerpo = cliente.get(PREFIJO + "/predicciones", params={"limit": 5000}).json()
    assert {f["campana_prueba"] for f in cuerpo["items"]} == {2019}
    assert cuerpo["total"] == 6


def test_los_filtros_de_provincias_y_exclusiones(cliente):
    assert [p["provincia_id"] for p in cliente.get(PREFIJO + "/provincias", params={"clasificacion": "selva"}).json()] == ["BBB"]
    assert [p["provincia_id"] for p in cliente.get(PREFIJO + "/provincias", params={"departamento": "PUNO"}).json()] == ["AAA"]
    filas = cliente.get(PREFIJO + "/cobertura/exclusiones", params={"motivo": "tensor_incompleto"}).json()
    assert [(f["provincia_id"], f["campana"]) for f in filas] == [("AAA", 2016)]


def test_el_embudo_conserva_los_niveles_sin_calcular_como_null(cliente):
    embudo = cliente.get(PREFIJO + "/cobertura/embudo").json()
    assert embudo[0] == {"nivel": "N0", "n": 10}
    assert embudo[-1] == {"nivel": "N5", "n": None}


def test_la_ejecucion_tecnica_de_he2_no_incluye_campanas_externas(cliente):
    he2 = cliente.get(PREFIJO + "/explicabilidad").json()["he2_tecnico"]
    assert he2["campanas_evaluadas"] == [2018, 2019]
    assert set(he2["contribuciones"]) == {"2018", "2019"}


def test_la_inferencia_devuelve_la_mediana_entre_semillas_y_no_inventa_shap(cliente):
    r = cliente.post(PREFIJO + "/predicciones/inferir", json={"provincia_id": "AAA", "campana": 2019}).json()
    assert r["y_pred"] == pytest.approx(1000.0)  # mediana de 900, 1100 y 1000
    assert r["y_b2"] == pytest.approx(950.0)
    assert r["anomalia"] == pytest.approx(50.0)
    assert r["y_real"] == pytest.approx(1000.0)
    assert r["shap"] is None
    assert r["modelo_version"] == "piloto-oficial-aaaaaaaa"


def test_el_tensor_editado_responde_501_hasta_que_exista_el_modelo(cliente):
    r = cliente.post(
        PREFIJO + "/predicciones/inferir",
        json={"tensor": {"valores": [[1.0] * 5] * 6, "altitud_media_m": 3800, "prop_quinua": 0.1}},
    )
    assert r.status_code == 501
    assert "serializado" in r.json()["detail"]


def test_una_ruta_inexistente_responde_problem_json(cliente):
    r = cliente.get(PREFIJO + "/no-existe")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["status"] == 404


def test_sin_los_artefactos_el_servicio_responde_503_con_instrucciones(tmp_path):
    r = TestClient(create_app(tmp_path)).get(PREFIJO + "/meta")
    assert r.status_code == 503
    assert r.headers["content-type"].startswith("application/problem+json")
    assert "run_pilot" in r.json()["detail"]


def test_cors_permite_solo_los_origenes_configurados(raiz_sintetica):
    cliente = TestClient(create_app(raiz_sintetica, origenes=["http://localhost:5173"]))
    permitido = cliente.get(PREFIJO + "/meta", headers={"Origin": "http://localhost:5173"})
    otro = cliente.get(PREFIJO + "/meta", headers={"Origin": "https://sitio-ajeno.example"})
    assert permitido.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in otro.headers


def test_el_servicio_es_de_solo_lectura(cliente):
    for metodo in ("put", "patch", "delete"):
        assert getattr(cliente, metodo)(PREFIJO + "/meta").status_code == 405
    assert cliente.post(PREFIJO + "/meta").status_code == 405


def test_la_salud_responde_sin_necesitar_datos(tmp_path):
    assert TestClient(create_app(tmp_path)).get("/salud").json() == {"estado": "ok"}
