"""Límite de uso y clave de API opcional (src/api/acceso.py)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app

META = "/api/v1/meta"


class Reloj:
    """Reloj manual para probar la ventana de un minuto sin esperar."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _cliente(raiz, **opciones) -> TestClient:
    return TestClient(create_app(raiz, **opciones))


# ---------------------------------------------------------------------------------------------------------------
# Límite de uso
# ---------------------------------------------------------------------------------------------------------------


def test_sobre_el_limite_responde_429_con_problem_json_y_retry_after(raiz_sintetica):
    c = _cliente(raiz_sintetica, limite_por_minuto=3, reloj=Reloj())
    assert [c.get(META).status_code for _ in range(3)] == [200, 200, 200]
    r = c.get(META)
    assert r.status_code == 429
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["status"] == 429 and "3 consultas por minuto" in r.json()["detail"]
    assert 1 <= int(r.headers["retry-after"]) <= 60


def test_la_ventana_se_libera_con_el_tiempo(raiz_sintetica):
    reloj = Reloj()
    c = _cliente(raiz_sintetica, limite_por_minuto=2, reloj=reloj)
    c.get(META), c.get(META)
    assert c.get(META).status_code == 429
    reloj.t += 61
    assert c.get(META).status_code == 200


def test_retry_after_indica_cuanto_falta_para_que_salga_la_consulta_mas_antigua(raiz_sintetica):
    reloj = Reloj()
    c = _cliente(raiz_sintetica, limite_por_minuto=1, reloj=reloj)
    c.get(META)
    reloj.t += 45
    assert int(c.get(META).headers["retry-after"]) == 15


def test_el_chequeo_de_salud_no_cuenta_ni_se_limita(raiz_sintetica):
    c = _cliente(raiz_sintetica, limite_por_minuto=1, reloj=Reloj())
    assert all(c.get("/salud").status_code == 200 for _ in range(10))
    assert c.get(META).status_code == 200  # la salud no consumió el cupo


def test_cero_desactiva_el_limite(raiz_sintetica):
    c = _cliente(raiz_sintetica, limite_por_minuto=0)
    assert all(c.get(META).status_code == 200 for _ in range(50))


def test_cada_cliente_tiene_su_propio_cupo_cuando_se_confia_en_el_proxy(raiz_sintetica):
    c = _cliente(raiz_sintetica, limite_por_minuto=1, confiar_proxy=True, reloj=Reloj())
    a = {"X-Forwarded-For": "203.0.113.7, 10.0.0.1"}
    b = {"X-Forwarded-For": "198.51.100.9"}
    assert c.get(META, headers=a).status_code == 200
    assert c.get(META, headers=a).status_code == 429
    assert c.get(META, headers=b).status_code == 200


def test_sin_confiar_en_el_proxy_no_se_puede_evadir_el_limite_cambiando_la_cabecera(raiz_sintetica):
    c = _cliente(raiz_sintetica, limite_por_minuto=1, confiar_proxy=False, reloj=Reloj())
    assert c.get(META, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert c.get(META, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_el_limite_por_defecto_es_generoso_para_una_pagina(raiz_sintetica):
    c = _cliente(raiz_sintetica)  # 300 por minuto
    assert all(c.get(META).status_code == 200 for _ in range(60))


# ---------------------------------------------------------------------------------------------------------------
# Clave de API
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cabeceras", [{}, {"X-API-Key": "otra"}, {"X-API-Key": ""}])
def test_con_clave_configurada_se_rechaza_la_ausente_o_incorrecta(raiz_sintetica, cabeceras):
    c = _cliente(raiz_sintetica, clave_api="secreta", limite_por_minuto=0)
    r = c.get(META, headers=cabeceras)
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/problem+json")
    assert "X-API-Key" in r.json()["detail"]


def test_con_la_clave_correcta_responde_normal(raiz_sintetica):
    c = _cliente(raiz_sintetica, clave_api="secreta", limite_por_minuto=0)
    assert c.get(META, headers={"X-API-Key": "secreta"}).status_code == 200


def test_la_clave_protege_tambien_la_documentacion_pero_no_la_salud(raiz_sintetica):
    c = _cliente(raiz_sintetica, clave_api="secreta", limite_por_minuto=0)
    assert c.get("/docs").status_code == 401
    assert c.get("/openapi.json").status_code == 401
    assert c.get("/salud").status_code == 200


def test_sin_clave_configurada_no_se_exige(raiz_sintetica):
    assert _cliente(raiz_sintetica, limite_por_minuto=0).get(META).status_code == 200


def test_los_intentos_con_clave_erronea_tambien_consumen_el_limite(raiz_sintetica):
    """Se limita antes de comprobar la clave: adivinarla a fuerza bruta se frena."""
    c = _cliente(raiz_sintetica, clave_api="secreta", limite_por_minuto=2, reloj=Reloj())
    assert [c.get(META, headers={"X-API-Key": "x"}).status_code for _ in range(3)] == [401, 401, 429]


def test_la_preflight_de_cors_no_exige_clave_ni_consume_cupo(raiz_sintetica):
    c = _cliente(raiz_sintetica, clave_api="secreta", limite_por_minuto=1, origenes=["http://localhost:5173"], reloj=Reloj())
    pre = c.options(
        META, headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "x-api-key"}
    )
    assert pre.status_code == 200
    assert "x-api-key" in pre.headers["access-control-allow-headers"].lower()
    assert c.get(META, headers={"X-API-Key": "secreta"}).status_code == 200  # el cupo seguía intacto


def test_los_errores_401_y_429_llevan_cabeceras_cors_para_que_el_navegador_pueda_leerlos(raiz_sintetica):
    origen = {"Origin": "http://localhost:5173"}
    c = _cliente(raiz_sintetica, clave_api="secreta", limite_por_minuto=1, origenes=["http://localhost:5173"], reloj=Reloj())
    r401 = c.get(META, headers=origen)
    assert r401.status_code == 401
    assert r401.headers.get("access-control-allow-origin") == "http://localhost:5173"
    r429 = c.get(META, headers=origen)
    assert r429.status_code == 429
    assert r429.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_las_variables_de_entorno_configuran_el_servicio(raiz_sintetica, monkeypatch):
    monkeypatch.setenv("QUINUAPROD_API_KEY", "desde-entorno")
    monkeypatch.setenv("QUINUAPROD_LIMITE_POR_MINUTO", "0")
    c = TestClient(create_app(raiz_sintetica))
    assert c.get(META).status_code == 401
    assert c.get(META, headers={"X-API-Key": "desde-entorno"}).status_code == 200
