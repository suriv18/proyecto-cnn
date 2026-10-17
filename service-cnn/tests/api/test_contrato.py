"""Las respuestas del servicio cumplen el contrato OpenAPI de la interfaz (app-quinuaprod/api/openapi.yaml).

El contrato es la fuente de verdad: aquí no se compara contra los modelos de la API sino contra el YAML.
"""
from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from tests.api.conftest import OPENAPI

PREFIJO = "/api/v1"


def _validar(contrato: dict, operacion: dict, respuesta) -> None:
    estado = str(respuesta.status_code)
    assert estado in operacion["responses"], f"el contrato no declara {estado} para esta operación"
    declarada = operacion["responses"][estado]
    if "$ref" in declarada:  # p. ej. '#/components/responses/Problema'
        destino = contrato
        for parte in declarada["$ref"].removeprefix("#/").split("/"):
            destino = destino[parte]
        declarada = destino
    contenido = declarada.get("content", {})
    media = respuesta.headers["content-type"].split(";")[0]
    assert media in contenido, f"tipo {media} no declarado para {estado}"
    esquema = {**contenido[media]["schema"], "components": contrato["components"]}
    errores = sorted(Draft202012Validator(esquema).iter_errors(respuesta.json()), key=lambda e: list(e.path))
    assert not errores, "; ".join(f"{list(e.path)}: {e.message}" for e in errores[:3])


def _gets_sin_parametros_obligatorios() -> list[str]:
    import yaml

    if not OPENAPI.exists():
        return []
    doc = yaml.safe_load(OPENAPI.read_text(encoding="utf-8"))
    rutas = []
    for ruta, ops in doc["paths"].items():
        op = ops.get("get")
        if op and not any(p.get("required") for p in op.get("parameters", [])):
            rutas.append(ruta)
    return rutas


RUTAS_GET = _gets_sin_parametros_obligatorios()


def test_el_contrato_tiene_operaciones_que_probar():
    assert len(RUTAS_GET) >= 12


@pytest.mark.parametrize("ruta", RUTAS_GET)
def test_get_cumple_el_contrato_con_datos_sinteticos(cliente, contrato, ruta):
    r = cliente.get(PREFIJO + ruta)
    _validar(contrato, contrato["paths"][ruta]["get"], r)
    assert r.status_code == 200


@pytest.mark.parametrize("ruta", RUTAS_GET)
def test_get_cumple_el_contrato_con_los_datos_reales(cliente_real, contrato, ruta):
    r = cliente_real.get(PREFIJO + ruta)
    _validar(contrato, contrato["paths"][ruta]["get"], r)


def test_predicciones_con_filtros_cumple_el_contrato(cliente, contrato):
    r = cliente.get(PREFIJO + "/predicciones", params={"modelo": "cnn1d", "campana_prueba": 2019, "limit": 2})
    _validar(contrato, contrato["paths"]["/predicciones"]["get"], r)
    assert r.json()["total"] == 3
    assert len(r.json()["items"]) == 2


@pytest.mark.parametrize(
    "cuerpo, estado",
    [
        ({"provincia_id": "AAA", "campana": 2019}, 200),
        ({"provincia_id": "AAA", "campana": 2022}, 422),
        ({"provincia_id": "AAA", "campana": 2016}, 404),
        ({"provincia_id": "AAA"}, 422),
        ({"tensor": {"valores": [[1.0] * 5] * 6, "altitud_media_m": 3800, "prop_quinua": 0.1}}, 501),
        ({"tensor": {"valores": [[1.0] * 4] * 6, "altitud_media_m": 3800, "prop_quinua": 0.1}}, 422),
        ({"tensor": {"valores": [[1.0] * 5] * 6, "altitud_media_m": 3800, "prop_quinua": -1}}, 422),
    ],
)
def test_inferir_cumple_el_contrato_en_cada_caso(cliente, contrato, cuerpo, estado):
    r = cliente.post(PREFIJO + "/predicciones/inferir", json=cuerpo)
    assert r.status_code == estado
    _validar(contrato, contrato["paths"]["/predicciones/inferir"]["post"], r)


@pytest.mark.parametrize(
    "ruta, params",
    [
        ("/predicciones", {"campana_prueba": 2021}),
        ("/predicciones", {"modelo": "no_existe"}),
        ("/predicciones", {"limit": 0}),
        ("/predicciones", {"limit": 5001}),
        ("/predicciones", {"offset": -1}),
    ],
)
def test_los_errores_de_predicciones_son_problem_json_y_estan_declarados(cliente, contrato, ruta, params):
    r = cliente.get(PREFIJO + ruta, params=params)
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("application/problem+json")
    _validar(contrato, contrato["paths"][ruta]["get"], r)
