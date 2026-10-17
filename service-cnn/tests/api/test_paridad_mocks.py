"""El servicio y los mocks de la interfaz entregan lo mismo.

Los fixtures de app-quinuaprod (`src/testing/mocks/fixtures`) son una instantánea de los mismos
artefactos. Si esta prueba falla, o cambió el pipeline (hay que re-exportar los fixtures) o cambió la
API (hay que revisar el contrato). Requiere los reports/ reales; sin ellos se omite.
"""
from __future__ import annotations

import json
import math

import pytest

from tests.api.conftest import FIXTURES

PREFIJO = "/api/v1"

# (ruta, fixture)
PARES = [
    ("/meta", "meta"),
    ("/provincias", "provincias"),
    ("/cobertura/embudo", "embudo"),
    ("/cobertura/exclusiones", "exclusiones"),
    ("/indicadores/y1", "indicadores_y1"),
    ("/indicadores/x1-x2", "indicadores_x1x2"),
    ("/indicadores/variabilidad-semillas", "variabilidad_semillas"),
    ("/explicabilidad", "explicabilidad"),
    ("/sensibilidades", "sensibilidades"),
    ("/sensibilidades/estratos", "sensibilidades_estratos"),
    ("/preespecificacion", "preespecificacion"),
]


def _iguales(a, b, ruta="") -> None:
    """Igualdad profunda; los fixtures redondean los decimales a 10 cifras, por eso se tolera."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"{ruta}: claves distintas {sorted(set(a) ^ set(b))[:4]}"
        for k in a:
            _iguales(a[k], b[k], f"{ruta}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{ruta}: {len(a)} frente a {len(b)} elementos"
        for i, (x, y) in enumerate(zip(a, b)):
            _iguales(x, y, f"{ruta}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), f"{ruta}: {a} frente a {b}"
    else:
        assert a == b, f"{ruta}: {a!r} frente a {b!r}"


def _fixture(nombre: str):
    ruta = FIXTURES / f"{nombre}.json"
    if not ruta.exists():
        pytest.skip(f"No existe el fixture {nombre}.json")
    return json.loads(ruta.read_text(encoding="utf-8"))


@pytest.mark.parametrize("ruta, nombre", PARES)
def test_la_api_coincide_con_el_fixture(cliente_real, ruta, nombre):
    _iguales(cliente_real.get(PREFIJO + ruta).json(), _fixture(nombre), nombre)


def test_las_predicciones_de_la_api_coinciden_con_el_fixture(cliente_real):
    items = cliente_real.get(PREFIJO + "/predicciones", params={"limit": 5000}).json()["items"]
    _iguales(items, _fixture("predicciones"), "predicciones")
