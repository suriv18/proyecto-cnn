"""Guarda de campañas del ensayo piloto (tarea T1 del plan del piloto).

El piloto decide parámetros de diseño (representación del objetivo, línea base,
SHAP, épocas) y por eso no puede ver las campañas externas: si una decisión se
apoyara en ellas, el contraste confirmatorio dejaría de ser independiente
(sección 4.12.1-4.12.2). Las campañas permitidas son las primeras `ventana_inicial`,
que son de entrenamiento en cualquier esquema de origen móvil con esa ventana.

Mientras el asesor no fije la ventana, la corrida técnica debe usar la MENOR ventana
plausible: con una ventana mayor, una campaña que hoy parece de entrenamiento podría
resultar externa.
"""
from __future__ import annotations

from typing import Iterable


class CampanaExternaError(ValueError):
    """Un componente del piloto recibió una campaña que no está permitida (externa)."""


def pilot_campaigns(primera_campana: int, ventana_inicial: int) -> list[int]:
    """Las `ventana_inicial` primeras campañas desde `primera_campana`."""
    if ventana_inicial < 1:
        raise ValueError(f"La ventana inicial debe ser positiva; recibido {ventana_inicial}.")
    return list(range(primera_campana, primera_campana + ventana_inicial))


def assert_only_allowed_campaigns(
    campanas: Iterable[int], permitidas: Iterable[int], contexto: str = ""
) -> None:
    """Falla si alguna campaña no está en `permitidas`.

    El mensaje nombra las campañas ofensoras (solo sus identificadores, nunca
    sus valores) y, si se indica, el componente que las recibió.

    Raises:
        CampanaExternaError: si hay campañas fuera de las permitidas.
    """
    permitidas_set = {int(c) for c in permitidas}
    externas = sorted({int(c) for c in campanas} - permitidas_set)
    if externas:
        donde = f" en {contexto}" if contexto else ""
        raise CampanaExternaError(
            f"Campañas no permitidas{donde}: {externas}. El piloto solo puede usar "
            f"{sorted(permitidas_set)}; las demás son campañas externas y no pueden "
            "influir en ninguna decisión de diseño (4.12.1)."
        )
