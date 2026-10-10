"""Corrección de Holm por multiplicidad (sección 3.3.5).

Procedimiento descendente de Holm: con m contrastes y los p ordenados de menor a mayor,
el p ajustado del i-ésimo es max_{j<=i} (m - j + 1) * p_(j), truncado en 1. Controla la
tasa de error por familia sin suponer independencia.

La función solo corrige valores p que ya existan; no los define. Qué contrastes forman la
familia y de dónde sale el p de cada uno es una decisión de E5 (ver docs/crisp/F4-evaluacion.md).
"""
from __future__ import annotations

import math


def holm_adjust(valores_p: dict[str, float]) -> dict[str, float]:
    """Valores p ajustados por Holm, en el mismo orden y con las mismas claves que la entrada.

    Raises:
        ValueError: si la familia está vacía o algún p no está entre 0 y 1.
    """
    if not valores_p:
        raise ValueError("La familia de contrastes está vacía.")
    for nombre, p in valores_p.items():
        if not (isinstance(p, (int, float)) and not math.isnan(p) and 0.0 <= p <= 1.0):
            raise ValueError(f"El valor p de '{nombre}' debe estar entre 0 y 1; recibido {p!r}.")

    m = len(valores_p)
    ordenados = sorted(valores_p, key=valores_p.__getitem__)
    ajustados: dict[str, float] = {}
    maximo_acumulado = 0.0
    for posicion, nombre in enumerate(ordenados):
        maximo_acumulado = max(maximo_acumulado, (m - posicion) * valores_p[nombre])
        ajustados[nombre] = min(1.0, maximo_acumulado)
    return {nombre: ajustados[nombre] for nombre in valores_p}


def holm_reject(valores_p: dict[str, float], alfa: float = 0.05) -> dict[str, bool]:
    """Qué hipótesis nulas se rechazan con Holm al nivel `alfa`."""
    ajustados = holm_adjust(valores_p)
    return {nombre: ajustados[nombre] <= alfa for nombre in valores_p}
