"""Crea un modelo CNN-1D sintético en el formato real, sin PyTorch.

Los pesos son aleatorios: sirven para probar la lectura, la inferencia y la API, no para predecir bien.
Así las pruebas del servicio corren en el CI sin instalar torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

V, T, K = 6, 5, 2
F1, F2, U = 3, 4, 4
SEMILLAS = [0, 1, 2]


def pesos_aleatorios(rng: np.random.Generator) -> dict[str, np.ndarray]:
    def bn(prefijo: str, n: int) -> dict[str, np.ndarray]:
        return {
            f"{prefijo}.weight": rng.uniform(0.5, 1.5, n), f"{prefijo}.bias": rng.normal(0, 0.1, n),
            f"{prefijo}.running_mean": rng.normal(0, 0.2, n), f"{prefijo}.running_var": rng.uniform(0.5, 1.5, n),
        }

    return {
        "bloque_convolucional.0.weight": rng.normal(0, 0.4, (F1, V, 3)), "bloque_convolucional.0.bias": rng.normal(0, 0.1, F1),
        **bn("bloque_convolucional.1", F1),
        "bloque_convolucional.4.weight": rng.normal(0, 0.4, (F2, F1, 3)), "bloque_convolucional.4.bias": rng.normal(0, 0.1, F2),
        **bn("bloque_convolucional.5", F2),
        "cabeza_densa.0.weight": rng.normal(0, 0.4, (U, F2 + K)), "cabeza_densa.0.bias": rng.normal(0, 0.1, U),
        "cabeza_densa.2.weight": rng.normal(0, 0.4, (1, U)), "cabeza_densa.2.bias": rng.normal(0, 0.1, 1),
    }


def crear_artefacto(destino: Path, vinculante: bool = True, semilla: int = 7) -> Path:
    rng = np.random.default_rng(semilla)
    arreglos: dict[str, np.ndarray] = {}
    for s in SEMILLAS:
        for nombre, valor in pesos_aleatorios(rng).items():
            arreglos[f"s{s}.{nombre}"] = valor
    arreglos["y_media"] = np.full(len(SEMILLAS), 5.0)
    arreglos["y_desviacion"] = np.full(len(SEMILLAS), 80.0)
    destino.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino / "pesos.npz", **arreglos)
    np.savez_compressed(
        destino / "preprocesador.npz",
        mediana=np.full((V, T), 10.0), media=np.full((V, T), 10.0), desviacion=np.full((V, T), 4.0),
        z_media=np.array([3800.0, 0.1]), z_desviacion=np.array([300.0, 0.05]),
    )
    (destino / "lineas_base.json").write_text(
        json.dumps({
            "media_global": 1000.0,
            "provincias": {
                "AAA": {"pendiente": 10.0, "intercepto": 900.0, "campana_referencia": 2016, "es_respaldo": False},
                "BBB": {"pendiente": 0.0, "intercepto": 2000.0, "campana_referencia": 2016, "es_respaldo": True},
            },
        }),
        encoding="utf-8",
    )
    (destino / "modelo.json").write_text(
        json.dumps({
            "formato": 1, "huella_pesos": "ab" * 32, "arquitectura": {
                "n_variables": V, "n_pasos_temporales": T, "n_covariables": K, "filtros_capa1": F1,
                "filtros_capa2": F2, "kernel_size": 3, "dropout": 0.1, "unidades_densa": U,
            },
            "semillas": SEMILLAS, "covariables": ["altitud_media_m", "prop_quinua"], "vinculante": vinculante,
            "modo": "oficial", "representacion": "anomalia", "campanas_entrenamiento": [2016, 2017, 2018, 2019],
        }),
        encoding="utf-8",
    )
    return destino
