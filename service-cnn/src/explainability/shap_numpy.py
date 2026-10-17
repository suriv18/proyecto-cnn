"""SHAP de la CNN-1D serializada, solo con NumPy (sin PyTorch ni la biblioteca `shap`).

Es el algoritmo de `shap.DeepExplainer`: DeepLIFT con la regla de reescalado (Shrikumar et al., 2017) y un
fondo de referencia. Para cada observación y cada referencia del fondo se propagan hacia atrás los
multiplicadores (la derivada en las capas lineales y el cociente de diferencias en las ReLU); el valor SHAP
es la media, sobre el fondo, de (entrada - referencia) × multiplicador. Se cumple la precisión local:

    suma de valores SHAP (tensor + covariables) = salida del modelo - media de la salida sobre el fondo.

La red es la de `QuinuaYieldCNN1D` en evaluación: Conv1d, BatchNorm (afín), ReLU, Conv1d, BatchNorm, ReLU,
promedio global en el tiempo, concatenación de covariables y dos capas densas. Los pesos son los que lee
`src/models/inferencia.py`. Las pruebas comparan este código con `shap.DeepExplainer`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.inferencia import EPS_BATCHNORM, _batchnorm, _conv1d, _relu

EPS_DIFERENCIA = 1e-7  # por debajo se usa la derivada, como hace shap


@dataclass(frozen=True)
class ShapSemilla:
    """Valores SHAP de una semilla, en la escala estandarizada de la salida del modelo."""

    tensor: np.ndarray  # (N, V, T)
    covariables: np.ndarray  # (N, K)
    base: float  # media de la salida sobre el fondo
    salida: np.ndarray  # (N,) salida del modelo en cada observación


def _pre_activaciones(p: dict[str, np.ndarray], x: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
    a1 = _batchnorm(_conv1d(x, p["bloque_convolucional.0.weight"], p["bloque_convolucional.0.bias"]), p, "bloque_convolucional.1")
    a2 = _batchnorm(_conv1d(_relu(a1), p["bloque_convolucional.4.weight"], p["bloque_convolucional.4.bias"]), p, "bloque_convolucional.5")
    resumen = _relu(a2).mean(axis=2)
    d = np.concatenate([resumen, z], axis=1) @ p["cabeza_densa.0.weight"].T + p["cabeza_densa.0.bias"]
    salida = (_relu(d) @ p["cabeza_densa.2.weight"].T + p["cabeza_densa.2.bias"])[:, 0]
    return {"a1": a1, "a2": a2, "d": d, "salida": salida}


def _multiplicador_relu(z_x: np.ndarray, z_r: np.ndarray) -> np.ndarray:
    """Regla de reescalado: (ReLU(x) - ReLU(r)) / (x - r); si x ≈ r, la derivada."""
    dif = z_x - z_r
    grande = np.abs(dif) > EPS_DIFERENCIA
    cociente = (np.maximum(z_x, 0.0) - np.maximum(z_r, 0.0)) / np.where(grande, dif, 1.0)
    return np.where(grande, cociente, (z_x > 0).astype(float))


def _conv1d_transpuesta(g_salida: np.ndarray, peso: np.ndarray) -> np.ndarray:
    """Gradiente de una convolución «same» respecto de su entrada: (P, O, T) y (O, C, K) -> (P, C, T)."""
    k = peso.shape[2]
    relleno = k // 2
    t = g_salida.shape[2]
    g_entrada = np.zeros((g_salida.shape[0], peso.shape[1], t))
    for i in range(k):
        desplazamiento = i - relleno  # la salida t depende de la entrada t + desplazamiento
        t0, t1 = max(0, -desplazamiento), min(t, t - desplazamiento)
        g_entrada[:, :, t0 + desplazamiento : t1 + desplazamiento] += np.einsum("oc,pot->pct", peso[:, :, i], g_salida[:, :, t0:t1])
    return g_entrada


def deeplift_semilla(
    p: dict[str, np.ndarray], x: np.ndarray, z: np.ndarray, fondo_x: np.ndarray, fondo_z: np.ndarray
) -> ShapSemilla:
    """SHAP (DeepLIFT-reescalado) de una semilla. `x`, `z` y el fondo ya van estandarizados."""
    n, b = x.shape[0], fondo_x.shape[0]
    obs, ref = _pre_activaciones(p, x, z), _pre_activaciones(p, fondo_x, fondo_z)
    t = x.shape[2]

    m1 = _multiplicador_relu(obs["a1"][:, None], ref["a1"][None])  # (N, B, F1, T)
    m2 = _multiplicador_relu(obs["a2"][:, None], ref["a2"][None])  # (N, B, F2, T)
    md = _multiplicador_relu(obs["d"][:, None], ref["d"][None])  # (N, B, U)

    escala1 = p["bloque_convolucional.1.weight"] / np.sqrt(p["bloque_convolucional.1.running_var"] + EPS_BATCHNORM)
    escala2 = p["bloque_convolucional.5.weight"] / np.sqrt(p["bloque_convolucional.5.running_var"] + EPS_BATCHNORM)

    # Hacia atrás: salida -> capa densa -> concatenación -> promedio global -> bloque convolucional.
    g_d = p["cabeza_densa.2.weight"][0][None, None, :] * md  # (N, B, U)
    g_combinado = np.einsum("nbu,uj->nbj", g_d, p["cabeza_densa.0.weight"])
    f2 = p["bloque_convolucional.4.weight"].shape[0]
    g_resumen, g_z = g_combinado[..., :f2], g_combinado[..., f2:]
    g_r2 = np.broadcast_to(g_resumen[..., None] / t, (n, b, f2, t))
    g_c2 = (g_r2 * m2 * escala2[None, None, :, None]).reshape(n * b, f2, t)
    f1 = p["bloque_convolucional.0.weight"].shape[0]
    g_r1 = _conv1d_transpuesta(g_c2, p["bloque_convolucional.4.weight"]).reshape(n, b, f1, t)
    g_c1 = (g_r1 * m1 * escala1[None, None, :, None]).reshape(n * b, f1, t)
    g_x = _conv1d_transpuesta(g_c1, p["bloque_convolucional.0.weight"]).reshape(n, b, *x.shape[1:])

    phi_x = ((x[:, None] - fondo_x[None]) * g_x).mean(axis=1)
    phi_z = ((z[:, None] - fondo_z[None]) * g_z).mean(axis=1)
    return ShapSemilla(tensor=phi_x, covariables=phi_z, base=float(ref["salida"].mean()), salida=obs["salida"])
