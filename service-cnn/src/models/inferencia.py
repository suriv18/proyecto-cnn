"""Inferencia de la CNN-1D serializada, solo con NumPy (sin PyTorch).

Reproduce el `forward` de `QuinuaYieldCNN1D` en modo evaluación: dos bloques Conv1d + BatchNorm + ReLU
(el dropout no actúa al evaluar), promedio global en el eje temporal, concatenación de las covariables
y una cabeza densa de dos capas. Lo escribe `src/models/exportacion.py`.

La salida del modelo es la ANOMALÍA respecto de la línea base provincial B2, ya devuelta a kg/ha. El
rendimiento se reconstruye sumando B2 (`LineaBase`), igual que el piloto.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS_BATCHNORM = 1e-5  # el valor por defecto de torch.nn.BatchNorm1d


@dataclass(frozen=True)
class LineaBase:
    """Línea base provincial B2: tendencia lineal ajustada solo con el entrenamiento."""

    media_global: float
    provincias: dict[str, dict]

    def predecir(self, provincia_id: str, campana: int) -> float | None:
        """B2 en kg/ha, o None si la provincia no tiene historia en el entrenamiento (no se inventa)."""
        b = self.provincias.get(provincia_id)
        if b is None:
            return None
        return float(b["intercepto"] + b["pendiente"] * (campana - b["campana_referencia"]))


def _conv1d(x: np.ndarray, peso: np.ndarray, sesgo: np.ndarray) -> np.ndarray:
    """Convolución 1D con relleno «same»: x (N, C, T), peso (O, C, K) -> (N, O, T)."""
    k = peso.shape[2]
    relleno = k // 2
    xp = np.pad(x, ((0, 0), (0, 0), (relleno, relleno)))
    t = x.shape[2]
    ventanas = np.stack([xp[:, :, i : i + t] for i in range(k)], axis=-1)  # (N, C, T, K)
    return np.einsum("nctk,ock->not", ventanas, peso) + sesgo[None, :, None]


def _batchnorm(x: np.ndarray, p: dict[str, np.ndarray], prefijo: str) -> np.ndarray:
    media, var = p[f"{prefijo}.running_mean"], p[f"{prefijo}.running_var"]
    gamma, beta = p[f"{prefijo}.weight"], p[f"{prefijo}.bias"]
    escala = gamma / np.sqrt(var + EPS_BATCHNORM)
    return (x - media[None, :, None]) * escala[None, :, None] + beta[None, :, None]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _adelante(p: dict[str, np.ndarray], x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Salida estandarizada del modelo para una semilla: (N,)."""
    h = _relu(_batchnorm(_conv1d(x, p["bloque_convolucional.0.weight"], p["bloque_convolucional.0.bias"]), p, "bloque_convolucional.1"))
    h = _relu(_batchnorm(_conv1d(h, p["bloque_convolucional.4.weight"], p["bloque_convolucional.4.bias"]), p, "bloque_convolucional.5"))
    resumen = h.mean(axis=2)  # promedio global en el eje temporal
    combinado = np.concatenate([resumen, z], axis=1)
    h = _relu(combinado @ p["cabeza_densa.0.weight"].T + p["cabeza_densa.0.bias"])
    return (h @ p["cabeza_densa.2.weight"].T + p["cabeza_densa.2.bias"])[:, 0]


@dataclass(frozen=True)
class ModeloCNN1D:
    tarjeta: dict
    pesos: list[dict[str, np.ndarray]]  # uno por semilla
    y_media: np.ndarray
    y_desviacion: np.ndarray
    mediana: np.ndarray
    media: np.ndarray
    desviacion: np.ndarray
    z_media: np.ndarray
    z_desviacion: np.ndarray
    linea_base: LineaBase

    @classmethod
    def cargar(cls, directorio: Path) -> ModeloCNN1D:
        directorio = Path(directorio)
        tarjeta = json.loads((directorio / "modelo.json").read_text(encoding="utf-8"))
        if tarjeta.get("formato") != 1:
            raise ValueError(f"Formato de modelo no soportado: {tarjeta.get('formato')!r}.")
        with np.load(directorio / "pesos.npz") as archivo:
            todo = {k: archivo[k] for k in archivo.files}
        pesos = []
        for semilla in tarjeta["semillas"]:
            prefijo = f"s{semilla}."
            pesos.append({k[len(prefijo):]: v for k, v in todo.items() if k.startswith(prefijo)})
        with np.load(directorio / "preprocesador.npz") as p:
            prep = {k: p[k] for k in p.files}
        base = json.loads((directorio / "lineas_base.json").read_text(encoding="utf-8"))
        return cls(
            tarjeta=tarjeta, pesos=pesos, y_media=todo["y_media"], y_desviacion=todo["y_desviacion"],
            mediana=prep["mediana"], media=prep["media"], desviacion=prep["desviacion"],
            z_media=prep["z_media"], z_desviacion=prep["z_desviacion"],
            linea_base=LineaBase(base["media_global"], base["provincias"]),
        )

    @property
    def forma_tensor(self) -> tuple[int, int]:
        a = self.tarjeta["arquitectura"]
        return a["n_variables"], a["n_pasos_temporales"]

    @property
    def n_covariables(self) -> int:
        return self.tarjeta["arquitectura"]["n_covariables"]

    @property
    def version(self) -> str:
        return f"cnn1d-{self.tarjeta['huella_pesos'][:8]}"

    def predecir_anomalia(self, x_crudo: np.ndarray, z_crudo: np.ndarray) -> np.ndarray:
        """Anomalía en kg/ha por semilla: (n_semillas, N).

        Args:
            x_crudo: (N, V, T) en unidades físicas, con NaN donde falte el dato (se imputa con la mediana
                del entrenamiento).
            z_crudo: (N, K) covariables sin estandarizar (altitud media y proporción de quinua).

        Raises:
            ValueError: si las formas no coinciden con las del modelo.
        """
        x = np.asarray(x_crudo, dtype=np.float64)
        z = np.asarray(z_crudo, dtype=np.float64)
        v, t = self.forma_tensor
        if x.ndim != 3 or x.shape[1:] != (v, t):
            raise ValueError(f"El tensor debe tener forma (N, {v}, {t}); recibido {x.shape}.")
        if z.ndim != 2 or z.shape != (x.shape[0], self.n_covariables):
            raise ValueError(f"Las covariables deben tener forma ({x.shape[0]}, {self.n_covariables}); recibido {z.shape}.")
        x_z = (np.where(np.isnan(x), self.mediana, x) - self.media) / self.desviacion
        z_z = (z - self.z_media) / self.z_desviacion
        return np.stack(
            [_adelante(p, x_z, z_z) * self.y_desviacion[i] + self.y_media[i] for i, p in enumerate(self.pesos)]
        )
