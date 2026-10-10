"""Constructor de pliegos de origen móvil con preprocesamiento sin fuga (tarea T2).

Para cada campaña de prueba, el entrenamiento son las campañas ANTERIORES y todo lo
que "aprende" algo se ajusta solo con ese entrenamiento (sección 4.12.2): la
mediana usada para imputar celdas vacías, la media y la desviación de la
estandarización y la línea base provincial (B2, 4.11). La validación interna usada
para elegir hiperparámetros es otro origen móvil DENTRO del entrenamiento, con su
propio preprocesamiento (`build_inner_folds`).

Antes de construir nada, `build_folds` comprueba con la guarda del piloto que no
llegue ninguna campaña no permitida (4.12.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.pilot_guard import assert_only_allowed_campaigns
from src.evaluation.splits import VentanaInicialInsuficienteError, expanding_window_splits
from src.features.target_transform import fit_provincial_baseline, predict_baseline


@dataclass(frozen=True)
class Preprocesador:
    """Estadísticas del entrenamiento, aplicadas sin recalcular a la prueba."""

    mediana: np.ndarray  # (V, T) para imputar celdas vacías
    media: np.ndarray  # (V, T)
    desviacion: np.ndarray  # (V, T), nunca 0
    z_media: np.ndarray  # (K,)
    z_desviacion: np.ndarray  # (K,)

    def transformar_x(self, x: np.ndarray) -> np.ndarray:
        imputado = np.where(np.isnan(x), self.mediana, x)
        return (imputado - self.media) / self.desviacion

    def transformar_z(self, z: np.ndarray) -> np.ndarray:
        return (z - self.z_media) / self.z_desviacion


@dataclass
class FoldData:
    """Un pliego: datos ya preprocesados de entrenamiento y prueba."""

    campana_prueba: int
    campanas_entrenamiento: list[int]
    X_train: np.ndarray
    Z_train: np.ndarray
    y_train: np.ndarray  # rendimiento en kg/ha (nivel)
    X_test: np.ndarray
    Z_test: np.ndarray
    y_test: np.ndarray
    provincia_train: np.ndarray
    provincia_test: np.ndarray
    campana_train: np.ndarray
    campana_test: np.ndarray
    baseline_train: np.ndarray  # B2 en kg/ha, ajustada solo con el entrenamiento
    baseline_test: np.ndarray
    n_sin_historia_test: int
    preprocesador: Preprocesador
    # datos sin preprocesar del entrenamiento, para reconstruir los pliegos internos
    obs_train: pd.DataFrame
    X_train_crudo: np.ndarray
    Z_train_crudo: np.ndarray

    @property
    def y_anomalia_train(self) -> np.ndarray:
        return self.y_train - self.baseline_train

    @property
    def y_anomalia_test(self) -> np.ndarray:
        return self.y_test - self.baseline_test


def restrict_to_campaigns(
    obs: pd.DataFrame, X: np.ndarray, Z: np.ndarray, permitidas: Iterable[int]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[int]]:
    """Conserva solo las campañas permitidas y devuelve cuáles se descartaron (solo sus ids)."""
    permitidas = {int(c) for c in permitidas}
    mascara = obs["campana_id"].isin(permitidas).to_numpy()
    descartadas = sorted({int(c) for c in obs.loc[~mascara, "campana_id"]})
    return obs[mascara].reset_index(drop=True), X[mascara], Z[mascara], descartadas


def load_pilot_dataset(
    directorio: str | Path, campanas_permitidas: Iterable[int]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[int]]:
    """Lee `data/processed` y descarta de inmediato las campañas no permitidas.

    El archivo contiene todas las campañas, así que se leen en memoria un instante;
    aquí mismo se descartan y solo se devuelve la lista de sus identificadores
    (nunca sus valores), de modo que ningún componente posterior pueda usarlas.

    Returns:
        `(observaciones, X, Z, campanas_descartadas)`, alineados por fila.

    Raises:
        ValueError: si el conjunto se generó sin covariables (`Z` vacía); el modelo
            las necesita (4.10.3).
    """
    directorio = Path(directorio)
    obs = pd.read_parquet(directorio / "observaciones.parquet")
    datos = np.load(directorio / "tensores.npz")
    X, Z = datos["X"], datos["Z"]
    if Z.shape[1] == 0:
        raise ValueError(
            "El conjunto procesado no tiene covariables (Z vacía). Regenérelo con "
            "scripts/run_preprocessing.py sin --sin-covariables."
        )
    return restrict_to_campaigns(obs, X, Z, campanas_permitidas)


def _ajustar_preprocesador(x_train: np.ndarray, z_train: np.ndarray) -> Preprocesador:
    mediana = np.nanmedian(x_train, axis=0)
    mediana = np.where(np.isnan(mediana), 0.0, mediana)  # celda vacía en todo el entrenamiento
    imputado = np.where(np.isnan(x_train), mediana, x_train)
    media = imputado.mean(axis=0)
    desviacion = imputado.std(axis=0)
    desviacion = np.where(desviacion < 1e-12, 1.0, desviacion)
    z_media = z_train.mean(axis=0) if z_train.shape[1] else np.zeros(0)
    z_desviacion = z_train.std(axis=0) if z_train.shape[1] else np.ones(0)
    z_desviacion = np.where(z_desviacion < 1e-12, 1.0, z_desviacion)
    return Preprocesador(mediana, media, desviacion, z_media, z_desviacion)


def _lineas_base(
    prov_train: np.ndarray, camp_train: np.ndarray, y_train: np.ndarray,
    prov_eval: np.ndarray, camp_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """B2 por provincia, ajustada solo con el entrenamiento.

    Una provincia de la evaluación sin historia usa la media de entrenamiento
    (se cuenta y se informa); no se inventa una tendencia.
    """
    historia = pd.DataFrame({"provincia_id": prov_train, "campana_id": camp_train, "rendimiento_kg_ha": y_train})
    ajustes = {p: fit_provincial_baseline(historia, p) for p in np.unique(prov_train)}
    media_global = float(np.mean(y_train))

    def predecir(provincias: np.ndarray, campanas: np.ndarray) -> tuple[np.ndarray, int]:
        salida, sin_historia = np.empty(len(provincias)), 0
        for i, (p, c) in enumerate(zip(provincias, campanas)):
            if p in ajustes:
                salida[i] = predict_baseline(ajustes[p], int(c))
            else:
                salida[i], sin_historia = media_global, sin_historia + 1
        return salida, sin_historia

    base_train, _ = predecir(prov_train, camp_train)
    base_eval, n_sin = predecir(prov_eval, camp_eval)
    return base_train, base_eval, n_sin


def _hacer_pliego(
    obs: pd.DataFrame, X: np.ndarray, Z: np.ndarray, idx_train: np.ndarray, idx_test: np.ndarray
) -> FoldData:
    o_tr, o_te = obs.iloc[idx_train].reset_index(drop=True), obs.iloc[idx_test].reset_index(drop=True)
    prep = _ajustar_preprocesador(X[idx_train], Z[idx_train])
    y_tr = o_tr["rendimiento_kg_ha"].to_numpy(dtype=float)
    y_te = o_te["rendimiento_kg_ha"].to_numpy(dtype=float)
    b_tr, b_te, n_sin = _lineas_base(
        o_tr["provincia_id"].to_numpy(), o_tr["campana_id"].to_numpy(), y_tr,
        o_te["provincia_id"].to_numpy(), o_te["campana_id"].to_numpy(),
    )
    return FoldData(
        campana_prueba=int(o_te["campana_id"].iloc[0]),
        campanas_entrenamiento=sorted(int(c) for c in o_tr["campana_id"].unique()),
        X_train=prep.transformar_x(X[idx_train]), Z_train=prep.transformar_z(Z[idx_train]), y_train=y_tr,
        X_test=prep.transformar_x(X[idx_test]), Z_test=prep.transformar_z(Z[idx_test]), y_test=y_te,
        provincia_train=o_tr["provincia_id"].to_numpy(), provincia_test=o_te["provincia_id"].to_numpy(),
        campana_train=o_tr["campana_id"].to_numpy(), campana_test=o_te["campana_id"].to_numpy(),
        baseline_train=b_tr, baseline_test=b_te, n_sin_historia_test=n_sin, preprocesador=prep,
        obs_train=o_tr, X_train_crudo=X[idx_train], Z_train_crudo=Z[idx_train],
    )


def build_folds(
    obs: pd.DataFrame, X: np.ndarray, Z: np.ndarray, ventana_inicial: int, campanas_permitidas: Iterable[int]
) -> list[FoldData]:
    """Pliegos de origen móvil: uno por cada campaña posterior a la ventana inicial.

    Raises:
        CampanaExternaError: si `obs` contiene campañas no permitidas (guarda del piloto).
        VentanaInicialInsuficienteError: si la ventana consume todas las campañas.
    """
    assert_only_allowed_campaigns(obs["campana_id"], campanas_permitidas, contexto="el constructor de pliegos")
    obs = obs.reset_index(drop=True)
    pliegos = []
    for s in expanding_window_splits(obs, "campana_id", ventana_inicial):
        pliegos.append(_hacer_pliego(obs, X, Z, s.entrenamiento.index.to_numpy(), s.prueba.index.to_numpy()))
    return pliegos


def build_inner_folds(fold: FoldData, ventana_interna: int) -> list[FoldData]:
    """Origen móvil DENTRO del entrenamiento de un pliego, para elegir hiperparámetros.

    Cada pliego interno reajusta su propio preprocesamiento con su propio
    entrenamiento. La campaña externa del pliego no interviene nunca. Si no hay
    campañas suficientes para la ventana interna, devuelve una lista vacía.
    """
    obs = fold.obs_train.reset_index(drop=True)
    try:
        splits = expanding_window_splits(obs, "campana_id", ventana_interna)
    except VentanaInicialInsuficienteError:
        return []
    return [
        _hacer_pliego(obs, fold.X_train_crudo, fold.Z_train_crudo, s.entrenamiento.index.to_numpy(), s.prueba.index.to_numpy())
        for s in splits
    ]
