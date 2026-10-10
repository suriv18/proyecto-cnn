"""Ejecutor por pliego: ajuste con presupuesto equiparado, semillas y benchmarks (tarea T4).

Para cada pliego externo de origen móvil:
  1. construye los pliegos internos (origen móvil DENTRO del entrenamiento);
  2. por cada uno de los 5 modelos ajustables busca hiperparámetros con Optuna (TPE)
     sobre el error medio de los pliegos internos, con el MISMO número de trials
     (4.12.3); la verdad de la campaña externa nunca interviene;
  3. reentrena con el mejor ajuste y las semillas que correspondan (10 en los
     estocásticos, una en los deterministas, 4.12.4) y predice la campaña externa;
  4. calcula los benchmarks B1, B2 y B3, que no se optimizan.

El objetivo se estandariza con la media y la desviación del entrenamiento para los 5
modelos (los espacios de búsqueda, p. ej. el `epsilon` de SVR, suponen esa escala) y
las predicciones se devuelven siempre en kg/ha, reconstruyendo la línea base cuando
la representación es la anomalía (4.11).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import optuna
import pandas as pd
import torch

from src.evaluation.fold_builder import FoldData, build_inner_folds
from src.models.benchmarks import fit_ndvi_benchmark, predict_ndvi_benchmark
from src.models.classic_models import build_classic_model
from src.models.cnn1d import CNN1DConfig, QuinuaYieldCNN1D
from src.models.cnn1d_trainer import TrainingConfig, train_cnn1d
from src.models.tuning import TuningResult, sugerir_hiperparametro
from src.models.tuning_cnn1d import _HIPERPARAMETROS_ARQUITECTURA, _HIPERPARAMETROS_ENTRENAMIENTO

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELOS = ("cnn1d", "elastic_net", "random_forest", "xgboost", "svr")
ESTOCASTICOS = frozenset({"cnn1d", "random_forest", "xgboost"})
REPRESENTACIONES = ("nivel", "anomalia")


def reconstruct_prediction(pred: np.ndarray, fold: FoldData, representacion: str) -> np.ndarray:
    """Devuelve la predicción en kg/ha: suma la línea base si se modeló la anomalía."""
    if representacion == "nivel":
        return pred
    if representacion == "anomalia":
        return pred + fold.baseline_test
    raise ValueError(f"representación debe ser una de {REPRESENTACIONES}; recibido '{representacion}'.")


def _objetivo(fold: FoldData, representacion: str) -> np.ndarray:
    reconstruct_prediction(np.zeros(len(fold.y_test)), fold, representacion)  # valida el nombre
    return fold.y_train if representacion == "nivel" else fold.y_anomalia_train


def _estandarizar(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    media, desviacion = float(y.mean()), float(y.std())
    desviacion = desviacion if desviacion > 1e-12 else 1.0
    return (y - media) / desviacion, media, desviacion


class Adaptador(Protocol):
    nombre: str
    estocastico: bool

    def ajustar_predecir(self, hp: dict, fold: FoldData, semilla: int, representacion: str) -> np.ndarray: ...


@dataclass
class TuningDetail(TuningResult):
    """`TuningResult` más todos los ensayos (hiperparámetros y RMSE de validación), para P9."""

    ensayos: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class AdaptadorClasico:
    nombre: str

    @property
    def estocastico(self) -> bool:
        return self.nombre in ESTOCASTICOS

    @staticmethod
    def _tabular(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.concatenate([x.reshape(len(x), -1), z], axis=1)

    def ajustar_predecir(self, hp: dict, fold: FoldData, semilla: int, representacion: str) -> np.ndarray:
        y_z, media, desviacion = _estandarizar(_objetivo(fold, representacion))
        modelo = build_classic_model(self.nombre, dict(hp), semilla)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # avisos de convergencia de ElasticNet con presupuestos pequeños
            modelo.fit(self._tabular(fold.X_train, fold.Z_train), y_z)
            pred_z = modelo.predict(self._tabular(fold.X_test, fold.Z_test))
        return reconstruct_prediction(np.asarray(pred_z) * desviacion + media, fold, representacion)


@dataclass
class AdaptadorCNN:
    epocas: int = 100
    nombre: str = "cnn1d"
    estocastico: bool = True
    # 'temporal': valida con la última campaña del entrenamiento (4.12.2, brecha G1 cerrada);
    # 'aleatoria': la partición anterior, solo para compararla en el piloto (P7).
    particion: str = "temporal"
    ultima_mejor_epoca: int = 0

    def entrenar(self, hp: dict, fold: FoldData, semilla: int, representacion: str) -> tuple[QuinuaYieldCNN1D, float, float]:
        """Entrena la CNN-1D y devuelve (modelo en modo eval, media y desviación del objetivo).

        La salida del modelo está en la escala estandarizada del objetivo; para pasarla a
        kg/ha se multiplica por la desviación, se suma la media y, si se modeló la anomalía,
        se suma la línea base (`reconstruct_prediction`). Se expone para el paso de SHAP.
        """
        y_z, media, desviacion = _estandarizar(_objetivo(fold, representacion))
        arquitectura = {k: v for k, v in hp.items() if k in _HIPERPARAMETROS_ARQUITECTURA}
        entrenamiento = {k: v for k, v in hp.items() if k in _HIPERPARAMETROS_ENTRENAMIENTO}
        config = CNN1DConfig(
            n_variables=fold.X_train.shape[1], n_pasos_temporales=fold.X_train.shape[2],
            n_covariables=fold.Z_train.shape[1], **arquitectura,
        )
        torch.manual_seed(semilla)
        modelo = QuinuaYieldCNN1D(config)
        x = torch.tensor(fold.X_train, dtype=torch.float32)
        z = torch.tensor(fold.Z_train, dtype=torch.float32)
        y = torch.tensor(y_z, dtype=torch.float32).unsqueeze(1)
        resultado = train_cnn1d(
            modelo, x, z, y,
            TrainingConfig(epocas_maximas=self.epocas, semilla=semilla, **entrenamiento),
            grupos_temporales=(
                torch.tensor(fold.campana_train, dtype=torch.long) if self.particion == "temporal" else None
            ),
        )
        self.ultima_mejor_epoca = resultado.mejor_epoca
        resultado.modelo.eval()
        return resultado.modelo, media, desviacion

    def ajustar_predecir(self, hp: dict, fold: FoldData, semilla: int, representacion: str) -> np.ndarray:
        modelo, media, desviacion = self.entrenar(hp, fold, semilla, representacion)
        with torch.no_grad():
            pred_z = modelo(
                torch.tensor(fold.X_test, dtype=torch.float32), torch.tensor(fold.Z_test, dtype=torch.float32)
            ).squeeze(1).numpy()
        return reconstruct_prediction(pred_z * desviacion + media, fold, representacion)


def build_adapter(nombre: str, epocas_cnn: int = 100, particion: str = "temporal") -> Adaptador:
    if particion not in ("temporal", "aleatoria"):
        raise ValueError(f"partición del early stopping debe ser 'temporal' o 'aleatoria'; recibido '{particion}'.")
    if nombre == "cnn1d":
        return AdaptadorCNN(epocas=epocas_cnn, particion=particion)
    if nombre in MODELOS:
        return AdaptadorClasico(nombre)
    raise ValueError(f"Modelo '{nombre}' no reconocido; use uno de {MODELOS}.")


def _rmse(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_real)) ** 2)))


def tune_model(
    adaptador: Adaptador, pliegos_internos: list[FoldData], espacio: dict, n_trials: int,
    representacion: str, semilla_tpe: int = 0, semilla_modelo: int = 0,
) -> TuningDetail:
    """Busca hiperparámetros minimizando el RMSE medio (kg/ha) de los pliegos internos.

    Solo intervienen pliegos internos del entrenamiento: la campaña externa del
    pliego no se ve (4.12.2). Todos los modelos reciben el mismo `n_trials`.
    """
    if not pliegos_internos:
        raise ValueError("No hay pliegos internos: el entrenamiento no tiene campañas suficientes para validar.")

    def objetivo(trial: optuna.Trial) -> float:
        hp = {n: sugerir_hiperparametro(trial, n, especificacion) for n, especificacion in espacio.items()}
        errores = [
            _rmse(interno.y_test, adaptador.ajustar_predecir(hp, interno, semilla_modelo, representacion))
            for interno in pliegos_internos
        ]
        return float(np.mean(errores))

    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=semilla_tpe))
    estudio.optimize(objetivo, n_trials=n_trials)
    ensayos = pd.DataFrame([{**e.params, "valor": e.value} for e in estudio.trials])
    return TuningDetail(
        mejores_hiperparametros=estudio.best_params,
        mejor_rmse_validacion=estudio.best_value,
        n_trials_ejecutados=len(estudio.trials),
        ensayos=ensayos,
    )


def benchmark_predictions(fold: FoldData, indice_ndvi_max: int) -> dict[str, np.ndarray]:
    """B1 (media histórica provincial), B2 (línea base provincial) y B3 (regresión con NDVI máximo).

    Los tres se ajustan solo con el entrenamiento y no se optimizan (4.12.3).
    """
    media_global = float(fold.y_train.mean())
    por_provincia = pd.Series(fold.y_train).groupby(fold.provincia_train).mean()
    b1 = np.array([por_provincia.get(p, media_global) for p in fold.provincia_test], dtype=float)

    def pico(x_crudo: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # filas con todas las fases vacías
            return np.nanmax(x_crudo[:, indice_ndvi_max, :], axis=1)

    ndvi_train = pico(fold.X_train_crudo)
    historia = pd.DataFrame({"ndvi_max": ndvi_train, "rendimiento_kg_ha": fold.y_train}).dropna()
    b3 = b1.copy()  # sin NDVI utilizable se recurre a B1
    if len(historia) >= 2:
        ajuste = fit_ndvi_benchmark(historia)
        ndvi_test = pico(fold.X_test_crudo)
        valido = ~np.isnan(ndvi_test)
        b3[valido] = predict_ndvi_benchmark(ajuste, ndvi_test[valido])
    return {"B1": b1, "B2": fold.baseline_test.copy(), "B3": b3}


@dataclass
class FoldResult:
    predicciones: pd.DataFrame  # larga: una fila por observación de prueba, modelo y semilla
    ajuste: pd.DataFrame  # una fila por modelo: presupuesto ejecutado, mejor ajuste y tiempos
    mejores_hiperparametros: dict[str, dict] = field(default_factory=dict)
    ensayos: dict[str, pd.DataFrame] = field(default_factory=dict)  # todos los ensayos por modelo (P9)


def run_fold(
    fold: FoldData, espacios: dict[str, dict], n_trials: int, n_semillas: int, ventana_interna: int,
    representacion: str, indice_ndvi_max: int, epocas_cnn: int = 100, semilla_tpe: int = 0,
    modelos: tuple[str, ...] = MODELOS,
) -> FoldResult:
    """Ajusta y evalúa los modelos y los benchmarks en un pliego externo."""
    internos = build_inner_folds(fold, ventana_interna)
    filas, ajustes, mejores_hp, ensayos = [], [], {}, {}

    def agregar(modelo: str, semilla: int, pred: np.ndarray) -> None:
        filas.append(pd.DataFrame({
            "campana_prueba": fold.campana_prueba, "modelo": modelo, "semilla": semilla,
            "provincia_id": fold.provincia_test, "y_real": fold.y_test, "y_pred": pred,
        }))

    for nombre in modelos:
        adaptador = build_adapter(nombre, epocas_cnn)
        t0 = time.perf_counter()
        ajuste = tune_model(adaptador, internos, espacios[nombre], n_trials, representacion, semilla_tpe)
        segundos_ajuste = time.perf_counter() - t0
        t0 = time.perf_counter()
        for semilla in range(n_semillas if adaptador.estocastico else 1):
            agregar(nombre, semilla, adaptador.ajustar_predecir(ajuste.mejores_hiperparametros, fold, semilla, representacion))
        ajustes.append({
            "campana_prueba": fold.campana_prueba, "modelo": nombre, "n_trials": ajuste.n_trials_ejecutados,
            "rmse_validacion_interna": ajuste.mejor_rmse_validacion, "hiperparametros": str(ajuste.mejores_hiperparametros),
            "segundos_ajuste": segundos_ajuste, "segundos_semillas": time.perf_counter() - t0,
        })
        mejores_hp[nombre] = dict(ajuste.mejores_hiperparametros)
        ensayos[nombre] = ajuste.ensayos

    for nombre, pred in benchmark_predictions(fold, indice_ndvi_max).items():
        agregar(nombre, 0, pred)

    return FoldResult(
        predicciones=pd.concat(filas, ignore_index=True), ajuste=pd.DataFrame(ajustes),
        mejores_hiperparametros=mejores_hp, ensayos=ensayos,
    )


def rmse_per_campaign(predicciones: pd.DataFrame) -> pd.DataFrame:
    """RMSE por modelo y campaña; las semillas se resumen con la mediana (4.12.4, 3.3).

    La unidad de los contrastes es la campaña externa y las semillas son réplicas
    computacionales, no observaciones independientes.
    """
    por_semilla = (
        predicciones.assign(_e=(predicciones["y_pred"] - predicciones["y_real"]) ** 2)
        .groupby(["modelo", "campana_prueba", "semilla"])["_e"].mean().pipe(np.sqrt)
        .rename("rmse").reset_index()
    )
    return por_semilla.groupby(["modelo", "campana_prueba"])["rmse"].median().reset_index()
