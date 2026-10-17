"""Serialización de la CNN-1D entrenada (formato portable, sin pickle).

El servicio de solo lectura (`src/api`) no debe depender de PyTorch para responder, así que el modelo
se guarda como arreglos (`.npz`) y metadatos (`.json`) que `src/models/inferencia.py` lee con NumPy:

    modelo.json          tarjeta del modelo: arquitectura, hiperparámetros, campañas, huella, advertencias
    pesos.npz            pesos y estadísticas de BatchNorm de cada semilla, más media y desviación del objetivo
    preprocesador.npz    mediana, media y desviación del tensor y de las covariables (solo del entrenamiento)
    fondo.npz            muestra del entrenamiento (estandarizada) que SHAP usa como referencia
    lineas_base.json     línea base provincial B2 ajustada solo con el entrenamiento

Anti-fuga (RNF1, 4.12.2): todo se ajusta con las campañas de entrenamiento del pliego que se exporta. Las
campañas externas no intervienen: el pliego ya llega sin ellas (`load_pilot_dataset` las descarta).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.evaluation.experiment_runner import build_adapter
from src.evaluation.fold_builder import FoldData
from src.explainability.shap_step import sample_background
from src.features.target_transform import fit_provincial_baseline

FORMATO = 1


def huella_de_pesos(arreglos: dict[str, np.ndarray]) -> str:
    """SHA-256 estable de los pesos (independiente de la compresión o de las marcas de tiempo del zip)."""
    h = hashlib.sha256()
    for nombre in sorted(arreglos):
        h.update(nombre.encode("utf-8"))
        h.update(np.ascontiguousarray(arreglos[nombre]).tobytes())
    return h.hexdigest()


def _lineas_base(pliego: FoldData) -> dict:
    historia = pliego.obs_train[["provincia_id", "campana_id", "rendimiento_kg_ha"]]
    provincias = {}
    for p in sorted(historia["provincia_id"].unique()):
        b = fit_provincial_baseline(historia, str(p))
        provincias[str(p)] = {
            "pendiente": b.pendiente,
            "intercepto": b.intercepto,
            "campana_referencia": b.campana_referencia,
            "es_respaldo": b.es_respaldo,
        }
    return {"media_global": float(pliego.y_train.mean()), "provincias": provincias}


def exportar_cnn1d(
    pliego: FoldData,
    hp: dict,
    representacion: str,
    epocas: int,
    n_semillas: int,
    destino: Path,
    contexto: dict,
    n_fondo: int = 100,
    semilla_fondo: int = 0,
) -> dict:
    """Entrena la CNN-1D con las semillas 0..n-1 sobre el entrenamiento del pliego y la guarda en `destino`.

    Es exactamente el entrenamiento del piloto (`AdaptadorCNN.entrenar`), de modo que las predicciones
    reproducen las del pliego. Devuelve la tarjeta del modelo que se escribió en `modelo.json`.

    Args:
        contexto: datos de procedencia para la tarjeta: `orden_variables`, `orden_fases`, `covariables`,
            `campanas_permitidas`, `sha256_tensores`, `modo` y `vinculante` (del piloto de origen).
    """
    if representacion != "anomalia":
        raise ValueError("El servicio solo sirve la representación en anomalía (la línea base B2 se suma al final).")
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    arreglos: dict[str, np.ndarray] = {}
    medias, desviaciones = [], []
    config = None
    for semilla in range(n_semillas):
        modelo, media, desviacion = build_adapter("cnn1d", epocas).entrenar(hp, pliego, semilla, representacion)
        config = modelo.config
        for nombre, valor in modelo.state_dict().items():
            if nombre.endswith("num_batches_tracked"):
                continue
            arreglos[f"s{semilla}.{nombre}"] = valor.detach().cpu().numpy().astype(np.float64)
        medias.append(media)
        desviaciones.append(desviacion)
    arreglos["y_media"] = np.array(medias)
    arreglos["y_desviacion"] = np.array(desviaciones)
    np.savez_compressed(destino / "pesos.npz", **arreglos)

    prep = pliego.preprocesador
    np.savez_compressed(
        destino / "preprocesador.npz",
        mediana=prep.mediana, media=prep.media, desviacion=prep.desviacion,
        z_media=prep.z_media, z_desviacion=prep.z_desviacion,
    )
    # Fondo de referencia de SHAP: muestra reproducible del ENTRENAMIENTO (nunca de la campaña de prueba), la
    # misma que usa el piloto (100 filas, semilla 0).
    fondo_x, fondo_z = sample_background(
        torch.tensor(pliego.X_train, dtype=torch.float32), torch.tensor(pliego.Z_train, dtype=torch.float32),
        n_fondo, semilla_fondo,
    )
    np.savez_compressed(destino / "fondo.npz", x=fondo_x.numpy().astype(np.float64), z=fondo_z.numpy().astype(np.float64))
    (destino / "lineas_base.json").write_text(json.dumps(_lineas_base(pliego), indent=1), encoding="utf-8")

    tarjeta = {
        "formato": FORMATO,
        "huella_pesos": huella_de_pesos(arreglos),
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "arquitectura": asdict(config),
        "representacion": representacion,
        "hiperparametros": {k: (float(v) if isinstance(v, float) else v) for k, v in hp.items()},
        "epocas_maximas": epocas,
        "semillas": list(range(n_semillas)),
        "campanas_entrenamiento": pliego.campanas_entrenamiento,
        "n_observaciones_entrenamiento": int(len(pliego.y_train)),
        "advertencia": (
            "Modelo del piloto entrenado con las campañas de entrenamiento indicadas. No es el modelo "
            "definitivo de la evaluación confirmatoria y sus resultados no son vinculantes salvo que el "
            "piloto de origen lo sea."
        ),
        **contexto,
    }
    (destino / "modelo.json").write_text(json.dumps(tarjeta, indent=1, ensure_ascii=False), encoding="utf-8")
    return tarjeta
