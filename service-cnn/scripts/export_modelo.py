"""Serializa la CNN-1D del piloto para que el servicio pueda predecir sobre un tensor.

Usa el MISMO entrenamiento que el piloto en su último pliego (campañas de entrenamiento anteriores a la
última campaña de prueba): mismos hiperparámetros (P7), misma representación (P4) y semillas 0..n-1. Los
toma de `reports/piloto_oficial/resultados.json`, de modo que el modelo servido es el que se evaluó.

Las campañas externas no se leen: `load_pilot_dataset` descarta de inmediato todo lo que no esté en las
campañas permitidas del piloto.

Uso (desde service-cnn/):
    .venv/Scripts/python.exe scripts/export_modelo.py
    .venv/Scripts/python.exe scripts/export_modelo.py --salida models/cnn1d_piloto --epocas-cnn 100

Al terminar compara las predicciones del modelo serializado con `predicciones.csv` del piloto; si no
coinciden, termina con error y no hay que servir ese modelo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.fold_builder import build_folds, load_pilot_dataset  # noqa: E402
from src.models.exportacion import exportar_cnn1d  # noqa: E402
from src.models.inferencia import ModeloCNN1D  # noqa: E402

TOLERANCIA_KG_HA = 1e-2  # float32 de PyTorch frente a float64 de NumPy


def verificar(modelo: ModeloCNN1D, pliego, predicciones_csv: Path) -> float | None:
    """Diferencia máxima (kg/ha) entre el modelo serializado y las predicciones del piloto para el pliego."""
    if not predicciones_csv.exists():
        return None
    piloto = pd.read_csv(predicciones_csv)
    piloto = piloto[(piloto["modelo"] == "cnn1d") & (piloto["campana_prueba"] == pliego.campana_prueba)]
    if piloto.empty:
        return None
    # El piloto guarda y_pred = B2 + anomalía por provincia y semilla.
    anomalia = modelo.predecir_anomalia(pliego.X_test_crudo, _z_crudo_prueba(pliego))
    maximo = 0.0
    for i, semilla in enumerate(modelo.tarjeta["semillas"]):
        esperado = piloto[piloto["semilla"] == semilla].set_index("provincia_id")["y_pred"]
        propio = pd.Series(pliego.baseline_test + anomalia[i], index=pliego.provincia_test)
        maximo = max(maximo, float(np.abs(propio - esperado.reindex(propio.index)).max()))
    return maximo


def _z_crudo_prueba(pliego) -> np.ndarray:
    # Las covariables de prueba ya estandarizadas se devuelven a su escala con las estadísticas del entrenamiento.
    p = pliego.preprocesador
    return pliego.Z_test * p.z_desviacion + p.z_media


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procesado", type=Path, default=Path("data/processed"))
    ap.add_argument("--piloto", type=Path, default=Path("reports/piloto_oficial"))
    ap.add_argument("--salida", type=Path, default=Path("models/cnn1d_piloto"))
    ap.add_argument("--ventana-piloto", type=int, default=3)
    ap.add_argument("--epocas-cnn", type=int, default=100, help="Igual que --epocas-cnn del piloto (por defecto 100).")
    ap.add_argument("--n-semillas", type=int, default=10)
    args = ap.parse_args()

    resultados = json.loads((args.piloto / "resultados.json").read_text(encoding="utf-8"))
    p0, representacion = resultados["p0"], resultados["p4"]["representacion_elegida"]
    hp = resultados["p7"]["hiperparametros_cnn"]
    permitidas = p0["campanas_permitidas"]

    obs, X, Z, descartadas = load_pilot_dataset(args.procesado, permitidas)
    print(f"Campañas permitidas {permitidas}; descartadas al cargar {descartadas}.")
    pliego = build_folds(obs, X, Z, args.ventana_piloto, permitidas)[-1]
    print(f"Pliego: entrena con {pliego.campanas_entrenamiento}, prueba {pliego.campana_prueba}. hp={hp}")

    metadata = json.loads((args.procesado / "metadata.json").read_text(encoding="utf-8"))
    tarjeta = exportar_cnn1d(
        pliego, hp, representacion, args.epocas_cnn, args.n_semillas, args.salida,
        contexto={
            "orden_variables": metadata["orden_variables"], "orden_fases": metadata["orden_fases"],
            "covariables": metadata["covariables"], "campanas_permitidas": permitidas,
            "campana_prueba_del_pliego": pliego.campana_prueba,
            "sha256_tensores": p0["sha256_tensores"], "modo": p0["modo"], "vinculante": p0["vinculante"],
        },
    )
    print(f"Modelo guardado en {args.salida} (huella {tarjeta['huella_pesos'][:12]}).")

    diferencia = verificar(ModeloCNN1D.cargar(args.salida), pliego, args.piloto / "predicciones.csv")
    if diferencia is None:
        print("No hay predicciones.csv del piloto para comparar; no se verificó.")
        return 0
    print(f"Diferencia máxima frente a las predicciones del piloto: {diferencia:.6f} kg/ha")
    if diferencia > TOLERANCIA_KG_HA:
        print(f"ERROR: supera {TOLERANCIA_KG_HA} kg/ha. El modelo serializado NO reproduce al piloto; no lo sirva.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
