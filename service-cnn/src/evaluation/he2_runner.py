"""Insumos de HE2 por campaña de prueba (sección 3.3.3-3.3.4).

Para cada pliego entrena la CNN-1D solo con las campañas anteriores a la de prueba y obtiene,
con varias semillas, (a) la contribución SHAP absoluta por fase y (b) el aumento de RMSE al
ocluir cada fase. Ambos se promedian entre semillas dentro de la campaña (3.3.3) y alimentan
`he2_evaluation.evaluate_he2`.

Las campañas que se pasan aquí son las que el llamador haya permitido: con el piloto serán de
entrenamiento (ensayo técnico no vinculante) y, en el confirmatorio, las externas.
"""
from __future__ import annotations

import numpy as np
import torch

from src.evaluation.experiment_runner import build_adapter
from src.evaluation.fold_builder import FoldData
from src.explainability.fidelity import occlusion_increments_by_phase
from src.explainability.shap_step import campaign_phase_contribution, explain_with_contingency, sample_background


def collect_he2_inputs(
    pliegos: list[FoldData], hp_cnn: dict, representacion: str, fases: list[str],
    n_semillas: int, epocas_cnn: int, n_fondo: int, n_explicadas: int, semilla: int,
    tolerancia_shap: float = 0.05,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Contribución SHAP por fase y aumento de RMSE por oclusión, por campaña de prueba.

    Returns:
        `(contribuciones, aumentos)`, cada uno `{campaña: {fase: valor}}` promediado entre semillas.
        Los aumentos están en la escala estandarizada del objetivo del modelo.

    Raises:
        ValueError: si no hay pliegos o las fases no coinciden con los pasos temporales.
    """
    if not pliegos:
        raise ValueError("Se requiere al menos un pliego para obtener los insumos de HE2.")
    if len(fases) != pliegos[0].X_train.shape[2]:
        raise ValueError(
            f"Las fases ({len(fases)}) deben coincidir con los pasos temporales del tensor ({pliegos[0].X_train.shape[2]})."
        )
    indices_por_fase = {fase: [i] for i, fase in enumerate(fases)}

    contribuciones: dict[int, dict[str, float]] = {}
    aumentos: dict[int, dict[str, float]] = {}
    for pliego in pliegos:
        x_tr, z_tr = (torch.tensor(a, dtype=torch.float32) for a in (pliego.X_train, pliego.Z_train))
        x_te, z_te = (torch.tensor(a, dtype=torch.float32) for a in (pliego.X_test, pliego.Z_test))
        xb, zb = sample_background(x_tr, z_tr, n_fondo, semilla)
        referencia = xb.mean(dim=(0, 2))  # valor de referencia por variable, el mismo fondo que SHAP
        n = min(n_explicadas, len(pliego.y_test))
        objetivo_prueba = pliego.y_test if representacion == "nivel" else pliego.y_anomalia_test

        por_semilla_c, por_semilla_a = [], []
        for s in range(n_semillas):
            modelo, media, desviacion = build_adapter("cnn1d", epocas_cnn).entrenar(hp_cnn, pliego, s, representacion)
            salida = explain_with_contingency(modelo, xb, zb, x_te[:n], z_te[:n], tolerancia_shap)
            por_semilla_c.append(campaign_phase_contribution(salida.explicacion, fases))
            y_z = torch.tensor((objetivo_prueba - media) / desviacion, dtype=torch.float32).unsqueeze(1)
            por_semilla_a.append(occlusion_increments_by_phase(modelo, x_te, z_te, y_z, referencia, indices_por_fase))

        campana = int(np.unique(pliego.campana_test)[0])
        contribuciones[campana] = {f: float(np.mean([d[f] for d in por_semilla_c])) for f in fases}
        aumentos[campana] = {f: float(np.mean([d[f] for d in por_semilla_a])) for f in fases}
    return contribuciones, aumentos
