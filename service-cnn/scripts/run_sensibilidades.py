"""Análisis de sensibilidad sobre las campañas de entrenamiento (4.14), SIN campañas externas.

Corre el protocolo del piloto (representación anomalía, 30 trials iguales para los 5 modelos, 10
semillas) sobre variantes del conjunto de datos y compara cada una con la base:
  base · sin_transicion · tensor_completo · horizonte_temprano   (mismo conjunto procesado)
  sin_arrastre_ndvi · con_exclusion_por_altitud                  (otro conjunto procesado)
Además, el error por estrato de participación de la quinua. Es exploratorio.

Uso:
    python scripts/run_sensibilidades.py --ventana-confirmatoria 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.fold_builder import load_pilot_dataset, restrict_to_campaigns  # noqa: E402
from src.evaluation.pilot_guard import pilot_campaigns  # noqa: E402
from src.evaluation.sensitivity import (  # noqa: E402
    VARIANTES,
    apply_variant,
    error_by_stratum,
    predict_sensitivity,
    summarize_sensitivity,
)
from src.evaluation.experiment_runner import rmse_per_campaign  # noqa: E402

MODELOS_EN_TABLA = ("cnn1d", "xgboost", "random_forest", "elastic_net", "svr", "B2", "B4")


def _cargar(directorio: Path, permitidas):
    obs, X, Z, _ = load_pilot_dataset(directorio, permitidas)
    todo = pd.read_parquet(directorio / "observaciones.parquet")
    datos = np.load(directorio / "tensores.npz")
    _, imp_plano, _, _ = restrict_to_campaigns(todo, datos["imputado"].reshape(len(todo), -1), datos["Z"], permitidas)
    return obs, X, Z, imp_plano.reshape(X.shape)


def _tabla_md(df: pd.DataFrame, decimales: int = 1) -> str:
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    filas = ["| " + " | ".join(f"{v:.{decimales}f}" if isinstance(v, (float, np.floating)) else str(v) for v in f) + " |" for f in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--procesado", type=Path, default=Path("data/processed"))
    p.add_argument("--procesado-sin-arrastre", type=Path, default=Path("data/processed_sens_sin_arrastre"))
    p.add_argument("--procesado-altitud", type=Path, default=Path("data/processed_sens_altitud"))
    p.add_argument("--hyperparams", type=Path, default=Path("configs/hyperparams.yaml"))
    p.add_argument("--salida", type=Path, default=Path("reports/sensibilidades"))
    p.add_argument("--ventana-confirmatoria", type=int, default=5)
    p.add_argument("--ventana-piloto", type=int, default=3)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--n-semillas", type=int, default=10)
    p.add_argument("--epocas-cnn", type=int, default=47)
    p.add_argument("--semilla", type=int, default=0)
    a = p.parse_args()

    espacios = {k: v for k, v in yaml.safe_load(a.hyperparams.read_text(encoding="utf-8")).items()
                if k in ("cnn1d", "elastic_net", "random_forest", "xgboost", "svr")}
    permitidas = pilot_campaigns(2016, a.ventana_confirmatoria)
    kw = dict(espacios=espacios, permitidas=permitidas, ventana_piloto=a.ventana_piloto, ventana_interna=2,
              n_trials=a.n_trials, n_semillas=a.n_semillas, epocas_cnn=a.epocas_cnn, indice_ndvi_max=5, semilla=a.semilla)

    a.salida.mkdir(parents=True, exist_ok=True)
    obs, X, Z, imp = _cargar(a.procesado, permitidas)
    tablas, predicciones = {}, {}
    for nombre in VARIANTES:
        o, x, z = apply_variant(nombre, obs, X, Z, imp)
        print(f"[{nombre}] {len(o)} observaciones", flush=True)
        pred = predict_sensitivity(o, x, z, **kw)
        predicciones[nombre] = pred
        tablas[nombre] = rmse_per_campaign(pred)
    for nombre, directorio in (("sin_arrastre_ndvi", a.procesado_sin_arrastre), ("con_exclusion_por_altitud", a.procesado_altitud)):
        o, x, z, _ = load_pilot_dataset(directorio, permitidas)
        print(f"[{nombre}] {len(o)} observaciones", flush=True)
        pred = predict_sensitivity(o, x, z, **kw)
        predicciones[nombre] = pred
        tablas[nombre] = rmse_per_campaign(pred)

    resumen = summarize_sensitivity(tablas)
    resumen.to_csv(a.salida / "sensibilidades_rmse.csv", index=False)
    estratos = error_by_stratum(predicciones["base"], obs, "prop_quinua", 3)
    estratos.to_csv(a.salida / "error_por_estrato_de_quinua.csv", index=False)

    vistos = resumen[resumen["modelo"].isin(MODELOS_EN_TABLA)]
    ancho = vistos.pivot_table(index=["variante", "modelo"], columns="campana_prueba", values="rmse").reset_index()
    ancho.columns = [str(c) if not isinstance(c, str) else c for c in ancho.columns]
    cnn = resumen[resumen["modelo"] == "cnn1d"][["variante", "campana_prueba", "rmse", "q_frente_a_b2", "q_frente_a_b4", "cambio_vs_base_pct"]]
    est = estratos[estratos["modelo"].isin(["cnn1d", "xgboost", "B2", "B4"])].pivot_table(index="modelo", columns="estrato", values="rmse", observed=True).reset_index()

    informe = f"""# Sensibilidades sobre las campañas de entrenamiento (exploratorio)

Campañas 2016-{max(permitidas)}; pliegos de prueba 2019 y 2020. Mismo protocolo del piloto oficial (anomalía, {a.n_trials} trials para los 5 modelos, {a.n_semillas} semillas). **No usa campañas externas y no decide ninguna hipótesis.**

## RMSE (kg/ha) por variante, modelo y campaña de prueba
{_tabla_md(ancho)}

## CNN-1D frente a las líneas base, por variante
`q_frente_a_*` es RMSE(CNN-1D)/RMSE(referencia) − 1; `cambio_vs_base_pct` es el cambio del RMSE respecto de la variante base.

{_tabla_md(cnn, 3)}

## Error por estrato de participación de la quinua (variante base; terciles de `prop_quinua`)
{_tabla_md(est)}
"""
    (a.salida / "informe_sensibilidades.md").write_text(informe, encoding="utf-8", newline="\n")
    print(f"Informe en {a.salida / 'informe_sensibilidades.md'}")


if __name__ == "__main__":
    main()
