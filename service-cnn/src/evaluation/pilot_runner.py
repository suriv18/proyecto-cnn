"""Orquestador del ensayo piloto (tarea T7; experimentos P0-P10 del plan).

Compone la guarda de campañas, el constructor de pliegos, el ejecutor por pliego, la
prueba de fuga, las reglas de decisión y el paso de SHAP. Garantías:

  - Solo se leen las campañas permitidas (las primeras `ventana_confirmatoria`); las
    demás se descartan al cargar y solo se registran sus identificadores (P0).
  - El modo `oficial` exige el gate de F2 cerrado; el modo `tecnico` (desarrollo) se
    marca NO VINCULANTE y no fija ningún parámetro de la preespecificación.
  - Los cinco modelos reciben el mismo presupuesto de ajuste (4.12.3).

Ver `docs/crisp/F3-plan-del-piloto.md` para el diseño y las reglas de decisión.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.evaluation.experiment_runner import MODELOS, build_adapter, rmse_per_campaign, run_fold
from src.evaluation.fold_builder import build_folds, load_pilot_dataset, restrict_to_campaigns
from src.evaluation.leak_test import run_permutation_test
from src.evaluation.pilot_experiments import (
    baseline_forms,
    border_check,
    decide_baseline_form,
    decide_representation,
    max_epochs_from_best,
    seed_variability,
)
from src.evaluation.pilot_guard import pilot_campaigns
from src.explainability.shap_aggregation import fase_de_maxima_contribucion
from src.explainability.shap_step import (
    campaign_phase_contribution,
    explain_with_contingency,
    explain_with_kernel_explainer,
    sample_background,
)

FASES_POR_DEFECTO = ["emergencia", "desarrollo_vegetativo", "floracion", "llenado_grano", "madurez"]
BENCHMARKS = ("B1", "B2", "B3", "B4")


class PilotGateError(RuntimeError):
    """Se intentó una corrida oficial del piloto sin el gate de F2 cerrado."""


@dataclass
class PilotConfig:
    directorio_procesado: Path
    directorio_salida: Path
    espacios: dict
    ventana_confirmatoria: int  # nº de campañas de la ventana inicial confirmatoria: fija las campañas permitidas
    ventana_piloto: int = 3
    ventana_interna: int = 2
    n_trials: int = 30
    n_semillas: int = 10
    epocas_cnn: int = 100
    n_permutaciones: int = 10
    n_explicadas: int = 30
    n_fondo_shap: int = 100
    tolerancia_shap: float = 0.05
    indice_ndvi_max: int = 5
    primera_campana: int = 2016
    modo: str = "tecnico"
    gate_f2_aprobado: bool = False
    semilla: int = 0
    ruta_preespecificacion: Path | None = Path("configs/preespecificacion.yaml")


@dataclass
class PilotResults:
    p0: dict
    p1: pd.DataFrame
    p2: pd.DataFrame
    p3: pd.DataFrame
    p4: dict
    p5: dict
    p6: pd.DataFrame
    p7: dict
    p8: dict
    p9: pd.DataFrame
    p10: pd.DataFrame
    predicciones: pd.DataFrame = field(default_factory=pd.DataFrame)


def _hp_central(espacio: dict) -> dict:
    hp = {}
    for nombre, (tipo, minimo, maximo) in espacio.items():
        medio = (minimo + maximo) / 2.0
        hp[nombre] = int(round(medio)) if tipo == "int" else float(medio)
    return hp


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def run_pilot(cfg: PilotConfig) -> PilotResults:
    """Ejecuta P0-P10 sobre los pliegos del piloto y devuelve los resultados.

    Raises:
        PilotGateError: en modo `oficial` sin el gate de F2 cerrado.
    """
    if cfg.modo == "oficial" and not cfg.gate_f2_aprobado:
        raise PilotGateError(
            "El modo oficial exige el gate de F2 cerrado (aprobación del asesor de las decisiones de E5). "
            "Use el modo técnico para desarrollo; sus resultados no son vinculantes."
        )

    # ---------------- P0: puerta de entrada ----------------
    permitidas = pilot_campaigns(cfg.primera_campana, cfg.ventana_confirmatoria)
    obs, X, Z, descartadas = load_pilot_dataset(cfg.directorio_procesado, permitidas)
    meta_ruta = Path(cfg.directorio_procesado) / "metadata.json"
    fases = json.loads(meta_ruta.read_text(encoding="utf-8")).get("orden_fases") if meta_ruta.exists() else None
    if not fases or len(fases) != X.shape[2]:
        fases = FASES_POR_DEFECTO if X.shape[2] == len(FASES_POR_DEFECTO) else [f"paso_{i}" for i in range(X.shape[2])]

    discrepancias, pendientes = None, None
    if cfg.ruta_preespecificacion is not None and Path(cfg.ruta_preespecificacion).exists():
        from src.evaluation.preregistration import check_consistency_with_code, load_preregistration, pending_decisions

        prereg = load_preregistration(cfg.ruta_preespecificacion)
        discrepancias = check_consistency_with_code(prereg)
        pendientes = {k: len(v) for k, v in pending_decisions(prereg).items()}

    pliegos = build_folds(obs, X, Z, cfg.ventana_piloto, permitidas)
    p0 = {
        "modo": cfg.modo,
        "vinculante": bool(cfg.modo == "oficial" and cfg.gate_f2_aprobado),
        "campanas_permitidas": permitidas,
        "campanas_descartadas": descartadas,
        "campanas_de_prueba": [p.campana_prueba for p in pliegos],
        "n_observaciones": int(len(obs)),
        "sha256_tensores": hashlib.sha256((Path(cfg.directorio_procesado) / "tensores.npz").read_bytes()).hexdigest(),
        "preespecificacion_discrepancias_con_el_codigo": discrepancias,
        "preespecificacion_pendientes": pendientes,
    }

    # ---------------- ajuste de los 5 modelos, en las dos representaciones ----------------
    por_rep: dict[str, list] = {}
    for rep in ("nivel", "anomalia"):
        por_rep[rep] = [
            run_fold(
                f, cfg.espacios, cfg.n_trials, cfg.n_semillas, cfg.ventana_interna, rep,
                cfg.indice_ndvi_max, cfg.epocas_cnn, cfg.semilla,
            )
            for f in pliegos
        ]
    preds = {rep: pd.concat([r.predicciones for r in rs], ignore_index=True) for rep, rs in por_rep.items()}
    rmse = {rep: rmse_per_campaign(p) for rep, p in preds.items()}

    # ---------------- P2 y P3 ----------------
    p2 = rmse["nivel"][rmse["nivel"]["modelo"].isin(BENCHMARKS)].reset_index(drop=True)
    p3 = pd.concat(
        [r.ajuste.assign(representacion=rep) for rep, rs in por_rep.items() for r in rs], ignore_index=True
    )

    # ---------------- P4 y P5 ----------------
    solo_modelos = {rep: t[t["modelo"].isin(MODELOS)] for rep, t in rmse.items()}
    p4 = decide_representation(solo_modelos["nivel"], solo_modelos["anomalia"])
    elegida = p4["decision"]
    tabla_b2 = pd.concat([baseline_forms(f) for f in pliegos], ignore_index=True)
    p5 = {**decide_baseline_form(tabla_b2), "tabla": tabla_b2}

    # ---------------- P6 ----------------
    p6 = seed_variability(preds[elegida])

    # ---------------- P1: prueba de fuga (último pliego) ----------------
    ultimo = pliegos[-1]
    hp_centrales = {m: _hp_central(cfg.espacios[m]) for m in MODELOS}
    p1 = run_permutation_test(
        ultimo, hp_centrales, MODELOS, cfg.n_permutaciones, cfg.semilla, epocas_cnn=cfg.epocas_cnn
    )

    # ---------------- P7: entrenamiento de la CNN (partición del early stopping) ----------------
    hp_cnn = por_rep[elegida][-1].mejores_hiperparametros["cnn1d"]
    filas7 = []
    for particion in ("temporal", "aleatoria"):
        for semilla in range(cfg.n_semillas):
            adaptador = build_adapter("cnn1d", cfg.epocas_cnn, particion)
            pred = adaptador.ajustar_predecir(hp_cnn, ultimo, semilla, elegida)
            filas7.append({
                "particion": particion, "semilla": semilla, "mejor_epoca": adaptador.ultima_mejor_epoca,
                "rmse_prueba": _rmse(ultimo.y_test, pred),
            })
    tabla7 = pd.DataFrame(filas7)
    p7 = {
        "tabla": tabla7,
        "epocas_maximas": max_epochs_from_best(tabla7.loc[tabla7.particion == "temporal", "mejor_epoca"].tolist()),
        "hiperparametros_cnn": hp_cnn,
    }

    # ---------------- P8: SHAP ----------------
    adaptador = build_adapter("cnn1d", cfg.epocas_cnn)
    modelo, _, _ = adaptador.entrenar(hp_cnn, ultimo, 0, elegida)
    x_tr, z_tr = (torch.tensor(a, dtype=torch.float32) for a in (ultimo.X_train, ultimo.Z_train))
    n = min(cfg.n_explicadas, len(ultimo.y_test))
    x_te, z_te = (torch.tensor(a[:n], dtype=torch.float32) for a in (ultimo.X_test, ultimo.Z_test))
    xb, zb = sample_background(x_tr, z_tr, cfg.n_fondo_shap, cfg.semilla)
    t0 = time.perf_counter()
    salida = explain_with_contingency(modelo, xb, zb, x_te, z_te, cfg.tolerancia_shap)
    segundos_deep = time.perf_counter() - t0
    t0 = time.perf_counter()
    kernel = explain_with_kernel_explainer(modelo, xb[:50], zb[:50], x_te, z_te)
    segundos_kernel = time.perf_counter() - t0
    c_usada = campaign_phase_contribution(salida.explicacion, fases)
    c_kernel = campaign_phase_contribution(kernel, fases)
    rho = float(stats.spearmanr([c_usada[f] for f in fases], [c_kernel[f] for f in fases]).statistic) if len(fases) > 2 else float("nan")
    p8 = {
        "variante": salida.variante, "tolerancia": salida.tolerancia, "error_deep_explainer": salida.error_deep_explainer,
        "n_explicadas": n, "n_fondo": int(xb.shape[0]), "spearman_por_fase_contra_kernel": rho,
        "fase_maxima": fase_de_maxima_contribucion(c_usada), "fase_maxima_kernel": fase_de_maxima_contribucion(c_kernel),
        "segundos_explicador": segundos_deep, "segundos_kernel": segundos_kernel,
        "contribucion_por_fase": c_usada,
    }

    # ---------------- P9: bordes de los espacios de búsqueda ----------------
    filas9 = []
    for modelo_nombre in MODELOS:
        ensayos = pd.concat([r.ensayos[modelo_nombre] for r in por_rep[elegida]], ignore_index=True)
        if len(ensayos) and cfg.espacios[modelo_nombre]:
            filas9.append(border_check(ensayos, cfg.espacios[modelo_nombre]).assign(modelo=modelo_nombre))
    p9 = pd.concat(filas9, ignore_index=True)[["modelo", "parametro", "lado", "fraccion_en_borde", "n_ensayos_evaluados"]]

    # ---------------- P10: curva de aprendizaje de XGBoost ----------------
    prueba = ultimo.campana_prueba
    anteriores = sorted(c for c in permitidas if c < prueba)
    hp_xgb = por_rep[elegida][-1].mejores_hiperparametros["xgboost"]
    filas10 = []
    for w in range(1, len(anteriores) + 1):
        usar = anteriores[-w:] + [prueba]
        o_sub, x_sub, z_sub, _ = restrict_to_campaigns(obs, X, Z, usar)
        pliego_w = build_folds(o_sub, x_sub, z_sub, w, usar)[-1]
        pred = build_adapter("xgboost").ajustar_predecir(hp_xgb, pliego_w, 0, elegida)
        filas10.append({"n_campanas_entrenamiento": w, "n_observaciones_entrenamiento": len(pliego_w.y_train), "rmse_xgboost": _rmse(pliego_w.y_test, pred)})
    p10 = pd.DataFrame(filas10)

    return PilotResults(
        p0=p0, p1=p1, p2=p2, p3=p3, p4={**p4, "representacion_elegida": elegida}, p5=p5, p6=p6, p7=p7,
        p8=p8, p9=p9, p10=p10, predicciones=preds[elegida],
    )


# ============================================================================ salidas
def _json_seguro(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return str(obj)


def save_pilot_results(res: PilotResults, directorio: str | Path) -> None:
    """Escribe los resultados en CSV/JSON y el informe en Markdown."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    res.p1.assign(r2_por_permutacion=res.p1["r2_por_permutacion"].astype(str)).to_csv(directorio / "p1_prueba_de_fuga.csv", index=False)
    res.p2.to_csv(directorio / "p2_referencias.csv", index=False)
    res.p3.to_csv(directorio / "p3_costo_y_ajuste.csv", index=False)
    res.p5["tabla"].to_csv(directorio / "p5_formas_de_b2.csv", index=False)
    res.p6.to_csv(directorio / "p6_variabilidad_por_semilla.csv", index=False)
    res.p7["tabla"].to_csv(directorio / "p7_entrenamiento_cnn.csv", index=False)
    res.p9.to_csv(directorio / "p9_bordes.csv", index=False)
    res.p10.to_csv(directorio / "p10_curva_de_aprendizaje.csv", index=False)
    res.predicciones.to_csv(directorio / "predicciones.csv", index=False)
    resumen = {
        "p0": res.p0, "p4": res.p4, "p5": {k: v for k, v in res.p5.items() if k != "tabla"},
        "p7": {k: v for k, v in res.p7.items() if k != "tabla"}, "p8": res.p8,
    }
    (directorio / "resultados.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=_json_seguro), encoding="utf-8")
    (directorio / "informe_piloto.md").write_text(render_pilot_report(res), encoding="utf-8", newline="\n")


def _tabla(df: pd.DataFrame, decimales: int = 3) -> str:
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    filas = ["| " + " | ".join(f"{v:.{decimales}f}" if isinstance(v, (float, np.floating)) else str(v) for v in f) + " |" for f in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


def render_pilot_report(res: PilotResults) -> str:
    """Informe del piloto en Markdown (el de la corrida técnica lleva el aviso de no vinculante)."""
    p0 = res.p0
    aviso = (
        "> **NO VINCULANTE.** Corrida técnica de desarrollo: no fija ningún parámetro de E5 ni cuenta como evidencia de la tesis. "
        "El piloto oficial se repite cuando el asesor cierre el gate de F2.\n"
        if not p0["vinculante"] else "> Piloto oficial: gate de F2 cerrado.\n"
    )
    p4, p5, p7, p8 = res.p4, res.p5, res.p7, res.p8
    pasa_fuga = bool(res.p1["pasa"].all())
    bordes = res.p9[res.p9["lado"] != "ninguno"]
    return f"""# Informe del ensayo piloto ({p0['modo']})

{aviso}
## P0 · Puerta de entrada
- Campañas permitidas: {p0['campanas_permitidas']}; **descartadas al cargar (solo sus identificadores): {', '.join(map(str, p0['campanas_descartadas'])) or 'ninguna'}**.
- Campañas de prueba del piloto: {p0['campanas_de_prueba']} · observaciones: {p0['n_observaciones']}.
- sha256 de `tensores.npz`: `{p0['sha256_tensores'][:16]}…`
- Pendientes de E5: {p0['preespecificacion_pendientes']} · discrepancias con el código: {p0['preespecificacion_discrepancias_con_el_codigo']}

## P1 · Prueba de fuga por permutación — {'PASA' if pasa_fuga else 'NO PASA'}
{_tabla(res.p1[['modelo', 'r2_mediano', 'r2_maximo', 'pasa']])}

## P2 · Referencias (RMSE en kg/ha)
{_tabla(res.p2)}

## P3 · Costo y presupuesto (igual para los 5 modelos)
{_tabla(res.p3[['representacion', 'modelo', 'n_trials', 'segundos_ajuste', 'segundos_semillas', 'rmse_validacion_interna']], 2)}

## P4 · Representación del objetivo
Nivel {p4['media_nivel']:.1f} kg/ha · anomalía {p4['media_anomalia']:.1f} kg/ha (diferencia relativa {100 * p4['diferencia_relativa']:.1f} %) → **{p4['decision']}**.

## P5 · Forma de B2
{_tabla(res.p5['tabla'])}
Decisión: **{p5['decision']}** (empate < 2 % → media).

## P6 · Variabilidad por semilla
{_tabla(res.p6)}

## P7 · Entrenamiento de la CNN-1D
{_tabla(p7['tabla'].groupby('particion').agg(mejor_epoca_mediana=('mejor_epoca', 'median'), rmse_mediano=('rmse_prueba', 'median')).reset_index())}
Épocas máximas propuestas: **{p7['epocas_maximas']}**.

## P8 · SHAP
Variante: **{p8['variante']}** · error de precisión local de DeepExplainer {p8['error_deep_explainer']:.4f} (tolerancia {p8['tolerancia']:.4f}) · Spearman por fase contra Kernel {p8['spearman_por_fase_contra_kernel']:.2f} · fase máxima: {p8['fase_maxima']} (Kernel: {p8['fase_maxima_kernel']}).

## P9 · Bordes de los espacios de búsqueda
{_tabla(bordes) if len(bordes) else 'Ningún hiperparámetro agrupa sus mejores ensayos en un borde.'}

## P10 · Curva de aprendizaje (XGBoost)
{_tabla(res.p10)}
"""
