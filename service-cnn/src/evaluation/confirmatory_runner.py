"""Orquestador de la evaluación confirmatoria (E5 congelada): HE1a, HE2c y la hipótesis general.

Compone lo que ya existe (pliegos de origen móvil, ajuste con presupuesto equiparado, SHAP, oclusión y las
reglas de decisión de `he1_evaluation` y `he2_evaluation`) en un procedimiento que se ejecuta sin ajuste
manual (control C-F4-4 del plan de F4). Dos modos:

  - `ensayo` (por defecto): ensaya el procedimiento completo con las campañas de entrenamiento del piloto
    (las `ventana` primeras de E5, 2016-2020). Las campañas externas se descartan al cargar y no se leen sus
    valores. Hace pasar como «externas» las últimas campañas permitidas con una ventana inicial menor (3 por
    defecto), de modo que el resultado NO es evidencia de la tesis: solo prueba el procedimiento.
  - `oficial`: la evaluación confirmatoria real sobre las campañas externas. Es irrepetible: exige E5
    congelada y coherente con el código, la huella de los datos registrada, un conjunto de datos no
    exploratorio, una confirmación explícita y un directorio de salida sin ejecuciones previas. Todos los
    parámetros salen de E5; la línea de comandos no puede cambiarlos.

B* (la mejor línea base entre B2 y B4) se elige con la validación interna del pliego inicial, que solo contiene
campañas de entrenamiento, antes de mirar cualquier campaña externa (3.3.1).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.experiment_runner import build_adapter, benchmark_predictions, reconstruct_prediction, rmse_per_campaign, run_fold
from src.evaluation.fold_builder import FoldData, build_folds, build_inner_folds, load_pilot_dataset
from src.evaluation.he1_evaluation import He1Result, evaluate_he1, select_best_baseline
from src.evaluation.he2_evaluation import He2Result, evaluate_he2
from src.evaluation.pilot_guard import pilot_campaigns
from src.evaluation.preregistration import (
    PreregistrationNotReadyError,
    assert_can_freeze,
    check_consistency_with_code,
    iter_parameters,
    load_preregistration,
)
from src.explainability.fidelity import occlusion_increments_by_phase
from src.explainability.shap_step import campaign_phase_contribution, explain_with_contingency, sample_background

FASES_POR_DEFECTO = ["emergencia", "desarrollo_vegetativo", "floracion", "llenado_grano", "madurez"]
ARCHIVO_DE_EJECUCION = "ejecucion_confirmatoria.json"
N_FONDO_SHAP = 100  # 4.15: muestra de hasta 100 observaciones del entrenamiento, semilla fija


class ConfirmatoryGateError(RuntimeError):
    """No se cumple una condición previa de la evaluación confirmatoria."""


@dataclass
class ConfirmatoryConfig:
    directorio_procesado: Path
    directorio_salida: Path
    espacios: dict
    modo: str = "ensayo"
    ruta_preespecificacion: Path = Path("configs/preespecificacion.yaml")
    confirmo_abrir_campanas_externas: bool = False
    # Solo en modo ensayo (en oficial deben ser None: salen de E5). Sirven para abaratar las pruebas.
    ventana_inicial: int | None = None
    n_trials: int | None = None
    n_semillas: int | None = None
    epocas_cnn: int | None = None
    n_permutaciones: int = 10000  # estabilidad (secundaria, descriptiva); no es un parámetro de decisión
    primera_campana: int = 2016
    semilla: int = 0
    tolerancia_shap: float | None = None


@dataclass
class ConfirmatoryResults:
    p0: dict
    predicciones: pd.DataFrame
    rmse_por_campana: pd.DataFrame
    ajuste: pd.DataFrame
    he1: He1Result
    he2: He2Result
    contribuciones: dict[int, dict[str, float]]
    aumentos_rmse: dict[int, dict[str, float]]
    b_star: str
    rmse_validacion_interna_b: dict[str, float]
    hipotesis_general_respaldada: bool
    verificaciones: dict = field(default_factory=dict)


def _parametro(prereg: dict, nombre: str):
    """Valor del único parámetro de E5 cuya ruta termina en `nombre`."""
    encontrados = [p["valor"] for ruta, p in iter_parameters(prereg) if ruta == nombre or ruta.endswith("." + nombre)]
    if len(encontrados) != 1:
        raise ConfirmatoryGateError(f"E5 debe tener exactamente un parámetro '{nombre}'; hay {len(encontrados)}.")
    return encontrados[0]


def parametros_de_e5(prereg: dict) -> dict:
    """Los parámetros de la evaluación, tal como los fijó E5."""
    return {
        "ventana_inicial": int(_parametro(prereg, "ventana_inicial_de_entrenamiento_campanas")),
        "ventana_interna": int(_parametro(prereg, "ventana_inicial_interna_campanas")),
        "n_trials": int(_parametro(prereg, "presupuesto_de_trials_por_modelo")),
        "n_semillas": int(_parametro(prereg, "n_replicas")),
        "epocas_cnn": int(_parametro(prereg, "epocas_maximas")),
        "representacion": str(_parametro(prereg, "representacion_de_modelado")),
        "margen": float(_parametro(prereg, "margen_no_inferioridad_relativo")),
        "alfa": float(_parametro(prereg, "nivel_de_significancia")),
        "n_fases_principales": int(_parametro(prereg, "n_fases_principales")),
        "tolerancia_shap": float(_parametro(prereg, "tolerancia_de_precision_local")),
        "horizonte": "principal",
    }


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def comprobar_puertas(cfg: ConfirmatoryConfig) -> tuple[dict, dict]:
    """Puerta de entrada: devuelve `(preespecificacion, parametros)` o lanza `ConfirmatoryGateError`.

    Las condiciones comunes se exigen siempre; las propias del modo oficial se suman a ellas.
    """
    if cfg.modo not in ("ensayo", "oficial"):
        raise ConfirmatoryGateError(f"El modo debe ser 'ensayo' u 'oficial'; recibido '{cfg.modo}'.")
    ruta = Path(cfg.ruta_preespecificacion)
    if not ruta.exists():
        raise ConfirmatoryGateError(f"No existe la preespecificación en {ruta}.")
    prereg = load_preregistration(ruta)
    meta = prereg["metadatos"]

    if meta["estado_documento"] != "CONGELADO" or not meta.get("fecha_congelamiento"):
        raise ConfirmatoryGateError("E5 no está congelada: la evaluación confirmatoria exige el registro congelado.")
    try:
        assert_can_freeze(prereg)
    except PreregistrationNotReadyError as exc:
        raise ConfirmatoryGateError(f"E5 tiene decisiones pendientes: {exc}") from exc
    discrepancias = check_consistency_with_code(prereg)
    if discrepancias:
        raise ConfirmatoryGateError("E5 no coincide con el código: " + "; ".join(discrepancias))

    tensores = Path(cfg.directorio_procesado) / "tensores.npz"
    if _sha256(tensores) != meta["sha256_conjunto_de_datos"]:
        raise ConfirmatoryGateError(
            "La huella de tensores.npz no coincide con la registrada en E5: los datos cambiaron después del congelamiento."
        )

    parametros = parametros_de_e5(prereg)
    if cfg.modo == "ensayo":
        for campo in ("ventana_inicial", "n_trials", "n_semillas", "epocas_cnn"):
            valor = getattr(cfg, campo)
            if valor is not None:
                parametros[campo] = int(valor)
        if cfg.tolerancia_shap is not None:
            parametros["tolerancia_shap"] = float(cfg.tolerancia_shap)
        return prereg, parametros

    # ---- modo oficial ----
    cambiados = [c for c in ("ventana_inicial", "n_trials", "n_semillas", "epocas_cnn", "tolerancia_shap") if getattr(cfg, c) is not None]
    if cambiados:
        raise ConfirmatoryGateError(
            f"En modo oficial los parámetros salen de E5 y no se pueden cambiar (se intentó cambiar: {cambiados})."
        )
    if not cfg.confirmo_abrir_campanas_externas:
        raise ConfirmatoryGateError(
            "La evaluación oficial abre las campañas externas una sola vez y no se puede deshacer: "
            "confírmelo explícitamente (--confirmo-abrir-campanas-externas)."
        )
    meta_datos = Path(cfg.directorio_procesado) / "metadata.json"
    metadatos = json.loads(meta_datos.read_text(encoding="utf-8")) if meta_datos.exists() else {}
    if metadatos.get("exploratorio", True):
        raise ConfirmatoryGateError(
            "El conjunto procesado es exploratorio (respaldo departamental con todo el histórico) y no es válido "
            "para la evaluación definitiva: regenérelo con scripts/run_preprocessing.py sin --exploratorio."
        )
    if (Path(cfg.directorio_salida) / ARCHIVO_DE_EJECUCION).exists():
        raise ConfirmatoryGateError(
            f"Ya hay una ejecución confirmatoria en {cfg.directorio_salida}: la evaluación oficial no se repite."
        )
    return prereg, parametros


def elegir_b_star(fold_inicial: FoldData, ventana_interna: int, indice_ndvi_max: int) -> tuple[str, dict[str, float]]:
    """B* con la validación interna del pliego inicial: solo campañas de entrenamiento, nunca las externas."""
    internos = build_inner_folds(fold_inicial, ventana_interna)
    if not internos:
        raise ConfirmatoryGateError("El pliego inicial no tiene pliegos internos para elegir B*.")
    rmse = {"B2": [], "B4": []}
    for interno in internos:
        predicciones = benchmark_predictions(interno, indice_ndvi_max)
        for base in rmse:
            rmse[base].append(float(np.sqrt(np.mean((predicciones[base] - interno.y_test) ** 2))))
    por_base = {base: float(np.mean(v)) for base, v in rmse.items()}
    return select_best_baseline(por_base), por_base


def _explicar_pliego(
    fold: FoldData, hp_cnn: dict, predicciones_cnn: pd.DataFrame, fases: list[str], parametros: dict, semilla_fondo: int,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Contribución SHAP por fase y aumento de RMSE por oclusión, promediados entre semillas (3.3.3).

    Explica TODAS las observaciones externas del pliego con la CNN-1D de cada semilla. Devuelve además la mayor
    diferencia entre la predicción de la CNN reentrenada y la de `run_fold`, para comprobar que se explica el
    mismo modelo que se evaluó.
    """
    representacion = parametros["representacion"]
    objetivo = fold.y_test if representacion == "nivel" else fold.y_anomalia_test
    indices_por_fase = {fase: [i] for i, fase in enumerate(fases)}
    x_tr, z_tr = (torch.tensor(a, dtype=torch.float32) for a in (fold.X_train, fold.Z_train))
    x_te, z_te = (torch.tensor(a, dtype=torch.float32) for a in (fold.X_test, fold.Z_test))
    xb, zb = sample_background(x_tr, z_tr, N_FONDO_SHAP, semilla_fondo)
    referencia = xb.mean(dim=(0, 2))

    contribuciones, aumentos, diferencia = [], [], 0.0
    for semilla in range(parametros["n_semillas"]):
        adaptador = build_adapter("cnn1d", parametros["epocas_cnn"])
        modelo, media, desviacion = adaptador.entrenar(hp_cnn, fold, semilla, representacion)
        with torch.no_grad():
            pred_z = modelo(x_te, z_te).squeeze(1).numpy()
        reconstruida = reconstruct_prediction(pred_z * desviacion + media, fold, representacion)
        previa = predicciones_cnn[predicciones_cnn["semilla"] == semilla]["y_pred"].to_numpy()
        diferencia = max(diferencia, float(np.max(np.abs(reconstruida - previa))))

        salida = explain_with_contingency(modelo, xb, zb, x_te, z_te, parametros["tolerancia_shap"])
        contribuciones.append(campaign_phase_contribution(salida.explicacion, fases))
        y_z = torch.tensor((objetivo - media) / desviacion, dtype=torch.float32).unsqueeze(1)
        aumentos.append(occlusion_increments_by_phase(modelo, x_te, z_te, y_z, referencia, indices_por_fase))

    promedio = lambda lista: {f: float(np.mean([d[f] for d in lista])) for f in fases}  # noqa: E731
    return promedio(contribuciones), promedio(aumentos), diferencia


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def marcar_ejecucion(salida: Path, estado: str, inicio: str, fin: str | None = None) -> None:
    """Escribe la marca de control de la corrida oficial; su sola existencia impide repetirla.

    `inicio` es la hora en que arrancó la corrida y se conserva en todas las escrituras.
    """
    marca = {"inicio": inicio, "estado": estado}
    if fin is not None:
        marca["fin"] = fin
    (Path(salida) / ARCHIVO_DE_EJECUCION).write_text(json.dumps(marca, indent=2), encoding="utf-8")


def run_confirmatory(cfg: ConfirmatoryConfig) -> ConfirmatoryResults:
    """Ejecuta la evaluación confirmatoria (o su ensayo) y devuelve los resultados y las decisiones.

    Raises:
        ConfirmatoryGateError: si no se cumple alguna condición previa (ver `comprobar_puertas`).
    """
    prereg, parametros = comprobar_puertas(cfg)
    salida = Path(cfg.directorio_salida)
    oficial = cfg.modo == "oficial"
    inicio = _ahora()
    if oficial:
        salida.mkdir(parents=True, exist_ok=True)
        marcar_ejecucion(salida, "iniciada", inicio)

    # En el ensayo solo se leen las campañas de entrenamiento del piloto; en la corrida oficial, todas.
    if oficial:
        permitidas = None
    else:
        permitidas = pilot_campaigns(cfg.primera_campana, int(_parametro(prereg, "ventana_inicial_de_entrenamiento_campanas")))
    obs, X, Z, descartadas = load_pilot_dataset(cfg.directorio_procesado, permitidas if permitidas is not None else range(1900, 2200))
    campanas_cargadas = sorted(int(c) for c in obs["campana_id"].unique())
    meta_ruta = Path(cfg.directorio_procesado) / "metadata.json"
    meta = json.loads(meta_ruta.read_text(encoding="utf-8")) if meta_ruta.exists() else {}
    variables = meta.get("orden_variables") or []
    fases = meta.get("orden_fases")
    if not fases or len(fases) != X.shape[2]:
        fases = FASES_POR_DEFECTO if X.shape[2] == len(FASES_POR_DEFECTO) else [f"paso_{i}" for i in range(X.shape[2])]
    indice_ndvi_max = variables.index("ndvi_max") if "ndvi_max" in variables else 5

    pliegos = build_folds(obs, X, Z, parametros["ventana_inicial"], campanas_cargadas)
    b_star, rmse_internos = elegir_b_star(pliegos[0], parametros["ventana_interna"], indice_ndvi_max)

    predicciones, ajustes, contribuciones, aumentos, diferencias = [], [], {}, {}, {}
    for pliego in pliegos:
        res = run_fold(
            pliego, cfg.espacios, parametros["n_trials"], parametros["n_semillas"], parametros["ventana_interna"],
            parametros["representacion"], indice_ndvi_max, parametros["epocas_cnn"], cfg.semilla,
        )
        predicciones.append(res.predicciones)
        ajustes.append(res.ajuste)
        cnn = res.predicciones[res.predicciones["modelo"] == "cnn1d"]
        c, a, d = _explicar_pliego(pliego, res.mejores_hiperparametros["cnn1d"], cnn, fases, parametros, cfg.semilla)
        contribuciones[pliego.campana_prueba], aumentos[pliego.campana_prueba], diferencias[pliego.campana_prueba] = c, a, d

    preds = pd.concat(predicciones, ignore_index=True)
    tabla_rmse = rmse_per_campaign(preds)
    he1 = evaluate_he1(tabla_rmse, parametros["margen"], b_star)
    he2 = evaluate_he2(
        contribuciones, aumentos, parametros["horizonte"], parametros["n_fases_principales"], cfg.n_permutaciones,
        cfg.semilla, parametros["alfa"],
    )
    hg = bool(he1.he1_respaldada and he2.he2_respaldada)  # hipótesis general conjuntiva (E5: hipotesis_general_conjuntiva)

    p0 = {
        "modo": cfg.modo,
        "vinculante": oficial,
        "e5": {k: prereg["metadatos"][k] for k in ("estado_documento", "version", "fecha_congelamiento", "commit_congelamiento")},
        "sha256_tensores": _sha256(Path(cfg.directorio_procesado) / "tensores.npz"),
        "parametros": parametros,
        "campanas_cargadas": campanas_cargadas,
        "campanas_descartadas_al_cargar": descartadas,
        "campanas_externas_evaluadas": [p.campana_prueba for p in pliegos],
        "n_observaciones_cargadas": int(len(obs)),
        "n_permutaciones_estabilidad": cfg.n_permutaciones,
        "fecha": datetime.now(timezone.utc).isoformat(),
    }
    resultados = ConfirmatoryResults(
        p0=p0, predicciones=preds, rmse_por_campana=tabla_rmse, ajuste=pd.concat(ajustes, ignore_index=True), he1=he1, he2=he2,
        contribuciones=contribuciones, aumentos_rmse=aumentos, b_star=b_star, rmse_validacion_interna_b=rmse_internos,
        hipotesis_general_respaldada=hg,
        verificaciones={"max_diferencia_prediccion_cnn_reentrenada_kg_ha": diferencias},
    )
    if oficial:
        marcar_ejecucion(salida, "completada", inicio, _ahora())
    return resultados


# ============================================================================ salidas
def _json_seguro(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return str(obj)


def save_confirmatory_results(res: ConfirmatoryResults, directorio: str | Path) -> None:
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    res.predicciones.to_csv(directorio / "predicciones.csv", index=False)
    res.rmse_por_campana.to_csv(directorio / "rmse_por_campana.csv", index=False)
    res.ajuste.to_csv(directorio / "ajuste.csv", index=False)
    pd.DataFrame(res.contribuciones).T.rename_axis("campana_prueba").to_csv(directorio / "shap_contribucion_por_fase.csv")
    pd.DataFrame(res.aumentos_rmse).T.rename_axis("campana_prueba").to_csv(directorio / "oclusion_aumento_rmse_por_fase.csv")
    resumen = {
        "p0": res.p0, "b_star": res.b_star, "rmse_validacion_interna_b": res.rmse_validacion_interna_b,
        "he1": res.he1, "he2": res.he2, "hipotesis_general_respaldada": res.hipotesis_general_respaldada,
        "verificaciones": res.verificaciones,
    }
    (directorio / "resultados.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=_json_seguro), encoding="utf-8")
    (directorio / "informe_confirmatorio.md").write_text(render_confirmatory_report(res), encoding="utf-8", newline="\n")


def _si_no(valor: bool) -> str:
    return "sí" if valor else "no"


def render_confirmatory_report(res: ConfirmatoryResults) -> str:
    p0, h1, h2 = res.p0, res.he1, res.he2
    if p0["vinculante"]:
        aviso = "> Evaluación confirmatoria oficial: E5 congelada, campañas externas evaluadas una sola vez.\n"
    else:
        aviso = (
            "> **ENSAYO, NO VINCULANTE.** Solo prueba el procedimiento con campañas de entrenamiento que se hacen pasar por externas. "
            "Sus decisiones no son evidencia de la tesis.\n"
        )
    filas_he1 = "\n".join(
        f"| {c} | {qx:+.3f} | {qb:+.3f} | {ss:+.3f} |"
        for c, qx, qb, ss in zip(h1.campanas, h1.q_xgboost, h1.q_bstar, h1.ss_por_campana)
    )
    filas_he2 = "\n".join(f"| {c} | {fase} |" for c, fase in h2.fases_maximas.items())
    return f"""# Evaluación confirmatoria ({p0['modo']})

{aviso}
## Puerta de entrada
- E5: {p0['e5']['estado_documento']} v{p0['e5']['version']} (congelada el {p0['e5']['fecha_congelamiento']}, commit `{str(p0['e5']['commit_congelamiento'])[:7]}`).
- SHA-256 de `tensores.npz`: `{p0['sha256_tensores'][:16]}…` (coincide con E5).
- Campañas cargadas: {p0['campanas_cargadas']} · descartadas al cargar: {p0['campanas_descartadas_al_cargar'] or 'ninguna'} · evaluadas como externas: {p0['campanas_externas_evaluadas']}.
- Parámetros (de E5): ventana inicial {p0['parametros']['ventana_inicial']}, {p0['parametros']['n_trials']} trials por modelo, {p0['parametros']['n_semillas']} semillas, {p0['parametros']['epocas_cnn']} épocas, representación `{p0['parametros']['representacion']}`, δ = {p0['parametros']['margen']}.

## B*
Elegida con la validación interna del pliego inicial (solo entrenamiento): RMSE de B2 {res.rmse_validacion_interna_b['B2']:.1f} kg/ha, de B4 {res.rmse_validacion_interna_b['B4']:.1f} kg/ha → **{res.b_star}**.

## HE1a · No inferioridad (δ = {p0['parametros']['margen']:.0%}) — {'RESPALDADA' if h1.he1a_respaldada else 'NO RESPALDADA'}
Regla exacta de signos: q < δ en todas las campañas (nivel {h1.nivel_regla_de_signos:.3f}).

| Campaña | q frente a XGBoost | q frente a {h1.b_star} | Skill score frente a {h1.b_star} |
|---|---|---|---|
{filas_he1}

No inferior a XGBoost: {_si_no(h1.no_inferior_a_xgboost)} · a {h1.b_star}: {_si_no(h1.no_inferior_a_bstar)}. HE1b (exploratoria): ganancia en {h1.n_campanas_con_ganancia} de {h1.n_campanas} campañas (p de signos {h1.p_signos_ganancia:.3f}).

## HE2c · Concordancia agronómica — {'RESPALDADA' if h2.he2_respaldada else 'NO RESPALDADA'}
Binomial exacta, p = {h2.p_concordancia:.4f}. Fase de máxima contribución por campaña:

| Campaña | Fase máxima |
|---|---|
{filas_he2}

Proporción de contribución del referente agronómico: {h2.proporcion_contribucion_referente:.3f} (supera el azar: {_si_no(h2.proporcion_referente_supera_azar)}). Secundarias descriptivas: W de Kendall {h2.w_kendall:.2f}, Jaccard {h2.jaccard_promedio:.2f}.

## Hipótesis general (conjuntiva: HE1 y HE2)
**{'RESPALDADA' if res.hipotesis_general_respaldada else 'NO RESPALDADA'}**.
"""
