"""Ensayo técnico de HE2 de extremo a extremo (NO VINCULANTE).

Entrena la CNN-1D en pliegos de origen móvil dentro de las campañas de entrenamiento
permitidas (por defecto 2016-2019: se evalúan las campañas 2018 y 2019 con ventana 2),
obtiene la contribución SHAP por fase y el aumento de RMSE por oclusión de cada fase, y
aplica `evaluate_he2`. Sirve para comprobar la cadena completa y dimensionar el costo; no
cuenta como evidencia de HE2, que solo se contrasta con las campañas externas.

Uso:
    python scripts/run_he2_tecnico.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.fold_builder import build_folds, load_pilot_dataset  # noqa: E402
from src.evaluation.he2_evaluation import evaluate_he2  # noqa: E402
from src.evaluation.he2_runner import collect_he2_conditional_increments, collect_he2_inputs  # noqa: E402
from src.explainability.he2_pvalues import fidelity_pvalue  # noqa: E402
from src.evaluation.pilot_guard import pilot_campaigns  # noqa: E402

FASES = ["emergencia", "desarrollo_vegetativo", "floracion", "llenado_grano", "madurez"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--procesado", type=Path, default=Path("data/processed"))
    p.add_argument("--piloto", type=Path, default=Path("reports/piloto_tecnico/resultados.json"))
    p.add_argument("--salida", type=Path, default=Path("reports/he2_tecnico"))
    p.add_argument("--ventana-confirmatoria", type=int, default=4)
    p.add_argument("--ventana-inicial", type=int, default=2)
    p.add_argument("--horizonte", choices=("principal", "temprano"), default="principal")
    p.add_argument("--n-semillas", type=int, default=10)
    p.add_argument("--n-fondo", type=int, default=100)
    p.add_argument("--n-explicadas", type=int, default=60)
    p.add_argument("--n-permutaciones", type=int, default=10000)
    p.add_argument("--semilla", type=int, default=2026)
    args = p.parse_args()

    piloto = json.loads(args.piloto.read_text(encoding="utf-8"))
    hp_cnn = piloto["p7"]["hiperparametros_cnn"]
    epocas = int(piloto["p7"]["epocas_maximas"])
    representacion = piloto["p4"]["decision"]

    permitidas = pilot_campaigns(2016, args.ventana_confirmatoria)
    obs, X, Z, descartadas = load_pilot_dataset(args.procesado, permitidas)
    fases = json.loads((args.procesado / "metadata.json").read_text(encoding="utf-8")).get("orden_fases") or FASES
    pliegos = build_folds(obs, X, Z, args.ventana_inicial, permitidas)

    t0 = time.perf_counter()
    contribuciones, aumentos = collect_he2_inputs(
        pliegos, hp_cnn, representacion, fases, args.n_semillas, epocas, args.n_fondo, args.n_explicadas, args.semilla,
    )
    segundos = time.perf_counter() - t0
    condicional = collect_he2_conditional_increments(pliegos, hp_cnn, representacion, fases, args.n_semillas, epocas, args.n_fondo, args.semilla)
    r = evaluate_he2(contribuciones, aumentos, args.horizonte, 3, args.n_permutaciones, args.semilla)

    maximas = [r.fases_maximas[c] for c in r.campanas]
    p_fid_cond = fidelity_pvalue([condicional[c] for c in r.campanas], maximas)
    args.salida.mkdir(parents=True, exist_ok=True)
    resultado = {
        "vinculante": False, "campanas_permitidas": list(map(int, permitidas)), "campanas_descartadas": list(map(int, descartadas)),
        "campanas_evaluadas": r.campanas, "representacion": representacion, "hp_cnn": hp_cnn, "epocas": epocas,
        "n_semillas": args.n_semillas, "segundos": segundos, "contribuciones": {str(k): v for k, v in contribuciones.items()},
        "aumentos_rmse": {str(k): v for k, v in aumentos.items()}, "fases_maximas": {str(k): v for k, v in r.fases_maximas.items()},
        "w_kendall": r.w_kendall, "jaccard_promedio": r.jaccard_promedio, "umbrales_de_estabilidad_cumplidos": r.umbrales_de_estabilidad_cumplidos,
        "p_valores": r.p_valores, "p_ajustados_secundarios": r.p_ajustados_secundarios, "rechazos_secundarios": r.rechazos_secundarios,
        "rechaza_concordancia": r.rechaza_concordancia, "secundarios_cumplidos": r.secundarios_cumplidos,
        "proporcion_contribucion_referente": r.proporcion_contribucion_referente, "proporcion_referente_supera_azar": r.proporcion_referente_supera_azar,
        "aumentos_rmse_condicional": {str(k): v for k, v in condicional.items()},
        "p_fidelidad_condicional": p_fid_cond,
        "he2_respaldada": r.he2_respaldada,
    }
    (args.salida / "resultados.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"HE2 técnico (NO VINCULANTE) sobre campañas {r.campanas} de entrenamiento; {segundos:.0f} s")
    print(f"Fases máximas: {r.fases_maximas}")
    print(f"W de Kendall {r.w_kendall:.2f} · Jaccard medio {r.jaccard_promedio:.2f} · umbrales de estabilidad: {r.umbrales_de_estabilidad_cumplidos}")
    print(f"  concordancia (primaria): p = {r.p_concordancia:.4f} · rechaza: {r.rechaza_concordancia}")
    for k in r.p_ajustados_secundarios:
        print(f"  {k} (secundaria): p = {r.p_valores[k]:.4f} · ajustado {r.p_ajustados_secundarios[k]:.4f} · rechaza: {r.rechazos_secundarios[k]}")
    print(f"  fidelidad con reemplazo condicional (sensibilidad): p = {p_fid_cond:.4f}")
    print(f"Proporción de contribución en el referente agronómico: {r.proporcion_contribucion_referente:.2f} (supera el azar: {r.proporcion_referente_supera_azar})")
    print(f"HE2 respaldada: {r.he2_respaldada}")


if __name__ == "__main__":
    main()
