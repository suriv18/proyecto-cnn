"""CLI del ensayo piloto (tarea T7): ejecuta P0-P10 y escribe resultados e informe.

Dos modos:
  --modo tecnico  (por defecto) desarrollo: NO VINCULANTE; no fija ningún parámetro de E5.
  --modo oficial  exige --gate-f2-aprobado (el asesor cerró las decisiones de E5).

Las campañas permitidas son las primeras `--ventana-confirmatoria` (2016, 2017, ...): son
las de entrenamiento en cualquier esquema con esa ventana. Mientras el asesor no fije la
ventana, el modo técnico usa 4, la menor plausible, para no ver una campaña que podría
resultar externa. Las demás campañas se descartan al cargar y solo se registran sus ids.

Uso:
    python scripts/run_pilot.py --ventana-confirmatoria 4
    python scripts/run_pilot.py --modo oficial --gate-f2-aprobado --ventana-confirmatoria 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.pilot_runner import PilotConfig, PilotGateError, run_pilot, save_pilot_results  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--procesado", type=Path, default=Path("data/processed"))
    p.add_argument("--hyperparams", type=Path, default=Path("configs/hyperparams.yaml"))
    p.add_argument("--salida", type=Path, default=None, help="Por defecto reports/piloto_<modo>")
    p.add_argument("--modo", choices=("tecnico", "oficial"), default="tecnico")
    p.add_argument("--gate-f2-aprobado", action="store_true")
    p.add_argument("--ventana-confirmatoria", type=int, default=4)
    p.add_argument("--ventana-piloto", type=int, default=3)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--n-semillas", type=int, default=10)
    p.add_argument("--epocas-cnn", type=int, default=100)
    p.add_argument("--n-permutaciones", type=int, default=10)
    p.add_argument("--n-explicadas", type=int, default=30)
    p.add_argument("--semilla", type=int, default=0)
    args = p.parse_args()

    espacios = yaml.safe_load(args.hyperparams.read_text(encoding="utf-8"))
    cfg = PilotConfig(
        directorio_procesado=args.procesado,
        directorio_salida=args.salida or Path(f"reports/piloto_{args.modo}"),
        espacios={k: v for k, v in espacios.items() if k in ("cnn1d", "elastic_net", "random_forest", "xgboost", "svr")},
        ventana_confirmatoria=args.ventana_confirmatoria, ventana_piloto=args.ventana_piloto,
        n_trials=args.n_trials, n_semillas=args.n_semillas, epocas_cnn=args.epocas_cnn,
        n_permutaciones=args.n_permutaciones, n_explicadas=args.n_explicadas, modo=args.modo,
        gate_f2_aprobado=args.gate_f2_aprobado, semilla=args.semilla,
    )
    try:
        resultados = run_pilot(cfg)
    except PilotGateError as exc:
        print(f"PILOTO NO LISTO: {exc}", file=sys.stderr)
        sys.exit(1)
    save_pilot_results(resultados, cfg.directorio_salida)
    p0 = resultados.p0
    print(f"Piloto {p0['modo']} ({'VINCULANTE' if p0['vinculante'] else 'NO VINCULANTE'}): informe en {cfg.directorio_salida / 'informe_piloto.md'}")
    print(f"Campañas permitidas {p0['campanas_permitidas']}; descartadas al cargar {p0['campanas_descartadas']}.")
    print(f"P1 fuga: {'pasa' if resultados.p1['pasa'].all() else 'NO PASA'} | P4: {resultados.p4['decision']} | P5: {resultados.p5['decision']} | "
          f"P7 épocas: {resultados.p7['epocas_maximas']} | P8: {resultados.p8['variante']}")


if __name__ == "__main__":
    main()
