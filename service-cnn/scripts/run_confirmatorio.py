"""CLI de la evaluación confirmatoria (E5 congelada): HE1a, HE2c y la hipótesis general.

Dos modos:
  --modo ensayo  (por defecto) prueba el procedimiento con las campañas de entrenamiento del piloto (2016-2020),
                 haciendo pasar por externas las últimas. NO VINCULANTE: no abre las campañas externas reales.
  --modo oficial la evaluación real sobre las campañas externas. Se hace UNA vez: exige E5 congelada y coherente
                 con el código, la huella de los datos registrada, un conjunto NO exploratorio y la confirmación
                 explícita. Los parámetros salen de E5; esta línea de comandos no puede cambiarlos.

Uso:
    python scripts/run_confirmatorio.py                          # ensayo con los parámetros de E5
    python scripts/run_confirmatorio.py --ventana-inicial 3      # ensayo: 2019 y 2020 como externas (por defecto)
    python scripts/run_confirmatorio.py --modo oficial --confirmo-abrir-campanas-externas
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.confirmatory_runner import (  # noqa: E402
    ConfirmatoryConfig,
    ConfirmatoryGateError,
    run_confirmatory,
    save_confirmatory_results,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--procesado", type=Path, default=Path("data/processed"))
    p.add_argument("--hyperparams", type=Path, default=Path("configs/hyperparams.yaml"))
    p.add_argument("--preespecificacion", type=Path, default=Path("configs/preespecificacion.yaml"))
    p.add_argument("--salida", type=Path, default=None, help="Por defecto reports/confirmatorio_<modo>")
    p.add_argument("--modo", choices=("ensayo", "oficial"), default="ensayo")
    p.add_argument("--confirmo-abrir-campanas-externas", action="store_true")
    p.add_argument("--semilla", type=int, default=0)
    # Solo para el ensayo (en oficial, cualquiera de estos es un error):
    p.add_argument("--ventana-inicial", type=int, default=None, help="Ensayo: 3 por defecto (externas simuladas 2019 y 2020)")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--n-semillas", type=int, default=None)
    p.add_argument("--epocas-cnn", type=int, default=None)
    args = p.parse_args()

    ventana = args.ventana_inicial
    if args.modo == "ensayo" and ventana is None:
        ventana = 3
    espacios = yaml.safe_load(args.hyperparams.read_text(encoding="utf-8"))
    cfg = ConfirmatoryConfig(
        directorio_procesado=args.procesado,
        directorio_salida=args.salida or Path(f"reports/confirmatorio_{args.modo}"),
        espacios={k: v for k, v in espacios.items() if k in ("cnn1d", "elastic_net", "random_forest", "xgboost", "svr")},
        modo=args.modo, ruta_preespecificacion=args.preespecificacion,
        confirmo_abrir_campanas_externas=args.confirmo_abrir_campanas_externas,
        ventana_inicial=ventana, n_trials=args.n_trials, n_semillas=args.n_semillas, epocas_cnn=args.epocas_cnn,
        semilla=args.semilla,
    )
    try:
        resultados = run_confirmatory(cfg)
    except ConfirmatoryGateError as exc:
        print(f"EVALUACIÓN NO LISTA: {exc}", file=sys.stderr)
        sys.exit(1)
    save_confirmatory_results(resultados, cfg.directorio_salida)
    p0, h1, h2 = resultados.p0, resultados.he1, resultados.he2
    print(f"Evaluación confirmatoria {p0['modo']} ({'VINCULANTE' if p0['vinculante'] else 'NO VINCULANTE'}): "
          f"informe en {cfg.directorio_salida / 'informe_confirmatorio.md'}")
    print(f"Campañas cargadas {p0['campanas_cargadas']}; descartadas al cargar {p0['campanas_descartadas_al_cargar']}; "
          f"externas evaluadas {p0['campanas_externas_evaluadas']}.")
    print(f"B* = {resultados.b_star} | HE1a: {'respaldada' if h1.he1a_respaldada else 'no respaldada'} | "
          f"HE2c: {'respaldada' if h2.he2_respaldada else 'no respaldada'} | "
          f"HG: {'respaldada' if resultados.hipotesis_general_respaldada else 'no respaldada'}")


if __name__ == "__main__":
    main()
