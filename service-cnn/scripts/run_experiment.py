"""CLI de la Actividad 8: evaluación predictiva (origen móvil, XGBoost, B2).

Referencia: secciones 4.12-4.14 de la tesis.

Estado actual: NO ejecutable todavía — depende de `data/processed/` con los
tensores V×T construidos (Actividad 4-6 del roadmap), que a su vez depende de
`run_preprocessing.py`. Este script queda como punto de integración
documentado: cuando `data/processed/` exista, aquí se orquestan en orden:
  1. `src.evaluation.splits.expanding_window_splits` (esquema confirmatorio)
  2. Por pliegue: `src.models.tuning.tune_classic_model` /
     `src.models.tuning_cnn1d.tune_cnn1d` con presupuesto equiparado
  3. `src.evaluation.metrics.*` por campaña, modelo y semilla
  4. `src.evaluation.hypothesis_tests.evaluate_non_inferiority` (HE1a) y
     `evaluate_superiority` (HE1b) sobre los RMSE agregados por campaña
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


class ExperimentNotReadyError(RuntimeError):
    """Los tensores procesados de entrada todavía no existen."""


def _verificar_datos_procesados(directorio_processed: Path) -> None:
    archivos_reales = (
        [
            p
            for p in directorio_processed.iterdir()
            if p.is_file() and p.name != ".gitkeep"
        ]
        if directorio_processed.exists()
        else []
    )
    if not archivos_reales:
        raise ExperimentNotReadyError(
            f"{directorio_processed} no contiene tensores procesados. La "
            "evaluación predictiva requiere el resultado de "
            "scripts/run_preprocessing.py (máscara + alineación fenológica + "
            "construcción de tensores V×T)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datos-processed", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--salida", type=Path, default=Path("reports/resultados_por_pliegue")
    )
    args = parser.parse_args()

    try:
        _verificar_datos_procesados(args.datos_processed)
    except ExperimentNotReadyError as exc:
        print(f"EXPERIMENTO NO LISTO: {exc}", file=sys.stderr)
        sys.exit(1)

    raise NotImplementedError(
        "La orquestación completa del experimento (splits + tuning + "
        "entrenamiento + métricas + contrastes) se completa cuando exista "
        "data/processed/ (Actividad 8 del roadmap, "
        "docs/02-arquitectura-tecnica.md §9). Todos los componentes ya están "
        "implementados y probados: src/evaluation/splits.py, "
        "src/models/tuning.py, src/models/tuning_cnn1d.py, "
        "src/evaluation/metrics.py, src/evaluation/hypothesis_tests.py."
    )


if __name__ == "__main__":
    main()
