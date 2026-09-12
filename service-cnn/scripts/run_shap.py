"""CLI de la Actividad 9: evaluación SHAP (estabilidad, fidelidad, concordancia).

Referencia: secciones 3.3.3-3.3.5, 4.15 de la tesis.

Estado actual: NO ejecutable todavía — depende del modelo CNN-1D entrenado
(salida de `run_experiment.py`). Este script queda como punto de integración
documentado: cuando exista un modelo entrenado y resultados por pliegue, aquí
se orquestan en orden:
  1. `src.explainability.shap_explainer.explain_with_deep_explainer` sobre
     cada campaña externa de prueba
  2. `validate_local_accuracy` — si falla, activar el plan de contingencia
     (KernelExplainer, docs/02-arquitectura-tecnica.md §1.1, no implementado
     aún porque DeepExplainer no ha fallado esa validación en las pruebas
     realizadas hasta ahora)
  3. `src.explainability.shap_aggregation.aggregate_shap_by_phase` y
     `fase_de_maxima_contribucion`, agregando semillas por campaña antes
  4. `src.explainability.stability.evaluate_stability` (HE2a)
  5. `src.explainability.fidelity.evaluate_fidelity_by_occlusion` (HE2b)
  6. `src.explainability.agronomic_concordance.evaluate_agronomic_concordance`
     (HE2c)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


class ShapEvaluationNotReadyError(RuntimeError):
    """Los resultados del experimento (modelo entrenado) todavía no existen."""


def _verificar_resultados_experimento(directorio_resultados: Path) -> None:
    archivos_reales = (
        [
            p
            for p in directorio_resultados.iterdir()
            if p.is_file() and p.name != ".gitkeep"
        ]
        if directorio_resultados.exists()
        else []
    )
    if not archivos_reales:
        raise ShapEvaluationNotReadyError(
            f"{directorio_resultados} no contiene resultados de experimento. "
            "La evaluación SHAP requiere el modelo CNN-1D entrenado, salida "
            "de scripts/run_experiment.py."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resultados-experimento",
        type=Path,
        default=Path("reports/resultados_por_pliegue"),
    )
    parser.add_argument(
        "--salida", type=Path, default=Path("reports/E6_interpretabilidad")
    )
    args = parser.parse_args()

    try:
        _verificar_resultados_experimento(args.resultados_experimento)
    except ShapEvaluationNotReadyError as exc:
        print(f"EVALUACIÓN SHAP NO LISTA: {exc}", file=sys.stderr)
        sys.exit(1)

    raise NotImplementedError(
        "La orquestación completa de la evaluación SHAP (explicación + "
        "agregación por fase + estabilidad + fidelidad + concordancia) se "
        "completa cuando exista un modelo entrenado (Actividad 9 del "
        "roadmap). Todos los componentes ya están implementados y probados "
        "en src/explainability/."
    )


if __name__ == "__main__":
    main()
