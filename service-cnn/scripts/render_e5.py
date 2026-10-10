"""Genera el texto de la preespecificación E5 desde `configs/preespecificacion.yaml`.

Verifica antes la coherencia con el código; si hay discrepancias, no escribe nada.

    python scripts/render_e5.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.preregistration import (  # noqa: E402
    check_consistency_with_code,
    load_preregistration,
    render_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=Path("configs/preespecificacion.yaml"))
    parser.add_argument("--salida", type=Path, default=Path("docs/crisp/E5-preespecificacion-BORRADOR.md"))
    args = parser.parse_args()

    prereg = load_preregistration(args.yaml)
    discrepancias = check_consistency_with_code(prereg)
    if discrepancias:
        print("La preespecificación no coincide con el código:", file=sys.stderr)
        for d in discrepancias:
            print(f"  - {d}", file=sys.stderr)
        sys.exit(1)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(render_markdown(prereg), encoding="utf-8", newline="\n")
    print(f"{args.salida}: escrito (coherente con el código)")


if __name__ == "__main__":
    main()
