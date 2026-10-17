"""Exporta una instantanea de los resultados REALES de service-cnn/ a fixtures JSON.

Uso (desde la raiz del repo):
    service-cnn/.venv/Scripts/python.exe app-quinuaprod/scripts/export_fixtures.py

No inventa datos: solo lee reports/, configs/ y data/processed/ de service-cnn.
Solo exporta campanas permitidas por el piloto oficial (2016-2020); las campanas
externas 2021-2025 no se leen ni se escriben.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SVC = ROOT / "service-cnn"
OUT = ROOT / "app-quinuaprod" / "src" / "testing" / "mocks" / "fixtures"
PILOTO = SVC / "reports" / "piloto_oficial"
CAMPANAS_EXTERNAS = set(range(2021, 2026))


def write(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8"
    )
    n = len(data) if hasattr(data, "__len__") else 1
    print(f"  {name}.json ({n})")


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


def main() -> int:
    resultados = json.loads((PILOTO / "resultados.json").read_text(encoding="utf-8"))
    permitidas = set(resultados["p0"]["campanas_permitidas"])
    assert not (permitidas & CAMPANAS_EXTERNAS), "el piloto expone campanas externas"

    he2 = json.loads((SVC / "reports/he2_tecnico/resultados.json").read_text(encoding="utf-8"))
    prov = pd.read_csv(SVC / "configs/provincias.csv")
    excl = pd.read_csv(SVC / "data/processed/exclusiones.csv")
    excl = excl[excl["campana_id"].isin(permitidas)]
    pred = pd.read_csv(PILOTO / "predicciones.csv")
    assert set(pred["campana_prueba"]) <= permitidas
    y1 = pd.read_csv(PILOTO / "p11_indicadores_y1.csv")
    p6 = pd.read_csv(PILOTO / "p6_variabilidad_por_semilla.csv")
    sens = pd.read_csv(SVC / "reports/sensibilidades/sensibilidades_rmse.csv")
    estr = pd.read_csv(SVC / "reports/sensibilidades/error_por_estrato_de_quinua.csv")
    meta = json.loads((SVC / "data/processed/metadata.json").read_text(encoding="utf-8"))
    embudo = json.loads(
        (SVC / "reports/E1_matriz_cobertura_enmendado_2016_2025.json").read_text(encoding="utf-8")
    )
    e5 = yaml.safe_load((SVC / "configs/preespecificacion.yaml").read_text(encoding="utf-8"))

    print("Exportando a", OUT)
    write("meta", {
        "modo": resultados["p0"]["modo"],
        "vinculante": resultados["p0"]["vinculante"],
        "campanas_permitidas": resultados["p0"]["campanas_permitidas"],
        "campanas_de_prueba": resultados["p0"]["campanas_de_prueba"],
        "n_observaciones": resultados["p0"]["n_observaciones"],
        "sha256_tensores": resultados["p0"]["sha256_tensores"],
        "estado_e5": e5["metadatos"]["estado_documento"],
        "version_e5": e5["metadatos"]["version"],
        "fecha_congelamiento_e5": e5["metadatos"]["fecha_congelamiento"],
        "orden_variables": meta["orden_variables"],
        "orden_fases": meta["orden_fases"],
        "representacion": resultados["p4"]["representacion_elegida"],
        "hiperparametros_cnn": resultados["p7"]["hiperparametros_cnn"],
    })
    write("provincias", records(prov))
    write("embudo", [{"nivel": k, "n": v} for k, v in embudo.items()])
    write("exclusiones", records(excl.rename(columns={"campana_id": "campana"})))
    write("predicciones", records(pred))
    write("indicadores_y1", records(y1))
    write("indicadores_x1x2", resultados["p11"])
    write("variabilidad_semillas", records(p6))
    write("explicabilidad", {
        "piloto_oficial": {
            "contribucion_por_fase": resultados["p8"]["contribucion_por_fase"],
            "fase_maxima": resultados["p8"]["fase_maxima"],
            "n_explicadas": resultados["p8"]["n_explicadas"],
            "campanas": resultados["p0"]["campanas_de_prueba"],
        },
        "he2_tecnico": {k: he2[k] for k in (
            "vinculante", "campanas_evaluadas", "contribuciones", "aumentos_rmse",
            "w_kendall", "jaccard_promedio", "he2_respaldada") if k in he2},
    })
    write("sensibilidades", records(sens))
    write("sensibilidades_estratos", records(estr))
    write("preespecificacion", {"metadatos": e5["metadatos"], "secciones": {
        k: v for k, v in e5.items() if k != "metadatos"}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
