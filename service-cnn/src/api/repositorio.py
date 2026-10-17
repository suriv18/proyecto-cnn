"""Lectura de los artefactos del piloto con la forma del contrato OpenAPI.

Regla de honestidad (RNF1): las campañas externas no se exponen nunca. La lista de campañas
permitidas sale de `reports/piloto_oficial/resultados.json` (p0) y se aplica aquí, en el servidor;
no se confía en que la interfaz las oculte.
"""
from __future__ import annotations

import json
import statistics
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.models.inferencia import ModeloCNN1D

Registro = dict[str, Any]


def _registros(df: pd.DataFrame) -> list[Registro]:
    """DataFrame -> lista de dicts con tipos nativos y NaN como None (JSON no admite NaN)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


class Repositorio:
    def __init__(self, raiz: Path) -> None:
        self.raiz = Path(raiz)
        self.reports = self.raiz / "reports"
        self.processed = self.raiz / "data" / "processed"
        self.configs = self.raiz / "configs"

    # --- lectura en bruto -------------------------------------------------------------------

    def _json(self, ruta: Path) -> Any:
        return json.loads(ruta.read_text(encoding="utf-8"))

    def _csv(self, ruta: Path) -> pd.DataFrame:
        if not ruta.exists():
            raise FileNotFoundError(2, "No existe el archivo", str(ruta.relative_to(self.raiz)))
        return pd.read_csv(ruta)

    def _leer_json(self, ruta: Path) -> Any:
        if not ruta.exists():
            raise FileNotFoundError(2, "No existe el archivo", str(ruta.relative_to(self.raiz)))
        return self._json(ruta)

    @cached_property
    def _resultados(self) -> Registro:
        return self._leer_json(self.reports / "piloto_oficial" / "resultados.json")

    @cached_property
    def _e5(self) -> Registro:
        ruta = self.configs / "preespecificacion.yaml"
        if not ruta.exists():
            raise FileNotFoundError(2, "No existe el archivo", str(ruta.relative_to(self.raiz)))
        return yaml.safe_load(ruta.read_text(encoding="utf-8"))

    @cached_property
    def modelo(self) -> ModeloCNN1D | None:
        """La CNN-1D serializada (scripts/export_modelo.py), o None si no se ha generado."""
        directorio = self.raiz / "models" / "cnn1d_piloto"
        return ModeloCNN1D.cargar(directorio) if (directorio / "modelo.json").exists() else None

    # --- campañas -------------------------------------------------------------------------

    @property
    def campanas_permitidas(self) -> list[int]:
        return list(self._resultados["p0"]["campanas_permitidas"])

    @property
    def campanas_de_prueba(self) -> list[int]:
        return list(self._resultados["p0"]["campanas_de_prueba"])

    def _solo_permitidas(self, df: pd.DataFrame, columna: str) -> pd.DataFrame:
        return df[df[columna].isin(self.campanas_permitidas)]

    # --- recursos del contrato ----------------------------------------------------------------

    def meta(self) -> Registro:
        p = self._resultados
        e5 = self._e5["metadatos"]
        metadata = self._leer_json(self.processed / "metadata.json")
        return {
            "modo": p["p0"]["modo"],
            "vinculante": p["p0"]["vinculante"],
            "campanas_permitidas": self.campanas_permitidas,
            "campanas_de_prueba": self.campanas_de_prueba,
            "n_observaciones": p["p0"]["n_observaciones"],
            "sha256_tensores": p["p0"]["sha256_tensores"],
            "estado_e5": e5["estado_documento"],
            "version_e5": str(e5["version"]),
            "fecha_congelamiento_e5": e5.get("fecha_congelamiento"),
            "orden_variables": metadata["orden_variables"],
            "orden_fases": metadata["orden_fases"],
            "representacion": p["p4"]["representacion_elegida"],
            "hiperparametros_cnn": p["p7"]["hiperparametros_cnn"],
        }

    def provincias(self) -> list[Registro]:
        return _registros(self._csv(self.configs / "provincias.csv"))

    def embudo(self) -> list[Registro]:
        datos = self._leer_json(self.reports / "E1_matriz_cobertura_enmendado_2016_2025.json")
        return [{"nivel": nivel, "n": n} for nivel, n in datos.items()]

    def exclusiones(self) -> list[Registro]:
        df = self._csv(self.processed / "exclusiones.csv").rename(columns={"campana_id": "campana"})
        return _registros(self._solo_permitidas(df, "campana"))

    def predicciones(self) -> pd.DataFrame:
        df = self._csv(self.reports / "piloto_oficial" / "predicciones.csv")
        return self._solo_permitidas(df, "campana_prueba")

    def indicadores_y1(self) -> list[Registro]:
        df = self._csv(self.reports / "piloto_oficial" / "p11_indicadores_y1.csv")
        return _registros(self._solo_permitidas(df, "campana_prueba"))

    def indicadores_x1_x2(self) -> Registro:
        return self._resultados["p11"]

    def variabilidad_semillas(self) -> list[Registro]:
        df = self._csv(self.reports / "piloto_oficial" / "p6_variabilidad_por_semilla.csv")
        return _registros(self._solo_permitidas(df, "campana_prueba"))

    def explicabilidad(self) -> Registro:
        p = self._resultados
        he2 = self._leer_json(self.reports / "he2_tecnico" / "resultados.json")
        permitidas = {str(c) for c in self.campanas_permitidas}
        por_campana = lambda d: {c: v for c, v in d.items() if c in permitidas}  # noqa: E731
        campanas_evaluadas = [c for c in he2["campanas_evaluadas"] if str(c) in permitidas]
        return {
            "piloto_oficial": {
                "contribucion_por_fase": p["p8"]["contribucion_por_fase"],
                "fase_maxima": p["p8"]["fase_maxima"],
                "n_explicadas": p["p8"]["n_explicadas"],
                "campanas": self.campanas_de_prueba,
            },
            "he2_tecnico": {
                "vinculante": he2["vinculante"],
                "campanas_evaluadas": campanas_evaluadas,
                "contribuciones": por_campana(he2["contribuciones"]),
                "aumentos_rmse": por_campana(he2["aumentos_rmse"]),
                "w_kendall": he2["w_kendall"],
                "jaccard_promedio": he2["jaccard_promedio"],
                "he2_respaldada": he2["he2_respaldada"],
            },
        }

    def sensibilidades(self) -> pd.DataFrame:
        df = self._csv(self.reports / "sensibilidades" / "sensibilidades_rmse.csv")
        return self._solo_permitidas(df, "campana_prueba")

    def sensibilidades_estratos(self) -> list[Registro]:
        return _registros(self._csv(self.reports / "sensibilidades" / "error_por_estrato_de_quinua.csv"))

    def preespecificacion(self) -> Registro:
        e5 = self._e5
        return {
            "metadatos": e5["metadatos"],
            "secciones": {k: v for k, v in e5.items() if k != "metadatos"},
        }

    # --- inferencia sobre un tensor editado ---------------------------------------------------

    def inferir_tensor(
        self, modelo: ModeloCNN1D, provincia_id: str, campana: int, valores: list[list[float]],
        altitud_media_m: float, prop_quinua: float,
    ) -> Registro | None:
        """Predicción de la CNN-1D serializada sobre un tensor editado.

        El modelo predice la anomalía respecto de B2, así que el rendimiento es B2(provincia, campaña) más la
        mediana de las anomalías de las semillas (igual que el piloto). Devuelve None si la provincia no tiene
        historia de entrenamiento: no se inventa una línea base.
        """
        y_b2 = modelo.linea_base.predecir(provincia_id, campana)
        if y_b2 is None:
            return None
        covariables = {"altitud_media_m": altitud_media_m, "prop_quinua": prop_quinua}
        z = np.array([[covariables[c] for c in modelo.tarjeta["covariables"]]], dtype=float)
        anomalias = modelo.predecir_anomalia(np.array([valores], dtype=float), z)[:, 0]
        anomalia = float(statistics.median(anomalias))
        return {
            "y_pred": y_b2 + anomalia,
            "y_b2": y_b2,
            "anomalia": anomalia,
            "y_real": None,
            "shap": None,
            "modelo_version": modelo.version,
            "vinculante": bool(modelo.tarjeta["vinculante"]),
        }

    # --- inferencia sobre una observación del piloto ------------------------------------------

    def inferir_observacion(self, provincia_id: str, campana: int) -> Registro | None:
        """Predicción ya calculada por el piloto: mediana entre semillas del CNN-1D y B2.

        No hay modelo serializado, así que solo se puede responder para las observaciones que el
        piloto evaluó. Devuelve None si la observación no está entre ellas.
        """
        df = self.predicciones()
        df = df[(df["provincia_id"] == provincia_id) & (df["campana_prueba"] == campana)]
        cnn = df[df["modelo"] == "cnn1d"]["y_pred"].tolist()
        b2 = df[df["modelo"] == "B2"]
        if not cnn or b2.empty:
            return None
        y_pred = float(statistics.median(cnn))
        y_b2 = float(b2.iloc[0]["y_pred"])
        p0 = self._resultados["p0"]
        return {
            "y_pred": y_pred,
            "y_b2": y_b2,
            "anomalia": y_pred - y_b2,
            "y_real": float(b2.iloc[0]["y_real"]),
            "shap": None,
            "modelo_version": f"piloto-{p0['modo']}-{p0['sha256_tensores'][:8]}",
            "vinculante": p0["vinculante"],
        }
