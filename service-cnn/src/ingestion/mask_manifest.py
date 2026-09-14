"""Cargador de umbrales de la máscara agrícola desde el manifiesto reproducible.

Referencia: data/manifest/mascara_agricola.yaml; sección 4.10.1 de la tesis
("el nombre, versión, resolución y fecha de cada capa, así como los
umbrales de altitud, pendiente y amplitud, se fijarán y registrarán antes
de la evaluación definitiva").

Este módulo NO importa `ee`/`geemap` ni toca la red: es lectura pura de un
archivo YAML ya existente, construyendo el `MaskThresholds` que consume
`src.preprocessing.mask.apply_agricultural_mask` y
`src.ingestion.gee_mask_pipeline.build_agricultural_mask_image`.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import yaml

from src.preprocessing.mask import MaskThresholds


class AmplitudNdviNoCalibradaError(ValueError):
    """`amplitud_ndvi_min` aún no se calibró contra datos reales de entrenamiento.

    A diferencia de `altitud_min_m`/`altitud_max_m`/`pendiente_max_grados`
    (criterios agronómicos externos, ya fijados con fuente citada en el
    manifiesto), la tesis (sección 4.10.1) exige que este umbral se calibre
    estadísticamente — no se puede inventar ni dejar en blanco.
    """


def load_mask_thresholds(ruta_yaml: str | Path) -> MaskThresholds:
    """Construye `MaskThresholds` desde el manifiesto de la máscara agrícola.

    Args:
        ruta_yaml: ruta al manifiesto (ej. `data/manifest/mascara_agricola.yaml`).

    Returns:
        `MaskThresholds` con los 4 valores leídos de la sección `umbrales`
        del manifiesto.

    Raises:
        AmplitudNdviNoCalibradaError: si `umbrales.amplitud_ndvi_min` es
            `null` en el manifiesto — indica que aún falta ejecutar
            `gee_mask_pipeline.sample_cells_for_calibration` +
            `mask.calibrate_ndvi_amplitude_threshold` sobre campañas de
            entrenamiento antes de poder usar la máscara en producción.

    Nota: si `umbrales.detalle.<nombre>.estado` empieza con "PROVISIONAL"
    para algún umbral, se emite un `UserWarning` (no bloquea la carga) —
    ver `_advertir_umbrales_provisionales`.
    """
    contenido = yaml.safe_load(Path(ruta_yaml).read_text(encoding="utf-8"))
    umbrales = contenido["umbrales"]

    if umbrales.get("amplitud_ndvi_min") is None:
        raise AmplitudNdviNoCalibradaError(
            "amplitud_ndvi_min no está calibrado en el manifiesto "
            f"({ruta_yaml}). Ejecute gee_mask_pipeline."
            "sample_cells_for_calibration + mask.calibrate_ndvi_amplitude_"
            "threshold sobre campañas de entrenamiento reales, y registre "
            "el resultado en el manifiesto antes de construir MaskThresholds."
        )

    _advertir_umbrales_provisionales(umbrales)

    return MaskThresholds(
        altitud_min_m=umbrales["altitud_min_m"],
        altitud_max_m=umbrales["altitud_max_m"],
        pendiente_max_grados=umbrales["pendiente_max_grados"],
        amplitud_ndvi_min=umbrales["amplitud_ndvi_min"],
    )


def _advertir_umbrales_provisionales(umbrales: dict) -> None:
    detalle = umbrales.get("detalle", {})
    for nombre_umbral, info in detalle.items():
        estado = info.get("estado", "") if isinstance(info, dict) else ""
        if isinstance(estado, str) and estado.startswith("PROVISIONAL"):
            warnings.warn(
                f"Umbral '{nombre_umbral}' es PROVISIONAL: {estado}",
                UserWarning,
                stacklevel=2,
            )
