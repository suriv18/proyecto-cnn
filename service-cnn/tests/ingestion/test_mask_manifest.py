"""Pruebas del cargador de umbrales de la máscara agrícola desde manifiesto.

Referencia: data/manifest/mascara_agricola.yaml, sección 4.10.1 de la tesis
("el nombre, versión, resolución y fecha de cada capa, así como los
umbrales de altitud, pendiente y amplitud, se fijarán y registrarán antes
de la evaluación definitiva").
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.ingestion.mask_manifest import (
    AmplitudNdviNoCalibradaError,
    load_mask_thresholds,
)
from src.preprocessing.mask import MaskThresholds


def _escribir_manifiesto(
    tmp_path: Path,
    altitud_min_m: float,
    altitud_max_m: float,
    pendiente_max_grados: float,
    amplitud_ndvi_min: float | None,
    estado_altitud: str | None = None,
    estado_pendiente: str | None = None,
    estado_amplitud: str | None = None,
) -> Path:
    """Construye un manifiesto YAML fixture con la MISMA estructura anidada
    del manifiesto real (umbrales.detalle.<nombre>.estado) — ver Global
    Constraints del plan para la estructura completa verificada."""
    detalle = {}
    if estado_altitud is not None:
        detalle["altitud"] = {"fuente": "fixture de prueba", "estado": estado_altitud}
    if estado_pendiente is not None:
        detalle["pendiente"] = {"fuente": "fixture de prueba", "estado": estado_pendiente}
    if estado_amplitud is not None:
        detalle["amplitud_ndvi"] = {"fuente": "fixture de prueba", "estado": estado_amplitud}

    umbrales = {
        "altitud_min_m": altitud_min_m,
        "altitud_max_m": altitud_max_m,
        "pendiente_max_grados": pendiente_max_grados,
        "amplitud_ndvi_min": amplitud_ndvi_min,
    }
    if detalle:
        umbrales["detalle"] = detalle

    ruta = tmp_path / "mascara_agricola.yaml"
    ruta.write_text(
        yaml.safe_dump({"umbrales": umbrales}, allow_unicode=True),
        encoding="utf-8",
    )
    return ruta


class TestLoadMaskThresholds:
    def test_carga_los_4_umbrales_correctamente(self, tmp_path):
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
        )

        resultado = load_mask_thresholds(ruta)

        assert resultado == MaskThresholds(
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
        )

    def test_acepta_ruta_como_string(self, tmp_path):
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
        )

        resultado = load_mask_thresholds(str(ruta))

        assert resultado.amplitud_ndvi_min == 0.15

    def test_amplitud_ndvi_min_nulo_lanza_error_explicito(self, tmp_path):
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=None,
        )

        with pytest.raises(AmplitudNdviNoCalibradaError, match="sample_cells_for_calibration"):
            load_mask_thresholds(ruta)

    def test_advierte_si_altitud_esta_marcada_provisional(self, tmp_path):
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
            estado_altitud="PROVISIONAL — pendiente de aprobación del asesor",
        )

        with pytest.warns(UserWarning, match="PROVISIONAL"):
            load_mask_thresholds(ruta)

    def test_advierte_si_pendiente_esta_marcada_provisional(self, tmp_path):
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
            estado_pendiente="PROVISIONAL — pendiente de aprobación del asesor",
        )

        with pytest.warns(UserWarning, match="PROVISIONAL"):
            load_mask_thresholds(ruta)

    def test_no_advierte_si_no_hay_seccion_detalle(self, tmp_path, recwarn):
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
        )

        load_mask_thresholds(ruta)

        assert len(recwarn) == 0

    def test_estado_pendiente_de_amplitud_no_genera_warning_extra(self, tmp_path, recwarn):
        """amplitud_ndvi_min ya vale 0.15 (calibrado) en este fixture, pero
        su bloque `detalle.amplitud_ndvi.estado` todavía dice "PENDIENTE"
        (posible desfase entre el valor numérico y la nota de estado, ej.
        si alguien actualiza el valor pero olvida actualizar el estado) —
        "PENDIENTE" no empieza con "PROVISIONAL", así que no debe generar
        warning; solo se advierte por estados que literalmente empiezan
        con "PROVISIONAL". Usa la fixture `recwarn` (no `pytest.warns(None)`,
        removido en pytest moderno) para verificar ausencia de warnings."""
        ruta = _escribir_manifiesto(
            tmp_path,
            altitud_min_m=2500,
            altitud_max_m=4000,
            pendiente_max_grados=14.04,
            amplitud_ndvi_min=0.15,
            estado_amplitud="PENDIENTE — requiere ejecutar la calibración contra datos reales de entrenamiento",
        )

        load_mask_thresholds(ruta)

        assert len(recwarn) == 0

    def test_carga_correctamente_el_manifiesto_real_del_proyecto(self):
        """El manifiesto real (ya versionado en el repo) es cargable: ruta,
        estructura YAML y campos íntegros, no solo los fixtures sintéticos.
        Desde la calibración del 2026-10-09 `amplitud_ndvi_min` tiene valor
        (percentil 5, provisional), y los tres umbrales siguen PROVISIONALES
        hasta la aprobación del asesor, por lo que la carga avisa de cada uno."""
        ruta_real = Path("data/manifest/mascara_agricola.yaml")
        with pytest.warns(UserWarning, match="PROVISIONAL") as avisos:
            umbrales = load_mask_thresholds(ruta_real)

        assert umbrales.amplitud_ndvi_min == pytest.approx(0.323)
        assert umbrales.altitud_min_m < umbrales.altitud_max_m
        assert len(avisos) == 3  # altitud, pendiente y amplitud_ndvi
