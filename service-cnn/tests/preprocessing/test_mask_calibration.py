"""Pruebas del resumen de calibración de `amplitud_ndvi_min` (sección 4.10.1).

El umbral se calibra solo con celdas de cultivo que ya cumplen altitud y
pendiente, y solo con campañas de entrenamiento (regla anti-fuga, 4.12.2).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.mask import summarize_amplitude_calibration


def _tabla() -> pd.DataFrame:
    filas = []
    # 10 celdas de cultivo elegibles en PUNO con amplitudes 0.1 ... 1.0
    for i in range(1, 11):
        filas.append(dict(provincia_id="AZANGARO", departamento="PUNO", campana_id=2016,
                          cobertura_suelo="cultivo", altitud_m=3800.0, pendiente_grados=3.0,
                          ndvi_max=i / 10, ndvi_min=0.0))
    # 10 celdas de cultivo elegibles en CUSCO con amplitudes 0.6 ... 1.5
    for i in range(1, 11):
        filas.append(dict(provincia_id="ANTA", departamento="CUSCO", campana_id=2016,
                          cobertura_suelo="cultivo", altitud_m=3400.0, pendiente_grados=3.0,
                          ndvi_max=0.5 + i / 10, ndvi_min=0.0))
    return pd.DataFrame(filas)


KW = dict(altitud_min_m=2500, altitud_max_m=4000, pendiente_max_grados=14.04,
          campanas_entrenamiento=[2016, 2017])


class TestSummarizeAmplitudeCalibration:
    def test_el_umbral_es_el_percentil_de_las_celdas_elegibles(self):
        resumen = summarize_amplitude_calibration(_tabla(), percentiles=[50], **KW)

        # 20 amplitudes: 0.1..1.0 y 0.6..1.5 -> mediana = 0.8
        assert resumen["umbrales"]["50"]["umbral"] == pytest.approx(0.8)
        assert resumen["n_elegibles"] == 20

    def test_ignora_celdas_que_no_son_cultivo_o_fallan_altitud_o_pendiente(self):
        tabla = _tabla()
        ruido = pd.DataFrame(
            [
                dict(provincia_id="X", departamento="PUNO", campana_id=2016,
                     cobertura_suelo="no_cultivo", altitud_m=3800.0, pendiente_grados=3.0,
                     ndvi_max=0.0, ndvi_min=0.0),
                dict(provincia_id="X", departamento="PUNO", campana_id=2016,
                     cobertura_suelo="cultivo", altitud_m=1000.0, pendiente_grados=3.0,
                     ndvi_max=0.0, ndvi_min=0.0),
                dict(provincia_id="X", departamento="PUNO", campana_id=2016,
                     cobertura_suelo="cultivo", altitud_m=3800.0, pendiente_grados=40.0,
                     ndvi_max=0.0, ndvi_min=0.0),
            ]
        )

        resumen = summarize_amplitude_calibration(
            pd.concat([tabla, ruido]), percentiles=[50], **KW
        )

        assert resumen["n_elegibles"] == 20
        assert resumen["n_celdas_cultivo"] == 22
        assert resumen["umbrales"]["50"]["umbral"] == pytest.approx(0.8)

    def test_informa_el_porcentaje_excluido_por_departamento(self):
        resumen = summarize_amplitude_calibration(_tabla(), percentiles=[50], **KW)

        excluidas = resumen["umbrales"]["50"]["excluidas_pct_por_departamento"]
        # umbral 0.8: PUNO (0.1..1.0) excluye 7 de 10; CUSCO (0.6..1.5) excluye 2 de 10
        assert excluidas["PUNO"] == pytest.approx(70.0)
        assert excluidas["CUSCO"] == pytest.approx(20.0)

    def test_rechaza_campanas_fuera_del_conjunto_de_entrenamiento(self):
        tabla = _tabla()
        tabla.loc[0, "campana_id"] = 2022  # campaña de prueba colada

        with pytest.raises(ValueError, match="entrenamiento"):
            summarize_amplitude_calibration(tabla, percentiles=[50], **KW)
