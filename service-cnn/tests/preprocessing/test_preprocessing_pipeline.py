"""Pruebas de la orquestación del preprocesamiento (secciones 4.10.2-4.10.4, 4.12.2).

Conecta alineación fenológica + tensor V×T + rendimiento objetivo y verifica
las garantías metodológicas: sin fuga en el respaldo departamental, rechazo
del esquema NDVI desactualizado y registro de exclusiones con su motivo.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.preprocessing_pipeline import (
    RawDatosImplausiblesError,
    RawSchemaDesactualizadaError,
    build_processed_dataset,
    save_processed_dataset,
    validate_raw_ranges,
    validate_raw_schema,
)

DURACION = {"emergencia": 3, "desarrollo_vegetativo": 5}
ORDEN_VARIABLES = ["precipitacion", "ndvi_mean"]
PROV_A_DEPTO = {"AZANGARO": "PUNO", "CHUCUITO": "PUNO", "LAMPA": "PUNO"}


def _series(celdas: list[tuple[str, int]]) -> pd.DataFrame:
    """Series diarias sintéticas que cubren de mediados de sep a mediados de dic
    del año de siembra, con las dos variables de ORDEN_VARIABLES."""
    filas = []
    for provincia, campana in celdas:
        for fecha in pd.date_range(f"{campana - 1}-09-15", f"{campana - 1}-12-15"):
            filas.append(
                dict(provincia_id=provincia, campana_id=campana, fecha=fecha.date(),
                     variable="precipitacion", valor=1.0)
            )
            filas.append(
                dict(provincia_id=provincia, campana_id=campana, fecha=fecha.date(),
                     variable="ndvi", valor_mean=0.1, valor_max=0.2)
            )
    return pd.DataFrame(filas)


def _siembra() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(provincia_id="AZANGARO", campana_id=2021, mes=10, superficie_ha=300.0),
            # Campaña de prueba con muchísima siembra en noviembre: no debe
            # influir en el respaldo calculado solo con entrenamiento.
            dict(provincia_id="AZANGARO", campana_id=2022, mes=11, superficie_ha=9000.0),
        ]
    )


def _rendimiento() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(provincia_id="AZANGARO", campana_id=2021, rendimiento_kg_ha=1200.0),
            dict(provincia_id="AZANGARO", campana_id=2022, rendimiento_kg_ha=1300.0),
            dict(provincia_id="CHUCUITO", campana_id=2022, rendimiento_kg_ha=900.0),
        ]
    )


def _construir(**kwargs):
    celdas = [("AZANGARO", 2021), ("AZANGARO", 2022), ("CHUCUITO", 2022)]
    argumentos = dict(
        series_crudas=_series(celdas),
        siembra_mensual=_siembra(),
        rendimiento=_rendimiento(),
        provincia_a_departamento=PROV_A_DEPTO,
        duracion_dias_por_fase=DURACION,
        orden_variables=ORDEN_VARIABLES,
        campanas_entrenamiento=[2021],
    )
    argumentos.update(kwargs)
    return build_processed_dataset(**argumentos)


class TestValidateRawSchema:
    def test_acepta_el_esquema_vigente(self):
        validate_raw_schema(_series([("AZANGARO", 2021)]))

    def test_rechaza_ndvi_con_esquema_viejo_de_columna_valor_unica(self):
        viejo = _series([("AZANGARO", 2021)]).drop(columns=["valor_mean", "valor_max"])
        viejo.loc[viejo["variable"] == "ndvi", "valor"] = 0.1

        with pytest.raises(RawSchemaDesactualizadaError, match="re-extra"):
            validate_raw_schema(viejo)


class TestValidateRawRanges:
    """Control de plausibilidad física de las series crudas.

    Habría detectado la regresión del reductor espacial (commit de73679): la
    precipitación diaria de una provincia salía como la SUMA de ~177 píxeles
    (hasta 85 970 mm) y la radiación como ~25 veces el valor real."""

    def test_acepta_series_fisicamente_plausibles(self):
        validate_raw_ranges(_series([("AZANGARO", 2021)]))

    def test_rechaza_precipitacion_diaria_imposible_por_suma_espacial(self):
        series = _series([("AZANGARO", 2021)])
        series.loc[series["variable"] == "precipitacion", "valor"] = 114.06 * 300  # suma de píxeles

        with pytest.raises(RawDatosImplausiblesError, match="precipitacion"):
            validate_raw_ranges(series)

    def test_rechaza_radiacion_solar_fuera_de_rango_fisico(self):
        series = _series([("AZANGARO", 2021)])
        filas = series[series["variable"] == "precipitacion"].copy()
        filas["variable"] = "radiacion_solar"
        filas["valor"] = 8.8e8  # J/m2 por dia: ~25 veces el maximo posible
        series = pd.concat([series, filas], ignore_index=True)

        with pytest.raises(RawDatosImplausiblesError, match="radiacion_solar"):
            validate_raw_ranges(series)

    def test_ignora_los_nulos_estructurales_de_columnas_que_no_aplican(self):
        # las filas de NDVI no tienen `valor` y las demás no tienen `valor_mean`
        validate_raw_ranges(_series([("AZANGARO", 2021)]))


class TestBuildProcessedDataset:
    def test_forma_del_tensor_y_alineacion_con_las_observaciones(self):
        ds = _construir()

        assert ds.X.shape == (3, 2, 2)  # N obs x V variables x T fases
        assert len(ds.observaciones) == 3
        assert set(ds.observaciones.columns) >= {
            "provincia_id", "campana_id", "rendimiento_kg_ha",
            "fuente_alineacion", "es_imputado", "fecha_siembra",
        }

    def test_el_rendimiento_corresponde_a_cada_fila_del_tensor(self):
        ds = _construir()

        fila = ds.observaciones.query("provincia_id == 'CHUCUITO' and campana_id == 2022")
        assert fila["rendimiento_kg_ha"].iloc[0] == 900.0

    def test_respaldo_usa_solo_campanas_de_entrenamiento(self):
        """CHUCUITO 2022 no tiene siembra ni NDVI utilizable: cae al respaldo
        de PUNO. Con entrenamiento=[2021] el mes modal es octubre, no el
        noviembre dominante de la campaña de prueba 2022."""
        ds = _construir(campanas_entrenamiento=[2021])

        fila = ds.observaciones.query("provincia_id == 'CHUCUITO'").iloc[0]
        assert fila["fuente_alineacion"] == "respaldo"
        assert fila["fecha_siembra"] == date(2021, 10, 16)

    def test_el_resultado_no_cambia_al_alterar_una_campana_de_prueba(self):
        base = _construir(campanas_entrenamiento=[2021])
        alterada_siembra = _siembra()
        alterada_siembra.loc[alterada_siembra["campana_id"] == 2022, "mes"] = 12
        alterada = _construir(
            siembra_mensual=alterada_siembra, campanas_entrenamiento=[2021]
        )

        a = base.observaciones.query("provincia_id == 'CHUCUITO'")["fecha_siembra"].iloc[0]
        b = alterada.observaciones.query("provincia_id == 'CHUCUITO'")["fecha_siembra"].iloc[0]
        assert a == b

    def test_marca_exploratorio_cuando_no_se_indican_campanas_de_entrenamiento(self):
        ds = _construir(campanas_entrenamiento=None)

        assert ds.metadata["exploratorio"] is True
        assert ds.metadata["campanas_entrenamiento"] is None

    def test_no_marca_exploratorio_con_campanas_de_entrenamiento(self):
        ds = _construir(campanas_entrenamiento=[2021])

        assert ds.metadata["exploratorio"] is False
        assert ds.metadata["campanas_entrenamiento"] == [2021]

    def test_registra_exclusion_por_variable_sin_ninguna_observacion(self):
        series = _series([("AZANGARO", 2021), ("AZANGARO", 2022)])  # sin CHUCUITO
        ds = _construir(series_crudas=series)

        assert len(ds.observaciones) == 2
        excluidas = ds.exclusiones.query("provincia_id == 'CHUCUITO'")
        assert len(excluidas) == 1
        assert excluidas["motivo"].iloc[0] in {
            "sin_alineacion_fenologica", "tensor_incompleto",
        }

    def test_diccionario_de_variables_describe_cada_fila_del_tensor(self):
        ds = _construir()

        assert list(ds.diccionario["nombre"]) == ORDEN_VARIABLES
        assert set(ds.diccionario["rol"]) == {"secuencial"}

    def test_metadata_registra_ordenes_de_variables_y_fases(self):
        ds = _construir()

        assert ds.metadata["orden_variables"] == ORDEN_VARIABLES
        assert ds.metadata["orden_fases"] == ["emergencia", "desarrollo_vegetativo"]


class TestAmbitoTerritorial:
    """Criterio de exclusión 4 (sección 4.7): se excluyen las provincias
    costeras bajo riego y de selva por régimen productivo no comparable; los
    casos de transición se conservan marcados hasta resolverse con evidencia."""

    CLASIF = {"AZANGARO": "altoandina", "CHUCUITO": "costa_riego", "LAMPA": "transicion"}

    def test_excluye_provincias_costeras_con_su_motivo(self):
        ds = _construir(clasificacion_por_provincia=self.CLASIF)

        assert set(ds.observaciones["provincia_id"]) == {"AZANGARO"}
        excluida = ds.exclusiones.query("provincia_id == 'CHUCUITO'")
        assert list(excluida["motivo"]) == ["fuera_del_ambito_altoandino"]
        assert "costa_riego" in excluida["detalle"].iloc[0]

    def test_conserva_los_casos_de_transicion_marcados(self):
        rendimiento = pd.concat(
            [_rendimiento(), pd.DataFrame([dict(provincia_id="LAMPA", campana_id=2021, rendimiento_kg_ha=1000.0)])]
        )
        series = _series([("AZANGARO", 2021), ("AZANGARO", 2022), ("CHUCUITO", 2022), ("LAMPA", 2021)])
        siembra = pd.concat(
            [_siembra(), pd.DataFrame([dict(provincia_id="LAMPA", campana_id=2021, mes=10, superficie_ha=50.0)])]
        )

        ds = _construir(
            series_crudas=series, rendimiento=rendimiento, siembra_mensual=siembra,
            clasificacion_por_provincia=self.CLASIF,
        )

        lampa = ds.observaciones.query("provincia_id == 'LAMPA'")
        assert len(lampa) == 1
        assert lampa["clasificacion"].iloc[0] == "transicion"

    def test_registra_en_metadata_las_clasificaciones_excluidas(self):
        ds = _construir(clasificacion_por_provincia=self.CLASIF)

        assert ds.metadata["clasificaciones_excluidas"] == ["costa_riego", "selva"]

    def test_sin_mapa_de_clasificacion_no_excluye_nada(self):
        ds = _construir()

        assert len(ds.observaciones) == 3
        assert ds.metadata["clasificaciones_excluidas"] == []


class TestAmbitoAltitudinal:
    """La provincia es ámbito altoandino operativo solo si la altitud mediana
    de su superficie de cultivo está dentro del rango de la máscara (4.5.3):
    fuera de él, el NDVI enmascarado no representa a la provincia."""

    ALT = {"AZANGARO": 3800.0, "CHUCUITO": 4300.0}

    def test_excluye_provincia_cuyo_cultivo_esta_fuera_del_rango_altitudinal(self):
        ds = _construir(altitud_mediana_cultivo_por_provincia=self.ALT, rango_altitudinal=(2500, 4000))

        assert set(ds.observaciones["provincia_id"]) == {"AZANGARO"}
        excluida = ds.exclusiones.query("provincia_id == 'CHUCUITO'")
        assert list(excluida["motivo"]) == ["fuera_del_ambito_altoandino"]
        assert "4300" in excluida["detalle"].iloc[0]

    def test_provincia_sin_dato_de_altitud_no_se_excluye(self):
        ds = _construir(altitud_mediana_cultivo_por_provincia={"AZANGARO": 3800.0}, rango_altitudinal=(2500, 4000))

        assert "CHUCUITO" in set(ds.observaciones["provincia_id"])

    def test_registra_el_rango_en_metadata(self):
        ds = _construir(altitud_mediana_cultivo_por_provincia=self.ALT, rango_altitudinal=(2500, 4000))

        assert ds.metadata["rango_altitudinal"] == [2500, 4000]


class TestCoberturaDeLaVentana:
    """Criterio de exclusión 2 (4.7): no se reconstruye una secuencia que
    exigiría extrapolar fuera de la serie extraída."""

    def test_excluye_observacion_cuyo_ciclo_excede_la_serie_extraida(self):
        # la serie sintética llega al 15-dic; siembra el 16-dic + 8 días de ciclo
        siembra = pd.DataFrame(
            [
                dict(provincia_id="AZANGARO", campana_id=2021, mes=10, superficie_ha=300.0),
                dict(provincia_id="AZANGARO", campana_id=2022, mes=12, superficie_ha=300.0),
            ]
        )
        ds = _construir(siembra_mensual=siembra, campanas_entrenamiento=[2021])

        excluida = ds.exclusiones.query("provincia_id == 'AZANGARO' and campana_id == 2022")
        assert list(excluida["motivo"]) == ["ciclo_fuera_de_ventana_de_extraccion"]

    def test_conserva_las_observaciones_cuyo_ciclo_esta_cubierto(self):
        ds = _construir()

        assert len(ds.observaciones) == 3
        assert "ciclo_fuera_de_ventana_de_extraccion" not in set(ds.exclusiones["motivo"])


class TestArrastreDeNdvi:
    def test_sin_arrastre_no_hay_celdas_imputadas(self):
        ds = _construir()

        assert ds.X_imputado.shape == ds.X.shape
        assert not ds.X_imputado.any()
        assert ds.metadata["max_dias_arrastre_ndvi"] == 0
        assert ds.metadata["celdas_imputadas"] == 0

    def test_el_guardado_incluye_la_mascara_de_imputacion(self, tmp_path):
        ds = _construir()

        save_processed_dataset(ds, tmp_path)

        datos = np.load(tmp_path / "tensores.npz")
        assert datos["imputado"].shape == datos["X"].shape


class TestSaveProcessedDataset:
    def test_escribe_los_cuatro_artefactos_y_se_puede_releer(self, tmp_path):
        ds = _construir()

        save_processed_dataset(ds, tmp_path)

        assert (tmp_path / "tensores.npz").exists()
        assert (tmp_path / "observaciones.parquet").exists()
        assert (tmp_path / "diccionario_variables.csv").exists()
        assert (tmp_path / "metadata.json").exists()
        recargado = np.load(tmp_path / "tensores.npz")["X"]
        np.testing.assert_array_equal(recargado, ds.X)
