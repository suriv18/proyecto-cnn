"""Pruebas de la preespecificación (E5): coherencia con el código y bloqueo del congelamiento.

La preespecificación (`configs/preespecificacion.yaml`) registra los umbrales, el
presupuesto y las reglas antes de evaluar. Si el código y el registro divergieran,
el registro dejaría de ser una garantía; estas pruebas lo detectan.
"""
from __future__ import annotations

import copy

import pytest

from src.evaluation.preregistration import (
    PreregistrationNotReadyError,
    assert_can_freeze,
    check_consistency_with_code,
    load_preregistration,
    pending_decisions,
    render_markdown,
)

RUTA = "configs/preespecificacion.yaml"


@pytest.fixture
def prereg() -> dict:
    return load_preregistration(RUTA)


class TestCoherenciaConElCodigo:
    def test_el_borrador_real_es_coherente_con_el_codigo(self, prereg):
        assert check_consistency_with_code(prereg) == []

    def test_detecta_un_umbral_de_estabilidad_distinto_del_codigo(self, prereg):
        alterado = copy.deepcopy(prereg)
        alterado["hipotesis"]["he2"]["estabilidad_w_kendall_minimo"]["valor"] = 0.80

        discrepancias = check_consistency_with_code(alterado)

        assert any("Kendall" in d for d in discrepancias)

    def test_detecta_fases_criticas_distintas_del_codigo(self, prereg):
        alterado = copy.deepcopy(prereg)
        alterado["hipotesis"]["he2"]["fases_criticas_horizonte_principal"]["valor"] = ["floracion"]

        assert any("fases críticas" in d for d in check_consistency_with_code(alterado))

    def test_detecta_un_presupuesto_distinto_del_del_codigo(self, prereg):
        alterado = copy.deepcopy(prereg)
        alterado["modelos"]["presupuesto_de_trials_por_modelo"]["valor"] = 50

        assert any("presupuesto" in d for d in check_consistency_with_code(alterado))

    def test_detecta_clasificaciones_excluidas_distintas_del_codigo(self, prereg):
        alterado = copy.deepcopy(prereg)
        alterado["datos"]["clasificaciones_excluidas"]["valor"] = ["costa_riego"]

        assert any("clasificaciones" in d for d in check_consistency_with_code(alterado))

    def test_detecta_modelos_estocasticos_y_deterministas_que_no_cubren_el_factor(self, prereg):
        alterado = copy.deepcopy(prereg)
        alterado["validacion"]["semillas"]["modelos_deterministas"]["valor"] = ["elastic_net"]

        assert any("modelos" in d for d in check_consistency_with_code(alterado))

    def test_detecta_un_numero_de_semillas_que_no_coincide_con_la_lista(self, prereg):
        alterado = copy.deepcopy(prereg)
        alterado["validacion"]["semillas"]["valores"]["valor"] = [0, 1, 2]

        assert any("semillas" in d for d in check_consistency_with_code(alterado))


class TestHistoriaMinimaDeLaLineaBase:
    """B2 solo usa tendencia lineal con un mínimo de campañas de historia; con menos
    cae a la media provincial y coincide con B1. El mínimo debe estar registrado."""

    def test_el_registro_coincide_con_la_constante_del_codigo(self, prereg):
        assert check_consistency_with_code(prereg) == []

    def test_detecta_una_historia_minima_distinta_del_codigo(self, prereg):
        alterado = copy.deepcopy(prereg)
        alterado["variable_objetivo"]["historia_minima_para_tendencia_campanas"]["valor"] = 3

        assert any("historia mínima" in d for d in check_consistency_with_code(alterado))

    def test_lo_decide_el_piloto(self, prereg):
        assert "variable_objetivo.historia_minima_para_tendencia_campanas" in pending_decisions(prereg)["por_decidir_piloto"]


class TestDecisionesPendientes:
    def test_lista_lo_que_requiere_al_asesor(self, prereg):
        pendientes = pending_decisions(prereg)

        assert "datos.periodo_de_campanas" in pendientes["por_decidir_asesor"]
        assert "validacion.ventana_inicial_de_entrenamiento_campanas" in pendientes["por_decidir_asesor"]

    def test_lista_lo_que_decide_el_piloto(self, prereg):
        pendientes = pending_decisions(prereg)

        assert "variable_objetivo.representacion_de_modelado" in pendientes["por_decidir_piloto"]
        assert "modelos.entrenamiento_cnn1d.early_stopping_particion" in pendientes["por_decidir_piloto"]

    def test_no_incluye_lo_ya_fijado_por_la_tesis(self, prereg):
        todos = [r for lista in pending_decisions(prereg).values() for r in lista]

        assert "hipotesis.nivel_de_significancia" not in todos


class TestCongelamiento:
    def test_no_se_puede_congelar_con_decisiones_pendientes(self, prereg):
        with pytest.raises(PreregistrationNotReadyError, match="por_decidir_asesor"):
            assert_can_freeze(prereg)

    def test_tampoco_se_puede_congelar_con_valores_sin_definir_aunque_todo_este_fijado(self, prereg):
        listo = copy.deepcopy(prereg)

        def fijar(nodo):
            if isinstance(nodo, dict) and "estado" in nodo and "valor" in nodo:
                nodo["estado"] = "fijado"
            elif isinstance(nodo, dict):
                for v in nodo.values():
                    fijar(v)

        fijar({k: v for k, v in listo.items() if k != "metadatos"})
        for seccion in [k for k in listo if k != "metadatos"]:
            fijar(listo[seccion])
        # los valores aún en `null` también impiden congelar
        with pytest.raises(PreregistrationNotReadyError, match="sin valor"):
            assert_can_freeze(listo)


class TestRenderizado:
    def test_el_texto_incluye_el_estado_y_la_lista_de_pendientes(self, prereg):
        texto = render_markdown(prereg)

        assert "BORRADOR" in texto
        assert "Decisiones pendientes" in texto
        assert "datos.periodo_de_campanas" in texto
        assert "fijado_por_tesis" in texto
