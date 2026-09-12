"""Pruebas de los esquemas de validación temporal anti-fuga (sección 4.12.1-4.12.2).

- Origen móvil (walk-forward): esquema confirmatorio principal — cada campaña
  externa se predice usando exclusivamente campañas anteriores.
- Leave-one-campaign-out (LOCO): complementario, puede usar campañas
  posteriores a la evaluada (por eso no es confirmatorio).
- Leave-one-department-out (LODO): análisis de robustez espacial.
"""
import pandas as pd
import pytest

from src.evaluation.splits import (
    VentanaInicialInsuficienteError,
    expanding_window_splits,
    leave_one_campaign_out_splits,
    leave_one_department_out_splits,
)


@pytest.fixture
def observaciones() -> pd.DataFrame:
    """19 campañas (2006-2024) en 2 departamentos, 2 provincias cada uno."""
    filas = []
    for campana in range(2006, 2025):
        for departamento, provincia in [
            ("Puno", "Azangaro"),
            ("Puno", "ElCollao"),
            ("Ayacucho", "Huamanga"),
            ("Ayacucho", "Cangallo"),
        ]:
            filas.append(
                {
                    "provincia_id": provincia,
                    "departamento": departamento,
                    "campana_id": campana,
                }
            )
    return pd.DataFrame(filas)


class TestExpandingWindowSplits:
    def test_entrenamiento_usa_solo_campanas_anteriores_a_la_externa(
        self, observaciones
    ):
        splits = expanding_window_splits(
            observaciones, columna_campana="campana_id", ventana_inicial_minima=5
        )
        for split in splits:
            max_entrenamiento = split.entrenamiento["campana_id"].max()
            campanas_prueba = split.prueba["campana_id"].unique()
            assert len(campanas_prueba) == 1
            assert max_entrenamiento < campanas_prueba[0]

    def test_primera_campana_externa_respeta_la_ventana_inicial_minima(
        self, observaciones
    ):
        splits = expanding_window_splits(
            observaciones, columna_campana="campana_id", ventana_inicial_minima=5
        )
        # 19 campañas (2006-2024), ventana inicial de 5 -> primera prueba es 2011
        primera_campana_prueba = splits[0].prueba["campana_id"].iloc[0]
        assert primera_campana_prueba == 2006 + 5

    def test_genera_un_split_por_cada_campana_posterior_a_la_ventana_inicial(
        self, observaciones
    ):
        splits = expanding_window_splits(
            observaciones, columna_campana="campana_id", ventana_inicial_minima=5
        )
        # 19 campañas totales, 5 consumidas por la ventana inicial -> 14 splits
        assert len(splits) == 14

    def test_ventana_de_entrenamiento_crece_monotonamente(self, observaciones):
        splits = expanding_window_splits(
            observaciones, columna_campana="campana_id", ventana_inicial_minima=5
        )
        tamanos = [len(s.entrenamiento) for s in splits]
        for anterior, siguiente in zip(tamanos, tamanos[1:]):
            assert siguiente > anterior

    def test_falla_si_la_ventana_inicial_excede_el_numero_de_campanas(
        self, observaciones
    ):
        with pytest.raises(VentanaInicialInsuficienteError):
            expanding_window_splits(
                observaciones, columna_campana="campana_id", ventana_inicial_minima=100
            )

    def test_no_hay_fuga_de_provincias_de_la_misma_campana_de_prueba(
        self, observaciones
    ):
        """Todas las provincias de la campaña externa deben ir juntas a prueba,
        nunca divididas entre entrenamiento y prueba (evita fuga por bloque)."""
        splits = expanding_window_splits(
            observaciones, columna_campana="campana_id", ventana_inicial_minima=5
        )
        for split in splits:
            campana_prueba = split.prueba["campana_id"].iloc[0]
            en_entrenamiento = split.entrenamiento[
                split.entrenamiento["campana_id"] == campana_prueba
            ]
            assert en_entrenamiento.empty


class TestLeaveOneCampaignOutSplits:
    def test_genera_un_split_por_cada_campana(self, observaciones):
        splits = leave_one_campaign_out_splits(observaciones, columna_campana="campana_id")
        campanas_unicas = observaciones["campana_id"].nunique()
        assert len(splits) == campanas_unicas

    def test_entrenamiento_puede_incluir_campanas_posteriores(self, observaciones):
        """A diferencia del origen móvil, LOCO es complementario precisamente
        porque el entrenamiento puede usar campañas posteriores a la evaluada."""
        splits = leave_one_campaign_out_splits(observaciones, columna_campana="campana_id")
        split_de_una_campana_temprana = next(
            s for s in splits if s.prueba["campana_id"].iloc[0] == 2007
        )
        campanas_entrenamiento = split_de_una_campana_temprana.entrenamiento[
            "campana_id"
        ]
        assert (campanas_entrenamiento > 2007).any()

    def test_prueba_y_entrenamiento_son_disjuntos_en_la_campana(self, observaciones):
        splits = leave_one_campaign_out_splits(observaciones, columna_campana="campana_id")
        for split in splits:
            campana_prueba = split.prueba["campana_id"].iloc[0]
            assert (split.entrenamiento["campana_id"] != campana_prueba).all()


class TestLeaveOneDepartmentOutSplits:
    def test_genera_un_split_por_cada_departamento(self, observaciones):
        splits = leave_one_department_out_splits(
            observaciones, columna_departamento="departamento"
        )
        assert len(splits) == observaciones["departamento"].nunique()

    def test_prueba_contiene_solo_el_departamento_excluido(self, observaciones):
        splits = leave_one_department_out_splits(
            observaciones, columna_departamento="departamento"
        )
        for split in splits:
            departamentos_prueba = split.prueba["departamento"].unique()
            assert len(departamentos_prueba) == 1

    def test_entrenamiento_excluye_por_completo_el_departamento_de_prueba(
        self, observaciones
    ):
        splits = leave_one_department_out_splits(
            observaciones, columna_departamento="departamento"
        )
        for split in splits:
            departamento_prueba = split.prueba["departamento"].iloc[0]
            assert (split.entrenamiento["departamento"] != departamento_prueba).all()
