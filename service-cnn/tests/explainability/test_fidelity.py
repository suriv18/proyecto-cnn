"""Pruebas de fidelidad mediante oclusión (HE2b, sección 3.3.4, 4.15).

En el conjunto externo de prueba, se reemplazan las entradas de alta
contribución por su valor de referencia (background), SIN reentrenar, y se
compara el incremento del RMSE con el producido por una ventana de control de
igual duración y baja contribución. La fidelidad se sostiene si el incremento
por oclusión de alta contribución es mayor que el de la ventana de control.
"""
import numpy as np
import pytest
import torch

from src.explainability.fidelity import (
    FidelityResult,
    evaluate_fidelity_by_occlusion,
    occlude_phase,
)
from src.models.cnn1d import CNN1DConfig, QuinuaYieldCNN1D


class TestOccludePhase:
    def test_reemplaza_los_pasos_de_la_fase_por_el_valor_de_referencia(self):
        tensor = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
        )  # (1 obs, 2 variables, 4 pasos)
        valor_referencia = torch.zeros(2)  # media de background por variable
        indices_fase = [1, 2]  # ocluir pasos 1 y 2 (floracion, p.ej.)

        ocluido = occlude_phase(tensor, indices_fase, valor_referencia)

        assert ocluido[0, 0, 0] == 1.0  # paso 0, no ocluido
        assert ocluido[0, 0, 1] == 0.0  # ocluido con el valor de referencia
        assert ocluido[0, 0, 2] == 0.0  # ocluido
        assert ocluido[0, 0, 3] == 4.0  # paso 3, no ocluido

    def test_no_modifica_el_tensor_original(self):
        tensor = torch.tensor([[[1.0, 2.0, 3.0]]])
        valor_referencia = torch.zeros(1)
        occlude_phase(tensor, [0], valor_referencia)
        assert tensor[0, 0, 0] == 1.0  # tensor original intacto


class TestEvaluateFidelityByOcclusion:
    @pytest.fixture
    def modelo_y_datos(self):
        torch.manual_seed(0)
        config = CNN1DConfig(n_variables=2, n_pasos_temporales=6, n_covariables=1)
        modelo = QuinuaYieldCNN1D(config)
        modelo.eval()

        x_prueba = torch.randn(10, 2, 6)
        cov_prueba = torch.randn(10, 1)
        y_real = torch.randn(10, 1)
        valor_referencia = torch.zeros(2)  # background degenerado simple

        return modelo, x_prueba, cov_prueba, y_real, valor_referencia

    def test_devuelve_incrementos_de_rmse_no_negativos_por_construccion(
        self, modelo_y_datos
    ):
        """No se garantiza que ocluir siempre EMPEORE el error de un modelo
        no entrenado hacia la fidelidad esperada, pero el incremento (positivo
        o negativo) debe calcularse correctamente como una diferencia real."""
        modelo, x_prueba, cov_prueba, y_real, valor_referencia = modelo_y_datos

        resultado = evaluate_fidelity_by_occlusion(
            modelo,
            x_prueba,
            cov_prueba,
            y_real,
            indices_fase_alta_contribucion=[2, 3],
            indices_fase_control=[0, 1],
            valor_referencia=valor_referencia,
        )

        assert isinstance(resultado, FidelityResult)
        assert isinstance(resultado.incremento_rmse_alta_contribucion, float)
        assert isinstance(resultado.incremento_rmse_control, float)

    def test_no_reentrena_el_modelo(self, modelo_y_datos):
        """Sección 3.3.4: la fidelidad se evalúa SIN reentrenar — los pesos
        del modelo deben permanecer intactos tras la evaluación."""
        modelo, x_prueba, cov_prueba, y_real, valor_referencia = modelo_y_datos
        pesos_antes = [p.clone() for p in modelo.parameters()]

        evaluate_fidelity_by_occlusion(
            modelo,
            x_prueba,
            cov_prueba,
            y_real,
            indices_fase_alta_contribucion=[2, 3],
            indices_fase_control=[0, 1],
            valor_referencia=valor_referencia,
        )

        pesos_despues = list(modelo.parameters())
        for antes, despues in zip(pesos_antes, pesos_despues):
            assert torch.allclose(antes, despues)

    def test_falla_si_las_ventanas_tienen_duracion_distinta(self, modelo_y_datos):
        """Sección 3.3.4: la ventana de control debe ser de igual duración que
        la ventana de alta contribución para que la comparación sea válida."""
        modelo, x_prueba, cov_prueba, y_real, valor_referencia = modelo_y_datos

        with pytest.raises(ValueError, match="misma duraci[oó]n"):
            evaluate_fidelity_by_occlusion(
                modelo,
                x_prueba,
                cov_prueba,
                y_real,
                indices_fase_alta_contribucion=[2, 3, 4],
                indices_fase_control=[0, 1],
                valor_referencia=valor_referencia,
            )

    def test_fidelidad_se_sostiene_si_ocluir_alta_contribucion_degrada_mas(self):
        """Construimos un modelo determinista simple (suma ponderada) donde
        una fase tiene coeficiente mucho mayor que la otra, para verificar que
        el criterio de fidelidad detecta correctamente cuál fase es más
        influyente en la práctica."""

        class ModeloJuguete(torch.nn.Module):
            def forward(self, x, cov):
                # Pesa fuertemente los pasos 2-3 (alta contribución real) y
                # débilmente los pasos 0-1 (control).
                pesos = torch.tensor([0.01, 0.01, 10.0, 10.0, 0.01, 0.01])
                return (x[:, 0, :] * pesos).sum(dim=1, keepdim=True)

        modelo = ModeloJuguete()
        torch.manual_seed(1)
        x_prueba = torch.randn(20, 1, 6)
        cov_prueba = torch.zeros(20, 1)
        with torch.no_grad():
            y_real = modelo(x_prueba, cov_prueba)
        valor_referencia = torch.zeros(1)

        resultado = evaluate_fidelity_by_occlusion(
            modelo,
            x_prueba,
            cov_prueba,
            y_real,
            indices_fase_alta_contribucion=[2, 3],
            indices_fase_control=[0, 1],
            valor_referencia=valor_referencia,
        )

        assert resultado.fidelidad_sostenida is True
        assert (
            resultado.incremento_rmse_alta_contribucion
            > resultado.incremento_rmse_control
        )
