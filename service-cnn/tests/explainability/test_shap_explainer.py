"""Pruebas de la generación de explicaciones SHAP sobre la CNN-1D (sección 4.15).

Las explicaciones se generan ÚNICAMENTE sobre observaciones externas de
prueba, post-entrenamiento. El conjunto de referencia/background se extrae
solo de campañas de entrenamiento. DeepExplainer es el explainer principal
(precedente: Joshi et al. 2025); KernelExplainer es el plan de contingencia
documentado en docs/02-arquitectura-tecnica.md §1.1 si DeepExplainer falla la
validación de "local accuracy" en el ensayo piloto (H2).
"""
import numpy as np
import pytest
import torch

from src.explainability.shap_explainer import (
    LocalAccuracyError,
    SHAPExplanation,
    explain_with_deep_explainer,
    validate_local_accuracy,
)
from src.models.cnn1d import CNN1DConfig, QuinuaYieldCNN1D


@pytest.fixture
def modelo_entrenado():
    torch.manual_seed(0)
    config = CNN1DConfig(n_variables=3, n_pasos_temporales=4, n_covariables=2)
    modelo = QuinuaYieldCNN1D(config)
    modelo.eval()
    return modelo, config


class TestExplainWithDeepExplainer:
    def test_devuelve_forma_predictor_por_paso_temporal_por_observacion(
        self, modelo_entrenado
    ):
        modelo, config = modelo_entrenado
        x_background = torch.randn(15, config.n_variables, config.n_pasos_temporales)
        cov_background = torch.randn(15, config.n_covariables)
        x_prueba = torch.randn(3, config.n_variables, config.n_pasos_temporales)
        cov_prueba = torch.randn(3, config.n_covariables)

        explicacion = explain_with_deep_explainer(
            modelo, x_background, cov_background, x_prueba, cov_prueba
        )

        assert isinstance(explicacion, SHAPExplanation)
        assert explicacion.valores_secuenciales.shape == (
            3,
            config.n_variables,
            config.n_pasos_temporales,
        )
        assert explicacion.valores_covariables.shape == (3, config.n_covariables)

    def test_background_se_extrae_solo_de_los_datos_provistos(self, modelo_entrenado):
        """El llamador es responsable de pasar únicamente campañas de
        entrenamiento como background (regla anti-fuga, sección 4.12.2) —
        esta función no filtra ni valida eso, solo lo consume."""
        modelo, config = modelo_entrenado
        x_background = torch.zeros(5, config.n_variables, config.n_pasos_temporales)
        cov_background = torch.zeros(5, config.n_covariables)
        x_prueba = torch.randn(2, config.n_variables, config.n_pasos_temporales)
        cov_prueba = torch.randn(2, config.n_covariables)

        # No debe fallar simplemente por tener un background degenerado (todo
        # ceros); es responsabilidad del llamador dar un background sensato.
        explicacion = explain_with_deep_explainer(
            modelo, x_background, cov_background, x_prueba, cov_prueba
        )
        assert explicacion.valores_secuenciales.shape[0] == 2


class TestValidateLocalAccuracy:
    def test_pasa_cuando_suma_de_shap_mas_base_reproduce_la_prediccion(
        self, modelo_entrenado
    ):
        """Local accuracy (propiedad fundamental de SHAP, Lundberg y Lee 2017):
        la suma de los valores SHAP más el valor base debe reproducir la
        predicción del modelo para esa observación. Sección 02-arquitectura-
        tecnica.md §1.1: riesgo documentado de violación en series temporales."""
        modelo, config = modelo_entrenado
        x_background = torch.randn(15, config.n_variables, config.n_pasos_temporales)
        cov_background = torch.randn(15, config.n_covariables)
        x_prueba = torch.randn(3, config.n_variables, config.n_pasos_temporales)
        cov_prueba = torch.randn(3, config.n_covariables)

        explicacion = explain_with_deep_explainer(
            modelo, x_background, cov_background, x_prueba, cov_prueba
        )

        modelo.eval()
        with torch.no_grad():
            predicciones_reales = modelo(x_prueba, cov_prueba).squeeze(-1).numpy()

        # No debe lanzar si la propiedad se cumple dentro de la tolerancia.
        validate_local_accuracy(
            explicacion, predicciones_reales, tolerancia=0.05
        )

    def test_falla_si_la_suma_no_reproduce_la_prediccion(self, modelo_entrenado):
        modelo, config = modelo_entrenado
        explicacion_corrupta = SHAPExplanation(
            valores_secuenciales=np.zeros((2, config.n_variables, config.n_pasos_temporales)),
            valores_covariables=np.zeros((2, config.n_covariables)),
            valor_base=np.array([0.0, 0.0]),
        )
        predicciones_muy_distintas = np.array([1000.0, -1000.0])

        with pytest.raises(LocalAccuracyError):
            validate_local_accuracy(
                explicacion_corrupta, predicciones_muy_distintas, tolerancia=0.05
            )
