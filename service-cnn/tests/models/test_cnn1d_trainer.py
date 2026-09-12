"""Pruebas del bucle de entrenamiento de la CNN-1D (sección 4.12.3-4.12.4).

Optimizador Adam con weight_decay (regularización L2, Sabo et al. 2023),
early stopping sobre un conjunto de validación interno, y reproducibilidad
mediante semilla explícita (10 semillas para la replicación estocástica,
sección 4.12.4).
"""
import numpy as np
import pytest
import torch

from src.models.cnn1d import CNN1DConfig, QuinuaYieldCNN1D
from src.models.cnn1d_trainer import TrainingConfig, train_cnn1d


def _datos_sinteticos(n_muestras: int, config: CNN1DConfig, semilla: int = 0):
    generador = torch.Generator().manual_seed(semilla)
    x_secuencial = torch.randn(
        n_muestras, config.n_variables, config.n_pasos_temporales, generator=generador
    )
    covariables = torch.randn(n_muestras, config.n_covariables, generator=generador)
    y = torch.randn(n_muestras, 1, generator=generador)
    return x_secuencial, covariables, y


class TestTrainCNN1D:
    def test_entrena_y_devuelve_un_modelo_con_pesos_actualizados(self):
        config = CNN1DConfig(n_variables=3, n_pasos_temporales=5, n_covariables=2)
        modelo = QuinuaYieldCNN1D(config)
        pesos_antes = [p.clone() for p in modelo.parameters()]

        x_seq, cov, y = _datos_sinteticos(30, config)
        resultado = train_cnn1d(
            modelo,
            x_secuencial=x_seq,
            covariables=cov,
            y=y,
            training_config=TrainingConfig(epocas_maximas=5, semilla=0),
        )

        pesos_despues = list(resultado.modelo.parameters())
        cambiaron = any(
            not torch.allclose(a, b) for a, b in zip(pesos_antes, pesos_despues)
        )
        assert cambiaron

    def test_reproducibilidad_con_la_misma_semilla(self):
        """Sección 4.12.4: las 10 semillas deben ser reproducibles — la misma
        semilla debe producir el mismo resultado de entrenamiento."""
        config = CNN1DConfig(n_variables=3, n_pasos_temporales=5, n_covariables=2)
        x_seq, cov, y = _datos_sinteticos(30, config, semilla=1)

        modelo_a = QuinuaYieldCNN1D(config)
        resultado_a = train_cnn1d(
            modelo_a, x_seq, cov, y, TrainingConfig(epocas_maximas=5, semilla=42)
        )

        modelo_b = QuinuaYieldCNN1D(config)
        resultado_b = train_cnn1d(
            modelo_b, x_seq, cov, y, TrainingConfig(epocas_maximas=5, semilla=42)
        )

        for p_a, p_b in zip(resultado_a.modelo.parameters(), resultado_b.modelo.parameters()):
            assert torch.allclose(p_a, p_b)

    def test_semillas_distintas_producen_resultados_distintos(self):
        config = CNN1DConfig(n_variables=3, n_pasos_temporales=5, n_covariables=2)
        x_seq, cov, y = _datos_sinteticos(30, config, semilla=1)

        modelo_a = QuinuaYieldCNN1D(config)
        resultado_a = train_cnn1d(
            modelo_a, x_seq, cov, y, TrainingConfig(epocas_maximas=5, semilla=1)
        )

        modelo_b = QuinuaYieldCNN1D(config)
        resultado_b = train_cnn1d(
            modelo_b, x_seq, cov, y, TrainingConfig(epocas_maximas=5, semilla=2)
        )

        diferentes = any(
            not torch.allclose(p_a, p_b)
            for p_a, p_b in zip(resultado_a.modelo.parameters(), resultado_b.modelo.parameters())
        )
        assert diferentes

    def test_registra_la_perdida_por_epoca(self):
        config = CNN1DConfig(n_variables=3, n_pasos_temporales=5, n_covariables=2)
        modelo = QuinuaYieldCNN1D(config)
        x_seq, cov, y = _datos_sinteticos(30, config)

        resultado = train_cnn1d(
            modelo, x_seq, cov, y, TrainingConfig(epocas_maximas=5, semilla=0)
        )

        assert len(resultado.historial_perdida) == 5
        assert all(np.isfinite(p) for p in resultado.historial_perdida)

    def test_early_stopping_detiene_antes_del_maximo_si_no_mejora(self):
        """No debe forzar `epocas_maximas` completas si la pérdida de
        validación deja de mejorar durante `paciencia` épocas consecutivas."""
        config = CNN1DConfig(n_variables=3, n_pasos_temporales=5, n_covariables=2)
        modelo = QuinuaYieldCNN1D(config)
        x_seq, cov, y = _datos_sinteticos(40, config)

        resultado = train_cnn1d(
            modelo,
            x_seq,
            cov,
            y,
            TrainingConfig(
                epocas_maximas=200,
                semilla=0,
                paciencia=3,
                fraccion_validacion=0.3,
            ),
        )

        assert len(resultado.historial_perdida) < 200

    def test_falla_si_fraccion_validacion_es_invalida(self):
        config = CNN1DConfig(n_variables=3, n_pasos_temporales=5, n_covariables=2)
        modelo = QuinuaYieldCNN1D(config)
        x_seq, cov, y = _datos_sinteticos(30, config)

        with pytest.raises(ValueError, match="fraccion_validacion"):
            train_cnn1d(
                modelo,
                x_seq,
                cov,
                y,
                TrainingConfig(epocas_maximas=5, semilla=0, fraccion_validacion=1.5),
            )
