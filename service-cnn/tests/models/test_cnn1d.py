"""Pruebas de la arquitectura CNN-1D (sección 2.3.2, Tabla 8; arquitectura de
referencia: Sabo et al. 2023, ver docs/02-arquitectura-tecnica.md §1.1).

Diseño parsimonioso: 2 capas convolucionales (kernel 2-3, 5-15 filtros),
batch normalization, global average pooling, dropout bajo, regularización L2
(vía weight_decay del optimizador, no aquí), como máximo 1 capa densa antes de
la salida. Las covariables estáticas (altitud) y dinámicas (proporción de
superficie con quinua) se concatenan DESPUÉS del bloque convolucional (sección
4.10.3), nunca como parte del tensor V×T de entrada.
"""
import torch

from src.models.cnn1d import CNN1DConfig, QuinuaYieldCNN1D


class TestCNN1DConfig:
    def test_valores_por_defecto_siguen_a_sabo_et_al(self):
        """Rango de filtros (5-15) y dropout (0.001-0.01) documentado como
        referencia arquitectónica principal en 02-arquitectura-tecnica.md."""
        config = CNN1DConfig(n_variables=4, n_pasos_temporales=5, n_covariables=2)
        assert 5 <= config.filtros_capa1 <= 15
        assert 5 <= config.filtros_capa2 <= 15
        assert 0.001 <= config.dropout <= 0.01
        assert config.kernel_size in (2, 3)


class TestQuinuaYieldCNN1DForward:
    def test_forma_de_salida_es_un_escalar_por_observacion(self):
        """La red predice un único valor (rendimiento en kg/ha o anomalía) por
        observación provincia-campaña."""
        config = CNN1DConfig(n_variables=4, n_pasos_temporales=5, n_covariables=2)
        modelo = QuinuaYieldCNN1D(config)

        batch_size = 8
        tensor_secuencial = torch.randn(batch_size, config.n_variables, config.n_pasos_temporales)
        covariables = torch.randn(batch_size, config.n_covariables)

        salida = modelo(tensor_secuencial, covariables)

        assert salida.shape == (batch_size, 1)

    def test_acepta_diferentes_longitudes_de_secuencia_temporal(self):
        """El uso de global average pooling permite que T varíe (ej. entre el
        horizonte principal de 30 días antes de cosecha y el horizonte
        temprano al cierre de floración, sección 4.12.1) sin cambiar la
        arquitectura ni el número de parámetros de las capas densas."""
        config = CNN1DConfig(n_variables=4, n_pasos_temporales=5, n_covariables=2)
        modelo = QuinuaYieldCNN1D(config)

        tensor_horizonte_temprano = torch.randn(4, config.n_variables, 3)
        covariables = torch.randn(4, config.n_covariables)

        salida = modelo(tensor_horizonte_temprano, covariables)
        assert salida.shape == (4, 1)

    def test_covariables_se_concatenan_despues_del_bloque_convolucional(self):
        """Verifica indirectamente el diseño de la sección 4.10.3: cambiar
        SOLO las covariables (con el tensor secuencial fijo) debe cambiar la
        salida — si las covariables no se usaran en absoluto, la salida sería
        idéntica en ambos casos."""
        config = CNN1DConfig(n_variables=4, n_pasos_temporales=5, n_covariables=2)
        modelo = QuinuaYieldCNN1D(config)
        modelo.eval()

        torch.manual_seed(0)
        tensor_secuencial = torch.randn(2, config.n_variables, config.n_pasos_temporales)
        covariables_a = torch.zeros(2, config.n_covariables)
        covariables_b = torch.ones(2, config.n_covariables) * 100.0

        with torch.no_grad():
            salida_a = modelo(tensor_secuencial, covariables_a)
            salida_b = modelo(tensor_secuencial, covariables_b)

        assert not torch.allclose(salida_a, salida_b)

    def test_numero_de_parametros_es_parsimonioso(self):
        """Sabo et al. (2023): del orden de 256 a 1800 parámetros totales para
        datasets de tamaño comparable (340-408 obs.) — un límite superior
        generoso aquí detecta si la arquitectura creció fuera del rango
        parsimonioso previsto para un dataset pequeño (~300-700 celdas)."""
        config = CNN1DConfig(n_variables=4, n_pasos_temporales=5, n_covariables=2)
        modelo = QuinuaYieldCNN1D(config)

        n_parametros = sum(p.numel() for p in modelo.parameters())
        assert n_parametros < 5000

    def test_usa_batch_normalization(self):
        config = CNN1DConfig(n_variables=4, n_pasos_temporales=5, n_covariables=2)
        modelo = QuinuaYieldCNN1D(config)
        tiene_batchnorm = any(
            isinstance(m, torch.nn.BatchNorm1d) for m in modelo.modules()
        )
        assert tiene_batchnorm

    def test_usa_global_average_pooling(self):
        config = CNN1DConfig(n_variables=4, n_pasos_temporales=5, n_covariables=2)
        modelo = QuinuaYieldCNN1D(config)
        tiene_gap = any(
            isinstance(m, torch.nn.AdaptiveAvgPool1d) for m in modelo.modules()
        )
        assert tiene_gap
