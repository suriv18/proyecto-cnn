"""Arquitectura CNN-1D para predicción de rendimiento de quinua.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 2.3.2 (X1,
modelado temporal mediante CNN-1D) y Tabla 8; docs/02-arquitectura-tecnica.md
§1.1 (justificación de la arquitectura de referencia).

Arquitectura parsimoniosa siguiendo a Sabo et al. (2023) — precedente elegido
por tener un dataset del mismo orden de magnitud (340-408 obs.) que el
esperado en esta tesis (~300-700 celdas), a diferencia de Li et al. (2025) o
Cheema et al. (2026), diseñadas para datasets 10-100x más grandes: 2 capas
convolucionales (kernel 2-3, 5-15 filtros), batch normalization, global
average pooling, dropout bajo (0.001-0.01), inicialización He Normal, como
máximo 1 capa densa antes de la salida.

Las covariables estáticas (altitud media de la superficie agrícola) y
dinámicas (proporción de superficie sembrada con quinua) se concatenan
DESPUÉS del bloque convolucional, nunca como parte del tensor V×T de entrada
(sección 4.10.3) — decisión de diseño explícita de la tesis, no una elección
libre de esta implementación.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CNN1DConfig:
    """Hiperparámetros de la arquitectura, con valores por defecto dentro del
    rango parsimonioso de Sabo et al. (2023). El espacio de búsqueda real de
    Optuna (sección 4.12.3) se define en `src/models/tuning.py`, no aquí."""

    n_variables: int
    n_pasos_temporales: int
    n_covariables: int
    filtros_capa1: int = 8
    filtros_capa2: int = 12
    kernel_size: int = 3
    dropout: float = 0.005
    unidades_densa: int = 8


class QuinuaYieldCNN1D(nn.Module):
    """CNN-1D + covariables estáticas/dinámicas para predecir rendimiento o
    anomalía de quinua (una sola salida escalar por observación)."""

    def __init__(self, config: CNN1DConfig):
        super().__init__()
        self.config = config

        padding = config.kernel_size // 2

        self.bloque_convolucional = nn.Sequential(
            nn.Conv1d(
                in_channels=config.n_variables,
                out_channels=config.filtros_capa1,
                kernel_size=config.kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(config.filtros_capa1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(
                in_channels=config.filtros_capa1,
                out_channels=config.filtros_capa2,
                kernel_size=config.kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(config.filtros_capa2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        # Global average pooling: reduce el eje temporal a 1, permitiendo que
        # T (número de pasos disponibles hasta el horizonte) varíe entre
        # observaciones sin cambiar la forma de las capas densas siguientes.
        self.pool_global = nn.AdaptiveAvgPool1d(1)

        self.cabeza_densa = nn.Sequential(
            nn.Linear(config.filtros_capa2 + config.n_covariables, config.unidades_densa),
            nn.ReLU(),
            nn.Linear(config.unidades_densa, 1),
        )

        self._inicializar_pesos()

    def _inicializar_pesos(self) -> None:
        """Inicialización He Normal (Sabo et al., 2023), apropiada para
        activaciones ReLU."""
        for modulo in self.modules():
            if isinstance(modulo, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(modulo.weight, nonlinearity="relu")
                if modulo.bias is not None:
                    nn.init.zeros_(modulo.bias)

    def forward(
        self, tensor_secuencial: torch.Tensor, covariables: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tensor_secuencial: forma (batch, n_variables, T) — el tensor V×T
                de la sección 4.10.3, T puede variar entre horizontes.
            covariables: forma (batch, n_covariables) — altitud media
                (estática) y proporción de superficie con quinua (dinámica),
                ya concatenadas por el llamador en un único vector por
                observación.

        Returns:
            Tensor de forma (batch, 1): la predicción escalar (rendimiento en
            kg/ha o anomalía, según la representación de la variable objetivo
            en uso — sección 4.11).
        """
        activaciones = self.bloque_convolucional(tensor_secuencial)
        resumen_temporal = self.pool_global(activaciones).squeeze(-1)
        combinado = torch.cat([resumen_temporal, covariables], dim=1)
        return self.cabeza_densa(combinado)
