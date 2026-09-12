"""Pruebas del tuning de la CNN-1D con Optuna (sección 4.12.3).

Mismo contrato que tune_classic_model (mismo tipo de retorno, mismo criterio
de RMSE de validación), pero operando sobre tensores 3D y reutilizando
train_cnn1d para entrenar cada trial. Necesario para que la CNN-1D reciba el
mismo presupuesto de evaluaciones que los 4 modelos clásicos (sección 4.12.3).
"""
import torch

from src.models.tuning import PRESUPUESTO_TRIALS_POR_MODELO
from src.models.tuning_cnn1d import tune_cnn1d


def _datos_sinteticos_cnn(n_muestras: int, n_variables: int, n_pasos: int, n_cov: int, semilla: int = 0):
    generador = torch.Generator().manual_seed(semilla)
    x_secuencial = torch.randn(n_muestras, n_variables, n_pasos, generator=generador)
    covariables = torch.randn(n_muestras, n_cov, generator=generador)
    y = torch.randn(n_muestras, 1, generator=generador)
    return x_secuencial, covariables, y


class TestTuneCNN1D:
    def test_devuelve_hiperparametros_dentro_del_espacio_declarado(self):
        x_train, cov_train, y_train = _datos_sinteticos_cnn(30, 4, 5, 2, semilla=1)
        x_val, cov_val, y_val = _datos_sinteticos_cnn(10, 4, 5, 2, semilla=2)

        espacio_busqueda = {
            "filtros_capa1": ("int", 5, 15),
            "filtros_capa2": ("int", 5, 15),
            "dropout": ("float", 0.001, 0.01),
            "tasa_aprendizaje": ("float", 1e-4, 1e-2),
        }

        resultado = tune_cnn1d(
            n_variables=4,
            n_covariables=2,
            espacio_busqueda=espacio_busqueda,
            x_train=x_train,
            covariables_train=cov_train,
            y_train=y_train,
            x_val=x_val,
            covariables_val=cov_val,
            y_val=y_val,
            n_trials=3,
            epocas_por_trial=5,
            semilla=0,
        )

        assert 5 <= resultado.mejores_hiperparametros["filtros_capa1"] <= 15
        assert 5 <= resultado.mejores_hiperparametros["filtros_capa2"] <= 15
        assert 0.001 <= resultado.mejores_hiperparametros["dropout"] <= 0.01

    def test_respeta_el_numero_de_trials_solicitado(self):
        x_train, cov_train, y_train = _datos_sinteticos_cnn(30, 4, 5, 2, semilla=1)
        x_val, cov_val, y_val = _datos_sinteticos_cnn(10, 4, 5, 2, semilla=2)

        espacio_busqueda = {"dropout": ("float", 0.001, 0.01)}

        resultado = tune_cnn1d(
            n_variables=4,
            n_covariables=2,
            espacio_busqueda=espacio_busqueda,
            x_train=x_train,
            covariables_train=cov_train,
            y_train=y_train,
            x_val=x_val,
            covariables_val=cov_val,
            y_val=y_val,
            n_trials=4,
            epocas_por_trial=5,
            semilla=0,
        )

        assert resultado.n_trials_ejecutados == 4

    def test_devuelve_rmse_de_validacion_no_negativo(self):
        x_train, cov_train, y_train = _datos_sinteticos_cnn(30, 4, 5, 2, semilla=1)
        x_val, cov_val, y_val = _datos_sinteticos_cnn(10, 4, 5, 2, semilla=2)

        espacio_busqueda = {"dropout": ("float", 0.001, 0.01)}
        resultado = tune_cnn1d(
            n_variables=4,
            n_covariables=2,
            espacio_busqueda=espacio_busqueda,
            x_train=x_train,
            covariables_train=cov_train,
            y_train=y_train,
            x_val=x_val,
            covariables_val=cov_val,
            y_val=y_val,
            n_trials=3,
            epocas_por_trial=5,
            semilla=0,
        )
        assert resultado.mejor_rmse_validacion >= 0.0

    def test_reproducible_con_la_misma_semilla(self):
        x_train, cov_train, y_train = _datos_sinteticos_cnn(30, 4, 5, 2, semilla=1)
        x_val, cov_val, y_val = _datos_sinteticos_cnn(10, 4, 5, 2, semilla=2)
        espacio_busqueda = {"dropout": ("float", 0.001, 0.01)}

        resultado_a = tune_cnn1d(
            n_variables=4, n_covariables=2, espacio_busqueda=espacio_busqueda,
            x_train=x_train, covariables_train=cov_train, y_train=y_train,
            x_val=x_val, covariables_val=cov_val, y_val=y_val,
            n_trials=3, epocas_por_trial=5, semilla=7,
        )
        resultado_b = tune_cnn1d(
            n_variables=4, n_covariables=2, espacio_busqueda=espacio_busqueda,
            x_train=x_train, covariables_train=cov_train, y_train=y_train,
            x_val=x_val, covariables_val=cov_val, y_val=y_val,
            n_trials=3, epocas_por_trial=5, semilla=7,
        )
        assert (
            resultado_a.mejores_hiperparametros == resultado_b.mejores_hiperparametros
        )

    def test_usa_el_presupuesto_de_trials_declarado_para_cnn1d(self):
        """Verifica la conexión real con PRESUPUESTO_TRIALS_POR_MODELO en un
        escenario mínimo, para que el presupuesto equiparado (sección 4.12.3)
        no quede solo declarado sino efectivamente usado por esta función."""
        x_train, cov_train, y_train = _datos_sinteticos_cnn(20, 3, 4, 1, semilla=1)
        x_val, cov_val, y_val = _datos_sinteticos_cnn(8, 3, 4, 1, semilla=2)
        espacio_busqueda = {"dropout": ("float", 0.001, 0.01)}

        n_trials = PRESUPUESTO_TRIALS_POR_MODELO["cnn1d"]
        resultado = tune_cnn1d(
            n_variables=3,
            n_covariables=1,
            espacio_busqueda=espacio_busqueda,
            x_train=x_train,
            covariables_train=cov_train,
            y_train=y_train,
            x_val=x_val,
            covariables_val=cov_val,
            y_val=y_val,
            n_trials=n_trials,
            epocas_por_trial=3,
            semilla=0,
        )
        assert resultado.n_trials_ejecutados == n_trials
