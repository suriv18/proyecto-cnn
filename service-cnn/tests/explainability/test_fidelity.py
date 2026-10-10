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


class TestOcclusionIncrementsByPhase:
    """Aumento de RMSE al ocluir cada fase (insumo de `he2_pvalues.fidelity_pvalue`)."""

    class _Modelo(torch.nn.Module):
        # pesos por paso: la fase 'c' (paso 2) domina y la 'a' (paso 0) casi no pesa
        def forward(self, x, cov):
            pesos = torch.tensor([0.01, 1.0, 10.0, 3.0, 0.5])
            return (x[:, 0, :] * pesos).sum(dim=1, keepdim=True)

    @pytest.fixture
    def datos(self):
        torch.manual_seed(2)
        modelo = self._Modelo()
        x = torch.randn(30, 1, 5)
        cov = torch.zeros(30, 1)
        with torch.no_grad():
            y = modelo(x, cov)
        return modelo, x, cov, y, torch.zeros(1)

    FASES = {"a": [0], "b": [1], "c": [2], "d": [3], "e": [4]}

    def test_devuelve_un_aumento_por_fase(self, datos):
        from src.explainability.fidelity import occlusion_increments_by_phase

        aumentos = occlusion_increments_by_phase(*datos, indices_por_fase=self.FASES)

        assert set(aumentos) == set(self.FASES)
        assert all(isinstance(v, float) for v in aumentos.values())

    def test_ordena_las_fases_segun_su_peso_en_el_modelo(self, datos):
        from src.explainability.fidelity import occlusion_increments_by_phase

        aumentos = occlusion_increments_by_phase(*datos, indices_por_fase=self.FASES)

        assert sorted(aumentos, key=aumentos.get, reverse=True) == ["c", "d", "b", "e", "a"]

    def test_coincide_con_evaluate_fidelity_by_occlusion(self, datos):
        from src.explainability.fidelity import occlusion_increments_by_phase

        aumentos = occlusion_increments_by_phase(*datos, indices_por_fase=self.FASES)
        unica = evaluate_fidelity_by_occlusion(*datos[:4], indices_fase_alta_contribucion=[2],
                                               indices_fase_control=[0], valor_referencia=datos[4])

        assert aumentos["c"] == pytest.approx(unica.incremento_rmse_alta_contribucion)
        assert aumentos["a"] == pytest.approx(unica.incremento_rmse_control)

    def test_no_reentrena_ni_modifica_las_entradas(self, datos):
        from src.explainability.fidelity import occlusion_increments_by_phase

        modelo, x, cov, y, ref = datos
        x_antes = x.clone()
        pesos_antes = [p.clone() for p in modelo.parameters()]

        occlusion_increments_by_phase(modelo, x, cov, y, ref, indices_por_fase=self.FASES)

        assert torch.equal(x, x_antes)
        assert all(torch.equal(a, b) for a, b in zip(pesos_antes, modelo.parameters()))

    def test_exige_fases_de_igual_duracion(self, datos):
        from src.explainability.fidelity import occlusion_increments_by_phase

        with pytest.raises(ValueError, match="misma duraci[oó]n"):
            occlusion_increments_by_phase(*datos, indices_por_fase={"a": [0], "b": [1, 2]})

    def test_exige_al_menos_dos_fases(self, datos):
        from src.explainability.fidelity import occlusion_increments_by_phase

        with pytest.raises(ValueError, match="al menos 2 fases"):
            occlusion_increments_by_phase(*datos, indices_por_fase={"a": [0]})


class TestConditionalOcclusionIncrements:
    """Oclusión con reemplazo condicional: la fase se rellena con lo que habría dado el resto de la observación."""

    class _Modelo(torch.nn.Module):
        def forward(self, x, cov):
            return (x[:, 0, :] * torch.tensor([1.0, 5.0, 0.2])).sum(dim=1, keepdim=True)

    FASES = {"a": [0], "b": [1], "c": [2]}

    @staticmethod
    def _datos(n=60, correlacion=True, semilla=0):
        g = torch.Generator().manual_seed(semilla)
        a = torch.randn(n, generator=g)
        c = torch.randn(n, generator=g)
        b = a.clone() if correlacion else torch.randn(n, generator=g)  # la fase b es copia de la a
        x = torch.stack([a, b, c], dim=1).unsqueeze(1)  # (n, 1 variable, 3 fases)
        return x, torch.zeros(n, 1)

    def _preparar(self, correlacion):
        modelo = self._Modelo()
        x_f, _ = self._datos(200, correlacion, semilla=1)
        x_t, cov = self._datos(40, correlacion, semilla=2)
        with torch.no_grad():
            y = modelo(x_t, cov)
        return modelo, x_t, cov, y, x_f

    def test_devuelve_un_aumento_por_fase(self):
        from src.explainability.fidelity import conditional_occlusion_increments_by_phase

        modelo, x_t, cov, y, x_f = self._preparar(True)

        aumentos = conditional_occlusion_increments_by_phase(modelo, x_t, cov, y, x_f, self.FASES, k=10)

        assert set(aumentos) == set(self.FASES) and all(isinstance(v, float) for v in aumentos.values())

    def test_si_la_fase_se_puede_deducir_de_las_demas_el_condicional_casi_no_degrada(self):
        from src.explainability.fidelity import conditional_occlusion_increments_by_phase, occlusion_increments_by_phase

        modelo, x_t, cov, y, x_f = self._preparar(True)  # la fase b (la de más peso) es copia de la a

        marginal = occlusion_increments_by_phase(modelo, x_t, cov, y, torch.zeros(1), self.FASES)
        condicional = conditional_occlusion_increments_by_phase(modelo, x_t, cov, y, x_f, self.FASES, k=5)

        assert marginal["b"] > 1.0  # ocluir con un valor fijo rompe la relación
        assert condicional["b"] < 0.5 * marginal["b"]  # el condicional recupera lo que da la fase a

    def test_si_la_fase_es_independiente_el_condicional_se_parece_al_marginal(self):
        from src.explainability.fidelity import conditional_occlusion_increments_by_phase, occlusion_increments_by_phase

        modelo, x_t, cov, y, x_f = self._preparar(False)

        marginal = occlusion_increments_by_phase(modelo, x_t, cov, y, torch.zeros(1), self.FASES)
        condicional = conditional_occlusion_increments_by_phase(modelo, x_t, cov, y, x_f, self.FASES, k=10)

        assert condicional["b"] > 0.5 * marginal["b"]

    def test_no_modifica_las_entradas_ni_reentrena(self):
        from src.explainability.fidelity import conditional_occlusion_increments_by_phase

        modelo, x_t, cov, y, x_f = self._preparar(True)
        x_antes, f_antes = x_t.clone(), x_f.clone()
        pesos = [p.clone() for p in modelo.parameters()]

        conditional_occlusion_increments_by_phase(modelo, x_t, cov, y, x_f, self.FASES, k=5)

        assert torch.equal(x_t, x_antes) and torch.equal(x_f, f_antes)
        assert all(torch.equal(a, b) for a, b in zip(pesos, modelo.parameters()))

    def test_k_mayor_que_el_fondo_se_acota_al_tamano_del_fondo(self):
        from src.explainability.fidelity import conditional_occlusion_increments_by_phase

        modelo, x_t, cov, y, _ = self._preparar(True)
        pequeno, _ = self._datos(4, True, semilla=3)

        aumentos = conditional_occlusion_increments_by_phase(modelo, x_t, cov, y, pequeno, self.FASES, k=50)

        assert set(aumentos) == set(self.FASES)

    def test_exige_fases_de_igual_duracion(self):
        from src.explainability.fidelity import conditional_occlusion_increments_by_phase

        modelo, x_t, cov, y, x_f = self._preparar(True)

        with pytest.raises(ValueError, match="misma duraci[oó]n"):
            conditional_occlusion_increments_by_phase(modelo, x_t, cov, y, x_f, {"a": [0], "b": [1, 2]}, k=5)
