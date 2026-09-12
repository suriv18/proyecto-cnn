"""Pruebas de la derivación de campaña agrícola desde año/mes calendario.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.4: una
campaña se identifica por su año de cosecha; el ciclo abarca el periodo que
inicia en el año t y finaliza con la cosecha en el año t+1. El archivo real
de MIDAGRI (SISAGRI.xlsx, ver configs/column_mapping.yaml) reporta año y mes
CALENDARIO, no campaña directamente — esta conversión es necesaria antes de
poder usar los datos en el pipeline (que opera sobre `campana_id`).

Convención adoptada (documentada explícitamente, sección 6.1: toda decisión
metodológica no trivial se registra): el año agrícola va de julio a junio.
Meses de julio a diciembre del año t pertenecen a la campaña cuyo año de
cosecha es t+1 (siembra reciente, cosecha esperada el año siguiente); meses
de enero a junio del año t pertenecen a la campaña cuyo año de cosecha es t
(cosecha del ciclo iniciado el año anterior). Esta convención es coherente
con el ciclo de la quinua en la sierra altoandina: siembra
septiembre-noviembre, cosecha abril-junio del año siguiente (sección 4.10.2).
"""
import pytest

from src.preprocessing.campaign_calendar import derive_campana_from_month


class TestDeriveCampanaFromMonth:
    def test_mes_de_julio_a_diciembre_pertenece_a_la_campana_del_anio_siguiente(self):
        """Julio-diciembre: siembra reciente, la cosecha se espera el año
        siguiente."""
        assert derive_campana_from_month(anio=2025, mes=7) == 2026
        assert derive_campana_from_month(anio=2025, mes=9) == 2026
        assert derive_campana_from_month(anio=2025, mes=12) == 2026

    def test_mes_de_enero_a_junio_pertenece_a_la_campana_del_mismo_anio(self):
        """Enero-junio: cosecha del ciclo iniciado el año anterior."""
        assert derive_campana_from_month(anio=2026, mes=1) == 2026
        assert derive_campana_from_month(anio=2026, mes=4) == 2026
        assert derive_campana_from_month(anio=2026, mes=6) == 2026

    def test_frontera_junio_diciembre_es_correcta(self):
        """El punto de quiebre exacto entre ambas mitades del año."""
        assert derive_campana_from_month(anio=2025, mes=6) == 2025
        assert derive_campana_from_month(anio=2025, mes=7) == 2026

    def test_ejemplo_completo_del_registro_real_majes_diciembre_2025(self):
        """Fila real observada en SISAGRI.xlsx: AÑO=2025, MES=12 (diciembre),
        QUINUA en Majes — debe caer en la campaña 2025/2026 (cosecha 2026)."""
        assert derive_campana_from_month(anio=2025, mes=12) == 2026

    def test_ejemplo_completo_del_registro_real_majes_enero_2026(self):
        """Fila real observada: AÑO=2026, MES=01 (enero) — misma campaña
        2025/2026 (cosecha 2026) que la fila de diciembre 2025 anterior,
        ambas parte del mismo ciclo productivo continuo en Majes."""
        assert derive_campana_from_month(anio=2026, mes=1) == 2026

    def test_falla_con_mes_fuera_de_rango(self):
        with pytest.raises(ValueError, match="mes"):
            derive_campana_from_month(anio=2025, mes=13)
        with pytest.raises(ValueError, match="mes"):
            derive_campana_from_month(anio=2025, mes=0)
