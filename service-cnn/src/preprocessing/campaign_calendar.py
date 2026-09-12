"""Derivación de campaña agrícola desde año/mes calendario.

Referencia: docs/Proyecto_Tesis_Maestria_UNMSM_WMSR.docx, sección 4.4: una
campaña se identifica por su año de cosecha. El archivo real de MIDAGRI
(SISAGRI.xlsx, ver configs/column_mapping.yaml) reporta año y mes calendario,
no campaña directamente.

Convención adoptada (decisión metodológica documentada, sección 6.1): el año
agrícola va de julio a junio. Meses de julio a diciembre del año t
pertenecen a la campaña cuyo año de cosecha es t+1; meses de enero a junio
del año t pertenecen a la campaña cuyo año de cosecha es t. Coherente con el
ciclo de la quinua en la sierra altoandina: siembra septiembre-noviembre,
cosecha abril-junio del año siguiente (sección 4.10.2).
"""
from __future__ import annotations

_MES_INICIO_SEGUNDA_MITAD_AGRICOLA = 7


def derive_campana_from_month(anio: int, mes: int) -> int:
    """Deriva el año de cosecha (campana_id) a partir de año y mes calendario.

    Args:
        anio: año calendario (ej. 2025).
        mes: mes calendario, 1-12.

    Returns:
        El año de cosecha (campana_id, sección 4.4): `anio` si `mes` está
        entre enero y junio; `anio + 1` si `mes` está entre julio y diciembre.

    Raises:
        ValueError: si `mes` no está en el rango 1-12.
    """
    if not (1 <= mes <= 12):
        raise ValueError(f"mes debe estar entre 1 y 12; recibido: {mes}")

    if mes >= _MES_INICIO_SEGUNDA_MITAD_AGRICOLA:
        return anio + 1
    return anio
