"""Orquestador de extracción real de series GEE por campaña (Actividad 3).

Referencia: docs/02-arquitectura-tecnica.md §2; docs/Proyecto_Tesis_Maestria_
UNMSM_WMSR.docx sección 4.9 (Tabla 7), 4.10.2 (alineación fenológica).

Conecta `GeeClient.extract_daily_series` (ya con reducer real y factores de
escala/offset aplicados) con las geometrías de `province_geometries.py` para
extraer, por cada campaña y cada variable de la Tabla 7, una ventana AMPLIA
de datos crudos: el año calendario completo del ciclo agrícola de la quinua
en la sierra (siembra oct-dic del año t-1, cosecha abr-jun del año t — sección
4.4). La alineación fenológica fina a fases BBCH (emergencia, floración, etc.)
ocurre después en `preprocessing/phenology.py`, que consume esta serie cruda
(incluyendo NDVI, usado como respaldo de nivel 2 para estimar la fecha de
siembra cuando no hay superficie sembrada mensual reportada).

No importa `ee` a nivel de módulo: recibe un `GeeClient` ya construido e
inicializado (real o doble de prueba), mismo patrón que el resto de
`ingestion/`.
"""
from __future__ import annotations

import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Optional

import pandas as pd


def dividir_en_meses(fecha_inicio: date, fecha_fin: date) -> list[tuple[date, date]]:
    """Divide un rango de fechas en trozos de un mes calendario cada uno.

    Necesario porque `ee.FeatureCollection.getInfo()` aborta con "Collection
    query aborted after accumulating over 5000 elements" cuando una sola
    llamada junta demasiadas filas — verificado contra la API real
    (proyecto cnn-sentinel): 80 provincias x 365 días diarios = 29 200
    elementos excede el límite; 80 x 31 = 2480 por mes queda cómodamente
    debajo.

    Raises:
        ValueError: si `fecha_inicio` es posterior a `fecha_fin`.
    """
    if fecha_inicio > fecha_fin:
        raise ValueError(
            f"fecha_inicio ({fecha_inicio}) no puede ser posterior a "
            f"fecha_fin ({fecha_fin})."
        )

    trozos: list[tuple[date, date]] = []
    cursor = fecha_inicio
    while cursor <= fecha_fin:
        ultimo_dia_mes = calendar.monthrange(cursor.year, cursor.month)[1]
        fin_mes = date(cursor.year, cursor.month, ultimo_dia_mes)
        fin_trozo = min(fin_mes, fecha_fin)
        trozos.append((cursor, fin_trozo))
        cursor = fin_trozo + timedelta(days=1)

    return trozos


def campana_a_ventana_extraccion(campana_id: int) -> tuple[date, date]:
    """Traduce una campaña (año de cosecha) a su ventana de extracción cruda.

    Se usa el año calendario completo del ciclo agrícola (1 jul del año
    anterior a 30 jun del año de cosecha) en vez de fechas de siembra
    exactas: estas últimas se estiman en `phenology.estimate_sowing_date`,
    que a su vez puede necesitar la propia serie NDVI extraída aquí (nivel 2
    de la jerarquía, sección 4.10.2) — extraer una ventana más angosta de
    antemano crearía una dependencia circular.
    """
    return date(campana_id - 1, 7, 1), date(campana_id, 6, 30)


def extract_all_provinces_all_variables(
    cliente: Any,
    geometrias_por_provincia: Any,
    variables: list[str],
    campanas: list[int],
    devolver_fallos: bool = False,
    max_workers: int = 1,
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]]:
    """Extrae todas las variables para todas las campañas, en una sola tabla.

    Args:
        cliente: `GeeClient` ya inicializado (real o doble de prueba), con un
            método `extract_daily_series(variable, geometrias_por_provincia,
            fecha_inicio, fecha_fin)`.
        geometrias_por_provincia: `ee.FeatureCollection` con las geometrías
            de las provincias a extraer (ver `province_geometries.py`).
        variables: claves de `gee_config` a extraer (ej. "precipitacion",
            "ndvi").
        campanas: años de cosecha a extraer (ej. range(2016, 2026)).
        devolver_fallos: si es True, en vez de propagar la primera excepción,
            continúa con las demás combinaciones variable×campaña×mes y
            devuelve `(resultado, fallos)` — una extracción de 10 campañas x
            5 variables x 12 meses tarda minutos contra la API real, y un
            fallo aislado (ej. cuota excedida en un solo mes) no debe
            descartar el resto del trabajo ya completado.
        max_workers: número de llamadas a `cliente.extract_daily_series` en
            vuelo simultáneamente. El cuello de botella real es espera de
            red hacia la API de Earth Engine (~78s por variable-mes con las
            80 provincias, verificado en producción), no cómputo local, así
            que un `ThreadPoolExecutor` reduce el tiempo total casi
            linealmente. 1 (por defecto) preserva el comportamiento
            estrictamente secuencial.

    Returns:
        Si `devolver_fallos` es False: DataFrame con columnas `provincia_id`,
        `fecha`, `valor`, `variable`, `campana_id` — concatenación de todas
        las combinaciones variable×campaña.
        Si `devolver_fallos` es True: tupla `(DataFrame, fallos)`, donde
        `fallos` es una lista de dicts `{variable, campana_id, fecha_inicio,
        fecha_fin, error}` por cada mes que lanzó una excepción.

    Nota de implementación: cada campaña se pagina internamente en meses
    (`dividir_en_meses`) antes de llamar a `cliente.extract_daily_series` —
    ver el docstring de esa función para el límite real de Earth Engine que
    motiva la paginación.
    """
    tareas: list[tuple[str, int, date, date]] = []
    for campana_id in campanas:
        fecha_inicio_campana, fecha_fin_campana = campana_a_ventana_extraccion(campana_id)
        meses = dividir_en_meses(fecha_inicio_campana, fecha_fin_campana)
        for variable in variables:
            for fecha_inicio, fecha_fin in meses:
                tareas.append((variable, campana_id, fecha_inicio, fecha_fin))

    def _ejecutar_tarea(tarea: tuple[str, int, date, date]) -> tuple[tuple, Any]:
        variable, campana_id, fecha_inicio, fecha_fin = tarea
        resultado_o_excepcion: Any
        try:
            resultado_o_excepcion = cliente.extract_daily_series(
                variable=variable,
                geometrias_por_provincia=geometrias_por_provincia,
                fecha_inicio=fecha_inicio,
                fecha_fin=fecha_fin,
            )
        except Exception as exc:  # noqa: BLE001 — se reporta, no se enmascara
            resultado_o_excepcion = exc
        return tarea, resultado_o_excepcion

    tablas: list[pd.DataFrame] = []
    fallos: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = [executor.submit(_ejecutar_tarea, tarea) for tarea in tareas]
        for futuro in as_completed(futuros):
            (variable, campana_id, fecha_inicio, fecha_fin), resultado = futuro.result()
            if isinstance(resultado, Exception):
                if not devolver_fallos:
                    raise resultado
                fallos.append(
                    {
                        "variable": variable,
                        "campana_id": campana_id,
                        "fecha_inicio": fecha_inicio.isoformat(),
                        "fecha_fin": fecha_fin.isoformat(),
                        "error": str(resultado),
                    }
                )
                continue

            tabla = resultado.copy()
            tabla["variable"] = variable
            tabla["campana_id"] = campana_id
            tablas.append(tabla)

    columnas = ["provincia_id", "fecha", "valor", "variable", "campana_id"]
    resultado_final = (
        pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame(columns=columnas)
    )

    if devolver_fallos:
        return resultado_final, fallos
    return resultado_final
