"""Preespecificación (E5): lectura, coherencia con el código y puerta de congelamiento.

La preespecificación es el registro fechado de hipótesis, márgenes, comparador,
umbrales de HE2, horizontes, espacios de búsqueda y reglas de depuración que se
fija ANTES de la evaluación definitiva (sección 4.18 de la tesis; hito H2).
`configs/preespecificacion.yaml` es la fuente única de verdad; este módulo:

  - la lee y lista lo que falta decidir (`pending_decisions`);
  - verifica que coincida con las constantes del código (`check_consistency_with_code`),
    de modo que el registro no pueda divergir en silencio de lo que se ejecuta;
  - impide congelarla mientras quede algo sin decidir (`assert_can_freeze`);
  - genera el texto legible (`render_markdown`).

Cada parámetro es `{valor, estado, fuente}`; ver los estados en el YAML.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import yaml

ESTADOS_PENDIENTES = ("por_decidir_asesor", "por_decidir_piloto", "propuesto")
_SECCIONES_SIN_PARAMETROS = {"metadatos"}


class PreregistrationNotReadyError(ValueError):
    """La preespecificación aún tiene decisiones pendientes o valores sin definir."""


def load_preregistration(ruta: str | Path) -> dict:
    return yaml.safe_load(Path(ruta).read_text(encoding="utf-8"))


def _es_parametro(nodo: Any) -> bool:
    return isinstance(nodo, dict) and "estado" in nodo and "valor" in nodo


def iter_parameters(prereg: dict) -> Iterator[tuple[str, dict]]:
    """Recorre los parámetros `{valor, estado, fuente}` con su ruta (`seccion.sub.nombre`)."""

    def recorrer(nodo: Any, ruta: str) -> Iterator[tuple[str, dict]]:
        if _es_parametro(nodo):
            yield ruta, nodo
        elif isinstance(nodo, dict):
            for clave, hijo in nodo.items():
                yield from recorrer(hijo, f"{ruta}.{clave}" if ruta else clave)

    for seccion, contenido in prereg.items():
        if seccion not in _SECCIONES_SIN_PARAMETROS:
            yield from recorrer(contenido, seccion)


def pending_decisions(prereg: dict) -> dict[str, list[str]]:
    """Rutas de los parámetros por estado pendiente (`ESTADOS_PENDIENTES`)."""
    resultado: dict[str, list[str]] = {estado: [] for estado in ESTADOS_PENDIENTES}
    for ruta, parametro in iter_parameters(prereg):
        if parametro["estado"] in resultado:
            resultado[parametro["estado"]].append(ruta)
    return resultado


def assert_can_freeze(prereg: dict) -> None:
    """Puerta del hito H2: no se congela con decisiones pendientes ni valores sin definir.

    Raises:
        PreregistrationNotReadyError: indicando cuántos parámetros siguen en cada
            estado pendiente y cuáles tienen `valor: null`.
    """
    problemas = []
    for estado, rutas in pending_decisions(prereg).items():
        if rutas:
            problemas.append(f"{len(rutas)} en {estado}")
    sin_valor = [ruta for ruta, p in iter_parameters(prereg) if p["valor"] is None]
    if sin_valor:
        problemas.append(f"{len(sin_valor)} sin valor ({', '.join(sin_valor[:3])}…)")
    if problemas:
        raise PreregistrationNotReadyError(
            "La preespecificación no puede congelarse: " + "; ".join(problemas) + "."
        )


def _valor(prereg: dict, *camino: str) -> Any:
    nodo = prereg
    for clave in camino:
        nodo = nodo[clave]
    return nodo["valor"]


def check_consistency_with_code(prereg: dict) -> list[str]:
    """Compara el registro con las constantes del código y devuelve las discrepancias.

    Una lista vacía significa que lo que se registró es lo que el código ejecuta.
    """
    from src.explainability import agronomic_concordance as concordancia
    from src.explainability import stability
    from src.models.tuning import PRESUPUESTO_TRIALS_POR_MODELO
    from src.features.target_transform import HISTORIA_MINIMA_CAMPANAS
    from src.preprocessing.preprocessing_pipeline import CLASIFICACIONES_EXCLUIDAS_POR_DEFECTO

    discrepancias: list[str] = []
    he2 = ("hipotesis", "he2")

    w = _valor(prereg, *he2, "estabilidad_w_kendall_minimo")
    if w != stability.UMBRAL_W_KENDALL:
        discrepancias.append(f"umbral W de Kendall: registro {w} frente a código {stability.UMBRAL_W_KENDALL}")
    j = _valor(prereg, *he2, "estabilidad_jaccard_minimo")
    if j != stability.UMBRAL_JACCARD:
        discrepancias.append(f"umbral de Jaccard: registro {j} frente a código {stability.UMBRAL_JACCARD}")

    principal = set(_valor(prereg, *he2, "fases_criticas_horizonte_principal"))
    if principal != set(concordancia.FASES_CRITICAS_PREESPECIFICADAS):
        discrepancias.append(
            f"fases críticas (principal): registro {sorted(principal)} frente a código "
            f"{sorted(concordancia.FASES_CRITICAS_PREESPECIFICADAS)}"
        )
    temprano = set(_valor(prereg, *he2, "fases_criticas_horizonte_temprano"))
    if temprano != set(concordancia._FASES_VALIDAS_POR_HORIZONTE["temprano"]):
        discrepancias.append(
            f"fases críticas (temprano): registro {sorted(temprano)} frente a código "
            f"{sorted(concordancia._FASES_VALIDAS_POR_HORIZONTE['temprano'])}"
        )

    factor = set(_valor(prereg, "modelos", "factor_algoritmo"))
    presupuesto = _valor(prereg, "modelos", "presupuesto_de_trials_por_modelo")
    del_codigo = set(PRESUPUESTO_TRIALS_POR_MODELO.values())
    if len(del_codigo) != 1 or presupuesto not in del_codigo:
        discrepancias.append(
            f"presupuesto de trials: registro {presupuesto} frente a código {dict(PRESUPUESTO_TRIALS_POR_MODELO)} "
            "(debe ser el mismo número para los 5 modelos, 4.12.3)"
        )
    if set(PRESUPUESTO_TRIALS_POR_MODELO) != factor:
        discrepancias.append(
            f"modelos del presupuesto {sorted(PRESUPUESTO_TRIALS_POR_MODELO)} distintos del factor {sorted(factor)}"
        )

    excluidas = set(_valor(prereg, "datos", "clasificaciones_excluidas"))
    if excluidas != set(CLASIFICACIONES_EXCLUIDAS_POR_DEFECTO):
        discrepancias.append(
            f"clasificaciones excluidas: registro {sorted(excluidas)} frente a código "
            f"{sorted(CLASIFICACIONES_EXCLUIDAS_POR_DEFECTO)}"
        )

    historia = _valor(prereg, "variable_objetivo", "historia_minima_para_tendencia_campanas")
    if historia != HISTORIA_MINIMA_CAMPANAS:
        discrepancias.append(
            f"historia mínima para la tendencia de B2: registro {historia} frente a código {HISTORIA_MINIMA_CAMPANAS}"
        )

    semillas = ("validacion", "semillas")
    estocasticos = set(_valor(prereg, *semillas, "modelos_estocasticos"))
    deterministas = set(_valor(prereg, *semillas, "modelos_deterministas"))
    if estocasticos | deterministas != factor or estocasticos & deterministas:
        discrepancias.append(
            "modelos estocásticos y deterministas deben repartirse sin solaparse todo el factor "
            f"({sorted(factor)}); registro {sorted(estocasticos)} + {sorted(deterministas)}"
        )
    n = _valor(prereg, *semillas, "n_replicas")
    valores = _valor(prereg, *semillas, "valores")
    if n != len(valores) or len(set(valores)) != len(valores):
        discrepancias.append(f"semillas: n_replicas={n} no coincide con {len(valores)} valores distintos")

    return discrepancias


def render_markdown(prereg: dict) -> str:
    """Texto legible de la preespecificación, generado desde el YAML (sin transcripción manual)."""
    meta = prereg["metadatos"]
    pendientes = pending_decisions(prereg)
    n_total = sum(1 for _ in iter_parameters(prereg))
    lineas = [
        "# E5 · Preespecificación (registro fechado)",
        "",
        f"**Estado:** {meta['estado_documento']} · **Versión:** {meta['version']} · **Borrador:** {meta['fecha_borrador']} · "
        f"**Congelado:** {meta['fecha_congelamiento'] or 'no'}",
        "",
        "> Generado desde `configs/preespecificacion.yaml` (fuente única de verdad) con `python scripts/render_e5.py`; "
        "no se edita a mano. Fase F3 de CRISP-ML(Q), hito H2 (sección 4.18 de la tesis).",
        "",
        f"Contiene **{n_total} parámetros**. Solo puede congelarse cuando no quede ninguno `propuesto` ni `por_decidir_*`; "
        "después, todo cambio exige una enmienda fechada y trazable.",
        "",
        "## Decisiones pendientes",
        "",
        "| Estado | Parámetros |",
        "|---|---|",
    ]
    for estado in ESTADOS_PENDIENTES:
        lineas.append(f"| `{estado}` | {len(pendientes[estado])} |")
    for estado, titulo in (
        ("por_decidir_asesor", "Requieren al asesor"),
        ("por_decidir_piloto", "Las decide el ensayo piloto (sin campañas externas)"),
        ("propuesto", "Propuestos, a revisar y fijar"),
    ):
        lineas += ["", f"### {titulo}", ""]
        for ruta in pendientes[estado]:
            parametro = dict(iter_parameters(prereg))[ruta]
            lineas.append(f"- `{ruta}` — {_texto_valor(parametro['valor'])} · _{parametro['fuente']}_")
    lineas += ["", "## Registro completo"]
    seccion_actual = None
    for ruta, parametro in iter_parameters(prereg):
        seccion = ruta.split(".")[0]
        if seccion != seccion_actual:
            seccion_actual = seccion
            lineas += ["", f"### {seccion}", "", "| Parámetro | Valor | Estado | Fuente |", "|---|---|---|---|"]
        nombre = ruta.split(".", 1)[1]
        lineas.append(f"| `{nombre}` | {_texto_valor(parametro['valor'])} | `{parametro['estado']}` | {parametro['fuente']} |")
    lineas += ["", f"**Política de desviaciones:** {_valor(prereg, 'politica_de_desviaciones', 'regla')}", ""]
    return "\n".join(lineas)


def _texto_valor(valor: Any) -> str:
    if valor is None:
        return "_sin definir_"
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    if isinstance(valor, (list, tuple)):
        return ", ".join(str(v) for v in valor)
    return str(valor).replace("|", "\\|")
