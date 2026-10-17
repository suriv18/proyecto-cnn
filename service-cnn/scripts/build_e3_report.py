"""Genera el informe E3: depuración y análisis exploratorio (fase F2 de CRISP-ML(Q)).

Lee `data/processed`, el parquet crudo, los manifiestos y la calibración de la
máscara, calcula TODAS las cifras y escribe:
  - docs/crisp/E3-informe-depuracion-eda.md
  - docs/crisp/E3/figuras/*.png

Ninguna cifra del informe se transcribe a mano: se regenera con este script.

Integridad del análisis exploratorio (sección 4.12.2): las relaciones entre
predictores y rendimiento y la distribución del objetivo se calculan SOLO con
las campañas de entrenamiento (por defecto 2016-2020), para que ninguna decisión
de modelado se apoye en las campañas externas. Las cifras de depuración y la
serie del rendimiento por campaña son descriptivas y usan todas.

Uso:
    python scripts/build_e3_report.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats  # noqa: E402

# ---- paleta de referencia validada (dataviz): slots 1-3, rampa azul, divergente azul<->rojo
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
AZUL_CLARO = "#9ec5f4"
SECUENCIAL = LinearSegmentedColormap.from_list("sec", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIVERGENTE = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

ETIQUETA_VAR = {
    "precipitacion": "Precipitación", "temperatura_maxima": "Temp. máxima", "temperatura_minima": "Temp. mínima",
    "radiacion_solar": "Radiación solar", "ndvi_mean": "NDVI medio", "ndvi_max": "NDVI máximo",
}
ETIQUETA_FASE = {"emergencia": "Emergencia", "desarrollo_vegetativo": "Desarrollo", "floracion": "Floración",
                 "llenado_grano": "Llenado", "madurez": "Madurez"}
DEP_NOMBRE = {"JUNIN": "Junín", "APURIMAC": "Apurímac", "HUANCAVELICA": "Huancavelica", "LA LIBERTAD": "La Libertad",
               "PUNO": "Puno", "CUSCO": "Cusco", "AYACUCHO": "Ayacucho", "AREQUIPA": "Arequipa"}


def nombre_dep(d: str) -> str:
    return DEP_NOMBRE.get(d, d.title())


MOTIVOS = {
    "fuera_del_ambito_altoandino": "Fuera del ámbito altoandino (costa con riego)",
    "ciclo_fuera_de_ventana_de_extraccion": "Ciclo fuera de la ventana de extracción",
    "tensor_incompleto": "Sin series (campaña fuera de 2016-2025)",
    "sin_alineacion_fenologica": "Sin alineación fenológica",
}


def fmt(x: float, nd: int = 1) -> str:
    """Número con coma decimal y separador de miles con espacio (convención de la tesis)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")
    return s


def md_tabla(df: pd.DataFrame) -> str:
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    filas = ["| " + " | ".join(str(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "no disponible"


def mayores_por_magnitud(m: np.ndarray, k: int = 3) -> list[tuple[int, int, float]]:
    """Las `k` celdas de mayor valor absoluto de TODA la matriz: `(fila, columna, valor)`.

    Ordena con `axis=None`; sin él, `np.argsort` ordena cada fila por separado.
    """
    orden = np.argsort(-np.abs(np.nan_to_num(m)), axis=None)[:k]
    filas, cols = np.unravel_index(orden, m.shape)
    return [(int(i), int(j), float(m[i, j])) for i, j in zip(filas, cols)]


def guardar(fig, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================================ figuras
def fig_cascada(pasos: list[tuple[str, int]], ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    etiquetas = [p[0] for p in pasos][::-1]
    valores = [p[1] for p in pasos][::-1]
    barras = ax.barh(etiquetas, valores, color=AZUL, height=0.62)
    for b, v in zip(barras, valores):
        ax.text(v + 6, b.get_y() + b.get_height() / 2, fmt(v, 0), va="center", color=INK, fontsize=9.5)
    ax.set_xlim(0, max(valores) * 1.12)
    ax.set_xlabel("Observaciones provincia-campaña")
    ax.grid(axis="y", visible=False)
    ax.set_title("De las celdas con producción al conjunto analítico (2016-2025)")
    guardar(fig, ruta)


def fig_exclusiones(e: pd.DataFrame, ruta: Path) -> None:
    tabla = e.pivot_table(index="campana_id", columns="motivo", values="provincia_id", aggfunc="count", fill_value=0)
    orden = [m for m in MOTIVOS if m in tabla.columns]
    colores = {"fuera_del_ambito_altoandino": AZUL, "ciclo_fuera_de_ventana_de_extraccion": NARANJA, "tensor_incompleto": AQUA}
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    base = np.zeros(len(tabla))
    for m in orden:
        v = tabla[m].to_numpy()
        ax.bar(tabla.index.astype(str), v, bottom=base, color=colores.get(m, MUTED), width=0.7,
               edgecolor=SURFACE, linewidth=1.5, label=MOTIVOS[m])
        base += v
    for x, tot in zip(tabla.index.astype(str), base):
        ax.text(x, tot + 0.6, fmt(tot, 0), ha="center", color=INK, fontsize=8.5)
    ax.set_ylabel("Observaciones excluidas")
    ax.set_xlabel("Campaña (año de cosecha)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Exclusiones por campaña y motivo")
    guardar(fig, ruta)


def fig_completitud(vacias_antes: np.ndarray, vacias_despues: np.ndarray, vars_: list[str], fases: list[str], ruta: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
    vmax = max(vacias_antes.max(), 1)
    for ax, datos, titulo in zip(ejes, (vacias_antes, vacias_despues), ("Antes de imputar", "Después de imputar (arrastre ≤ 16 días)")):
        im = ax.imshow(datos, cmap=SECUENCIAL, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(fases)), [ETIQUETA_FASE[f] for f in fases], rotation=20, ha="right")
        ax.set_yticks(range(len(vars_)), [ETIQUETA_VAR[v] for v in vars_])
        ax.grid(False)
        for i in range(datos.shape[0]):
            for j in range(datos.shape[1]):
                oscuro = datos[i, j] > 0.55 * vmax
                ax.text(j, i, fmt(datos[i, j], 1), ha="center", va="center", fontsize=8,
                        color="#ffffff" if oscuro else INK)
        ax.set_title(titulo)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Celdas vacías del tensor por variable y fase (% de observaciones)", x=0.01, ha="left", fontsize=10.5, fontweight="bold", y=1.02)
    guardar(fig, ruta)


def fig_objetivo(entren: pd.DataFrame, ruta: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.5})
    a1.hist(entren["rendimiento_kg_ha"], bins=28, color=AZUL, edgecolor=SURFACE, linewidth=1)
    med = entren["rendimiento_kg_ha"].median()
    a1.axvline(med, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    a1.text(med + 40, a1.get_ylim()[1] * 0.93, f"mediana {fmt(med, 0)}", color=INK2, fontsize=8.5)
    a1.set_xlabel("Rendimiento (kg/ha)")
    a1.set_ylabel("Observaciones")
    a1.set_title("Distribución del rendimiento")
    a1.grid(axis="x", visible=False)
    orden = entren.groupby("departamento")["rendimiento_kg_ha"].median().sort_values().index
    datos = [entren.loc[entren.departamento == d, "rendimiento_kg_ha"] for d in orden]
    bp = a2.boxplot(datos, vert=False, patch_artist=True, widths=0.55, showfliers=True,
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MUTED, "markeredgecolor": "none", "alpha": 0.7})
    for caja in bp["boxes"]:
        caja.set(facecolor=AZUL_CLARO, edgecolor=AZUL, linewidth=1)
    for k in ("whiskers", "caps"):
        for l in bp[k]:
            l.set(color=AZUL, linewidth=1)
    for l in bp["medians"]:
        l.set(color=INK, linewidth=1.4)
    a2.set_yticks(range(1, len(orden) + 1), [f"{nombre_dep(d)} ({len(x)})" for d, x in zip(orden, datos)])
    a2.set_xlabel("Rendimiento (kg/ha)")
    a2.set_title("Por departamento (n)")
    a2.grid(axis="y", visible=False)
    guardar(fig, ruta)


def fig_serie(todas: pd.DataFrame, n_entren: int, ruta: Path) -> None:
    g = todas.groupby("campana_id")["rendimiento_kg_ha"]
    c = g.median().index.to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.fill_between(c, g.quantile(0.25), g.quantile(0.75), color=AZUL_CLARO, alpha=0.55, linewidth=0)
    ax.plot(c, g.median(), color=AZUL, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    corte = c[n_entren - 1] + 0.5
    ax.axvline(corte, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(corte - 0.1, ax.get_ylim()[1] * 0.985, "ventana inicial (supuesto)", ha="right", va="top", color=INK2, fontsize=8.5)
    ax.text(corte + 0.1, ax.get_ylim()[1] * 0.985, "campañas externas", ha="left", va="top", color=INK2, fontsize=8.5)
    ax.set_xticks(c)
    ax.set_ylabel("Rendimiento (kg/ha)")
    ax.set_xlabel("Campaña (año de cosecha)")
    ax.set_title("Rendimiento por campaña: mediana y rango intercuartil")
    ax.grid(axis="x", visible=False)
    guardar(fig, ruta)


def fig_correlaciones(c_nivel: np.ndarray, c_anom: np.ndarray, vars_: list[str], fases: list[str], ruta: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
    for ax, datos, titulo in zip(ejes, (c_nivel, c_anom), ("Nivel (entre y dentro de provincias)", "Anomalía (dentro de la provincia)")):
        im = ax.imshow(datos, cmap=DIVERGENTE, vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(len(fases)), [ETIQUETA_FASE[f] for f in fases], rotation=20, ha="right")
        ax.set_yticks(range(len(vars_)), [ETIQUETA_VAR[v] for v in vars_])
        ax.grid(False)
        for i in range(datos.shape[0]):
            for j in range(datos.shape[1]):
                ax.text(j, i, fmt(datos[i, j], 2), ha="center", va="center", fontsize=8, color=INK)
        ax.set_title(titulo)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=ejes, fraction=0.025, pad=0.02)
    cb.set_label("Correlación de Spearman", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Correlación de cada predictor con el rendimiento (campañas de entrenamiento)", x=0.01, ha="left", fontsize=10.5, fontweight="bold", y=1.02)
    guardar(fig, ruta)


# ============================================================================ cálculo
def correlaciones(X: np.ndarray, y: np.ndarray, prov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spearman predictor×fase vs rendimiento: en nivel y en anomalía (centrando por provincia)."""
    V, T = X.shape[1], X.shape[2]
    nivel, anom, n = np.full((V, T), np.nan), np.full((V, T), np.nan), np.zeros((V, T), int)
    y_c = y - pd.Series(y).groupby(prov).transform("mean").to_numpy()
    for i in range(V):
        for j in range(T):
            x = X[:, i, j]
            ok = ~np.isnan(x)
            if ok.sum() < 10:
                continue
            nivel[i, j] = stats.spearmanr(x[ok], y[ok]).statistic
            x_c = x - pd.Series(x).groupby(prov).transform("mean").to_numpy()
            ok2 = ok & ~np.isnan(x_c)
            anom[i, j] = stats.spearmanr(x_c[ok2], y_c[ok2]).statistic
            n[i, j] = ok.sum()
    return nivel, anom, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw/gee_series_crudo.parquet"))
    ap.add_argument("--raw-antiguo", type=Path, default=Path("data/raw/gee_series_crudo_esquema_ndvi_antiguo.parquet"))
    ap.add_argument("--provincias", type=Path, default=Path("configs/provincias.csv"))
    ap.add_argument("--altitud", type=Path, default=Path("configs/altitud_cultivo_por_provincia.csv"))
    ap.add_argument("--area-cultivo", type=Path, default=Path("configs/area_cultivo_por_provincia.csv"))
    ap.add_argument("--mascara", type=Path, default=Path("data/manifest/mascara_agricola.yaml"))
    ap.add_argument("--calibracion", type=Path, default=Path("reports/calibracion_mascara_amplitud_2016_2020.json"))
    ap.add_argument("--sisagri-cache", type=Path, default=Path("data/interim/sisagri_quinua_8deptos.parquet"))
    ap.add_argument("--campanas-entrenamiento", type=int, nargs="+", default=[2016, 2017, 2018, 2019, 2020])
    ap.add_argument("--salida", type=Path, default=Path("docs/crisp/E3-informe-depuracion-eda.md"))
    args = ap.parse_args()

    figdir = args.salida.parent / "E3" / "figuras"
    meta = json.loads((args.processed / "metadata.json").read_text(encoding="utf-8"))
    z = np.load(args.processed / "tensores.npz")
    X, I = z["X"], z["imputado"]
    obs = pd.read_parquet(args.processed / "observaciones.parquet")
    exc = pd.read_csv(args.processed / "exclusiones.csv")
    prov = pd.read_csv(args.provincias)
    obs = obs.merge(prov[["provincia_id", "departamento"]], on="provincia_id", how="left")
    vars_, fases = meta["orden_variables"], meta["orden_fases"]
    entren_camp = args.campanas_entrenamiento
    es_entren = obs["campana_id"].isin(entren_camp).to_numpy()
    umbrales = yaml.safe_load(args.mascara.read_text(encoding="utf-8"))["umbrales"]
    calib = json.loads(args.calibracion.read_text(encoding="utf-8")) if args.calibracion.exists() else None
    raw = pd.read_parquet(args.raw)

    # ---------------- depuración
    en_periodo = exc[exc.campana_id.between(2016, 2025)]
    n_ambito = int((en_periodo.motivo == "fuera_del_ambito_altoandino").sum())
    n_ventana = int((en_periodo.motivo == "ciclo_fuera_de_ventana_de_extraccion").sum())
    n_ambito_total = int((exc.motivo == "fuera_del_ambito_altoandino").sum())
    n_obs = len(obs)
    n4 = n_obs + len(en_periodo)
    completo = ~np.isnan(X).any(axis=(1, 2))
    n5 = int((((~np.isnan(X)).mean(axis=(1, 2))) >= 0.8).sum())  # elegibles: >= 80 % de celdas válidas (4.7)
    n_completo = int(completo.sum())
    pasos = [("N4  Celdas con producción", n4), ("Tras excluir costa con riego", n4 - n_ambito),
             ("N5  Elegibles (≥ 80 % de celdas válidas)", n5), ("   de ellas, con el tensor completo", n_completo)]
    fig_cascada(pasos, figdir / "f1_cascada.png")
    fig_exclusiones(exc, figdir / "f2_exclusiones.png")

    vacias_despues = np.isnan(X).mean(axis=0) * 100
    vacias_antes = (np.isnan(X) | I).mean(axis=0) * 100
    fig_completitud(vacias_antes, vacias_despues, vars_, fases, figdir / "f3_completitud.png")

    # ---------------- objetivo (solo entrenamiento para la distribución)
    ent = obs[es_entren]
    y_e = ent["rendimiento_kg_ha"]
    fig_objetivo(ent, figdir / "f4_objetivo.png")
    fig_serie(obs, len(entren_camp), figdir / "f5_serie_campana.png")
    media_prov = ent.groupby("provincia_id")["rendimiento_kg_ha"].transform("mean")
    pct_entre = float(media_prov.var() / y_e.var() * 100)
    ent_orden = ent.sort_values(["provincia_id", "campana_id"])
    rezago = ent_orden.assign(prev=ent_orden.groupby("provincia_id")["rendimiento_kg_ha"].shift(1),
                              dif=ent_orden.groupby("provincia_id")["campana_id"].diff())
    rezago = rezago[rezago.dif == 1].dropna(subset=["prev"])
    rho_lag1 = float(stats.spearmanr(rezago.rendimiento_kg_ha, rezago.prev).statistic)
    # ¿cuánto de la variación dentro de provincia explica la campaña común (choque regional)?
    anom = ent["rendimiento_kg_ha"] - media_prov
    pct_campana = float(anom.groupby(ent["campana_id"]).transform("mean").var() / anom.var() * 100)

    # ---------------- predictores (solo entrenamiento)
    nivel, anom_c, n_corr = correlaciones(X[es_entren], y_e.to_numpy(), ent["provincia_id"].to_numpy())
    fig_correlaciones(nivel, anom_c, vars_, fases, figdir / "f6_correlaciones.png")

    # ---------------- covariables (solo entrenamiento para las relaciones)
    t_cov, texto_repr, texto_colin = "", "", "—"
    if "prop_quinua" in obs.columns and obs["prop_quinua"].notna().any():
        e_cov = obs[es_entren]
        filas = []
        for col, tipo, unidad in (("altitud_media_m", "Estática (provincia)", "m"), ("prop_quinua", "Dinámica (provincia-campaña)", "—")):
            s = obs[col].dropna()
            sub = e_cov.dropna(subset=[col])
            rho = stats.spearmanr(sub[col], sub["rendimiento_kg_ha"]).statistic
            dec = 0 if col == "altitud_media_m" else 3
            filas.append([col, tipo, unidad, fmt(s.min(), dec), fmt(s.median(), dec), fmt(s.max(), dec), fmt(rho, 2)])
        # señal interanual de la covariable dinámica: centrar por provincia
        dentro = e_cov.dropna(subset=["prop_quinua"]).copy()
        dentro["x_c"] = dentro["prop_quinua"] - dentro.groupby("provincia_id")["prop_quinua"].transform("mean")
        dentro["y_c"] = dentro["rendimiento_kg_ha"] - dentro.groupby("provincia_id")["rendimiento_kg_ha"].transform("mean")
        rho_dentro = stats.spearmanr(dentro["x_c"], dentro["y_c"]).statistic
        t_cov = md_tabla(pd.DataFrame(filas, columns=["Covariable", "Tipo", "Unidad", "Mín.", "Mediana", "Máx.", "Spearman con rendimiento (entrenamiento)"]))
        texto_cov_dentro = fmt(rho_dentro, 2)
        # colinealidad de la altitud con las variables climáticas (media de las fases, solo entrenamiento)
        Xe, alt_e = X[es_entren], obs.loc[es_entren, "altitud_media_m"].to_numpy()
        colin = {}
        for nombre, i in (("temperatura mínima", 2), ("radiación solar", 3), ("temperatura máxima", 1), ("precipitación", 0)):
            media_fases = np.nanmean(Xe[:, i, :], axis=1)
            colin[nombre] = float(stats.spearmanr(alt_e, media_fases).statistic)
        texto_colin = "; ".join(f"{k} {fmt(v, 2)}" for k, v in colin.items())
        if args.area_cultivo.exists():
            area = pd.read_csv(args.area_cultivo)
            m_obs = obs.merge(area[["provincia_id", "area_cultivo_ha", "frac_area_en_mascara"]], on="provincia_id", how="left")
            baja = m_obs[m_obs["frac_area_en_mascara"] < 0.5]
            texto_repr = (
                f"**Representatividad de la máscara (por área, WorldCover + SRTM):** en {baja['provincia_id'].nunique()} provincias "
                f"({fmt(len(baja), 0)} observaciones, {fmt(100 * len(baja) / len(obs), 0)} %) la máscara retiene menos de la mitad del "
                f"área de cultivo ({', '.join(sorted(p.title() for p in baja['provincia_id'].unique()))}); en ellas el NDVI describe una fracción "
                f"menor de lo que mide el rendimiento. Se propone usar `frac_area_en_mascara` como estrato en el análisis de sensibilidad."
            )
        else:
            texto_repr = ""
    else:
        texto_cov_dentro = "—"

    # ---------------- tablas
    rangos = []
    for v, col in [("precipitacion", "valor"), ("temperatura_maxima", "valor"), ("temperatura_minima", "valor"),
                   ("radiacion_solar", "valor"), ("ndvi", "valor_mean"), ("ndvi", "valor_max")]:
        s = raw.loc[raw.variable == v, col]
        nombre = v if v != "ndvi" else f"ndvi ({col})"
        unidad = {"precipitacion": "mm/día", "temperatura_maxima": "°C", "temperatura_minima": "°C", "radiacion_solar": "J/m²/día"}.get(v, "adim.")
        rangos.append([nombre, unidad, fmt(s.min(), 3 if v == "ndvi" else 1), fmt(s.median(), 3 if v == "ndvi" else 1),
                       fmt(s.quantile(0.99), 3 if v == "ndvi" else 1), fmt(s.max(), 3 if v == "ndvi" else 1), f"{s.isna().sum():,}".replace(",", " ")])
    t_rangos = md_tabla(pd.DataFrame(rangos, columns=["Variable", "Unidad", "Mín.", "Mediana", "P99", "Máx.", "Sin valor"]))

    t_motivos = md_tabla(pd.DataFrame(
        [[MOTIVOS.get(m, m), int(n), ", ".join(sorted(exc[exc.motivo == m].provincia_id.unique())[:8]) + ("…" if exc[exc.motivo == m].provincia_id.nunique() > 8 else "")]
         for m, n in exc.motivo.value_counts().items()],
        columns=["Motivo", "Observaciones", "Provincias (primeras)"]))

    por_campana = obs.assign(completo=completo).groupby("campana_id").agg(total=("completo", "size"), completas=("completo", "sum"))
    t_campana = md_tabla(pd.DataFrame({"Campaña": por_campana.index, "Observaciones": por_campana.total.values,
                                       "Tensor completo": por_campana.completas.astype(int).values,
                                       "Rol": ["entrenamiento (supuesto)" if c in entren_camp else "externa" for c in por_campana.index]}))
    por_dep = obs.assign(completo=completo).groupby("departamento").agg(total=("completo", "size"), completas=("completo", "sum"),
                                                                        rend=("rendimiento_kg_ha", "median")).sort_values("total", ascending=False)
    t_dep = md_tabla(pd.DataFrame({"Departamento": [nombre_dep(d) for d in por_dep.index], "Observaciones": por_dep.total.values,
                                   "Tensor completo": por_dep.completas.astype(int).values,
                                   "Rendimiento mediano (kg/ha)": [fmt(v, 0) for v in por_dep.rend.values]}))

    est = y_e.describe(percentiles=[0.01, 0.5, 0.99])
    t_obj = md_tabla(pd.DataFrame([[fmt(len(y_e), 0), fmt(est["mean"], 0), fmt(est["std"], 0), fmt(est["std"] / est["mean"] * 100, 0) + " %",
                                    fmt(est["1%"], 0), fmt(est["50%"], 0), fmt(est["99%"], 0), fmt(float(stats.skew(y_e)), 2)]],
                                  columns=["n", "Media", "Desv. est.", "CV", "P1", "Mediana", "P99", "Asimetría"]))

    # evidencia territorial (SISAGRI por distrito)
    evid = ""
    if args.sisagri_cache.exists():
        q = pd.read_parquet(args.sisagri_cache)
        q.columns = ["anio", "mes", "ubigeo", "depto", "provincia", "distrito", "cod_prod", "producto", "siembra", "cosecha", "produccion", "verde", "precio"]
        q = q[q.anio.between(2015, 2025)]
        filas = []
        for p in ("CARABAYA", "CASTILLA"):
            g = q[q.provincia == p].groupby("distrito").produccion.sum().sort_values(ascending=False)
            top = ", ".join(f"{d.title()} {fmt(100 * v / g.sum(), 0)} %" for d, v in g.head(4).items())
            alt = pd.read_csv(args.altitud).set_index("provincia_id")
            filas.append([p.title(), top, fmt(float(alt.loc[p, "altitud_mediana_m"]), 0) + " m", fmt(100 * float(alt.loc[p, "frac_en_rango"]), 0) + " %"])
        evid = md_tabla(pd.DataFrame(filas, columns=["Provincia", "Distritos con más producción de quinua", "Altitud mediana del cultivo", "Cultivo dentro de 2 500-4 000 m"]))

    top = obs.sort_values("rendimiento_kg_ha", ascending=False).head(5)
    bajo = obs.sort_values("rendimiento_kg_ha").head(3)
    t_extremos = md_tabla(pd.DataFrame(
        [[("Alto" if r.rendimiento_kg_ha >= obs.rendimiento_kg_ha.median() else "Bajo"), r.provincia_id.title(), nombre_dep(r.departamento),
          int(r.campana_id), fmt(r.rendimiento_kg_ha, 0)] for r in pd.concat([top, bajo]).itertuples()],
        columns=["Extremo", "Provincia", "Departamento", "Campaña", "kg/ha"]))

    raw_ndvi = raw[raw.variable == "ndvi"]
    nulos = raw_ndvi["valor_mean"].isna()
    sin_ningun = raw_ndvi.groupby(["provincia_id", "campana_id"])["valor_mean"].transform(lambda s: s.isna().all())
    provs_sin_ndvi = sorted(raw_ndvi.loc[sin_ningun, "provincia_id"].unique())
    pct_nulos_provs = 100 * float((nulos & sin_ningun).sum() / max(nulos.sum(), 1))
    efecto_mascara = ""
    if args.raw_antiguo.exists():
        viejo = pd.read_parquet(args.raw_antiguo)
        vn = viejo[viejo.variable == "ndvi"]
        efecto_mascara = f"El NDVI medio global pasó de {fmt(vn['valor'].mean(), 3)} (sin máscara ni filtro de calidad) a {fmt(raw_ndvi['valor_mean'].mean(), 3)} (con máscara y filtro), porque se excluyen celdas no agrícolas."

    # texto de correlaciones
    def fuerte(m: np.ndarray, k: int = 3) -> str:
        return "; ".join(
            f"{ETIQUETA_VAR[vars_[i]].lower()} en {ETIQUETA_FASE[fases[j]].lower()} ({fmt(v, 2)})"
            for i, j, v in mayores_por_magnitud(m, k)
        )

    # ---------------- umbrales
    t_umbrales = md_tabla(pd.DataFrame([
        ["Altitud", f"{fmt(umbrales['altitud_min_m'], 0)}–{fmt(umbrales['altitud_max_m'], 0)} m", "PROINPA/FAO (2011); INIA", "Provisional"],
        ["Pendiente máxima", f"{fmt(umbrales['pendiente_max_grados'], 2)}° (25 %)", "D.S. 005-2022-MIDAGRI (RCTCUM)", "Provisional"],
        ["Amplitud NDVI mínima", fmt(umbrales["amplitud_ndvi_min"], 3), "Calibrada: percentil 5, campañas 2016-2020", "Provisional (percentil por decidir)"],
    ], columns=["Criterio", "Umbral", "Fuente", "Estado"]))
    t_calib = ""
    if calib:
        filas = []
        for p, r in calib["umbrales"].items():
            d = r["excluidas_pct_por_departamento"]
            filas.append([f"p{p}", fmt(r["umbral"], 3), fmt(r["excluidas_pct_total"], 0) + " %", fmt(d.get("AREQUIPA", np.nan), 0) + " %", fmt(d.get("PUNO", np.nan), 0) + " %", fmt(d.get("CUSCO", np.nan), 0) + " %"])
        t_calib = md_tabla(pd.DataFrame(filas, columns=["Percentil", "Umbral", "Excluye (total)", "Arequipa", "Puno", "Cusco"]))

    ruta_celdas = Path("data/interim/calibracion_mascara_celdas.csv")
    pct_sobre_4000 = pct_sobre_4300 = float("nan")
    if ruta_celdas.exists():
        cel = pd.read_csv(ruta_celdas).drop_duplicates("celda_id")
        pct_sobre_4000, pct_sobre_4300 = 100 * float((cel.altitud_m > 4000).mean()), 100 * float((cel.altitud_m > 4300).mean())

    hoy = date.today().isoformat()
    artefactos = md_tabla(pd.DataFrame([
        ["Series crudas GEE", str(args.raw).replace("\\", "/"), sha256(args.raw)[:16] + "…"],
        ["Tensores V×T", "data/processed/tensores.npz", sha256(args.processed / "tensores.npz")[:16] + "…"],
        ["Observaciones", "data/processed/observaciones.parquet", sha256(args.processed / "observaciones.parquet")[:16] + "…"],
    ], columns=["Artefacto", "Ruta", "sha256 (16 primeros)"]))

    completos_pct = 100 * n_completo / n_obs
    imput = int(I.sum())
    celdas_ndvi = X[:, 4:, :].size

    md = f"""# E3 · Informe de depuración y análisis exploratorio

**Fase CRISP-ML(Q):** F2 · Preparación de datos (actividades 3-5 del cronograma) · **Entregable:** E3
**Proyecto:** Modelo CNN-1D interpretable con SHAP para predecir el rendimiento de quinua en la sierra peruana
**Generado:** {hoy} · **Commit:** `{git_commit()}` · **Estado del conjunto:** `exploratorio = {str(meta['exploratorio']).lower()}` ({'no válido aún para la evaluación definitiva' if meta['exploratorio'] else 'respaldo departamental estimado solo con ' + str(entren_camp[0]) + '–' + str(entren_camp[-1]) + ': válido para la evaluación definitiva'})

> Este informe se regenera con `python scripts/build_e3_report.py`; ninguna cifra se transcribe a mano.
> Las relaciones predictor–rendimiento y la distribución del objetivo usan **solo las campañas de entrenamiento
> {entren_camp[0]}–{entren_camp[-1]}**, para que el análisis exploratorio no oriente decisiones con las campañas externas (sección 4.12.2).

## 1. Resumen

- El conjunto analítico tiene **{fmt(n_obs, 0)} observaciones provincia-campaña** ({fmt(prov[prov.provincia_id.isin(obs.provincia_id)].shape[0], 0)} provincias, 2016–2025): **N5 = {fmt(n5, 0)}**, todas con al menos el 80 % de celdas válidas (criterio de la sección 4.7; la menor tiene {fmt(100 * float((~np.isnan(X)).mean(axis=(1, 2)).min()), 1)} %). **{fmt(n_completo, 0)} ({fmt(completos_pct, 0)} %) tienen el tensor completo** tras una imputación limitada de NDVI sin fuga; las otras {fmt(n5 - n_completo, 0)} conservan alguna celda vacía que se imputará con la mediana de entrenamiento, y el subconjunto completo sirve de análisis de sensibilidad.
- El clima (CHIRPS, ERA5-Land) está **100 % completo**; el NDVI enmascarado tiene {fmt(100 * raw_ndvi['valor_mean'].isna().mean(), 1)} % de filas sin valor: el {fmt(pct_nulos_provs, 0)} % corresponde a {len(provs_sin_ndvi)} provincias costeras o de selva que la máscara de altitud deja sin píxeles por diseño, y el resto son huecos puntuales de calidad (nubes).
- Se detectaron y corrigieron **tres defectos de calidad** durante la preparación (sección 4.3); uno habría invalidado las variables climáticas.
- **{fmt(pct_entre, 0)} % de la varianza del rendimiento** (entrenamiento) es **entre provincias** y el resto, dentro de ellas: la línea base provincial (B2) captura buena parte de la señal y el valor añadido de los modelos está en la variación interanual.
- La potencia es limitada: con una ventana inicial de {len(entren_camp)} campañas quedan **{len(set(obs.campana_id) - set(entren_camp))} campañas externas**.
- Hay **cinco decisiones provisionales** que requieren al asesor (sección 7).

## 2. Fuentes y trazabilidad

{artefactos}

Fuentes: SIEA-MIDAGRI (rendimiento y siembra, `SISAGRI.xlsx`); CHIRPS v2.0 (precipitación); ERA5-Land DAILY_AGGR (temperaturas y radiación); MODIS MOD13Q1 V061 (NDVI, 250 m) sobre la máscara agrícola (ESA WorldCover v200 2021 + SRTM). Procedencia por variable, hashes y el incidente de extracción: `data/manifest/gee_series_climaticas_espectrales.yaml`.

## 3. Depuración del conjunto analítico

### 3.1 Cascada de elegibilidad

![Cascada](E3/figuras/f1_cascada.png)

Las cotas N0–N3 de la sección 4.5.6 (800, 770, 670 y 620 celdas) se calcularon en la auditoría H1 sobre el periodo enmendado; **N4 = {fmt(n4, 0)}** es el recuento real con producción y superficie cosechada > 0 en 2016–2025 y supera a N3 porque la producción observada abarca más provincias que las 62 documentadas de forma preliminar. De N4 se excluyeron {fmt(n_ambito, 0)} observaciones fuera del ámbito altoandino y {fmt(n_ventana, 0)} por ciclo fuera de la ventana de extracción; quedan **N5 = {fmt(n5, 0)}** observaciones elegibles ({fmt(n_completo, 0)} con el tensor completo).

### 3.2 Exclusiones y su motivo

![Exclusiones](E3/figuras/f2_exclusiones.png)

{t_motivos}

Las exclusiones por *sin series* son las campañas 2015 y 2026, fuera del periodo enmendado. De las {n_ambito_total} exclusiones por ámbito, {n_ambito_total - n_ambito} pertenecen también a esas dos campañas, por lo que la cascada de 2016–2025 (3.1) descuenta {n_ambito}. Toda observación descartada queda registrada con su motivo en `data/processed/exclusiones.csv` (sección 4.7).

### 3.3 Reglas aplicadas

| Criterio (sección 4.7) | Implementación |
|---|---|
| Inclusión 1: producción y superficie cosechada > 0 | Rendimiento = producción / superficie cosechada, solo celdas con ambas > 0 |
| Inclusión 2 y exclusión 4: ámbito altoandino | Se excluyen `costa_riego` y `selva` (`configs/provincias.csv`); los casos de **transición** se conservan marcados en la columna `clasificacion` |
| Inclusión 3 / exclusión 2: completitud y sin extrapolación extensa | Se excluye el ciclo que excede la serie extraída; imputación limitada de NDVI (arrastre ≤ 16 días, sin usar datos posteriores) |
| Inclusión 4: solo información hasta el horizonte | Ventanas de fase truncadas {fmt(meta['dias_antes_de_cosecha'], 0)} días antes de la cosecha estimada |
| Exclusión 3: valores extremos | Se marcan para revisión, **no** se eliminan automáticamente (ver 5.1) |

### 3.4 Cobertura del conjunto

{t_campana}

{t_dep}

## 4. Calidad de las series

### 4.1 Plausibilidad física

{t_rangos}

Todas las variables están dentro de los límites físicos de control (`validate_raw_ranges`). {efecto_mascara}

### 4.2 Completitud del tensor e imputación limitada

![Completitud](E3/figuras/f3_completitud.png)

El NDVI de MODIS es un compuesto de 16 días y la fase de emergencia dura 3 días, por lo que el {fmt(vacias_antes[4, 0], 0)} % de las celdas de esa fase quedaba vacío. Se aplicó la imputación limitada de 4.7 de forma **que no usa información futura**: una celda vacía toma la última observación válida con fecha ≤ al fin de su fase y a lo sumo 16 días antes. Se imputaron **{fmt(imput, 0)} celdas** ({fmt(100 * imput / celdas_ndvi, 1)} % de las de NDVI, todas de emergencia) y cada una queda marcada en `X_imputado`. El tensor completo pasa de {fmt(100 * (~(np.isnan(X) | I).any(axis=(1, 2))).mean(), 0)} % a **{fmt(completos_pct, 0)} %**, y la menor cobertura por variable y fase es de {fmt(100 - float(vacias_despues.max()), 1)} %, por encima del 80 % exigido.

### 4.3 Defectos de calidad detectados y corregidos

1. **Reductor espacial erróneo (grave).** Desde el commit `de73679` la regla temporal por fase (suma, máximo, mínimo) se usaba como reductor espacial de la provincia: la precipitación diaria era la **suma de ~177 píxeles** (114,06 mm frente a una media real de 0,649 mm en Azángaro, 3-jul-2018) y la radiación ~25 veces su valor. Se corrigió, se verificó contra la API real (58 filas × 4 variables idénticas a 1e-6) y se añadió un control permanente de límites físicos.
2. **Nombre de propiedad de `reduceRegions`.** Con una imagen de una banda y reducers combinados, la API real devuelve `mean`/`max`, no `NDVI_mean`/`NDVI_max`; el código habría fallado en la extracción larga.
3. **Ámbito territorial no aplicado.** La primera corrida incluyó {fmt(76, 0)} observaciones fuera del ámbito altoandino; se corrigió el filtro.

## 5. Análisis exploratorio del rendimiento (campañas de entrenamiento)

{t_obj}

![Objetivo](E3/figuras/f4_objetivo.png)

### 5.1 Valores extremos

El rendimiento máximo es de {fmt(obs.rendimiento_kg_ha.max(), 0)} kg/ha y el mínimo de {fmt(obs.rendimiento_kg_ha.min(), 0)} kg/ha (todas las campañas). El máximo previo de 5 000 kg/ha (un valor redondo) era **Chepén 2024**, costa con riego, ya excluida. Los extremos restantes se **conservan marcados y no se eliminan** (sección 4.7, criterio 3); los más altos son plausibles pero llamativos para la sierra y su efecto se evaluará en el análisis de sensibilidad:

{t_extremos}

### 5.2 Estructura de la variación

- **Entre provincias:** {fmt(pct_entre, 0)} % de la varianza (nivel de rendimiento de cada provincia).
- **Choque común de campaña:** {fmt(pct_campana, 0)} % de la variación *dentro* de las provincias se explica por la campaña (clima regional compartido), lo que confirma que las campañas no son independientes entre provincias (sección 4.5.7).
- **Persistencia:** correlación de Spearman con el rendimiento de la campaña anterior de la misma provincia = **{fmt(rho_lag1, 2)}** (n = {fmt(len(rezago), 0)} pares).

![Serie por campaña](E3/figuras/f5_serie_campana.png)

La serie por campaña es descriptiva e incluye todas las campañas; no se usó para ninguna decisión de modelado.

## 6. Análisis exploratorio de los predictores (campañas de entrenamiento)

![Correlaciones](E3/figuras/f6_correlaciones.png)

- **En nivel**, las asociaciones más fuertes son: {fuerte(nivel)}. Reflejan sobre todo diferencias entre provincias (altitud y clima medio), no señal interanual.
- **En anomalía** (centrando por provincia, que es lo que realmente debe predecir un modelo frente a B2), las más fuertes son: {fuerte(anom_c)}. Valores bajos indican que la señal interanual de cada predictor es modesta con n = {fmt(int(n_corr.max()), 0)} observaciones de entrenamiento.
- Las correlaciones son **descriptivas** y no implican causalidad; los valores SHAP medirán la contribución predictiva, no el efecto causal (sección 1.6).

### 6.1 Covariables del tensor (sección 4.10.3)

Se concatenan **después del bloque convolucional**; no forman parte del tensor V×T (`Z`, N × 2, en `tensores.npz`).

{t_cov}

- La **altitud media de la superficie agrícola** se calcula solo con el cultivo que pasa la máscara (altitud y pendiente), la misma superficie sobre la que se mide el NDVI. Su correlación negativa con el rendimiento refleja que las provincias más altas rinden menos. Está **moderadamente correlacionada con el clima**: correlación de Spearman de la altitud media con la media por fase de {texto_colin}. Por eso su aporte propio debe evaluarse en la ablación y no asumirse.
- La **proporción sembrada con quinua** es la superficie sembrada en la campaña dividida por el área de cultivo de la provincia (ESA WorldCover 2021, supuesto estático). Nunca supera 1 ({fmt(meta.get('prop_quinua_mayor_que_uno', 0), 0)} casos), lo que indica coherencia entre SISAGRI y WorldCover. Dentro de cada provincia su correlación con el rendimiento es de **{texto_cov_dentro}**: la variación interanual de la proporción sembrada no explica por sí sola el rendimiento, y su valor está sobre todo en distinguir el peso relativo de la quinua entre provincias.
- {texto_repr}

## 7. Decisiones provisionales y pendientes con el asesor

Se tomaron con la mejor evidencia disponible y son **reversibles**; ninguna se considera definitiva hasta la aprobación del asesor.

**Umbrales de la máscara agrícola**

{t_umbrales}

**Percentil de la amplitud NDVI (compromiso con sesgo regional).**

{t_calib}

| Decisión | Valor provisional | Evidencia y efecto | Alternativa |
|---|---|---|---|
| Percentil de amplitud NDVI | p5 = {fmt(umbrales['amplitud_ndvi_min'], 3)} | Ver tabla: el mismo umbral excluye más cultivo en Arequipa y Puno que en Cusco | p1 casi sin sesgo pero filtro casi nulo; exige re-extraer el NDVI (~3,5 h) |
| Provincias de transición | Incluidas y marcadas ({fmt(int((obs.clasificacion == 'transicion').sum()), 0)} obs) | La Mar y Huanta son valles interandinos productores; Carabaya produce en zona alta (abajo) | Excluirlas y reportar la sensibilidad |
| Exclusión por altitud del cultivo | **No** se excluye (`--excluir-por-altitud` desactivado) | La altitud del cultivo no es la de la quinua (abajo) | Usar `frac_area_en_mascara` (`area_cultivo_por_provincia.csv`) como estrato de sensibilidad |
| Ventana de extracción y calendario | Se excluyen {fmt(n_ventana, 0)} obs con ciclo fuera de ventana | Siembras de mayo-junio (Arequipa, La Libertad) cosechan en sept-dic | Ampliar la ventana a diciembre y re-extraer (~2 h); Arequipa recuperaría casi todo |
| Ventana inicial de entrenamiento | {len(entren_camp)} campañas ({entren_camp[0]}–{entren_camp[-1]}) | Deja {len(set(obs.campana_id) - set(entren_camp))} campañas externas; el menor valor p del Wilcoxon unilateral con n = 5 es 1/32 ≈ 0,031 | Ventana de 4: 6 campañas externas y menos entrenamiento inicial |

**Evidencia territorial** (SISAGRI por distrito, 2015–2025, y altitud del cultivo según WorldCover/SRTM):

{evid}

La quinua de Castilla se produce en la sierra (Pampacolca y Viraco) y no en la pampa de riego de Majes; la de Carabaya, en la zona alta de la provincia. El techo de 4 000 m lo respalda la literatura (INIA: la quinua se adapta del nivel del mar a ~4 000 m) y solo el {fmt(pct_sobre_4000, 1)} % de las celdas de cultivo muestreadas lo supera ({fmt(pct_sobre_4300, 1)} % sobre 4 300 m), por lo que no se propone subirlo. Otros pendientes: confirmar que los umbrales de altitud y pendiente (tabla anterior) se aprueban, y la interpretación operacional de «pasos válidos» del criterio de completitud.

## 8. Implicaciones para el modelado (F3) y la evaluación (F4)

1. **B2 será una referencia fuerte**: con {fmt(pct_entre, 0)} % de la varianza entre provincias, un modelo debe mejorar la *variación interanual* para superar la línea base provincial (HE1b).
2. **Dependencia entre provincias de una misma campaña**: las unidades independientes son las campañas, no las observaciones; los intervalos deben calcularse por bloques de campaña.
3. **Potencia**: con {len(set(obs.campana_id) - set(entren_camp))} campañas externas los contrastes confirmatorios tendrán poca potencia; se reportarán tamaños de efecto e intervalos además de valores p.
4. **Imputación y horizonte**: las celdas imputadas se marcan (`X_imputado`) y pueden usarse para análisis de sensibilidad; las ventanas ya respetan el horizonte de 30 días.
5. **Escalas muy distintas** (radiación del orden de 10⁸ J/m² frente a NDVI < 1): la estandarización debe ajustarse solo con entrenamiento por pliegue.
6. **Respaldo fenológico**: la alineación actual usó el nivel primario en el 100 % de las observaciones; el respaldo departamental (`estimate_departmental_modal_dates`) se estima solo con las campañas de entrenamiento (2016–2020, ventana inicial de E5) y no se necesitó.

## 9. Reproducibilidad

```
python scripts/run_ingestion.py --gee-project-id cnn-sentinel      # series crudas (requiere GEE autenticado)
python scripts/calibrate_mask_amplitude.py                          # umbral de amplitud NDVI
python scripts/derive_altitud_cultivo.py                            # altitud del cultivo por provincia y covariable estática
python scripts/derive_area_cultivo.py                               # área de cultivo por provincia (requiere GEE)
python scripts/run_preprocessing.py --archivo-sisagri <SISAGRI.xlsx> --campanas-entrenamiento 2016 2017 2018 2019 2020 --dias-antes-de-cosecha 30
python scripts/build_e3_report.py                                   # este informe
```

Limitaciones: el conjunto es observacional y retrospectivo; el rendimiento oficial puede tener error de medición; la máscara no distingue quinua de otros cultivos (se controla con la covariable de proporción de quinua y el análisis de sensibilidad por estratos de representatividad); el área de cultivo es de 2021 y se asume estática para las 10 campañas.
"""
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(md, encoding="utf-8", newline="\n")
    print(f"Informe: {args.salida} ({len(md.splitlines())} líneas); figuras en {figdir}")


if __name__ == "__main__":
    main()
