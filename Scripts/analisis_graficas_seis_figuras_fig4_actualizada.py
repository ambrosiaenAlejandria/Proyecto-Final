
"""
Cafeterías de especialidad en Bogotá · Haciendo Economía 2026-2
Genera las 6 gráficas del análisis y las guarda en una carpeta.
 
Pregunta: ¿Cómo se asocian el storytelling y la ambientación de las cafeterías
de especialidad con su diferenciación percibida, la recordación y la preferencia
declarada de sus clientes?
 
USO (desde la carpeta del proyecto):
    pip install pandas numpy scipy matplotlib openpyxl
    python analisis_graficas_corregido.py
    python analisis_graficas_corregido.py --datos ruta/a/Base_de_datos.xlsx --salida figuras
 
Por defecto busca el Excel en la carpeta del script o en ./datos/ (cualquier
archivo que empiece por "Base_de_datos") y escribe en ./figuras/.
 
Decisiones metodológicas (según el Plan de análisis):
  * Unidad de análisis = el café. Las encuestas se resumen en un valor por café.
  * Asociaciones solo con cafés con >= MIN_ENC encuestas completas (Fig. 2-6).
  * Spearman + IC 95% bootstrap (remuestreo de cafés). Con ~7 cafés es EXPLORATORIO.
  * Los datos originales no se modifican: la limpieza ocurre aquí, en memoria.
"""
import argparse
import glob
import os
import sys
import unicodedata
import warnings
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
 
warnings.filterwarnings("ignore")
 
# ----------------------------------------------------------------- parámetros
MIN_ENC = 6          # mínimo de encuestas completas para incluir un café
N_BOOT = 5000        # remuestreos bootstrap
SEED = 42            # semilla -> resultados idénticos en cada ejecución
DPI = 200
 
# Paleta (validada para daltonismo): storytelling = azul, ambientación = naranja
C_STORY, C_AMB = "#2a78d6", "#eb6834"
INK, INK2, GRID, ABSENT, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#e3e2dd", "#fcfcfb"
 
# Nombres cortos para etiquetas (si un café no está, se usa su nombre completo)
NOMBRE_CORTO = {
    "C01": "Silo", "C02": "Oishii", "C03": "BCR LAB", "C04": "Casa Green",
    "C05": "Orilla", "C06": "Nossa", "C07": "Amor Patrio", "C08": "Café Nos",
    "C09": "Serendipiarte",
}
ITEMS_ST = {"st1": "Origen en la carta", "st2": "Proceso visible", "st3": "Historia visible",
            "st4": "Personal informado", "st5": "Catas / experiencias", "st6": "Empaque con origen"}
ITEMS_AM = {"am1": "Identidad visual", "am2": "Música", "am3": "Aroma", "am4": "Zonas diferenciadas",
            "am5": "Mobiliario cómodo", "am6": "Barra a la vista", "am7": "Decoración alusiva"}
 
# Variables de resultado por café: columna -> (etiqueta corta, etiqueta eje)
RESULTADOS = {
    "per_distintivo": ("Diferenciación", "Diferenciación percibida (prom. 1–5)"),
    "pct_recuerda": ("Recordación", "% que recuerda un dato (%)"),
    "nps": ("Recomendación", "Recomendaría (prom. 0–10)"),
    "prob_volver": ("Prob. de volver", "Probabilidad de volver (prom. 1–5)"),
    "pct_primera": ("Primera opción", "% que lo declara primera opción"),
}
 
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10,
})
 
 
# ------------------------------------------------------------------ utilidades
def norm(s):
    """minúsculas, sin tildes, guiones normalizados (arregla '1–3' vs '1-3', 'Pasó' vs 'Paso')."""
    if pd.isna(s):
        return np.nan
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.replace("–", "-").replace("—", "-").strip().lower()
 
 
def coma(x, d=2):
    return "—" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")
 
 
def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return spearmanr(x, y)[0]
 
 
def rho_boot(x, y, n_boot=N_BOOT, seed=SEED):
    """Spearman e IC 95% (percentil) remuestreando cafés."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r0 = rho(x, y)
    rng = np.random.default_rng(seed)
    n, vals = len(x), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        vals.append(np.corrcoef(rankdata(x[i]), rankdata(y[i]))[0, 1])
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return r0, lo, hi
 
 
def encontrar_excel(ruta):
    if ruta:
        return ruta
    base = os.path.dirname(os.path.abspath(__file__))
    for carpeta in (base, os.path.join(base, "datos"), os.getcwd(), os.path.join(os.getcwd(), "datos")):
        hallados = sorted(glob.glob(os.path.join(carpeta, "Base_de_datos*.xlsx")))
        if hallados:
            return hallados[0]
    sys.exit("No encontré el Excel. Ponlo junto al script (o en ./datos/) o usa --datos ruta.xlsx")
 
 
# ------------------------------------------------- 1. carga y limpieza de datos
def cargar(ruta):
    ficha = pd.read_excel(ruta, sheet_name="Ficha_Cafe")
    enc = pd.read_excel(ruta, sheet_name="Encuestas")
 
    # --- Ficha de observación: solo cafés reales con los 13 ítems registrados
    items = list(ITEMS_ST) + list(ITEMS_AM)
    ficha = ficha[ficha["cafe_id"].astype(str).str.match(r"^C\d+$")].copy()
    for c in items + ["google_rating", "google_resenas", "tiene_instagram"]:
        ficha[c] = pd.to_numeric(ficha[c], errors="coerce")
    ficha = ficha.dropna(subset=items)                    # excluye reservas sin visitar
    ficha["idx_story"] = ficha[list(ITEMS_ST)].sum(axis=1)  # se recalculan (0–6)
    ficha["idx_amb"] = ficha[list(ITEMS_AM)].sum(axis=1)    # (0–7)
    ficha["corto"] = ficha.apply(lambda r: NOMBRE_CORTO.get(r.cafe_id, r.nombre_cafe), axis=1)
 
    # --- Encuestas: sin fila EJEMPLO, solo con consentimiento = 1
    enc = enc[enc["cafe_id"].isin(ficha["cafe_id"])].copy()
    enc["consentimiento"] = pd.to_numeric(enc["consentimiento"], errors="coerce")
    enc = enc[enc["consentimiento"] == 1]
    num = ["per_ambiente", "per_distintivo", "per_origen", "per_historia", "per_calidad",
           "per_atencion", "per_descripcion", "recuerda_dato", "nps_recomienda", "prob_volver"]
    for c in num:                                          # algunas celdas vienen como texto
        enc[c] = pd.to_numeric(enc[c], errors="coerce")
    enc["_conocio"] = enc["como_conocio"].map(norm)
    enc["_digital"] = enc["_conocio"].map(
        lambda s: np.nan if pd.isna(s) else float(s in ("redes sociales", "google maps u otra app")))
    enc["_primera"] = enc["primera_opcion"].map(norm).map(
        lambda s: np.nan if pd.isna(s) else float(s == "es mi primera opcion"))
    return ficha.reset_index(drop=True), enc.reset_index(drop=True)
 
 
def resumen_por_cafe(ficha, enc):
    g = enc.groupby("cafe_id")
    r = pd.DataFrame({
        "n_enc": g.size(),
        "per_distintivo": g["per_distintivo"].mean(),
        "per_ambiente": g["per_ambiente"].mean(),
        "per_historia": g["per_historia"].mean(),
        "pct_recuerda": g["recuerda_dato"].mean() * 100,
        "pct_origen": g["per_origen"].mean() * 100,
        "nps": g["nps_recomienda"].mean(),
        "prob_volver": g["prob_volver"].mean(),
        "pct_primera": g["_primera"].mean() * 100,
        "pct_digital": g["_digital"].mean() * 100,
    })
    cols = ["cafe_id", "corto", "nombre_cafe", "zona_muestreo", "idx_story", "idx_amb",
            "tiene_instagram", "google_rating", "google_resenas"]
    df = ficha[cols].merge(r, left_on="cafe_id", right_index=True, how="left")
    df["n_enc"] = df["n_enc"].fillna(0).astype(int)
    df["incluido"] = df["n_enc"] >= MIN_ENC
    return df
 
 
# ----------------------------------------------------------- elementos comunes
def cierre(fig, titulo, nota, nombre, carpeta):
    fig.suptitle(titulo, x=0.02, y=0.985, ha="left", va="top",
                 fontsize=13.5, fontweight="bold", color=INK)
    # Reserva una franja superior para el título/leyenda y una inferior para la nota.
    fig.tight_layout(rect=(0.025, 0.085, 0.985, 0.865))
    fig.text(0.02, 0.018, nota, fontsize=8.3, color=INK2, ha="left", va="bottom")
    ruta = os.path.join(carpeta, nombre)
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    print("  guardada:", ruta)
 
 
def etiquetar(ax, xs, ys, nombres, dx=6, dy=5):
    """Etiqueta cada punto; los que coinciden en (x, y) se apilan en una sola etiqueta."""
    grupos = {}
    for x, y, n in zip(xs, ys, nombres):
        grupos.setdefault((round(float(x), 3), round(float(y), 3)), []).append(n)
    for (x, y), ns in grupos.items():
        ax.annotate("\n".join(ns), (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color=INK2, va="bottom")
 
 
def dispersion(ax, d, xc, yc, color, xlabel, ylabel, ylim=None, xlim=None, xticks=None, logx=False):
    """Un punto por café; tamaño = nº de encuestas; ρ de Spearman en el título."""
    d = d.dropna(subset=[xc, yc])
    sizes = 40 + 18 * d["n_enc"].clip(lower=1)
    ax.scatter(d[xc], d[yc], s=sizes, color=color, edgecolor=SURFACE, linewidth=1.6, zorder=3)
    etiquetar(ax, d[xc], d[yc], d["corto"])
    ax.text(0.02, 0.97, f"ρ = {coma(rho(d[xc], d[yc]))}  (n = {len(d)} cafés)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK,
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.88, pad=2.0),
            zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    else:
        lo, hi = d[xc].min(), d[xc].max()
        pad = (hi - lo) * 0.18 or 1
        ax.set_xlim(lo - pad, hi + pad * 1.7) if not logx else ax.set_xlim(lo * 0.6, hi * 2.2)
    if xticks is not None:
        ax.set_xticks(xticks)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        lo, hi = d[yc].min(), d[yc].max()
        pad = (hi - lo) * 0.2 or 1
        ax.set_ylim(lo - pad, hi + pad)
    ax.grid(axis="x", visible=False)
 
 
# -------------------------------------------------------------------- FIGURAS
def fig1_practicas(ficha, df, carpeta):
    """Qué hace cada café: 13 prácticas observadas."""
    d = ficha.merge(df[["cafe_id", "n_enc"]], on="cafe_id")
    d = d.sort_values(["idx_story", "idx_amb", "cafe_id"], ascending=[False, False, True]).reset_index(drop=True)
    cols = list(ITEMS_ST) + list(ITEMS_AM)
    xpos = {c: i + (0.7 if c.startswith("am") else 0) for i, c in enumerate(cols)}
 
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for r, row in d.iterrows():
        for c in cols:
            on = row[c] == 1
            col = (C_STORY if c.startswith("st") else C_AMB) if on else ABSENT
            ax.add_patch(Rectangle((xpos[c] - 0.5, r - 0.5), 1, 1, facecolor=col, edgecolor=SURFACE, lw=2.5))
            ax.text(xpos[c], r, "✓" if on else "", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        xr = max(xpos.values()) + 1.2
        ax.text(xr, r, f"{int(row.idx_story)}", ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(xr + 1.4, r, f"{int(row.idx_amb)}", ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(xr, -0.95, "idx_story\n(0–6)", ha="center", va="bottom", fontsize=8.5, color=C_STORY, fontweight="bold")
    ax.text(xr + 1.4, -0.95, "idx_amb\n(0–7)", ha="center", va="bottom", fontsize=8.5, color=C_AMB, fontweight="bold")
 
    nombres = {**ITEMS_ST, **ITEMS_AM}
    ax.set_xticks([xpos[c] for c in cols])
    ax.set_xticklabels([nombres[c] for c in cols], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.cafe_id} {r.corto}" + ("*" if r.n_enc < MIN_ENC else "") for r in d.itertuples()], fontsize=10)
    ax.set_xlim(-0.6, xr + 2.2)
    ax.set_ylim(len(d) - 0.5, -0.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.legend(handles=[Rectangle((0, 0), 1, 1, color=C_STORY, label="Storytelling presente"),
                        Rectangle((0, 0), 1, 1, color=C_AMB, label="Ambientación presente"),
                        Rectangle((0, 0), 1, 1, color=ABSENT, label="Se buscó y no está")],
               loc="upper center", bbox_to_anchor=(0.52, 0.925), ncol=3,
               frameon=False, fontsize=9, handlelength=1.2)
    s, a = df["idx_story"], df["idx_amb"]
    cierre(fig,
           f"Fig. 1 · Los cafés difieren mucho en storytelling ({int(s.min())}–{int(s.max())} de 6) "
           f"y casi nada en ambientación ({int(a.min())}–{int(a.max())} de 7)",
           f"* Menos de {MIN_ENC} encuestas completas: se muestra aquí, pero se excluye de las asociaciones (Fig. 2–6). "
           "Fuente: ficha de observación (1 visita por café).",
           "fig1_practicas_por_cafe.png", carpeta)
 
 
def tabla_correlaciones(df):
    d = df[df["incluido"]]
    filas = []
    for idx, nom in (("idx_story", "Storytelling"), ("idx_amb", "Ambientación")):
        for col, (lab, _) in RESULTADOS.items():
            r, lo, hi = rho_boot(d[idx], d[col])
            filas.append({"indice": nom, "idx": idx, "resultado": lab, "col": col, "rho": r, "ic_inf": lo, "ic_sup": hi, "n_cafes": len(d)})
    return pd.DataFrame(filas)
 
 
def fig2_forest(tc, carpeta):
    """Resumen: ρ de Spearman con IC 95% bootstrap para cada par índice-resultado."""
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    labs = [v[0] for v in RESULTADOS.values()]
    for k, lab in enumerate(labs):
        for off, idx, col in ((-0.17, "Storytelling", C_STORY), (0.17, "Ambientación", C_AMB)):
            f = tc[(tc.resultado == lab) & (tc.indice == idx)].iloc[0]
            y = k + off
            if not np.isnan(f.ic_inf):
                ax.hlines(y, f.ic_inf, f.ic_sup, color=col, lw=2.2, alpha=0.55, zorder=2)
            ax.scatter(f.rho, y, s=75, color=col, edgecolor=SURFACE, linewidth=1.6, zorder=3)
            ax.text(1.04, y, coma(f.rho), va="center", fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=10.5)
    ax.set_ylim(len(labs) - 0.5, -0.5)
    ax.set_xlim(-1.05, 1.2)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("Correlación de Spearman (ρ) · línea = IC 95% bootstrap")
    ax.grid(axis="y", visible=False)
    fig.legend(handles=[Line2D([], [], marker="o", color=C_STORY, lw=2.2, label="idx_story (storytelling)"),
                        Line2D([], [], marker="o", color=C_AMB, lw=2.2, label="idx_amb (ambientación)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2,
               frameon=False, fontsize=9.5)
    sig = int(((tc.ic_inf > 0) | (tc.ic_sup < 0)).sum())
    n = int(tc.n_cafes.iloc[0])
    cierre(fig,
           f"Fig. 2 · Asociación de cada índice con los resultados: {sig} de {len(tc)} intervalos excluyen el 0 (n = {n} cafés)",
           f"Exploratorio: {n} cafés con ≥{MIN_ENC} encuestas; IC por bootstrap de {N_BOOT:,}".replace(",", ".")
           + " remuestreos de cafés. Asociación, no causalidad.",
           "fig2_correlaciones_spearman.png", carpeta)
 
 
def fig3_dispersion(df, carpeta):
    d = df[df["incluido"]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    outs = [("per_distintivo", "Diferenciación percibida (prom. 1–5)", (2.5, 5.2)),
            ("pct_recuerda", "% que recuerda un dato de origen/historia", (-8, 105)),
            ("prob_volver", "Probabilidad de volver (prom. 1–5)", (2.5, 5.3))]
    for j, (col, lab, yl) in enumerate(outs):
        dispersion(axes[0, j], d, "idx_story", col, C_STORY, "Índice de storytelling (0–6)", lab,
                   ylim=yl, xlim=(-0.5, 7.6), xticks=range(0, 7))
        dispersion(axes[1, j], d, "idx_amb", col, C_AMB, "Índice de ambientación (0–7)", lab,
                   ylim=yl, xlim=(4.6, 7.8), xticks=[5, 6, 7])
    cierre(fig,
           "Fig. 3 · Storytelling (arriba) y ambientación (abajo) frente a diferenciación, recordación y preferencia",
           "Un punto = un café; tamaño = nº de encuestas. Puntos con igual valor se apilan en una etiqueta. "
           "Con 7 cafés, la lectura es exploratoria.",
           "fig3_dispersion_resultados.png", carpeta)
 
 
def fig4_perfil(df, carpeta):
    """Perfil de cada café: ¿cómo le va en posicionamiento y preferencia según cuánto storytelling y ambientación hace?"""
    d = df[df["incluido"]].sort_values(["idx_story", "idx_amb", "cafe_id"],
                                       ascending=[False, False, True]).reset_index(drop=True)
    cols = list(RESULTADOS)
    cmap = LinearSegmentedColormap.from_list("perfil", ["#f3f1ec", "#8d99a4", "#1f2d3a"])
    x_idx = {"idx_story": 0, "idx_amb": 1}
    x_out = {c: 2.8 + j for j, c in enumerate(cols)}

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for r, row in d.iterrows():
        # Índices (prácticas observadas): intensidad proporcional al máximo posible
        for c, col, mx in (("idx_story", C_STORY, 6), ("idx_amb", C_AMB, 7)):
            ax.add_patch(Rectangle((x_idx[c] - 0.5, r - 0.5), 1, 1, facecolor=col,
                                   alpha=0.25 + 0.75 * row[c] / mx, edgecolor=SURFACE, lw=2.5))
            ax.text(x_idx[c], r, f"{int(row[c])}", ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white" if row[c] / mx > 0.45 else INK)
        # Resultados: color = posición relativa del café en esa columna; número = valor real
        for c in cols:
            lo, hi = d[c].min(), d[c].max()
            z = 0.5 if hi == lo else (row[c] - lo) / (hi - lo)
            ax.add_patch(Rectangle((x_out[c] - 0.5, r - 0.5), 1, 1, facecolor=cmap(z),
                                   edgecolor=SURFACE, lw=2.5))
            txt = f"{row[c]:.0f}%" if c.startswith("pct") else coma(row[c], 1 if c == "nps" else 2)
            ax.text(x_out[c], r, txt, ha="center", va="center", fontsize=10.5,
                    color="white" if z > 0.55 else INK)

    ax.set_xticks(list(x_idx.values()) + list(x_out.values()))
    ax.set_xticklabels(["Índice\nstorytelling\n(0–6)", "Índice\nambientación\n(0–7)"]
                       + [RESULTADOS[c][0].replace(" de ", " de\n") for c in cols], fontsize=9.5)
    for t, col in zip(ax.get_xticklabels()[:2], (C_STORY, C_AMB)):
        t.set_color(col)
        t.set_fontweight("bold")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.corto}  (n = {r.n_enc})" for r in d.itertuples()], fontsize=10.5)
    ax.set_xlim(-0.6, max(x_out.values()) + 0.6)
    ax.set_ylim(len(d) - 0.5, -0.5)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cierre(fig,
           "Fig. 4 · Perfil de cada café (ordenados de más a menos storytelling): ningún café lidera en todos los resultados",
           "Color de los resultados = posición relativa del café en esa columna entre los cafés incluidos (claro = más bajo, oscuro = más alto); "
           "el número es el valor real.\nDiferenciación y prob. de volver: prom. 1–5; recomendación: prom. 0–10; recordación y primera opción: %. "
           "Descriptivo, sin pruebas de hipótesis.",
           "fig4_perfil_por_cafe.png", carpeta)



def fig5_validacion(df, carpeta):
    """¿Lo que se observó en el local coincide con lo que percibe el cliente?"""
    d = df[df["incluido"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    dispersion(axes[0], d, "idx_story", "pct_origen", C_STORY, "Índice de storytelling (0–6)",
               "% a quien le explicaron el origen del café", ylim=(-5, 108), xlim=(-0.5, 7.6), xticks=range(0, 7))
    dispersion(axes[1], d, "idx_story", "per_historia", C_STORY, "Índice de storytelling (0–6)",
               "La historia le resulta interesante (prom. 1–5)", ylim=(1.5, 4.6), xlim=(-0.5, 7.6), xticks=range(0, 7))
    dispersion(axes[2], d, "idx_amb", "per_ambiente", C_AMB, "Índice de ambientación (0–7)",
               "El ambiente es agradable (prom. 1–5)", ylim=(3.5, 5.2), xlim=(4.6, 7.8), xticks=[5, 6, 7])
    cierre(fig,
           "Fig. 5 · Validación: lo que observamos en el local frente a lo que percibe el cliente",
           "Si los índices miden algo real, deberían asociarse con la percepción correspondiente (origen/historia ↔ storytelling; "
           "ambiente ↔ ambientación). Un punto = un café; tamaño = nº de encuestas.",
           "fig5_validacion_percepcion.png", carpeta)
 
 
def fig6_digital(df, carpeta):
    """Experiencia física vs. presencia digital (Instagram / Google Maps)."""
    d = df[df["incluido"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    dispersion(axes[0], df, "google_resenas", "idx_story", C_STORY, "Reseñas en Google Maps (escala log)",
               "Índice de storytelling (0–6)", ylim=(-0.8, 7), logx=True)
    # En el primer panel se incluyen todos los cafés, no solo los que superan MIN_ENC.
    for txt in axes[0].texts:
        if txt.get_text().startswith("ρ ="):
            txt.set_text(txt.get_text().replace(f"n = {len(df)}", f"n = {len(df)}, todos"))
    dispersion(axes[1], d, "idx_story", "pct_digital", C_STORY, "Índice de storytelling (0–6)",
               "% de clientes que llegó por redes o Google Maps", ylim=(-8, 60), xlim=(-0.5, 7.6), xticks=range(0, 7))
    dispersion(axes[2], d, "google_resenas", "pct_digital", INK2, "Reseñas en Google Maps (escala log)",
               "% de clientes que llegó por redes o Google Maps", ylim=(-8, 60), logx=True)
    cierre(fig,
           "Fig. 6 · Experiencia física vs. presencia digital: ¿los cafés con más storytelling son más visibles en línea?",
           "Presencia digital = reseñas en Google Maps (dato público) y % de clientes que conoció el café por redes sociales o Google Maps (encuesta). "
           "Un punto = un café; tamaño = nº de encuestas.",
           "fig6_presencia_digital.png", carpeta)
 
 
# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Genera las 6 gráficas del análisis de cafeterías.")
    ap.add_argument("--datos", help="Ruta al Excel de la base de datos")
    ap.add_argument("--salida", default="figuras", help="Carpeta de salida (por defecto: figuras)")
    a = ap.parse_args()
 
    ruta = encontrar_excel(a.datos)
    os.makedirs(a.salida, exist_ok=True)
    print(f"Leyendo: {ruta}")
    ficha, enc = cargar(ruta)
    df = resumen_por_cafe(ficha, enc)
    print(f"  {len(ficha)} cafés con ficha · {len(enc)} encuestas · "
          f"{int(df.incluido.sum())} cafés con ≥{MIN_ENC} encuestas (usados en las asociaciones)")
 
    # tablas de respaldo para trazabilidad
    df.to_csv(os.path.join(a.salida, "resumen_por_cafe.csv"), index=False, encoding="utf-8-sig")
    tc = tabla_correlaciones(df)
    tc.drop(columns="col").to_csv(os.path.join(a.salida, "correlaciones_spearman.csv"), index=False, encoding="utf-8-sig")
 
    print("Generando gráficas…")
    fig1_practicas(ficha, df, a.salida)
    fig2_forest(tc, a.salida)
    fig3_dispersion(df, a.salida)
    fig4_perfil(df, a.salida)
    fig5_validacion(df, a.salida)
    fig6_digital(df, a.salida)
    print("Listo.")
 
 
if __name__ == "__main__":
    main()