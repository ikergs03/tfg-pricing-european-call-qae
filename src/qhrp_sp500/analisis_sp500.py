#!/usr/bin/env python3
"""QHRP vs HRP clasico sobre los paneles del S&P 500 (data/datos_raffinot), con todo el historico.

Mismo analisis que src/qhrp_figure3_4/fig3_4.ipynb:
  1. Fig. 3: matriz de correlacion sin ordenar / ordenada por HRP clasico / ordenada por QHRP.
  2. Fig. 4: matriz de distancias cuanticas sin ordenar / ordenada por QHRP.
  3. Grupos: se corta cada arbol en k grupos (fcluster maxclust) y, si hay
     metadatos de sector, se calcula pureza y entropia sectorial por grupo,
     el solapamiento (Jaccard) entre grupos clasicos y cuanticos y los activos
     que cambian de grupo.

Uso:
    python analisis_sp500.py --panel sectores
    python analisis_sp500.py --panel acciones --k 11
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster

from qhrp_core import (
    average_density_matrix,
    classical_linkage,
    frobenius_distance_matrix,
    quantum_linkage,
    rolling_features,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data" / "datos_raffinot"
PANELS = {
    "sectores": DATA_DIR / "sectors_10" / "v1" / "returns.csv",
    "acciones": DATA_DIR / "stocks_357_candidate" / "v1" / "returns.csv",
}
META_ACCIONES = SCRIPT_DIR / "datos" / "asset_metadata_357.csv"  # sector de cada accion (obtener_sectores.py)


##############################################
# Datos
##############################################
def load_panel(panel: str):
    returns = pd.read_csv(PANELS[panel], index_col="date", parse_dates=True).drop(columns=["previous_date"])
    if returns.isna().any().any():
        raise ValueError("El panel tiene huecos")
    if panel == "acciones":
        meta = pd.read_csv(META_ACCIONES, dtype=str).set_index("ticker").loc[returns.columns]
        labels = [n.split(" DEAD")[0].title() for n in meta["name"]]
        sectors = meta["sector"].tolist()
    else:
        labels = returns.columns.tolist()
        sectors = None
    return returns, labels, sectors


##############################################
# Grupos, pureza y comparacion
##############################################
def normalized_entropy(values: pd.Series) -> float:
    counts = values.value_counts()
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def group_table(groups: np.ndarray, order: list, labels: list, sectors: list | None) -> pd.DataFrame:
    """Una fila por grupo, numerados en el orden en que aparecen en la matriz ordenada."""
    rank = {g: r + 1 for r, g in enumerate(dict.fromkeys(groups[order]))}
    rows = []
    for g, r in rank.items():
        members = [i for i in order if groups[i] == g]
        row = {"grupo": r, "n_activos": len(members)}
        if sectors is not None:
            s = pd.Series([sectors[i] for i in members])
            row.update(
                sector_dominante=s.value_counts().index[0],
                pureza=round(s.value_counts().iloc[0] / len(s), 4),
                entropia=round(normalized_entropy(s), 4),
                n_sectores=s.nunique(),
            )
        row["activos"] = " | ".join(labels[i] for i in members)
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_purity(groups: np.ndarray, sectors: list) -> float:
    df = pd.DataFrame({"g": groups, "s": sectors})
    return float(df.groupby("g")["s"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df))


def random_purity(groups: np.ndarray, sectors: list, n_iter: int = 1000, seed: int = 0) -> float:
    """Pureza ponderada esperada si los activos se repartiesen al azar en grupos de los mismos tamanos."""
    rng = np.random.default_rng(seed)
    return float(np.mean([weighted_purity(groups, list(rng.permutation(sectors))) for _ in range(n_iter)]))


def jaccard_table(g_class: np.ndarray, g_quant: np.ndarray, rank_c: dict, rank_q: dict) -> pd.DataFrame:
    rows = []
    for c in np.unique(g_class):
        sc = set(np.where(g_class == c)[0])
        for q in np.unique(g_quant):
            sq = set(np.where(g_quant == q)[0])
            inter = len(sc & sq)
            if inter:
                rows.append({
                    "grupo_clasico": rank_c[c], "grupo_cuantico": rank_q[q], "interseccion": inter,
                    "jaccard": round(inter / len(sc | sq), 4), "tam_clasico": len(sc), "tam_cuantico": len(sq),
                })
    return pd.DataFrame(rows).sort_values(["jaccard", "interseccion"], ascending=False)


##############################################
# Figuras
##############################################
def _sector_strip(ax, order: list, sectors: list, palette: dict) -> None:
    colors = [palette[sectors[i]] for i in order]
    for k, c in enumerate(colors):
        ax.add_patch(plt.Rectangle((k, -0.025 * len(order)), 1, 0.02 * len(order), color=c, clip_on=False))


def plot_fig3(corr, order_c, order_q, labels, sectors, path: Path) -> None:
    n = len(labels)
    show = n <= 40
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(2, 2)
    panels = [
        (fig.add_subplot(gs[0, :]), list(range(n)), "(a) Matriz de correlacion (sin ordenar)"),
        (fig.add_subplot(gs[1, 0]), order_c, "(b) Ordenacion HRP clasica"),
        (fig.add_subplot(gs[1, 1]), order_q, "(c) Ordenacion HRP cuantica"),
    ]
    palette = None
    if sectors is not None:
        palette = dict(zip(sorted(set(sectors)), sns.color_palette("tab20", len(set(sectors)))))
    for ax, order, title in panels:
        lab = [labels[i] for i in order] if show else False
        sns.heatmap(corr[np.ix_(order, order)], cmap="coolwarm", vmin=-0.2, vmax=1.0, square=True,
                    xticklabels=lab, yticklabels=lab, ax=ax)
        ax.set_title(title, fontsize=12, pad=22 if palette is not None else 6)
        if palette is not None:
            _sector_strip(ax, order, sectors, palette)
    if palette is not None:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
        fig.legend(handles, palette.keys(), loc="lower center", ncol=6, fontsize=9, title="Sector (franja bajo cada matriz)")
    fig.tight_layout(rect=(0, 0.05 if palette else 0, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fig4(d_q, order_q, labels, path: Path) -> None:
    n = len(labels)
    show = n <= 40
    off = d_q[np.triu_indices_from(d_q, 1)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, order, title in (
        (axes[0], list(range(n)), "(a) Matriz de distancia cuantica (sin ordenar)"),
        (axes[1], order_q, "(b) Matriz de distancia cuantica (ordenacion QHRP)"),
    ):
        lab = [labels[i] for i in order] if show else False
        sns.heatmap(d_q[np.ix_(order, order)], cmap="coolwarm", vmin=0, vmax=off.max(), square=True,
                    xticklabels=lab, yticklabels=lab, ax=ax)
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


SECTOR_ES = {
    "Basic Materials": "Materiales", "Communication Services": "Comunicaciones",
    "Consumer Cyclical": "Consumo ciclico", "Consumer Defensive": "Consumo basico", "Energy": "Energia",
    "Financial Services": "Finanzas", "Healthcare": "Salud", "Industrials": "Industria",
    "Real Estate": "Inmobiliario", "Technology": "Tecnologia", "Utilities": "Servicios publicos",
}


def sector_runs(order: list, sectors: list, labels: list, min_len: int = 4) -> pd.DataFrame:
    """Rachas de activos seguidos del mismo sector en el orden (los 'cuadrados' de la Fig. 3)."""
    seq = [sectors[i] for i in order]
    rows, start = [], 0
    for k in range(1, len(seq) + 1):
        if k == len(seq) or seq[k] != seq[start]:
            if k - start >= min_len:
                rows.append({"desde": start, "hasta": k, "n_activos": k - start, "sector": seq[start],
                             "activos": " | ".join(labels[i] for i in order[start:k])})
            start = k
    return pd.DataFrame(rows, columns=["desde", "hasta", "n_activos", "sector", "activos"])


def same_sector_neighbours(order: list, sectors: list) -> float:
    seq = np.array([sectors[i] for i in order])
    return float((seq[1:] == seq[:-1]).mean())


def plot_fig3_sector_runs(corr, methods: list, path: Path) -> None:
    """methods: [(titulo, order, runs)]. Fig. 3 ordenada con las rachas del mismo sector recuadradas y nombradas."""
    fig, axes = plt.subplots(1, len(methods), figsize=(9 * len(methods), 8.5))
    for ax, (title, order, runs) in zip(axes, methods):
        n = len(order)
        sns.heatmap(corr[np.ix_(order, order)], cmap="coolwarm", vmin=-0.2, vmax=1.0, square=True,
                    xticklabels=False, yticklabels=False, ax=ax, cbar=False)
        min_gap = n / 30  # separacion minima entre etiquetas, en filas
        last_y = -np.inf
        for row in runs.itertuples():
            ax.add_patch(plt.Rectangle((row.desde, row.desde), row.n_activos, row.n_activos,
                                       fill=False, ec="black", lw=1.4))
            y_box = (row.desde + row.hasta) / 2
            y_lab = max(y_box, last_y + min_gap)
            last_y = y_lab
            ax.annotate(f"{SECTOR_ES.get(row.sector, row.sector)} ({row.n_activos})",
                        xy=(n, y_box), xytext=(n * 1.03, y_lab), fontsize=9, va="center",
                        annotation_clip=False,
                        arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.6})
        ax.set_title(f"{title}\n{len(runs)} rachas de >=4 empresas seguidas del mismo sector "
                     f"({runs['n_activos'].sum()} de {n} empresas)", fontsize=11)
    fig.tight_layout(w_pad=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


##############################################
# CLI
##############################################
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QHRP vs HRP clasico en los paneles S&P 500.")
    parser.add_argument("--panel", choices=sorted(PANELS), default="sectores")
    parser.add_argument("--k", type=int, default=None, help="Numero de grupos (por defecto 3 sectores / 11 acciones)")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "resultados")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k = args.k or (3 if args.panel == "sectores" else 11)
    out = args.out_dir / args.panel
    out.mkdir(parents=True, exist_ok=True)

    returns, labels, sectors = load_panel(args.panel)
    n = len(labels)
    print(f"Panel {args.panel}: {n} activos, {len(returns)} dias ({returns.index[0].date()} -> {returns.index[-1].date()})")

    # HRP clasico
    corr = returns.corr().to_numpy()
    z_c, order_c, _ = classical_linkage(corr)

    # QHRP
    t0 = time.perf_counter()
    r = returns.to_numpy()
    rhos = [average_density_matrix(rolling_features(r[:, j]), alpha=args.alpha) for j in range(n)]
    d_q = frobenius_distance_matrix(rhos)
    z_q, order_q = quantum_linkage(d_q)
    print(f"Matrices de densidad + distancias: {time.perf_counter() - t0:.1f}s")

    pd.DataFrame(d_q, index=labels, columns=labels).to_csv(out / "distancias_cuanticas.csv")
    plot_fig3(corr, order_c, order_q, labels, sectors, out / "fig3_correlacion_ordenada.png")
    plot_fig4(d_q, order_q, labels, out / "fig4_distancia_cuantica.png")

    if args.panel == "sectores":
        lines = []
        for name, z, order in (("HRP clasico", z_c, order_c), ("QHRP", z_q, order_q)):
            lines.append(f"== {name}\nOrden: {[labels[i] for i in order]}")
            for kk in range(2, 6):
                g = fcluster(z, t=kk, criterion="maxclust")
                tab = group_table(g, order, labels, None)
                lines.append(f"  k={kk}: " + "   ".join(f"[{a}]" for a in tab["activos"]))
        text = "\n".join(lines)
        (out / "grupos.txt").write_text(text + "\n", encoding="utf-8")
        print("\n" + text)
    else:
        g_c = fcluster(z_c, t=k, criterion="maxclust")
        g_q = fcluster(z_q, t=k, criterion="maxclust")
        tab_c = group_table(g_c, order_c, labels, sectors)
        tab_q = group_table(g_q, order_q, labels, sectors)
        tab_c.to_csv(out / f"grupos_clasico_k{k}.csv", index=False)
        tab_q.to_csv(out / f"grupos_cuantico_k{k}.csv", index=False)

        rank_c = {g: r + 1 for r, g in enumerate(dict.fromkeys(g_c[order_c]))}
        rank_q = {g: r + 1 for r, g in enumerate(dict.fromkeys(g_q[order_q]))}
        jac = jaccard_table(g_c, g_q, rank_c, rank_q)
        jac.to_csv(out / f"jaccard_clasico_vs_cuantico_k{k}.csv", index=False)

        resumen = pd.DataFrame({
            "metodo": ["HRP clasico", "QHRP"],
            "n_grupos": [len(tab_c), len(tab_q)],
            "grupo_mas_grande": [tab_c["n_activos"].max(), tab_q["n_activos"].max()],
            "pureza_media": [tab_c["pureza"].mean(), tab_q["pureza"].mean()],
            "pureza_ponderada": [weighted_purity(g_c, sectors), weighted_purity(g_q, sectors)],
            "pureza_azar": [random_purity(g_c, sectors), random_purity(g_q, sectors)],
            "entropia_media": [tab_c["entropia"].mean(), tab_q["entropia"].mean()],
        }).round(4)
        resumen.to_csv(out / f"resumen_k{k}.csv", index=False)

        # Rachas del mismo sector en el orden (lo que usa la biseccion recursiva)
        runs_c = sector_runs(order_c, sectors, labels)
        runs_q = sector_runs(order_q, sectors, labels)
        runs_c.assign(metodo="HRP clasico").to_csv(out / "rachas_sector_clasico.csv", index=False)
        runs_q.assign(metodo="QHRP").to_csv(out / "rachas_sector_cuantico.csv", index=False)
        plot_fig3_sector_runs(corr, [("(a) HRP clasico", order_c, runs_c), ("(b) HRP cuantico", order_q, runs_q)],
                              out / "fig3_grupos_sector.png")
        rng = np.random.default_rng(0)
        azar = np.mean([same_sector_neighbours(list(rng.permutation(n)), sectors) for _ in range(1000)])
        print("\n== Vecinos del mismo sector en el orden")
        print(f"HRP clasico {same_sector_neighbours(order_c, sectors):.2f} | "
              f"QHRP {same_sector_neighbours(order_q, sectors):.2f} | azar {azar:.2f}")

        pd.set_option("display.width", 200)
        for name, tab in (("HRP clasico", tab_c), ("QHRP", tab_q)):
            print(f"\n== Grupos {name} (k={k})")
            print(tab.drop(columns="activos").to_string(index=False))
        print("\n== Resumen")
        print(resumen.to_string(index=False))
        print("\n== Mayores coincidencias clasico-cuantico (Jaccard)")
        print(jac.head(10).to_string(index=False))

    print(f"\nResultados en: {out}")


if __name__ == "__main__":
    main()
