#!/usr/bin/env python3
"""Variante del QHRP: comparar los activos dia a dia en vez de promediar primero.

QHRP original (qhrp_core / analisis_sp500.py):
    rho_i = media_t |psi_i(t)><psi_i(t)|        d_ij = Frobenius(rho_i, rho_j)
  Al promediar antes de comparar se pierde que dia es cual, asi que la distancia
  no ve si dos activos se mueven a la vez.

Variante (este archivo):
    d_ij = media_t Frobenius(|psi_i(t)><psi_i(t)|, |psi_j(t)><psi_j(t)|)
         = media_t 0.5 * sqrt(2 * (1 - |<psi_i(t)|psi_j(t)>|^2))
  Se comparan los estados de los dos activos el mismo dia. Mismo circuito,
  mismas 6 features y mismo enlace de Ward que el original.

--escala minmax    angulos escalados entre el min y el max de cada feature (como el original)
--escala percentil angulos segun el percentil de cada valor en su serie (no depende de los extremos)

Uso:
    python qhrp_dia_a_dia.py --panel sectores
    python qhrp_dia_a_dia.py --panel acciones --escala percentil
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

from analisis_sp500 import (
    group_table,
    load_panel,
    plot_fig3,
    plot_fig3_sector_runs,
    random_purity,
    same_sector_neighbours,
    sector_runs,
    weighted_purity,
)
from qhrp_core import classical_linkage, quantum_linkage, rolling_features, states_from_theta

SCRIPT_DIR = Path(__file__).resolve().parent


##############################################
# Angulos del feature-map
##############################################
def feature_angles(features: np.ndarray, escala: str, alpha: float) -> np.ndarray:
    """(T, P) features de un activo -> (T, P) angulos."""
    if escala == "minmax":
        x_min, x_max = features.min(axis=0), features.max(axis=0)
        span = np.where(x_max > x_min, x_max - x_min, 1.0)
        return np.where(x_max > x_min, alpha * math.pi * (features - x_min) / span, 0.0)
    if escala == "percentil":
        ranks = stats.rankdata(features, axis=0)
        return alpha * math.pi * (ranks - 1) / max(len(features) - 1, 1)
    raise ValueError(f"escala desconocida: {escala}")


##############################################
# Distancia dia a dia
##############################################
def daily_distance_matrix(angles: list, block: int = 32) -> np.ndarray:
    """angles: lista de (T, P) por activo. Devuelve la media temporal de la distancia de Frobenius diaria."""
    n_assets, t_total = len(angles), angles[0].shape[0]
    d = np.zeros((n_assets, n_assets))
    for start in range(0, t_total, block):
        stop = min(start + block, t_total)
        states = np.stack([states_from_theta(a[start:stop]) for a in angles])  # (N, B, 64)
        per_day = states.transpose(1, 0, 2)  # (B, N, 64)
        overlap = per_day @ per_day.conj().transpose(0, 2, 1)  # (B, N, N)
        fidelity = np.clip(np.abs(overlap) ** 2, 0.0, 1.0)
        d += (0.5 * np.sqrt(2.0 * (1.0 - fidelity))).sum(axis=0)
    d /= t_total
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


##############################################
# CLI
##############################################
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QHRP comparando los activos dia a dia.")
    parser.add_argument("--panel", choices=["sectores", "acciones"], default="sectores")
    parser.add_argument("--escala", choices=["minmax", "percentil"], default="minmax")
    parser.add_argument("--k", type=int, default=None, help="Numero de grupos (por defecto 3 sectores / 11 acciones)")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "resultados_dia_a_dia")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k = args.k or (3 if args.panel == "sectores" else 11)
    out = args.out_dir / f"{args.panel}_{args.escala}"
    out.mkdir(parents=True, exist_ok=True)

    returns, labels, sectors = load_panel(args.panel)
    n = len(labels)
    r = returns.to_numpy()
    corr = returns.corr().to_numpy()
    print(f"Panel {args.panel}: {n} activos, {len(returns)} dias | escala {args.escala}")

    _, order_c, d_c = classical_linkage(corr)

    t0 = time.perf_counter()
    angles = [feature_angles(rolling_features(r[:, j]), args.escala, args.alpha) for j in range(n)]
    d_dia = daily_distance_matrix(angles)
    z_dia, order_dia = quantum_linkage(d_dia)
    print(f"Distancias dia a dia: {time.perf_counter() - t0:.1f}s")
    pd.DataFrame(d_dia, index=labels, columns=labels).to_csv(out / "distancias_dia_a_dia.csv")

    # Cuanto se parece la nueva distancia a la de correlacion del HRP clasico (Spearman sobre los pares)
    rho_corr = stats.spearmanr(squareform(d_dia, checks=False), squareform(d_c, checks=False)).correlation
    print(f"Parecido con la distancia de correlacion (Spearman): {rho_corr:.2f}")

    plot_fig3(corr, order_c, order_dia, labels, sectors, out / "fig3_correlacion_ordenada.png")

    if args.panel == "sectores":
        lines = [f"Orden QHRP dia a dia: {[labels[i] for i in order_dia]}"]
        for kk in range(2, 6):
            tab = group_table(fcluster(z_dia, t=kk, criterion="maxclust"), order_dia, labels, None)
            lines.append(f"  k={kk}: " + "   ".join(f"[{a}]" for a in tab["activos"]))
        text = "\n".join(lines)
        (out / "grupos.txt").write_text(text + "\n", encoding="utf-8")
        print(text)
        return

    # QHRP original, leido de los resultados de analisis_sp500.py
    d_orig = pd.read_csv(SCRIPT_DIR / "resultados" / "acciones" / "distancias_cuanticas.csv", index_col=0).to_numpy()
    _, order_orig = quantum_linkage(d_orig)

    methods = [("HRP clasico", order_c), ("QHRP original", order_orig), (f"QHRP dia a dia ({args.escala})", order_dia)]
    runs = {name: sector_runs(order, sectors, labels) for name, order in methods}
    plot_fig3_sector_runs(corr, [(name, order, runs[name]) for name, order in methods], out / "fig3_grupos_sector.png")
    pd.concat([df.assign(metodo=name) for name, df in runs.items()]).to_csv(out / "rachas_sector.csv", index=False)

    g_dia = fcluster(z_dia, t=k, criterion="maxclust")
    group_table(g_dia, order_dia, labels, sectors).to_csv(out / f"grupos_k{k}.csv", index=False)

    rng = np.random.default_rng(0)
    azar = np.mean([same_sector_neighbours(list(rng.permutation(n)), sectors) for _ in range(1000)])
    resumen = pd.DataFrame({
        "metodo": [name for name, _ in methods] + ["azar"],
        "vecinos_mismo_sector": [same_sector_neighbours(order, sectors) for _, order in methods] + [azar],
        "rachas_>=4": [len(runs[name]) for name, _ in methods] + [np.nan],
        "empresas_en_rachas": [runs[name]["n_activos"].sum() for name, _ in methods] + [np.nan],
    }).round(3)
    resumen.to_csv(out / "resumen.csv", index=False)
    print("\n" + resumen.to_string(index=False))
    print(f"\nQHRP dia a dia, {k} grupos: pureza ponderada {weighted_purity(g_dia, sectors):.3f} "
          f"(azar {random_purity(g_dia, sectors):.3f}), grupo mas grande {np.bincount(g_dia).max()}")
    print(f"\nResultados en: {out}")


if __name__ == "__main__":
    main()
