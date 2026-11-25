#!/usr/bin/env python3
"""HCAA (Raffinot, 2018, "Hierarchical Clustering-Based Asset Allocation") con QHRP como agrupacion.

Reproduce el esquema del articulo y anade el QHRP como forma de construir el arbol:
  1. Arbol jerarquico de los activos con la ventana de 252 dias anteriores.
     - Clasicos (como el articulo): distancia sqrt(2 (1 - rho)) con enlace simple (SL),
       completo (CL), medio (AL) o Ward (WM).
     - QHRP original: distancia de Frobenius entre matrices de densidad promedio, Ward.
     - QHRP nuevo: distancia cuantica dia a dia con escala por percentiles, Ward y enlace medio.
  2. Numero de grupos con el indice Gap (Tibshirani et al., 2001), minimo 2 grupos.
  3. Pesos: 50 % a cada rama en cada division del arbol hasta llegar a un grupo;
     dentro de cada grupo, pesos iguales.

Metricas del articulo: ASR (Pezier y White), CEQ (gamma = 1), MDD, TO y SSPW, con rf = 0.

Uso:
    python hcaa_sp500.py --panel sectores --paso 1
    python hcaa_sp500.py --panel acciones --paso 21
    python hcaa_sp500.py --panel acciones --paso 1 --solo-clasicos
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kurtosis, skew

from analisis_sp500 import load_panel
from qhrp_core import average_density_matrix, frobenius_distance_matrix, rolling_features
from qhrp_dia_a_dia import daily_distance_matrix, feature_angles

SCRIPT_DIR = Path(__file__).resolve().parent
WINDOW = 252
ALPHA = 2.0
CLASSICAL = {"SL": "single", "CL": "complete", "AL": "average", "WM": "ward"}
QUANTUM = ["QHRP original WM", "QHRP nuevo WM", "QHRP nuevo AL"]

# Datos compartidos con los procesos hijos (se rellenan en main antes de crear el pool)
_R: np.ndarray | None = None
_FEATURES: list | None = None
_OPTS: dict = {}


##############################################
# Indice Gap sobre una matriz de distancias
##############################################
def _embed(d: np.ndarray) -> np.ndarray:
    """Escalado multidimensional clasico: puntos cuyas distancias euclideas reproducen d."""
    n = len(d)
    j = np.eye(n) - 1.0 / n
    b = -0.5 * j @ (d**2) @ j
    vals, vecs = np.linalg.eigh(b)
    keep = vals > 1e-10 * vals.max()
    return vecs[:, keep] * np.sqrt(vals[keep])


def _log_w(d2: np.ndarray, labels: np.ndarray) -> float:
    """log de la dispersion intra-grupo W_k = sum_r (1 / 2 n_r) sum_{i,j en r} d_ij^2."""
    w = 0.0
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        w += d2[np.ix_(idx, idx)].sum() / (2 * len(idx))
    return float(np.log(max(w, 1e-300)))


def gap_number_of_clusters(d: np.ndarray, z: np.ndarray, method: str, k_max: int,
                           n_ref: int, rng: np.random.Generator) -> int:
    """Menor k >= 2 con Gap(k) >= Gap(k+1) - s_{k+1} (Tibshirani et al., 2001, referencia 'b')."""
    ks = list(range(2, k_max + 1))
    d2 = d**2
    log_w = np.array([_log_w(d2, fcluster(z, k, criterion="maxclust")) for k in ks])

    x = _embed(d)
    x = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    xp = x @ vt.T
    lo, hi = xp.min(axis=0), xp.max(axis=0)
    ref = np.zeros((n_ref, len(ks)))
    for b in range(n_ref):
        zr = rng.uniform(lo, hi, size=xp.shape) @ vt
        dr = pdist(zr)
        lr = linkage(dr, method=method)
        dr2 = squareform(dr) ** 2
        ref[b] = [_log_w(dr2, fcluster(lr, k, criterion="maxclust")) for k in ks]

    gap = ref.mean(axis=0) - log_w
    s = ref.std(axis=0) * np.sqrt(1 + 1 / n_ref)
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - s[i + 1]:
            return ks[i]
    return ks[-1]


##############################################
# Pesos HCAA
##############################################
def hcaa_weights(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """50 % a cada rama hasta llegar a un grupo; pesos iguales dentro del grupo."""
    n = len(labels)
    leaves = {i: [i] for i in range(n)}
    for i, (a, b) in enumerate(z[:, :2].astype(int)):
        leaves[n + i] = leaves[a] + leaves[b]
    w = np.zeros(n)
    stack = [(2 * n - 2, 1.0)]
    while stack:
        node, cap = stack.pop()
        members = leaves[node]
        if len(set(labels[members])) == 1:
            w[members] = cap / len(members)
        else:
            a, b = z[node - n, :2].astype(int)
            stack += [(a, cap / 2), (b, cap / 2)]
    return w


##############################################
# Una fecha de rebalanceo
##############################################
def _tree_distances(start: int, end: int) -> dict:
    train = _R[start:end]
    corr = np.corrcoef(train, rowvar=False)
    d_corr = np.sqrt(np.clip(2 * (1 - corr), 0.0, None))
    np.fill_diagonal(d_corr, 0.0)
    out = {name: (d_corr, method) for name, method in CLASSICAL.items()}
    if not _OPTS["solo_clasicos"]:
        feats = [f[start:end] for f in _FEATURES]
        d_orig = frobenius_distance_matrix([average_density_matrix(f, ALPHA) for f in feats])
        d_new = daily_distance_matrix([feature_angles(f, "percentil", ALPHA) for f in feats])
        out["QHRP original WM"] = (d_orig, "ward")
        out["QHRP nuevo WM"] = (d_new, "ward")
        out["QHRP nuevo AL"] = (d_new, "average")
    return out


def run_date(end: int):
    rng = np.random.default_rng(end)
    weights, n_clusters = {}, {}
    for name, (d, method) in _tree_distances(end - WINDOW, end).items():
        z = linkage(squareform(d, checks=False), method=method)
        if _OPTS["k_fijos"]:
            # Mismo numero de grupos para todos los metodos: solo cambia que activos van juntos
            for k in _OPTS["k_fijos"]:
                weights[f"{name} k={k}"] = hcaa_weights(z, fcluster(z, k, criterion="maxclust"))
                n_clusters[f"{name} k={k}"] = k
        else:
            k = gap_number_of_clusters(d, z, method, _OPTS["k_max"], _OPTS["n_ref"], rng)
            weights[name] = hcaa_weights(z, fcluster(z, k, criterion="maxclust"))
            n_clusters[name] = k
    return end, weights, n_clusters


##############################################
# Metricas del articulo (rf = 0, anualizadas con 252 dias)
##############################################
def metrics(r: np.ndarray, w_hist: np.ndarray) -> dict:
    sr = r.mean() / r.std(ddof=1) * np.sqrt(252)
    asr = sr * (1 + skew(r) / 6 * sr - (kurtosis(r, fisher=False) - 3) / 24 * sr**2)
    ceq = (r.mean() * 252 - 0.5 * r.var(ddof=1) * 252) * 100
    wealth = np.cumprod(1 + r)
    mdd = -(wealth / np.maximum.accumulate(wealth) - 1).min() * 100
    to = np.abs(np.diff(w_hist, axis=0)).sum(axis=1).mean() * 100 if len(w_hist) > 1 else np.nan
    sspw = (w_hist**2).sum(axis=1).mean()
    return {"ASR": asr, "Sharpe": sr, "CEQ": ceq, "MDD": mdd, "TO": to, "SSPW": sspw}


##############################################
# CLI
##############################################
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HCAA de Raffinot con agrupaciones clasicas y QHRP.")
    parser.add_argument("--panel", choices=["sectores", "acciones"], default="sectores")
    parser.add_argument("--paso", type=int, default=1, help="Dias entre rebalanceos (el articulo usa 1)")
    parser.add_argument("--solo-clasicos", action="store_true")
    parser.add_argument("--k-max", type=int, default=None, help="Maximo de grupos para el Gap (def. 9 / 30)")
    parser.add_argument("--n-ref", type=int, default=10, help="Muestras de referencia del indice Gap")
    parser.add_argument("--k-fijos", type=str, default=None,
                        help="Numeros de grupos fijos separados por comas (p. ej. 2,3,4,5) en vez del indice Gap")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "resultados_hcaa")
    return parser.parse_args()


def main() -> None:
    global _R, _FEATURES, _OPTS
    args = parse_args()
    returns, labels, _ = load_panel(args.panel)
    _R = returns.to_numpy()
    n = _R.shape[1]
    _OPTS = {"solo_clasicos": args.solo_clasicos, "k_max": args.k_max or min(n - 1, 30 if n > 30 else 9),
             "n_ref": args.n_ref,
             "k_fijos": [int(k) for k in args.k_fijos.split(",")] if args.k_fijos else None}
    if not args.solo_clasicos:
        _FEATURES = [rolling_features(_R[:, j]) for j in range(n)]

    tag = f"{args.panel}_paso{args.paso}" + ("_clasicos" if args.solo_clasicos else "") \
        + (f"_k{args.k_fijos.replace(',', '-')}" if args.k_fijos else "")
    out = args.out_dir / tag
    out.mkdir(parents=True, exist_ok=True)

    ends = list(range(WINDOW, len(_R), args.paso))
    print(f"Panel {args.panel}: {n} activos | {len(ends)} rebalanceos (ventana {WINDOW}, cada {args.paso} dias) "
          f"| k_max {_OPTS['k_max']} | {args.workers} procesos", flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(run_date, ends, chunksize=4 if args.paso == 1 else 1), start=1):
            results.append(res)
            if i % max(1, len(ends) // 20) == 0 or i == len(ends):
                print(f"  {i}/{len(ends)} rebalanceos ({time.time() - t0:.0f}s)", flush=True)

    methods = list(results[0][1]) + ["EW"]
    oos = {m: [] for m in methods}
    w_hist = {m: [] for m in methods}
    for end, weights, _ in results:
        stop = min(end + args.paso, len(_R))
        weights = {**weights, "EW": np.full(n, 1.0 / n)}
        for m in methods:
            oos[m].append(_R[end:stop] @ weights[m])
            w_hist[m].append(weights[m])

    tabla = pd.DataFrame({m: metrics(np.concatenate(oos[m]), np.array(w_hist[m])) for m in methods}).T
    tabla["grupos_medio"] = [np.mean([res[2][m] for res in results]) if m != "EW" else np.nan for m in methods]
    tabla = tabla.round(4)
    tabla.to_csv(out / "metricas.csv")

    index = returns.index[ends[0]:]
    oos_df = pd.DataFrame({m: np.concatenate(oos[m]) for m in methods}, index=index[: len(np.concatenate(oos["EW"]))])
    oos_df.to_csv(out / "retornos_oos.csv")
    pd.DataFrame({m: [res[2][m] for res in results] for m in methods if m != "EW"},
                 index=returns.index[ends]).to_csv(out / "numero_grupos.csv")

    if _OPTS["k_fijos"]:
        base = tabla.drop(index="EW").copy()
        base["metodo"] = [i.rsplit(" k=", 1)[0] for i in base.index]
        base["k"] = [int(i.rsplit(" k=", 1)[1]) for i in base.index]
        for col in ("ASR", "Sharpe", "CEQ", "MDD"):
            pivot = base.pivot(index="metodo", columns="k", values=col)
            pivot.to_csv(out / f"resumen_{col}.csv")
            print(f"\n== {col} por metodo y numero de grupos (EW: {tabla.loc['EW', col]:.4f})")
            print(pivot.round(4).to_string())
        print(f"\nResultados en: {out}")
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    wealth = (1 + oos_df).cumprod()
    for m in methods:
        quantum = m.startswith("QHRP")
        ax.plot(wealth.index, wealth[m], lw=2.2 if quantum else 1.1, ls="-" if quantum or m == "EW" else "--",
                label=f"{m} (ASR {tabla.loc[m, 'ASR']:.3f})")
    ax.set_yscale("log")
    ax.set_ylabel("Valor de 1 USD invertido (escala log)")
    ax.set_title(f"HCAA fuera de muestra: {args.panel} del S&P 500, ventana {WINDOW} dias, "
                 f"rebalanceo cada {args.paso} dia(s)")
    ax.grid(color="#dddddd", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "evolucion_cartera.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n" + tabla.to_string())
    print(f"\nResultados en: {out}")


if __name__ == "__main__":
    main()
