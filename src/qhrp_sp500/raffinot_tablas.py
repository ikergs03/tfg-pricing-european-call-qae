#!/usr/bin/env python3
"""Tablas 2 y 4 de Raffinot (2018) con todos sus metodos y, al final, los QHRP.

Mismo esquema que el articulo:
  - Ventana movil de 252 dias y rebalanceo diario (--paso 1).
  - Metodos de presupuesto de riesgo (Roncalli, 2013), con la covarianza muestral de la ventana:
      MV   minima varianza (solo largos)
      MDP  cartera mas diversificada (Choueifaty): minima varianza sobre la matriz de
           correlacion y despues w_i proporcional a y_i / sigma_i
      ERC  igual contribucion al riesgo, b_i = 1/N (descenso por coordenadas, CCD)
      IVRB presupuestos b_i proporcionales a sigma_i^-1 (CCD). La formula (15) del articulo
           pone sigma^-2, pero sus notas lo llaman "inverse-volatility" y solo sigma^-1
           reproduce su tabla 2 (ASR 0,428, MDD 49,0, TO 0,474, SSPW 0,121)
  - Metodos de agrupacion jerarquica (HCAA): distancia sqrt(2 (1 - rho)) con enlace
    simple (SL), completo (CL), medio (AL) o Ward (WM); numero de grupos con el indice Gap.
  - QHRP como agrupacion dentro del mismo HCAA: original (Ward) y nuevo (Ward y medio).
  - Criterios: ASR, CEQ (gamma = 1), MDD, TO y SSPW; marcas MCS a (p >= 0,20) y b (p >= 0,70).

  - DBHT (Musmeci et al., 2015) con la implementacion de Riskfolio-Lib, que decide por si
    misma los grupos y la jerarquia; los pesos, igual que en los demas metodos HCAA.

El indice Gap se evalua desde k = 1, pero, como dice el articulo ("we consider at least two
clusters"), si elige un unico grupo se usan dos; lo mismo con el DBHT.

Requiere riskfolio-lib, instalado en su propio entorno (raffinot-env) para no cambiar
las versiones de numpy y pandas de qiskit-env.

Uso (desde src/qhrp_sp500):
    ../../raffinot-env/bin/python raffinot_tablas.py --panel sectores
    ../../raffinot-env/bin/python raffinot_tablas.py --panel acciones --paso 21
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

from riskfolio.src.DBHT import DBHTs

from analisis_sp500 import load_panel
from hcaa_sp500 import _embed, hcaa_weights, metrics
from mcs_hcaa import run_mcs, stationary_bootstrap_indices
from qhrp_core import average_density_matrix, frobenius_distance_matrix, rolling_features
from qhrp_dia_a_dia import daily_distance_matrix, feature_angles

SCRIPT_DIR = Path(__file__).resolve().parent
WINDOW = 252
ALPHA = 2.0
RISK_BASED = ["MV", "MDP", "ERC", "IVRB"]
CLASSICAL = {"SL": "single", "CL": "complete", "AL": "average", "WM": "ward"}
QUANTUM = ["QHRP original WM", "QHRP nuevo WM", "QHRP nuevo AL"]
ORDER = ["EW"] + RISK_BASED + list(CLASSICAL) + ["DBHT"] + QUANTUM

_R: np.ndarray | None = None
_FEATURES: list | None = None
_OPTS: dict = {}


##############################################
# Presupuesto de riesgo
##############################################
def _project_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


def min_variance_long_only(cov: np.ndarray, iters: int = 5000, tol: float = 1e-12) -> np.ndarray:
    """min w' S w con sum(w) = 1, w >= 0, por gradiente proyectado acelerado (FISTA)."""
    n = len(cov)
    step = 1.0 / (2.0 * np.linalg.eigvalsh(cov)[-1])
    w = y = np.full(n, 1.0 / n)
    t = 1.0
    for _ in range(iters):
        w_new = _project_simplex(y - step * 2.0 * cov @ y)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = w_new + (t - 1) / t_new * (w_new - w)
        if np.abs(w_new - w).sum() < tol:
            w = w_new
            break
        w, t = w_new, t_new
    return w


def risk_budgeting(cov: np.ndarray, budgets: np.ndarray, sweeps: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """Contribuciones al riesgo proporcionales a budgets (descenso por coordenadas, Griveau-Billion et al.)."""
    diag = np.diag(cov)
    y = 1.0 / np.sqrt(diag)
    for _ in range(sweeps):
        y_old = y.copy()
        for i in range(len(y)):
            c = cov[i] @ y - diag[i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4 * diag[i] * budgets[i])) / (2 * diag[i])
        if np.abs(y - y_old).max() < tol * y.max():
            break
    return y / y.sum()


def risk_based_weights(train: np.ndarray) -> dict:
    cov = np.cov(train, rowvar=False)
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)
    y = min_variance_long_only(corr) / sigma
    n = len(cov)
    return {
        "MV": min_variance_long_only(cov),
        "MDP": y / y.sum(),
        "ERC": risk_budgeting(cov, np.full(n, 1.0 / n)),
        "IVRB": ivrb_weights(cov),
    }


def ivrb_weights(cov: np.ndarray) -> np.ndarray:
    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    return risk_budgeting(cov, inv_vol / inv_vol.sum())


##############################################
# Indice Gap (Tibshirani et al., 2001), admitiendo k = 1
##############################################
def _log_w_all(d2: np.ndarray, z: np.ndarray, ks: list) -> np.ndarray:
    """log W_k para cada k, con W_k = sum_r (1 / 2 n_r) sum_{i,j en r} d_ij^2."""
    out = np.empty(len(ks))
    for i, k in enumerate(ks):
        labels = fcluster(z, k, criterion="maxclust") - 1
        h = np.zeros((len(labels), labels.max() + 1))
        h[np.arange(len(labels)), labels] = 1.0
        within = (h * (d2 @ h)).sum(axis=0)
        out[i] = np.log(max((within / (2 * h.sum(axis=0))).sum(), 1e-300))
    return out


def gap_k(d: np.ndarray, z: np.ndarray, method: str, k_max: int, n_ref: int, rng: np.random.Generator) -> int:
    ks = list(range(1, k_max + 1))
    log_w = _log_w_all(d**2, z, ks)
    x = _embed(d)
    x = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    xp = x @ vt.T
    lo, hi = xp.min(axis=0), xp.max(axis=0)
    ref = np.empty((n_ref, len(ks)))
    for b in range(n_ref):
        dr = pdist(rng.uniform(lo, hi, size=xp.shape) @ vt)
        ref[b] = _log_w_all(squareform(dr) ** 2, linkage(dr, method=method), ks)
    gap = ref.mean(axis=0) - log_w
    s = ref.std(axis=0) * np.sqrt(1 + 1 / n_ref)
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - s[i + 1]:
            return ks[i]
    return ks[-1]


##############################################
# Una fecha de rebalanceo
##############################################
def run_date(end: int):
    rng = np.random.default_rng(end)
    train = _R[end - WINDOW:end]
    weights = risk_based_weights(train)
    n_clusters = {}

    corr = np.corrcoef(train, rowvar=False)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    d_corr = np.sqrt(np.clip(2 * (1 - corr), 0.0, None))
    trees = {name: (d_corr, method) for name, method in CLASSICAL.items()}

    # DBHT (Riskfolio-Lib): decide el solo los grupos y la jerarquia; similitud = 1 + rho
    dbht = DBHTs(d_corr, 1.0 + corr)
    groups = np.asarray(dbht[0]).ravel()
    if len(np.unique(groups)) < 2:  # el articulo considera al menos dos grupos
        groups = fcluster(dbht[-1], 2, criterion="maxclust")
    weights["DBHT"] = hcaa_weights(dbht[-1], groups)
    n_clusters["DBHT"] = len(np.unique(groups))
    if not _OPTS["sin_cuanticos"]:
        feats = [f[end - WINDOW:end] for f in _FEATURES]
        d_orig = frobenius_distance_matrix([average_density_matrix(f, ALPHA) for f in feats])
        d_new = daily_distance_matrix([feature_angles(f, "percentil", ALPHA) for f in feats])
        trees.update({"QHRP original WM": (d_orig, "ward"), "QHRP nuevo WM": (d_new, "ward"),
                      "QHRP nuevo AL": (d_new, "average")})

    for name, (d, method) in trees.items():
        z = linkage(squareform(d, checks=False), method=method)
        k = max(gap_k(d, z, method, _OPTS["k_max"], _OPTS["n_ref"], rng), 2)  # al menos dos grupos
        weights[name] = hcaa_weights(z, fcluster(z, k, criterion="maxclust"))
        n_clusters[name] = k
    return end, weights, n_clusters


##############################################
# Tabla con el formato del articulo
##############################################
def paper_table(tabla: pd.DataFrame, mcs: pd.DataFrame) -> pd.DataFrame:
    def mark(value: float, p: float, fmt: str) -> str:
        sup = "b" if p >= 0.70 else ("a" if p >= 0.20 else "")
        return f"{value:{fmt}}" + (f"^{sup}" if sup else "")

    rows = {}
    for m in tabla.index:
        rows[m] = {
            "ASR": mark(tabla.loc[m, "ASR"], mcs.loc[m, "p_MCS_ASR"], ".3f"),
            "CEQ": mark(tabla.loc[m, "CEQ"], mcs.loc[m, "p_MCS_CEQ"], ".2f"),
            "MDD": f"{tabla.loc[m, 'MDD']:.1f}",
            "TO": "–" if m == "EW" else f"{tabla.loc[m, 'TO']:.3f}",
            "SSPW": f"{tabla.loc[m, 'SSPW']:.4f}" if tabla.loc[m, "SSPW"] < 0.01 else f"{tabla.loc[m, 'SSPW']:.3f}",
        }
    return pd.DataFrame(rows).T[["ASR", "CEQ", "MDD", "TO", "SSPW"]]


def paper_table_png(tab: pd.DataFrame, path: Path, title: str, note: str) -> None:
    """Tabla con el aspecto de las del articulo; filas QHRP sombreadas al final."""
    cols = ["", *tab.columns]
    rows = [[m, *[v.replace("^a", "ᵃ").replace("^b", "ᵇ") for v in tab.loc[m]]] for m in tab.index]
    n = len(rows)
    xs = [0.02, 0.36, 0.50, 0.64, 0.78, 0.92]
    fig = plt.figure(figsize=(7.6, 0.32 * n + 1.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    top = 1 - 0.95 / (0.32 * n + 1.5)
    row_h = (top - 0.7 / (0.32 * n + 1.5)) / (n + 1)
    ax.text(0.02, top + row_h * 0.6, title, fontsize=11.5, weight="bold", va="bottom")
    ax.plot([0.02, 0.98], [top, top], color="black", lw=1.2)
    y = top - row_h / 2
    for x, c in zip(xs, cols):
        ax.text(x, y, c, ha="left" if x == xs[0] else "center", va="center", fontsize=11, style="italic", weight="bold")
    ax.plot([0.02, 0.98], [top - row_h, top - row_h], color="black", lw=0.8)
    for i, row in enumerate(rows):
        y_mid = top - row_h * (i + 1.5)
        if row[0].startswith("QHRP"):
            ax.add_patch(plt.Rectangle((0.02, y_mid - row_h / 2), 0.96, row_h, color="#ecebe7", lw=0))
        for x, v in zip(xs, row):
            ax.text(x, y_mid, v, ha="left" if x == xs[0] else "center", va="center", fontsize=11)
    bottom = top - row_h * (n + 1)
    ax.plot([0.02, 0.98], [bottom, bottom], color="black", lw=1.2)
    ax.text(0.02, bottom - row_h * 0.4, note, fontsize=8, color="#444444", va="top", wrap=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


##############################################
# CLI
##############################################
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tablas de Raffinot (2018) con todos los metodos y los QHRP.")
    parser.add_argument("--panel", choices=["sectores", "acciones"], default="sectores")
    parser.add_argument("--paso", type=int, default=1, help="Dias entre rebalanceos (el articulo usa 1)")
    parser.add_argument("--sin-cuanticos", action="store_true")
    parser.add_argument("--k-max", type=int, default=None, help="Maximo de grupos del Gap (def. 9 / 30)")
    parser.add_argument("--n-ref", type=int, default=10)
    parser.add_argument("--B", type=int, default=1000, help="Remuestreos del MCS")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "resultados_raffinot")
    parser.add_argument("--rehacer-ivrb", action="store_true",
                        help="Recalcular solo IVRB sobre resultados ya guardados (sin repetir el resto)")
    parser.add_argument("--k-fijo", type=str, default=None, metavar="METODOS:K",
                        help="Recalcular esos metodos con k grupos fijo sobre resultados guardados, "
                             "p. ej. 'WM,QHRP original WM:2'")
    return parser.parse_args()


def main() -> None:
    global _R, _FEATURES, _OPTS
    args = parse_args()
    returns, _, _ = load_panel(args.panel)
    _R = returns.to_numpy()
    n = _R.shape[1]
    _OPTS = {"sin_cuanticos": args.sin_cuanticos, "n_ref": args.n_ref,
             "k_max": args.k_max or min(n - 1, 30 if n > 30 else 9)}
    if not args.sin_cuanticos and not args.rehacer_ivrb:  # --k-fijo con QHRP tambien las necesita
        _FEATURES = [rolling_features(_R[:, j]) for j in range(n)]

    out = args.out_dir / f"{args.panel}_paso{args.paso}"
    out.mkdir(parents=True, exist_ok=True)
    if args.rehacer_ivrb:
        redo_ivrb(args, out)
        return
    if args.k_fijo:
        names, k = args.k_fijo.rsplit(":", 1)
        redo_fixed_k(args, out, [m.strip() for m in names.split(",")], int(k))
        return
    ends = list(range(WINDOW, len(_R), args.paso))
    print(f"Panel {args.panel}: {n} activos | {len(ends)} rebalanceos | k_max {_OPTS['k_max']} "
          f"| {args.workers} procesos", flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(run_date, ends, chunksize=2), start=1):
            results.append(res)
            if i % max(1, len(ends) // 40) == 0 or i == len(ends):
                el = time.time() - t0
                print(f"  {i}/{len(ends)} rebalanceos ({el:.0f}s, faltan ~{el / i * (len(ends) - i) / 60:.0f} min)",
                      flush=True)

    methods = [m for m in ORDER if m == "EW" or m in results[0][1]]
    oos = {m: [] for m in methods}
    w_hist = {m: [] for m in methods}
    for end, weights, _ in results:
        stop = min(end + args.paso, len(_R))
        weights = {**weights, "EW": np.full(n, 1.0 / n)}
        for m in methods:
            oos[m].append(_R[end:stop] @ weights[m])
            w_hist[m].append(weights[m])

    tabla = pd.DataFrame({m: metrics(np.concatenate(oos[m]), np.array(w_hist[m])) for m in methods}).T
    n_days = sum(len(x) for x in oos["EW"])
    oos_df = pd.DataFrame({m: np.concatenate(oos[m]) for m in methods},
                          index=returns.index[ends[0]: ends[0] + n_days])
    oos_df.to_csv(out / "retornos_oos.csv")
    k_df = pd.DataFrame({m: [res[2][m] for res in results] for m in results[0][2]}, index=returns.index[ends])
    k_df.to_csv(out / "numero_grupos.csv")
    write_tables(args, out, tabla[["ASR", "Sharpe", "CEQ", "MDD", "TO", "SSPW"]], oos_df, k_df)


def redo_ivrb(args, out: Path) -> None:
    """Recalcula solo IVRB sobre una ejecucion ya guardada y rehace tablas y MCS."""
    ends = list(range(WINDOW, len(_R), args.paso))
    oos_df = pd.read_csv(out / "retornos_oos.csv", index_col=0, parse_dates=True)
    k_df = pd.read_csv(out / "numero_grupos.csv", index_col=0, parse_dates=True)
    tabla = pd.read_csv(out / "metricas.csv", index_col=0)[["ASR", "Sharpe", "CEQ", "MDD", "TO", "SSPW"]]
    weights = [ivrb_weights(np.cov(_R[e - WINDOW:e], rowvar=False)) for e in ends]
    oos_df["IVRB"] = np.concatenate([_R[e:min(e + args.paso, len(_R))] @ w for e, w in zip(ends, weights)])
    tabla.loc["IVRB"] = metrics(oos_df["IVRB"].to_numpy(), np.array(weights))
    oos_df.to_csv(out / "retornos_oos.csv")
    write_tables(args, out, tabla, oos_df, k_df)


def _fixed_k_date(task):
    """Pesos HCAA con k fijo para los metodos pedidos en una fecha."""
    end, names, k = task
    train = _R[end - WINDOW:end]
    corr = np.corrcoef(train, rowvar=False)
    d_corr = np.sqrt(np.clip(2 * (1 - 0.5 * (corr + corr.T)), 0.0, None))
    np.fill_diagonal(d_corr, 0.0)
    out = {}
    for name in names:
        if name in CLASSICAL:
            d, method = d_corr, CLASSICAL[name]
        elif name == "QHRP original WM":
            feats = [f[end - WINDOW:end] for f in _FEATURES]
            d, method = frobenius_distance_matrix([average_density_matrix(f, ALPHA) for f in feats]), "ward"
        else:
            feats = [f[end - WINDOW:end] for f in _FEATURES]
            d = daily_distance_matrix([feature_angles(f, "percentil", ALPHA) for f in feats])
            method = "ward" if name.endswith("WM") else "average"
        z = linkage(squareform(d, checks=False), method=method)
        out[name] = hcaa_weights(z, fcluster(z, k, criterion="maxclust"))
    return out


def redo_fixed_k(args, out: Path, names: list, k: int) -> None:
    """Recalcula los metodos indicados con k grupos fijo sobre una ejecucion guardada y rehace tablas y MCS."""
    ends = list(range(WINDOW, len(_R), args.paso))
    oos_df = pd.read_csv(out / "retornos_oos.csv", index_col=0, parse_dates=True)
    k_df = pd.read_csv(out / "numero_grupos.csv", index_col=0, parse_dates=True)
    tabla = pd.read_csv(out / "metricas.csv", index_col=0)[["ASR", "Sharpe", "CEQ", "MDD", "TO", "SSPW"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        res = list(pool.map(_fixed_k_date, [(e, names, k) for e in ends]))
    for name in names:
        weights = [r[name] for r in res]
        oos_df[name] = np.concatenate([_R[e:min(e + args.paso, len(_R))] @ w for e, w in zip(ends, weights)])
        tabla.loc[name] = metrics(oos_df[name].to_numpy(), np.array(weights))
        k_df[name] = k
    oos_df.to_csv(out / "retornos_oos.csv")
    k_df.to_csv(out / "numero_grupos.csv")
    write_tables(args, out, tabla, oos_df, k_df)


def write_tables(args, out: Path, tabla: pd.DataFrame, oos_df: pd.DataFrame, k_df: pd.DataFrame) -> None:
    methods = [m for m in ORDER if m in oos_df.columns]
    tabla = tabla.loc[methods]
    idx = stationary_bootstrap_indices(len(oos_df), args.B, 21.0, np.random.default_rng(0))
    mcs = run_mcs(oos_df[methods], idx)
    tabla = tabla.join(mcs[["p_MCS_ASR", "p_MCS_CEQ"]])
    tabla["k_medio"] = [k_df[m].mean() if m in k_df else np.nan for m in methods]
    tabla.round(4).to_csv(out / "metricas.csv")

    tab = paper_table(tabla, mcs)
    tab.to_csv(out / "tabla_articulo.csv")
    periodo = f"{oos_df.index[0]:%B %Y}–{oos_df.index[-1]:%B %Y}"
    panel_name = "S&P 500 Sectors" if args.panel == "sectores" else "Individual Stocks"
    paper_table_png(tab, out / "tabla_articulo.png",
                    f"Investment Strategies Comparison: {panel_name} ({periodo})",
                    "ASR, CEQ (%), MDD (%), TO (%) y SSPW como en el articulo. a / b: metodo en el conjunto de "
                    "mejores modelos de la prueba MCS (p >= 0,20 / p >= 0,70). En gris, los QHRP anadidos. "
                    f"Ventana de {WINDOW} dias, rebalanceo cada {args.paso} dia(s).")

    pd.set_option("display.width", 200)
    print("\n" + tab.to_string())
    print("\nGrupos medios (Gap): " + ", ".join(f"{m} {k_df[m].mean():.1f}" for m in k_df))
    print(f"\nResultados en: {out}")


if __name__ == "__main__":
    main()
