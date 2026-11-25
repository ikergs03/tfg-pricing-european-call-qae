#!/usr/bin/env python3
"""Backtest walk-forward sobre los paneles del S&P 500 (data/datos_raffinot).

Mismo esquema que src/results/result.ipynb (Cap. 6 de la memoria):
  - 756 dias de entrenamiento para estimar los pesos, 21 dias de prueba fuera de muestra,
    y la ventana avanza 21 dias hasta cubrir todo el historico.
  - Sharpe anualizado con rf = 1 %, PSR (Sharpe objetivo 0) por ventana y Sharpe OOS acumulado.
  - Pesos HRP por biseccion recursiva con la covarianza de la ventana de entrenamiento.
  - HRP clasico: misma ordenacion que classical_hrp_ordering de quantum_hrp.py
    (distancia sqrt((1 - corr) / 2), enlace simple, cuasi-diagonalizacion).

Metodos: HRP clasico, QHRP original, QHRP dia a dia (escala por percentiles),
Markowitz (inverso de la varianza) y pesos iguales.

Uso:
    python backtest_sp500.py --panel sectores
    python backtest_sp500.py --panel acciones --workers 6
"""

from __future__ import annotations

import os

# Cada proceso usa pocos hilos de BLAS; si no, los procesos en paralelo se pisan entre si
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
from scipy.cluster.hierarchy import linkage
from scipy.stats import kurtosis, norm, skew

from analisis_sp500 import load_panel
from qhrp_core import (
    average_density_matrix,
    frobenius_distance_matrix,
    quantum_linkage,
    quasi_diagonal,
    rolling_features,
)
from qhrp_dia_a_dia import daily_distance_matrix, feature_angles

SCRIPT_DIR = Path(__file__).resolve().parent
METHODS = ["HRP clasico", "QHRP original", "QHRP dia a dia", "Markowitz", "Pesos iguales"]
RF = 0.01
ALPHA = 2.0

# Datos compartidos con los procesos hijos (se rellenan en main antes de crear el pool)
_R: np.ndarray | None = None
_FEATURES: list | None = None


##############################################
# Metricas (mismas formulas que quantum_hrp.py)
##############################################
def sharpe_ratio(returns, rf=RF, annualization_factor=252):
    rf = rf / annualization_factor
    std_ret = np.std(returns, ddof=1)
    daily_sr = (np.mean(returns) - rf) / std_ret if std_ret != 0 else 0
    return daily_sr * np.sqrt(annualization_factor)


def probabilistic_sharpe_ratio(returns, target_sr=0, rf=RF, annualization_factor=252):
    target_sr = target_sr / np.sqrt(annualization_factor)
    n = len(returns)
    sr_daily = sharpe_ratio(returns, rf, annualization_factor) / np.sqrt(annualization_factor)
    sample_skew = skew(returns, bias=False)
    sample_kurt = kurtosis(returns, fisher=False, bias=False)
    denom = np.sqrt(1 - sample_skew * sr_daily + ((sample_kurt - 1) / 4) * sr_daily**2)
    if not denom > 0:
        return 1.0 if sr_daily > target_sr else 0.0
    return float(norm.cdf((sr_daily - target_sr) * np.sqrt(n - 1) / denom))


def max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1 + returns)
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())


##############################################
# Pesos
##############################################
def _cluster_variance(cov: np.ndarray, items: list) -> float:
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov: np.ndarray, order: list) -> np.ndarray:
    w = np.ones(cov.shape[0])
    clusters = [list(order)]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for k in range(0, len(clusters), 2):
            left, right = clusters[k], clusters[k + 1]
            var_l, var_r = _cluster_variance(cov, left), _cluster_variance(cov, right)
            alpha = 1.0 - var_l / (var_l + var_r)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    return w / w.sum()


def classical_order(train: np.ndarray) -> list:
    """Igual que classical_hrp_ordering de quantum_hrp.py (y que el codigo original de HRP)."""
    dist = np.sqrt(np.clip((1 - np.corrcoef(train, rowvar=False)) / 2.0, 0.0, None))
    return [int(i) for i in quasi_diagonal(linkage(dist, method="single"))]


##############################################
# Una ventana
##############################################
def run_window(bounds):
    train_start, train_end, test_end = bounds
    train, test = _R[train_start:train_end], _R[train_end:test_end]
    feats = [f[train_start:train_end] for f in _FEATURES]
    cov = np.cov(train, rowvar=False)

    t0 = time.perf_counter()
    order_orig = quantum_linkage(frobenius_distance_matrix([average_density_matrix(f, ALPHA) for f in feats]))[1]
    t_orig = time.perf_counter() - t0

    t0 = time.perf_counter()
    order_dia = quantum_linkage(daily_distance_matrix([feature_angles(f, "percentil", ALPHA) for f in feats]))[1]
    t_dia = time.perf_counter() - t0

    inv_var = 1.0 / np.diag(cov)
    weights = {
        "HRP clasico": hrp_weights(cov, classical_order(train)),
        "QHRP original": hrp_weights(cov, order_orig),
        "QHRP dia a dia": hrp_weights(cov, order_dia),
        "Markowitz": inv_var / inv_var.sum(),
        "Pesos iguales": np.full(train.shape[1], 1.0 / train.shape[1]),
    }
    return {m: test @ w for m, w in weights.items()}, t_orig, t_dia


##############################################
# CLI
##############################################
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest walk-forward QHRP vs HRP en el S&P 500.")
    parser.add_argument("--panel", choices=["sectores", "acciones"], default="sectores")
    parser.add_argument("--train", type=int, default=756)
    parser.add_argument("--test", type=int, default=21)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "resultados_backtest")
    return parser.parse_args()


def main() -> None:
    global _R, _FEATURES
    args = parse_args()
    out = args.out_dir / args.panel
    out.mkdir(parents=True, exist_ok=True)

    returns, labels, _ = load_panel(args.panel)
    _R = returns.to_numpy()
    _FEATURES = [rolling_features(_R[:, j]) for j in range(_R.shape[1])]
    t_total = len(_R)
    bounds = [(s, s + args.train, s + args.train + args.test)
              for s in range(0, t_total - args.train - args.test + 1, args.test)]
    print(f"Panel {args.panel}: {len(labels)} activos | {len(bounds)} ventanas "
          f"({args.train} dias entrenamiento, {args.test} prueba) | {args.workers} procesos")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for k, res in enumerate(pool.map(run_window, bounds), start=1):
            results.append(res)
            if k % 10 == 0 or k == len(bounds):
                print(f"  {k}/{len(bounds)} ventanas ({time.time() - t0:.0f}s)", flush=True)

    # Tablas
    rows = []
    for m in METHODS:
        per_window = [res[0][m] for res in results]
        oos = np.concatenate(per_window)
        sr = np.array([sharpe_ratio(x) for x in per_window])
        rows.append({
            "metodo": m,
            "sharpe_medio_ventana": sr.mean(),
            "sharpe_desv_ventana": sr.std(),
            "psr_medio": np.mean([probabilistic_sharpe_ratio(x) for x in per_window]),
            "sharpe_oos_acumulado": sharpe_ratio(oos),
            "rentabilidad_anual": (np.prod(1 + oos) ** (252 / len(oos)) - 1),
            "volatilidad_anual": oos.std(ddof=1) * np.sqrt(252),
            "max_drawdown": max_drawdown(oos),
        })
    tabla = pd.DataFrame(rows).set_index("metodo").round(4)
    tabla.to_csv(out / "metricas.csv")

    test_index = returns.index[bounds[0][1]: bounds[-1][2]]
    oos_df = pd.DataFrame({m: np.concatenate([res[0][m] for res in results]) for m in METHODS}, index=test_index)
    oos_df.to_csv(out / "retornos_oos.csv")
    tiempos = pd.DataFrame({"QHRP original": [r[1] for r in results], "QHRP dia a dia": [r[2] for r in results]})
    tiempos.describe().round(3).to_csv(out / "tiempos_por_ventana.csv")

    # Figura: evolucion de 1 USD invertido
    fig, ax = plt.subplots(figsize=(11, 5.5))
    wealth = (1 + oos_df).cumprod()
    for m in METHODS:
        ax.plot(wealth.index, wealth[m], lw=2 if "QHRP" in m or m == "HRP clasico" else 1.2,
                label=f"{m} (Sharpe {tabla.loc[m, 'sharpe_oos_acumulado']:.2f})")
    ax.set_yscale("log")
    ax.set_ylabel("Valor de 1 USD invertido (escala log)")
    ax.set_title(f"Backtest fuera de muestra: {args.panel} del S&P 500 "
                 f"({test_index[0].year}-{test_index[-1].year}, rebalanceo cada {args.test} dias)")
    ax.grid(color="#dddddd", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "evolucion_cartera.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n" + tabla.to_string())
    print("\nTiempo medio por ventana (s): " + ", ".join(f"{c} {tiempos[c].mean():.2f}" for c in tiempos))
    print(f"\nResultados en: {out}")


if __name__ == "__main__":
    main()
