#!/usr/bin/env python3
"""Model Confidence Set (Hansen, Lunde y Nason, 2011) sobre los resultados de hcaa_sp500.py.

Como en el articulo de Raffinot (2018): para cada criterio (ASR y CEQ) se busca el
conjunto de metodos que no se pueden distinguir del mejor, con bootstrap.

  - Bootstrap estacionario (Politis y Romano) de los dias fuera de muestra, con bloques
    de longitud media --bloque, aplicando los mismos indices a todos los metodos.
  - Perdida de cada metodo = -criterio (ASR o CEQ) calculado sobre la muestra remuestreada.
  - Estadistico T_max; en cada paso se elimina el metodo con mayor t_i mientras p < alpha.
  - p-valor MCS de cada metodo: el maximo de los p-valores hasta su eliminacion.

Con columnas del tipo "metodo k=5" se hace un MCS por cada numero de grupos k
(con pesos iguales, EW, incluido en todos).

Uso:
    python mcs_hcaa.py resultados_hcaa/sectores_paso1_k2-3-4-5
    python mcs_hcaa.py resultados_hcaa/acciones_paso21_k5-10-20 --B 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

ALPHAS = (0.20, 0.70)  # niveles que reporta el articulo


##############################################
# Criterios (mismas formulas que hcaa_sp500.metrics, vectorizadas por columnas)
##############################################
def asr(r: np.ndarray) -> np.ndarray:
    sr = r.mean(axis=0) / r.std(axis=0, ddof=1) * np.sqrt(252)
    return sr * (1 + skew(r, axis=0) / 6 * sr - (kurtosis(r, axis=0, fisher=False) - 3) / 24 * sr**2)


def ceq(r: np.ndarray) -> np.ndarray:
    return (r.mean(axis=0) * 252 - 0.5 * r.var(axis=0, ddof=1) * 252) * 100


CRITERIA = {"ASR": asr, "CEQ": ceq}


##############################################
# Bootstrap y MCS
##############################################
def stationary_bootstrap_indices(t: int, n_boot: int, block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty((n_boot, t), dtype=np.int64)
    idx[:, 0] = rng.integers(0, t, n_boot)
    restart = rng.random((n_boot, t)) < 1.0 / block
    fresh = rng.integers(0, t, (n_boot, t))
    for s in range(1, t):
        idx[:, s] = np.where(restart[:, s], fresh[:, s], (idx[:, s - 1] + 1) % t)
    return idx


def model_confidence_set(loss: np.ndarray, loss_boot: np.ndarray) -> np.ndarray:
    """loss: (m,) perdida de cada modelo; loss_boot: (B, m). Devuelve el p-valor MCS de cada modelo."""
    m = len(loss)
    alive = list(range(m))
    pvals = np.zeros(m)
    p_running = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        d = loss[a] - loss[a].mean()
        d_boot = loss_boot[:, a] - loss_boot[:, a].mean(axis=1, keepdims=True)
        sd = d_boot.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, np.inf)
        t_stat = d / sd
        t_boot = ((d_boot - d) / sd).max(axis=1)
        p = float((t_boot >= t_stat.max()).mean())
        p_running = max(p_running, p)
        worst = a[np.argmax(t_stat)]
        pvals[worst] = p_running
        alive.remove(worst)
    pvals[alive[0]] = 1.0
    return pvals


def run_mcs(returns: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    r = returns.to_numpy()
    rows = {}
    for name, fn in CRITERIA.items():
        value = fn(r)
        boot = np.stack([fn(r[i]) for i in idx])
        rows[name] = value
        rows[f"p_MCS_{name}"] = model_confidence_set(-value, -boot)
    out = pd.DataFrame(rows, index=returns.columns)
    for name in CRITERIA:
        for alpha in ALPHAS:
            out[f"{name}_en_M{int(alpha * 100)}%"] = out[f"p_MCS_{name}"] >= alpha
    return out


##############################################
# CLI
##############################################
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model Confidence Set sobre los retornos de hcaa_sp500.py.")
    parser.add_argument("carpeta", type=Path, help="Carpeta con retornos_oos.csv")
    parser.add_argument("--B", type=int, default=1000, help="Numero de remuestreos bootstrap")
    parser.add_argument("--bloque", type=float, default=21.0, help="Longitud media de bloque (dias)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    returns = pd.read_csv(args.carpeta / "retornos_oos.csv", index_col=0, parse_dates=True)
    rng = np.random.default_rng(args.seed)
    idx = stationary_bootstrap_indices(len(returns), args.B, args.bloque, rng)

    ks = sorted({int(c.rsplit(" k=", 1)[1]) for c in returns.columns if " k=" in c})
    groups = {f"k={k}": [c for c in returns.columns if c.endswith(f" k={k}")] + ["EW"] for k in ks} \
        if ks else {"todos": list(returns.columns)}

    pd.set_option("display.width", 200)
    tablas = []
    for label, cols in groups.items():
        tab = run_mcs(returns[cols], idx)
        tab.index = [c.rsplit(" k=", 1)[0] for c in tab.index]
        tab.insert(0, "grupo", label)
        tablas.append(tab)
        show = tab[["ASR", "p_MCS_ASR", "CEQ", "p_MCS_CEQ"]].sort_values("p_MCS_ASR", ascending=False)
        print(f"\n== {label}  (p_MCS >= 0.20: no se distingue del mejor; >= 0.70: claramente entre los mejores)")
        print(show.round(3).to_string())

    pd.concat(tablas).to_csv(args.carpeta / "mcs.csv")
    print(f"\nGuardado en: {args.carpeta / 'mcs.csv'}")


if __name__ == "__main__":
    main()
