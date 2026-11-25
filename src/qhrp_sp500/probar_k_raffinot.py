#!/usr/bin/env python3
"""Prueba numeros de grupos fijos (k) para SL, CL, AL, WM y DBHT y los compara con el articulo.

raffinot_tablas.py elige k con el indice Gap (minimo 2), pero el articulo no detalla como lo
aplica y sus metodos de agrupacion no salen identicos. Aqui se fija el mismo k todos los dias,
para cada k posible, y se mide cuanto se parece cada resultado a las tablas 2 y 4 del articulo.

Distancia al articulo: media de las diferencias relativas en ASR, CEQ, MDD, TO y SSPW.

Salida: resultados_raffinot/probar_k/<panel>_paso<paso>.csv (todas las combinaciones)
        y por pantalla el k mas parecido para cada metodo.

Uso (desde src/qhrp_sp500):
    ../../raffinot-env/bin/python probar_k_raffinot.py --panel sectores
    ../../raffinot-env/bin/python probar_k_raffinot.py --panel acciones --paso 21 --sin-dbht
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from riskfolio.src.DBHT import DBHTs
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from analisis_sp500 import load_panel
from hcaa_sp500 import hcaa_weights, metrics

SCRIPT_DIR = Path(__file__).resolve().parent
WINDOW = 252
CLASSICAL = {"SL": "single", "CL": "complete", "AL": "average", "WM": "ward"}
COLS = ["ASR", "CEQ", "MDD", "TO", "SSPW"]
PAPER = {  # tablas 2 y 4 del articulo
    "sectores": {"SL": (0.418, 6.53, 53.4, 0.669, 0.115), "CL": (0.430, 6.87, 51.0, 0.817, 0.114),
                 "AL": (0.421, 6.62, 52.6, 0.762, 0.114), "WM": (0.415, 6.49, 52.9, 0.883, 0.149),
                 "DBHT": (0.449, 7.43, 59.0, 3.450, 0.122)},
    "acciones": {"SL": (0.492, 19.2, 43.1, 32.4, 0.0552), "CL": (0.473, 16.5, 47.4, 33.7, 0.0151),
                 "AL": (0.520, 19.5, 46.1, 33.6, 0.041), "WM": (0.572, 14.2, 51.2, 33.4, 0.0051),
                 "DBHT": (0.591, 13.5, 47.6, 39.4, 0.0056)},
}

_R: np.ndarray | None = None
_OPTS: dict = {}


def run_date(end: int) -> dict:
    train = _R[end - WINDOW:end]
    corr = np.corrcoef(train, rowvar=False)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    d = np.sqrt(np.clip(2 * (1 - corr), 0.0, None))
    trees = {name: linkage(squareform(d, checks=False), method=m) for name, m in CLASSICAL.items()}
    out = {}
    if not _OPTS["sin_dbht"]:
        dbht = DBHTs(d, 1.0 + corr)
        trees["DBHT"] = dbht[-1]
        out[("DBHT", "propio")] = hcaa_weights(dbht[-1], np.asarray(dbht[0]).ravel())
    for name, z in trees.items():
        for k in _OPTS["ks"]:
            out[(name, k)] = hcaa_weights(z, fcluster(z, k, criterion="maxclust"))
    return out


def main() -> None:
    global _R, _OPTS
    parser = argparse.ArgumentParser(description="Prueba numeros de grupos fijos frente al articulo.")
    parser.add_argument("--panel", choices=["sectores", "acciones"], default="sectores")
    parser.add_argument("--paso", type=int, default=1)
    parser.add_argument("--sin-dbht", action="store_true")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    returns, _, _ = load_panel(args.panel)
    _R = returns.to_numpy()
    n = _R.shape[1]
    ks = list(range(1, n)) if n <= 10 else [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]
    _OPTS = {"ks": ks, "sin_dbht": args.sin_dbht}
    ends = list(range(WINDOW, len(_R), args.paso))
    print(f"Panel {args.panel}: {len(ends)} rebalanceos, k = {ks}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_date, ends, chunksize=8 if args.paso == 1 else 1))

    rows = []
    for key in results[0]:
        r = np.concatenate([_R[e:min(e + args.paso, len(_R))] @ res[key] for e, res in zip(ends, results)])
        m = metrics(r, np.array([res[key] for res in results]))
        paper = PAPER[args.panel][key[0]]
        dist = np.mean([abs(m[c] - p) / abs(p) for c, p in zip(COLS, paper)])
        rows.append({"metodo": key[0], "k": key[1], **{c: m[c] for c in COLS}, "distancia_articulo": dist})
    df = pd.DataFrame(rows)
    out = SCRIPT_DIR / "resultados_raffinot" / "probar_k"
    out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / f"{args.panel}_paso{args.paso}.csv", index=False)

    pd.set_option("display.width", 200)
    for name in df["metodo"].unique():
        sub = df[df["metodo"] == name].sort_values("distancia_articulo")
        print(f"\n== {name}   articulo: " + "  ".join(f"{c} {v}" for c, v in zip(COLS, PAPER[args.panel][name])))
        print(sub.head(4).round(4).to_string(index=False))
    print(f"\nTodas las combinaciones en: {out / f'{args.panel}_paso{args.paso}.csv'}")


if __name__ == "__main__":
    main()
