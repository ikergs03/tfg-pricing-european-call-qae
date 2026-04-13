#!/usr/bin/env python3
"""Comparativa QHRP 6q vs 5q en todos los activos con pipeline identico al baseline.

Objetivo: que el escenario 6q reproduzca el mismo flujo de la Fig. 4 de referencia,
y que el 5q cambie exclusivamente en una cosa: se elimina una feature.

Pipeline fijo en este script:
1) Universo completo tras limpieza/alineamiento (sin preseleccion de activos).
2) Features "transformacion matematica" (las antiguas del notebook):
   r, r^2, |r|^1.5, sign(r)*log1p(|r|), tanh(r), sign(r)*sqrt(|r|+1e-6)
3) Matriz de distancia con Frobenius del paper en escala 1.0
4) Ordenamiento cuantico con Ward

Solo cambia entre escenarios:
- 6q: usa las 6 features.
- 5q: elimina la feature indicada en --drop-feature-index.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

if str(SCRIPT_DIR / "src") not in sys.path:
    sys.path.append(str(SCRIPT_DIR / "src"))

from analyze_qubits_6_vs_5 import (  # noqa: E402
    PROJECT_ROOT,
    ScenarioResult,
    average_density_matrix,
    block_contrast,
    compare_scenarios,
)


LEGACY_FEATURE_NAMES = [
    "retorno",
    "retorno_cuadrado",
    "abs_ret_1p5",
    "sign_log1p_abs_ret",
    "tanh_ret",
    "sign_sqrt_abs_ret",
]


def load_all_assets_returns(project_root: Path, days: int) -> tuple[list[str], pd.DataFrame]:
    """Cargar returns limpios/alineados conservando todo el universo disponible."""
    returns_path = project_root / "data" / "returns.csv"
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.dropna(axis=0, how="any")
    returns = returns.dropna(axis=1, how="all")

    # Mismo criterio de limpieza usado en el resto de scripts/notebooks.
    std = returns.std(axis=0)
    returns = returns.loc[:, std > 1e-8]

    if days > 0:
        returns = returns.tail(min(days, returns.shape[0]))

    asset_names = returns.columns.tolist()
    return asset_names, returns


def build_legacy_math_features(returns_df: pd.DataFrame) -> np.ndarray:
    """Construye las 6 features antiguas de transformacion matematica."""
    r = returns_df.to_numpy()
    t_total, n_assets = r.shape
    features = np.zeros((n_assets, t_total, 6), dtype=float)

    for j in range(n_assets):
        for t in range(t_total):
            x = r[t, j]
            features[j, t, 0] = x
            features[j, t, 1] = x**2
            features[j, t, 2] = abs(x) ** 1.5
            features[j, t, 3] = np.log1p(abs(x)) * np.sign(x)
            features[j, t, 4] = np.tanh(x)
            features[j, t, 5] = np.sign(x) * np.sqrt(abs(x) + 1e-6)

    return features


def scenario_features_exact(
    features_tensor: np.ndarray,
    n_qubits: int,
    drop_feature_index: int,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Selecciona features para 6q o 5q manteniendo el resto del pipeline intacto."""
    p = features_tensor.shape[2]
    if p != 6:
        raise ValueError(f"Este script espera 6 features base y encontro p={p}")

    if n_qubits == 6:
        keep_idx = list(range(6))
    elif n_qubits == 5:
        if not (0 <= drop_feature_index < 6):
            raise ValueError(f"drop_feature_index debe estar en [0, 5], recibido {drop_feature_index}")
        keep_idx = [i for i in range(6) if i != drop_feature_index]
    else:
        raise ValueError("Este script solo soporta n_qubits=6 o n_qubits=5")

    names = [LEGACY_FEATURE_NAMES[i] for i in keep_idx]
    return features_tensor[:, :, keep_idx], names, keep_idx


def frobenius_distance_scale1(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Distancia Frobenius en escala 1.0 (formula de la tesis)."""
    diff = rho1 - rho2
    return float(0.5 * np.sqrt(np.trace(diff.conj().T @ diff).real))


def compute_distance_matrix_paper_scale1(rho_list: list[np.ndarray]) -> np.ndarray:
    """Matriz de distancias usando la metrica Frobenius del baseline."""
    n_assets = len(rho_list)
    d = np.zeros((n_assets, n_assets), dtype=float)

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            val = frobenius_distance_scale1(rho_list[i], rho_list[j])
            d[i, j] = val
            d[j, i] = val

    return d


def quantum_ordering_ward(d: np.ndarray) -> np.ndarray:
    """Ordenamiento cuantico Ward tal como se usa en el notebook base."""
    z = linkage(squareform(d), method="ward")
    return leaves_list(z)


def run_scenario_exact(name: str, features_tensor: np.ndarray, returns_work: pd.DataFrame, alpha: float) -> ScenarioResult:
    """Ejecuta un escenario (6q/5q) con pipeline exacto baseline."""
    n_assets, _, n_qubits = features_tensor.shape
    print(f"[{name}] Calculando densidades para {n_assets} activos con {n_qubits} qubits...")

    start = time.perf_counter()
    rho_list = []
    for i in range(n_assets):
        rho_list.append(average_density_matrix(features_tensor[i], alpha=alpha))
        if (i + 1) % 10 == 0 or (i + 1) == n_assets:
            print(f"[{name}]   activos procesados: {i + 1}/{n_assets}")

    distance = compute_distance_matrix_paper_scale1(rho_list)
    order = quantum_ordering_ward(distance)

    corr = np.corrcoef(returns_work.to_numpy(), rowvar=False)
    corr_ordered = corr[np.ix_(order, order)]

    dist_ordered = distance[np.ix_(order, order)]
    band = max(3, n_assets // 10)
    block_raw = block_contrast(distance, band=band)
    block_ordered = block_contrast(dist_ordered, band=band)

    elapsed_s = time.perf_counter() - start

    return ScenarioResult(
        name=name,
        n_qubits=n_qubits,
        used_features=[],
        distance=distance,
        order=order,
        corr_ordered=corr_ordered,
        elapsed_s=elapsed_s,
        block_raw=block_raw,
        block_ordered=block_ordered,
    )


def quasi_diagonal(linkage_matrix: np.ndarray) -> list[int]:
    """Cuasi-diagonalizacion usada por el ordenamiento HRP clasico."""
    linkage_matrix = linkage_matrix.astype(int)
    sorted_items = pd.Series([linkage_matrix[-1, 0], linkage_matrix[-1, 1]])
    num_items = int(linkage_matrix[-1, 3])

    while sorted_items.max() >= num_items:
        sorted_items.index = range(0, sorted_items.shape[0] * 2, 2)
        dataframe = sorted_items[sorted_items >= num_items]
        i = dataframe.index
        j = dataframe.values - num_items
        sorted_items[i] = linkage_matrix[j, 0]
        right = pd.Series(linkage_matrix[j, 1], index=i + 1)
        sorted_items = pd.concat([sorted_items, right]).sort_index()
        sorted_items.index = range(sorted_items.shape[0])

    return [int(x) for x in sorted_items.tolist()]


def classical_ordered_corr(corr: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Ordenamiento estilo HRP clasico a partir de la matriz de correlacion."""
    dist = np.sqrt((1 - corr) / 2.0)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    dist = np.nan_to_num(dist)

    link = linkage(squareform(dist), method="single")
    order = quasi_diagonal(link)
    corr_ord = corr[np.ix_(order, order)]
    return corr_ord, order


def build_summary_text_all_assets(
    asset_names: list[str],
    returns_all: pd.DataFrame,
    days_requested: int,
    drop_feature_index: int,
    names_6q: list[str],
    names_5q: list[str],
    result_6q: ScenarioResult,
    result_5q: ScenarioResult,
    cmp_metrics: dict,
    shuffled: bool,
    fig3_path: Path,
    fig4_path: Path,
) -> str:
    tri6 = np.triu_indices_from(result_6q.distance, 1)
    tri5 = np.triu_indices_from(result_5q.distance, 1)
    off6 = result_6q.distance[tri6]
    off5 = result_5q.distance[tri5]

    mode_txt = "con shuffle" if shuffled else "sin shuffle"
    dropped_name = LEGACY_FEATURE_NAMES[drop_feature_index]

    if days_requested > 0:
        horizon_note = f"ultimos {returns_all.shape[0]} dias (solicitado --days={days_requested})"
    else:
        horizon_note = f"horizonte completo comun ({returns_all.shape[0]} dias)"

    head_assets = ", ".join(asset_names[:25])

    lines = [
        "ANALISIS 6 VS 5 QUBITS (SIMULACION IDEAL) - TODOS LOS ACTIVOS",
        f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Pipeline aplicado (igual al baseline Fig. 4)",
        "- Features: transformacion matematica antigua (6 features)",
        "- Distancia: Frobenius scale=1.0",
        "- Ordenamiento cuantico: Ward",
        f"- Modo de orden de activos: {mode_txt}",
        "",
        "Contexto del experimento",
        f"- Universo: todos los activos tras limpieza (N={returns_all.shape[1]})",
        f"- Horizonte temporal: {horizon_note}",
        f"- Regla 5 qubits: se elimina la feature indice {drop_feature_index} ({dropped_name})",
        "",
        "Escenario 6 qubits",
        f"- Features usadas: {', '.join(names_6q)}",
        f"- Tiempo simulacion: {result_6q.elapsed_s:.2f} s",
        f"- Rango distancias (off-diagonal): [{off6.min():.4f}, {off6.max():.4f}]",
        f"- Contraste bloques sin ordenar: {result_6q.block_raw:.4f}",
        f"- Contraste bloques ordenado: {result_6q.block_ordered:.4f}",
        f"- Mejora de estructura: {result_6q.block_ordered - result_6q.block_raw:+.4f}",
        "",
        "Escenario 5 qubits",
        f"- Features usadas: {', '.join(names_5q)}",
        f"- Tiempo simulacion: {result_5q.elapsed_s:.2f} s",
        f"- Rango distancias (off-diagonal): [{off5.min():.4f}, {off5.max():.4f}]",
        f"- Contraste bloques sin ordenar: {result_5q.block_raw:.4f}",
        f"- Contraste bloques ordenado: {result_5q.block_ordered:.4f}",
        f"- Mejora de estructura: {result_5q.block_ordered - result_5q.block_raw:+.4f}",
        "",
        "Comparativa 6 vs 5",
        f"- Correlacion Pearson entre distancias: {cmp_metrics['distance_pearson']:.4f}",
        f"- Correlacion Spearman entre distancias: {cmp_metrics['distance_spearman']:.4f}",
        f"- Spearman entre rankings (orden de activos): {cmp_metrics['order_spearman']:.4f}",
        f"- Kendall tau entre rankings: {cmp_metrics['order_kendall']:.4f}",
        f"- Distorsion relativa Frobenius: {cmp_metrics['relative_frobenius_distortion']:.4f}",
        "",
        f"- Fig 3 (correlaciones) guardada en: {fig3_path}",
        f"- Fig 4 (distancias) guardada en: {fig4_path}",
        "",
        f"Primeros 25 activos (de {len(asset_names)}):",
        f"- {head_assets}",
    ]

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QHRP todos los activos: 6q vs 5q con pipeline identico al baseline"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=756,
        help="Dias a conservar desde el tramo final. Usa 0 para el horizonte comun completo.",
    )
    parser.add_argument("--alpha", type=float, default=2.0, help="Parametro alpha del feature-map")
    parser.add_argument(
        "--shuffle-assets",
        action="store_true",
        help="Permutar activos antes de graficar (por defecto desactivado para reproducir baseline).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Semilla de la permutacion cuando --shuffle-assets esta activo.",
    )
    parser.add_argument(
        "--drop-feature-index",
        type=int,
        default=5,
        help="Indice de feature a eliminar en 5 qubits (default 5 = sign_sqrt_abs_ret)",
    )
    parser.add_argument(
        "--fig4-vmax",
        type=float,
        default=0.55,
        help="Vmax fijo para Fig.4 (baseline usa 0.55)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "results_6_vs_5_qubits_all_assets",
        help="Directorio de salida para figuras e informes",
    )
    return parser.parse_args()


def save_fig3_corr(
    path: Path,
    corr_raw: np.ndarray,
    corr_classical: np.ndarray,
    corr_quantum_6q: np.ndarray,
    corr_quantum_5q: np.ndarray,
    shuffled: bool,
) -> None:
    """Figura 3 en formato 3+3: fila 6q y fila 5q."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    tag = "shuffled" if shuffled else "baseline"
    mats = [
        [corr_raw, corr_classical, corr_quantum_6q],
        [corr_raw, corr_classical, corr_quantum_5q],
    ]
    titles = [
        [
            f"(a) 6q {tag}: correlacion sin ordenar",
            f"(b) 6q {tag}: correlacion ordenada HRP clasica",
            f"(c) 6q {tag}: correlacion ordenada HRP cuantica",
        ],
        [
            f"(d) 5q {tag}: correlacion sin ordenar",
            f"(e) 5q {tag}: correlacion ordenada HRP clasica",
            f"(f) 5q {tag}: correlacion ordenada HRP cuantica",
        ],
    ]

    for r in range(2):
        for c in range(3):
            sns.heatmap(
                mats[r][c],
                cmap="coolwarm",
                vmin=-0.2,
                vmax=1.0,
                square=True,
                cbar=True,
                ax=axes[r, c],
            )
            axes[r, c].set_title(titles[r][c])
            axes[r, c].tick_params(labelsize=5)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_fig4_distances(path: Path, result_6q: ScenarioResult, result_5q: ScenarioResult, vmax: float, shuffled: bool) -> None:
    """Figura 4: distancias 6q y 5q antes/despues de ordenar, con escala baseline."""
    d6_ordered = result_6q.distance[np.ix_(result_6q.order, result_6q.order)]
    d5_ordered = result_5q.distance[np.ix_(result_5q.order, result_5q.order)]
    tag = "shuffled" if shuffled else "baseline"

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    sns.heatmap(result_6q.distance, cmap="coolwarm", vmin=0, vmax=vmax, square=True, cbar=True, ax=axes[0, 0])
    axes[0, 0].set_title(f"(a) Distancia cuantica 6q sin ordenar ({tag})")
    axes[0, 0].tick_params(labelsize=5)

    sns.heatmap(d6_ordered, cmap="coolwarm", vmin=0, vmax=vmax, square=True, cbar=True, ax=axes[0, 1])
    axes[0, 1].set_title(f"(b) Distancia cuantica 6q ordenada ({tag})")
    axes[0, 1].tick_params(labelsize=5)

    sns.heatmap(result_5q.distance, cmap="coolwarm", vmin=0, vmax=vmax, square=True, cbar=True, ax=axes[1, 0])
    axes[1, 0].set_title(f"(c) Distancia cuantica 5q sin ordenar ({tag})")
    axes[1, 0].tick_params(labelsize=5)

    sns.heatmap(d5_ordered, cmap="coolwarm", vmin=0, vmax=vmax, square=True, cbar=True, ax=axes[1, 1])
    axes[1, 1].set_title(f"(d) Distancia cuantica 5q ordenada ({tag})")
    axes[1, 1].tick_params(labelsize=5)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando todos los activos (tras limpieza/alineamiento)...")
    asset_names, returns_all = load_all_assets_returns(PROJECT_ROOT, days=args.days)
    print(f"Universo -> activos={returns_all.shape[1]}, dias={returns_all.shape[0]}")

    # Por defecto, no se permuta: se busca reproducir baseline lo maximo posible.
    if args.shuffle_assets:
        rng = np.random.default_rng(args.shuffle_seed)
        perm = rng.permutation(returns_all.shape[1])
        returns_work = returns_all.iloc[:, perm]
        asset_names_work = [asset_names[i] for i in perm]
        print(f"Activos shuffled con seed={args.shuffle_seed}; primeros 10 indices={perm[:10].tolist()}")
    else:
        returns_work = returns_all
        asset_names_work = asset_names
        perm = None
        print("Activos sin shuffle (modo baseline).")

    # Pipeline exacto baseline: features antiguas de transformacion matematica.
    features_work = build_legacy_math_features(returns_work)

    tensor_6q, names_6q, _ = scenario_features_exact(
        features_work,
        n_qubits=6,
        drop_feature_index=args.drop_feature_index,
    )
    tensor_5q, names_5q, _ = scenario_features_exact(
        features_work,
        n_qubits=5,
        drop_feature_index=args.drop_feature_index,
    )

    result_6q = run_scenario_exact("6q_all_assets", tensor_6q, returns_work, alpha=args.alpha)
    result_6q.used_features = names_6q

    result_5q = run_scenario_exact("5q_all_assets", tensor_5q, returns_work, alpha=args.alpha)
    result_5q.used_features = names_5q

    cmp_metrics = compare_scenarios(result_6q, result_5q)

    corr_raw = np.corrcoef(returns_work.to_numpy(), rowvar=False)
    corr_classical, order_classical = classical_ordered_corr(corr_raw)

    suffix = "shuffled" if args.shuffle_assets else "baseline"
    fig3_path = out_dir / f"fig3_corr_6_vs_5_all_assets_{suffix}_3x3.png"
    fig4_path = out_dir / f"fig4_distance_6_vs_5_all_assets_{suffix}.png"

    save_fig3_corr(
        fig3_path,
        corr_raw,
        corr_classical,
        result_6q.corr_ordered,
        result_5q.corr_ordered,
        shuffled=args.shuffle_assets,
    )
    save_fig4_distances(
        fig4_path,
        result_6q,
        result_5q,
        vmax=args.fig4_vmax,
        shuffled=args.shuffle_assets,
    )

    summary_text = build_summary_text_all_assets(
        asset_names=asset_names_work,
        returns_all=returns_work,
        days_requested=args.days,
        drop_feature_index=args.drop_feature_index,
        names_6q=names_6q,
        names_5q=names_5q,
        result_6q=result_6q,
        result_5q=result_5q,
        cmp_metrics=cmp_metrics,
        shuffled=args.shuffle_assets,
        fig3_path=fig3_path,
        fig4_path=fig4_path,
    )

    summary_path = out_dir / "summary_6_vs_5_qubits_all_assets.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    metrics_payload = {
        "mode": "all_assets_after_cleaning",
        "pipeline": {
            "features": "legacy_math_features",
            "distance": "frobenius_scale1",
            "ordering": "ward",
            "shuffle_assets": args.shuffle_assets,
            "shuffle_seed": args.shuffle_seed if args.shuffle_assets else None,
            "fig4_vmax": args.fig4_vmax,
        },
        "asset_count": len(asset_names),
        "asset_names": asset_names,
        "shape_returns": list(returns_work.shape),
        "shape_features": list(features_work.shape),
        "days_requested": args.days,
        "drop_feature_index": args.drop_feature_index,
        "used_features_6q": names_6q,
        "used_features_5q": names_5q,
        "timing_seconds": {
            "six_qubits": result_6q.elapsed_s,
            "five_qubits": result_5q.elapsed_s,
        },
        "block_contrast": {
            "six_qubits_raw": result_6q.block_raw,
            "six_qubits_ordered": result_6q.block_ordered,
            "five_qubits_raw": result_5q.block_raw,
            "five_qubits_ordered": result_5q.block_ordered,
        },
        "figure_paths": {
            "fig3_correlations": str(fig3_path),
            "fig4_distances": str(fig4_path),
        },
        "classical_order_head": order_classical[:10],
        "comparison": cmp_metrics,
        "perm_head": perm[:10].tolist() if perm is not None else None,
    }

    json_path = out_dir / "metrics_6_vs_5_qubits_all_assets.json"
    json_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("\nFinalizado.")
    print(f"- Resumen: {summary_path}")
    print(f"- Metricas: {json_path}")
    print(f"- Figura:  {fig3_path}")
    print(f"- Figura:  {fig4_path}")


if __name__ == "__main__":
    main()
