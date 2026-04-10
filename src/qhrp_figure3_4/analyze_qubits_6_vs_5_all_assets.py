#!/usr/bin/env python3
"""Comparativa QHRP 6q vs 5q en simulacion ideal usando todos los activos.

Este script replica la comparacion del piloto, pero sin preseleccion de activos:
- Mantiene todo el universo disponible tras limpieza/alineamiento.
- Compara codificacion de 6 qubits frente a 5 qubits en el mismo horizonte.
- Exporta resumen, metricas JSON y figuras comparativas.

A diferencia de analyze_qubits_6_vs_5.py, esta variante no aplica la prueba
de preseleccion de 40 activos por liquidez.

Politica de figuras en este script:
- Las figuras se generan sobre un orden aleatorio de activos (seed fija),
  para mantener el enfoque de robustez tipo shuffled del notebook.
- La Fig. 3 se exporta en formato 3+3 (seis paneles: fila 6q + fila 5q).

Por defecto usa 90 dias para ser comparable con el piloto 40x90.
Usa --days 0 para ejecutar el horizonte comun completo.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from analyze_qubits_6_vs_5 import (  # noqa: E402
    FEATURE_NAMES,
    PROJECT_ROOT,
    compare_scenarios,
    run_scenario,
    save_plot,
    scenario_features,
)


def load_all_assets(project_root: Path, days: int) -> tuple[list[str], pd.DataFrame, np.ndarray]:
    """Cargar datos limpios/alineados y conservar todos los activos tras limpieza."""
    returns_path = project_root / "data" / "returns.csv"
    features_path = project_root / "data" / "features.npy"

    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    features = np.load(features_path)

    original_columns = returns.columns.tolist()
    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.dropna(axis=0, how="any")
    returns = returns.dropna(axis=1, how="all")

    # Mantener la misma limpieza que en notebook/scripts para evitar correlaciones inestables.
    std = returns.std(axis=0)
    returns = returns.loc[:, std > 1e-8]

    surviving_idx = [original_columns.index(c) for c in returns.columns]
    features = features[surviving_idx]

    t_common = min(returns.shape[0], features.shape[1])
    returns = returns.tail(t_common)
    features = features[:, -t_common:, :]

    if days > 0:
        t_use = min(days, t_common)
        returns = returns.tail(t_use)
        features = features[:, -t_use:, :]

    # Importante: conservar todo el universo limpio (sin ranking ni filtro de activos).
    asset_names = returns.columns.tolist()
    return asset_names, returns, features


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
    features_all: np.ndarray,
    days_requested: int,
    drop_feature_index: int,
    names_6q: list[str],
    names_5q: list[str],
    result_6q,
    result_5q,
    cmp_metrics: dict,
    shuffle_seed: int,
    fig3_path: Path,
    fig4_path: Path,
    legacy_plot_path: Path,
) -> str:
    tri6 = np.triu_indices_from(result_6q.distance, 1)
    tri5 = np.triu_indices_from(result_5q.distance, 1)
    off6 = result_6q.distance[tri6]
    off5 = result_5q.distance[tri5]

    state_dim_6 = 2 ** result_6q.n_qubits
    state_dim_5 = 2 ** result_5q.n_qubits
    dropped_name = FEATURE_NAMES[drop_feature_index] if drop_feature_index < len(FEATURE_NAMES) else f"f{drop_feature_index + 1}"

    if days_requested > 0:
        horizon_note = f"ultimos {returns_all.shape[0]} dias (solicitado --days={days_requested})"
    else:
        horizon_note = f"horizonte completo comun ({returns_all.shape[0]} dias)"

    head_assets = ", ".join(asset_names[:25])

    lines = [
        "ANALISIS 6 VS 5 QUBITS (SIMULACION IDEAL) - TODOS LOS ACTIVOS",
        f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Contexto del experimento",
        f"- Universo: todos los activos tras limpieza (N={returns_all.shape[1]})",
        f"- Horizonte temporal: {horizon_note}",
        f"- Tensor de entrada: {features_all.shape}",
        f"- Regla 5 qubits: se elimina la feature indice {drop_feature_index} ({dropped_name})",
        f"- Version de figuras: shuffled (seed={shuffle_seed})",
        "",
        "Escenario 6 qubits",
        f"- Features usadas: {', '.join(names_6q)}",
        f"- Dimension Hilbert por estado: {state_dim_6}",
        f"- Tiempo simulacion: {result_6q.elapsed_s:.2f} s",
        f"- Rango distancias (off-diagonal): [{off6.min():.4f}, {off6.max():.4f}]",
        f"- Contraste bloques sin ordenar: {result_6q.block_raw:.4f}",
        f"- Contraste bloques ordenado: {result_6q.block_ordered:.4f}",
        f"- Mejora de estructura: {result_6q.block_ordered - result_6q.block_raw:+.4f}",
        "",
        "Escenario 5 qubits",
        f"- Features usadas: {', '.join(names_5q)}",
        f"- Dimension Hilbert por estado: {state_dim_5}",
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
        f"- Fig 3 shuffled (3+3 paneles) guardada en: {fig3_path}",
        f"- Fig 4 shuffled (distancias) guardada en: {fig4_path}",
        f"- Figura comparativa legacy shuffled (2x3) guardada en: {legacy_plot_path}",
        "",
        f"Primeros 25 activos (de {len(asset_names)}):",
        f"- {head_assets}",
    ]

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QHRP todos los activos: simulacion 6 qubits vs 5 qubits")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Dias a conservar desde el tramo final. Usa 0 para el horizonte comun completo.",
    )
    parser.add_argument("--alpha", type=float, default=2.0, help="Parametro alpha del feature-map")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Semilla de la permutacion aleatoria de activos usada en figuras.",
    )
    parser.add_argument(
        "--drop-feature-index",
        type=int,
        default=5,
        help="Indice de feature a eliminar en 5 qubits (por defecto 5 = curtosis)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "results_6_vs_5_qubits_all_assets",
        help="Directorio de salida para figuras e informes",
    )
    return parser.parse_args()


def save_fig3_shuffled_3x3(
    path: Path,
    corr_shuf: np.ndarray,
    corr_classical_shuf: np.ndarray,
    corr_quantum_6q: np.ndarray,
    corr_quantum_5q: np.ndarray,
) -> None:
    """Figura 3 en formato 3+3 (fila 6q y fila 5q), sobre universo shuffled."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    mats = [
        [corr_shuf, corr_classical_shuf, corr_quantum_6q],
        [corr_shuf, corr_classical_shuf, corr_quantum_5q],
    ]
    titles = [
        [
            "(a) 6q shuffled: correlacion sin ordenar",
            "(b) 6q shuffled: correlacion ordenada HRP clasica",
            "(c) 6q shuffled: correlacion ordenada HRP cuantica",
        ],
        [
            "(d) 5q shuffled: correlacion sin ordenar",
            "(e) 5q shuffled: correlacion ordenada HRP clasica",
            "(f) 5q shuffled: correlacion ordenada HRP cuantica",
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


def save_fig4_distances(path: Path, result_6q, result_5q) -> None:
    """Figura 4: distancias 6q y 5q antes/despues de ordenar (shuffled)."""
    tri6 = np.triu_indices_from(result_6q.distance, 1)
    tri5 = np.triu_indices_from(result_5q.distance, 1)
    off_all = np.concatenate([result_6q.distance[tri6], result_5q.distance[tri5]])

    vmax_dist = np.percentile(off_all, 97)
    if vmax_dist <= 0:
        vmax_dist = 1.0

    d6_ordered = result_6q.distance[np.ix_(result_6q.order, result_6q.order)]
    d5_ordered = result_5q.distance[np.ix_(result_5q.order, result_5q.order)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    sns.heatmap(result_6q.distance, cmap="coolwarm", vmin=0, vmax=vmax_dist, square=True, cbar=True, ax=axes[0, 0])
    axes[0, 0].set_title("(a) Distancia cuantica 6q sin ordenar (shuffled)")
    axes[0, 0].tick_params(labelsize=5)

    sns.heatmap(d6_ordered, cmap="coolwarm", vmin=0, vmax=vmax_dist, square=True, cbar=True, ax=axes[0, 1])
    axes[0, 1].set_title("(b) Distancia cuantica 6q ordenada (shuffled)")
    axes[0, 1].tick_params(labelsize=5)

    sns.heatmap(result_5q.distance, cmap="coolwarm", vmin=0, vmax=vmax_dist, square=True, cbar=True, ax=axes[1, 0])
    axes[1, 0].set_title("(c) Distancia cuantica 5q sin ordenar (shuffled)")
    axes[1, 0].tick_params(labelsize=5)

    sns.heatmap(d5_ordered, cmap="coolwarm", vmin=0, vmax=vmax_dist, square=True, cbar=True, ax=axes[1, 1])
    axes[1, 1].set_title("(d) Distancia cuantica 5q ordenada (shuffled)")
    axes[1, 1].tick_params(labelsize=5)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Corrida de universo completo: contraste directo frente al piloto con preseleccion.
    print("Cargando todos los activos (tras limpieza/alineamiento)...")
    asset_names, returns_all, features_all = load_all_assets(PROJECT_ROOT, days=args.days)
    print(f"Universo -> activos={returns_all.shape[1]}, dias={returns_all.shape[0]}, features={features_all.shape[2]}")

    # Version shuffled para figuras (determinista con semilla).
    rng = np.random.default_rng(args.shuffle_seed)
    perm = rng.permutation(returns_all.shape[1])
    returns_work = returns_all.iloc[:, perm]
    features_work = features_all[perm]
    asset_names_work = [asset_names[i] for i in perm]
    print(f"Activos shuffled con seed={args.shuffle_seed}; primeros 10 indices={perm[:10].tolist()}")

    tensor_6q, names_6q, _ = scenario_features(features_work, n_qubits=6, drop_feature_index=args.drop_feature_index)
    tensor_5q, names_5q, _ = scenario_features(features_work, n_qubits=5, drop_feature_index=args.drop_feature_index)

    result_6q = run_scenario("6q_all_assets_shuffled", tensor_6q, returns_work, alpha=args.alpha)
    result_6q.used_features = names_6q

    result_5q = run_scenario("5q_all_assets_shuffled", tensor_5q, returns_work, alpha=args.alpha)
    result_5q.used_features = names_5q

    cmp_metrics = compare_scenarios(result_6q, result_5q)

    corr_shuf = np.corrcoef(returns_work.to_numpy(), rowvar=False)
    corr_classical_shuf, order_classical_shuf = classical_ordered_corr(corr_shuf)

    fig3_path = out_dir / "fig3_corr_6_vs_5_all_assets_shuffled_3x3.png"
    fig4_path = out_dir / "fig4_distance_6_vs_5_all_assets_shuffled.png"
    legacy_plot_path = out_dir / "qubits_6_vs_5_all_assets_heatmaps_shuffled.png"

    save_fig3_shuffled_3x3(
        fig3_path,
        corr_shuf,
        corr_classical_shuf,
        result_6q.corr_ordered,
        result_5q.corr_ordered,
    )
    save_fig4_distances(fig4_path, result_6q, result_5q)
    save_plot(legacy_plot_path, result_6q, result_5q)

    summary_text = build_summary_text_all_assets(
        asset_names=asset_names_work,
        returns_all=returns_work,
        features_all=features_work,
        days_requested=args.days,
        drop_feature_index=args.drop_feature_index,
        names_6q=names_6q,
        names_5q=names_5q,
        result_6q=result_6q,
        result_5q=result_5q,
        cmp_metrics=cmp_metrics,
        shuffle_seed=args.shuffle_seed,
        fig3_path=fig3_path,
        fig4_path=fig4_path,
        legacy_plot_path=legacy_plot_path,
    )

    summary_path = out_dir / "summary_6_vs_5_qubits_all_assets.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    metrics_payload = {
        "mode": "all_assets_after_cleaning",
        "asset_count": len(asset_names),
        "asset_names": asset_names,
        "shape_returns": list(returns_all.shape),
        "shape_features": list(features_all.shape),
        "days_requested": args.days,
        "shuffled": {
            "enabled": True,
            "seed": args.shuffle_seed,
            "permutation_head": perm[:10].tolist(),
        },
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
            "legacy_combined": str(legacy_plot_path),
        },
        "classical_order_shuffled_head": order_classical_shuf[:10],
        "comparison": cmp_metrics,
    }

    json_path = out_dir / "metrics_6_vs_5_qubits_all_assets.json"
    json_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("\nFinalizado.")
    print(f"- Resumen: {summary_path}")
    print(f"- Metricas: {json_path}")
    print(f"- Figura:  {fig3_path}")
    print(f"- Figura:  {fig4_path}")
    print(f"- Figura:  {legacy_plot_path}")


if __name__ == "__main__":
    main()
