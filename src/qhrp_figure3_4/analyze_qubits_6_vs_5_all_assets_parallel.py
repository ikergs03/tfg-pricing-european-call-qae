#!/usr/bin/env python3
"""Comparativa QHRP 6q vs 5q en todos los activos, version paralela.

Identico a analyze_qubits_6_vs_5_all_assets.py en pipeline y resultados:
solo cambia como se calcula la matriz de densidad promedio de cada activo.

En el script original, `run_scenario_exact` simula el circuito de cada
activo uno detras de otro (bucle secuencial en Python). Como cada activo
es independiente entre si, aqui se reparte esa lista de activos entre
varios procesos con ProcessPoolExecutor para aprovechar todos los nucleos
de la maquina.

Uso: mismos argumentos que el script base, mas --workers.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_qubits_6_vs_5_all_assets import (
    PROJECT_ROOT,
    SCRIPT_DIR,
    ScenarioResult,
    average_density_matrix,
    block_contrast,
    build_legacy_math_features,
    build_summary_text_all_assets,
    classical_ordered_corr,
    compare_scenarios,
    compute_distance_matrix_paper_scale1,
    load_all_assets_returns,
    quantum_ordering_ward,
    save_fig3_corr,
    save_fig4_distances,
    save_index_map,
    scenario_features_exact,
)


def _compute_one_asset_density(asset_index: int, asset_features: np.ndarray, alpha: float) -> tuple[int, np.ndarray]:
    """Simula el circuito de un unico activo. Vive a nivel de modulo para poder enviarse a cada proceso worker (pickling)."""
    rho = average_density_matrix(asset_features, alpha=alpha)
    return asset_index, rho


def run_scenario_parallel(
    name: str,
    features_tensor: np.ndarray,
    returns_work: pd.DataFrame,
    alpha: float,
    n_workers: int,
) -> ScenarioResult:
    """Version paralela de run_scenario_exact: reparte los activos entre procesos."""
    n_assets, _, n_qubits = features_tensor.shape
    print(f"[{name}] Calculando densidades para {n_assets} activos con {n_qubits} qubits usando {n_workers} procesos...")

    start = time.perf_counter()
    rho_by_index: dict[int, np.ndarray] = {}
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_compute_one_asset_density, i, features_tensor[i], alpha): i
            for i in range(n_assets)
        }
        for future in as_completed(futures):
            idx, rho = future.result()
            rho_by_index[idx] = rho
            done += 1
            if done % 10 == 0 or done == n_assets:
                print(f"[{name}]   activos procesados: {done}/{n_assets}")

    rho_list = [rho_by_index[i] for i in range(n_assets)]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QHRP todos los activos: 6q vs 5q con pipeline identico al baseline (version paralela)"
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
        default=SCRIPT_DIR / "results_6_vs_5_qubits_all_assets_parallel",
        help="Directorio de salida para figuras e informes",
    )
    parser.add_argument(
        "--axis-labels",
        type=str,
        choices=["none", "sparse", "all"],
        default="sparse",
        help=(
            "Etiquetas de ejes en los heatmaps: "
            "none (sin etiquetas), sparse (subset legible), all (todos los activos)."
        ),
    )
    parser.add_argument(
        "--max-axis-labels",
        type=int,
        default=30,
        help="Numero maximo de etiquetas por eje cuando --axis-labels=sparse.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Numero de procesos worker para paralelizar el calculo de densidades (default: todos los nucleos).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando todos los activos (tras limpieza/alineamiento)...")
    asset_names, returns_all = load_all_assets_returns(PROJECT_ROOT, days=args.days)
    print(f"Universo -> activos={returns_all.shape[1]}, dias={returns_all.shape[0]}")

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

    result_6q = run_scenario_parallel(
        "6q_all_assets", tensor_6q, returns_work, alpha=args.alpha, n_workers=args.workers
    )
    result_6q.used_features = names_6q

    result_5q = run_scenario_parallel(
        "5q_all_assets", tensor_5q, returns_work, alpha=args.alpha, n_workers=args.workers
    )
    result_5q.used_features = names_5q

    cmp_metrics = compare_scenarios(result_6q, result_5q)

    corr_raw = np.corrcoef(returns_work.to_numpy(), rowvar=False)
    corr_classical, order_classical = classical_ordered_corr(corr_raw)

    asset_names_raw = asset_names_work
    asset_names_classical = [asset_names_work[i] for i in order_classical]
    asset_names_quantum_6q = [asset_names_work[i] for i in result_6q.order]
    asset_names_quantum_5q = [asset_names_work[i] for i in result_5q.order]

    suffix = "shuffled" if args.shuffle_assets else "baseline"
    fig3_path = out_dir / f"fig3_corr_6_vs_5_all_assets_{suffix}_3x3.png"
    fig4_path = out_dir / f"fig4_distance_6_vs_5_all_assets_{suffix}.png"
    map_raw_path = out_dir / f"index_map_raw_{suffix}.csv"
    map_classical_path = out_dir / f"index_map_classical_{suffix}.csv"
    map_q6_path = out_dir / f"index_map_quantum_6q_{suffix}.csv"
    map_q5_path = out_dir / f"index_map_quantum_5q_{suffix}.csv"

    save_fig3_corr(
        fig3_path,
        corr_raw,
        corr_classical,
        result_6q.corr_ordered,
        result_5q.corr_ordered,
        shuffled=args.shuffle_assets,
        asset_names_raw=asset_names_raw,
        asset_names_classical=asset_names_classical,
        asset_names_quantum_6q=asset_names_quantum_6q,
        asset_names_quantum_5q=asset_names_quantum_5q,
        axis_labels_mode=args.axis_labels,
        max_axis_labels=args.max_axis_labels,
    )
    save_fig4_distances(
        fig4_path,
        result_6q,
        result_5q,
        vmax=args.fig4_vmax,
        shuffled=args.shuffle_assets,
        asset_names_raw=asset_names_raw,
        asset_names_quantum_6q=asset_names_quantum_6q,
        asset_names_quantum_5q=asset_names_quantum_5q,
        axis_labels_mode=args.axis_labels,
        max_axis_labels=args.max_axis_labels,
    )

    save_index_map(map_raw_path, asset_names_raw, list(range(len(asset_names_raw))))
    save_index_map(map_classical_path, asset_names_classical, order_classical)
    save_index_map(map_q6_path, asset_names_quantum_6q, result_6q.order.tolist())
    save_index_map(map_q5_path, asset_names_quantum_5q, result_5q.order.tolist())

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
        "mode": "all_assets_after_cleaning_parallel",
        "pipeline": {
            "features": "legacy_math_features",
            "distance": "frobenius_scale1",
            "ordering": "ward",
            "shuffle_assets": args.shuffle_assets,
            "shuffle_seed": args.shuffle_seed if args.shuffle_assets else None,
            "fig4_vmax": args.fig4_vmax,
            "workers": args.workers,
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
            "index_map_raw": str(map_raw_path),
            "index_map_classical": str(map_classical_path),
            "index_map_quantum_6q": str(map_q6_path),
            "index_map_quantum_5q": str(map_q5_path),
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
    print(f"- Mapa indices: {map_raw_path}")
    print(f"- Mapa indices: {map_classical_path}")
    print(f"- Mapa indices: {map_q6_path}")
    print(f"- Mapa indices: {map_q5_path}")


if __name__ == "__main__":
    main()
