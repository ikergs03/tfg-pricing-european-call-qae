#!/usr/bin/env python3
"""Analisis de sensibilidad QHRP en simulacion ideal: 6 qubits vs 5 qubits.

Este script se mantiene separado del notebook para facilitar reproducibilidad,
trazabilidad y exportacion de resultados.

Flujo:
1) Cargar y limpiar retornos + tensor de features precomputado.
2) Ejecutar una prueba piloto con preseleccion de activos (por defecto 40x90):
    primero se prioriza liquidez reciente y, si faltan activos, se completa por
    volatilidad reciente.
3) Ejecutar dos escenarios en simulacion ideal:
    - 6 qubits (6 features).
    - 5 qubits (se elimina una feature; por defecto curtosis).
4) Comparar matrices de distancia, estabilidad del orden y estructura de bloques.
5) Guardar figuras, resumen de texto y metricas en disco.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
MODULE_DIR = SCRIPT_DIR / "src"

if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from quantum_hrp_hardware import average_density_matrix, compute_distance_matrix, quantum_ordering


FEATURE_NAMES = [
    "retorno",
    "media_movil",
    "desv_std_movil",
    "retorno_acumulado",
    "asimetria",
    "curtosis",
]


@dataclass
class ScenarioResult:
    name: str
    n_qubits: int
    used_features: List[str]
    distance: np.ndarray
    order: np.ndarray
    corr_ordered: np.ndarray
    elapsed_s: float
    block_raw: float
    block_ordered: float


def block_contrast(distance_matrix: np.ndarray, band: int = 4) -> float:
    """Valores altos indican estructura de bloques mas limpia cerca de diagonal."""
    n = distance_matrix.shape[0]
    ii, jj = np.indices((n, n))
    off_diag = ii != jj
    near = (np.abs(ii - jj) <= band) & off_diag
    far = (np.abs(ii - jj) > band)
    return float(np.mean(distance_matrix[far]) - np.mean(distance_matrix[near]))


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _float_or_nan(value: float) -> float:
    if value is None:
        return float("nan")
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return val if np.isfinite(val) else float("nan")


def load_and_align_returns_features(project_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    returns_path = project_root / "data" / "returns.csv"
    features_path = project_root / "data" / "features.npy"

    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    features = np.load(features_path)

    original_columns = returns.columns.tolist()
    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.dropna(axis=0, how="any")
    returns = returns.dropna(axis=1, how="all")

    std = returns.std(axis=0)
    returns = returns.loc[:, std > 1e-8]

    surviving_idx = [original_columns.index(c) for c in returns.columns]
    features = features[surviving_idx]

    # Si returns y features no tienen el mismo largo temporal, usar el comun mas reciente.
    t_common = min(returns.shape[0], features.shape[1])
    returns = returns.tail(t_common)
    features = features[:, -t_common:, :]

    return returns, features


def select_assets_pilot(
    returns: pd.DataFrame,
    features: np.ndarray,
    project_root: Path,
    target_assets: int,
    target_days: int,
) -> tuple[List[str], pd.DataFrame, np.ndarray, str]:
    """Construye el subconjunto piloto (p.ej. 40x90) con preseleccion explicita.

    Regla de preseleccion:
    1) Ranking principal por liquidez reciente en la ventana objetivo,
       usando mediana de volumen monetario diario (precio x volumen).
    2) Si no hay suficientes activos por datos faltantes, completar con
       ranking de mayor volatilidad reciente.

    Devuelve activos seleccionados, returns y features ya alineados al piloto.
    """
    # Paso A: acotar horizonte temporal (90 dias por defecto).
    t_hw = min(target_days, returns.shape[0], features.shape[1])
    returns_recent = returns.tail(t_hw)

    prices = pd.read_csv(project_root / "data" / "prices.csv", index_col=0, low_memory=False)
    volumes = pd.read_csv(project_root / "data" / "volumes.csv", index_col=0, low_memory=False)

    prices.index = pd.to_datetime(prices.index, format="%Y-%m-%d", errors="coerce")
    volumes.index = pd.to_datetime(volumes.index, format="%Y-%m-%d", errors="coerce")
    prices = prices[~prices.index.isna()]
    volumes = volumes[~volumes.index.isna()]

    prices = prices.apply(pd.to_numeric, errors="coerce")
    volumes = volumes.apply(pd.to_numeric, errors="coerce")

    common_cols = [c for c in returns_recent.columns if c in prices.columns and c in volumes.columns]

    selection_note = "median_dollar_volume_90d"
    selected_assets: List[str] = []

    if common_cols:
        # Paso B (criterio principal): seleccionar por liquidez reciente.
        returns_liq = returns_recent[common_cols]
        prices_recent = prices.reindex(returns_liq.index).reindex(columns=common_cols)
        volumes_recent = volumes.reindex(returns_liq.index).reindex(columns=common_cols)

        dollar_volume = prices_recent * volumes_recent
        liq_score = dollar_volume.median(axis=0, skipna=True)
        liq_score = liq_score.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        selected_assets = liq_score.index[: min(target_assets, len(liq_score))].tolist()

    if len(selected_assets) < target_assets:
        # Paso C (respaldo): completar por volatilidad reciente si faltan activos.
        missing = target_assets - len(selected_assets)
        vol_score = returns_recent.std(axis=0).sort_values(ascending=False)
        fallback = [a for a in vol_score.index if a not in selected_assets][:missing]
        selected_assets = selected_assets + fallback
        selection_note = "median_dollar_volume_90d + volatility_fallback"

    # Paso D: fijar tamano final del piloto y alinear filas de features.
    selected_assets = selected_assets[: min(target_assets, len(selected_assets))]

    selected_idx = [returns.columns.get_loc(c) for c in selected_assets]
    returns_hw = returns_recent[selected_assets]
    features_hw = features[selected_idx][:, -t_hw:, :]

    return selected_assets, returns_hw, features_hw, selection_note


def scenario_features(
    features_hw: np.ndarray,
    n_qubits: int,
    drop_feature_index: int,
) -> tuple[np.ndarray, List[str], List[int]]:
    p = features_hw.shape[2]
    if n_qubits == p:
        keep_idx = list(range(p))
        names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"f{i + 1}" for i in keep_idx]
        return features_hw.copy(), names, keep_idx

    if n_qubits != p - 1:
        raise ValueError(
            f"Unsupported n_qubits={n_qubits}. This script supports {p} or {p - 1} qubits."
        )

    if not (0 <= drop_feature_index < p):
        raise ValueError(f"drop_feature_index must be in [0, {p - 1}] and got {drop_feature_index}")

    keep_idx = [i for i in range(p) if i != drop_feature_index]
    names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"f{i + 1}" for i in keep_idx]
    return features_hw[:, :, keep_idx], names, keep_idx


def run_scenario(name: str, features_tensor: np.ndarray, returns_hw: pd.DataFrame, alpha: float) -> ScenarioResult:
    n_assets, _, n_qubits = features_tensor.shape
    print(f"[{name}] Building density matrices for {n_assets} assets with {n_qubits} qubits...")

    start = time.perf_counter()
    rho_list = []
    for i in range(n_assets):
        rho_list.append(average_density_matrix(features_tensor[i], alpha=alpha))
        if (i + 1) % 10 == 0 or (i + 1) == n_assets:
            print(f"[{name}]   assets processed: {i + 1}/{n_assets}")

    distance = compute_distance_matrix(rho_list)
    order = quantum_ordering(distance)
    corr = np.corrcoef(returns_hw.to_numpy(), rowvar=False)
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


def compare_scenarios(base_6q: ScenarioResult, alt_5q: ScenarioResult) -> dict:
    tri = np.triu_indices_from(base_6q.distance, 1)
    d6 = base_6q.distance[tri]
    d5 = alt_5q.distance[tri]

    pos6 = np.empty(base_6q.order.shape[0], dtype=int)
    pos5 = np.empty(alt_5q.order.shape[0], dtype=int)
    pos6[base_6q.order] = np.arange(base_6q.order.shape[0])
    pos5[alt_5q.order] = np.arange(alt_5q.order.shape[0])

    frob_rel = np.linalg.norm(alt_5q.distance - base_6q.distance, ord="fro")
    frob_rel /= max(np.linalg.norm(base_6q.distance, ord="fro"), 1e-12)

    dist_spearman = _float_or_nan(spearmanr(d6, d5).correlation)
    order_spearman = _float_or_nan(spearmanr(pos6, pos5).correlation)
    order_kendall = _float_or_nan(kendalltau(pos6, pos5).correlation)

    return {
        "distance_pearson": _float_or_nan(_safe_corrcoef(d6, d5)),
        "distance_spearman": dist_spearman,
        "order_spearman": order_spearman,
        "order_kendall": order_kendall,
        "relative_frobenius_distortion": _float_or_nan(float(frob_rel)),
    }


def save_plot(path: Path, result_6q: ScenarioResult, result_5q: ScenarioResult) -> None:
    tri6 = np.triu_indices_from(result_6q.distance, 1)
    tri5 = np.triu_indices_from(result_5q.distance, 1)

    off_all = np.concatenate([result_6q.distance[tri6], result_5q.distance[tri5]])
    vmax_dist = np.percentile(off_all, 97)
    if vmax_dist <= 0:
        vmax_dist = 1.0

    d6_ordered = result_6q.distance[np.ix_(result_6q.order, result_6q.order)]
    d5_ordered = result_5q.distance[np.ix_(result_5q.order, result_5q.order)]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    sns.heatmap(result_6q.distance, cmap="coolwarm", vmin=0, vmax=vmax_dist, square=True, cbar=True, ax=axes[0, 0])
    axes[0, 0].set_title("6 qubits: distancia sin ordenar")
    axes[0, 0].tick_params(labelsize=5)

    sns.heatmap(d6_ordered, cmap="coolwarm", vmin=0, vmax=vmax_dist, square=True, cbar=True, ax=axes[0, 1])
    axes[0, 1].set_title("6 qubits: distancia ordenada")
    axes[0, 1].tick_params(labelsize=5)

    sns.heatmap(result_6q.corr_ordered, cmap="coolwarm", vmin=-0.2, vmax=1.0, square=True, cbar=True, ax=axes[0, 2])
    axes[0, 2].set_title("6 qubits: correlacion ordenada")
    axes[0, 2].tick_params(labelsize=5)

    sns.heatmap(result_5q.distance, cmap="coolwarm", vmin=0, vmax=vmax_dist, square=True, cbar=True, ax=axes[1, 0])
    axes[1, 0].set_title("5 qubits: distancia sin ordenar")
    axes[1, 0].tick_params(labelsize=5)

    sns.heatmap(d5_ordered, cmap="coolwarm", vmin=0, vmax=vmax_dist, square=True, cbar=True, ax=axes[1, 1])
    axes[1, 1].set_title("5 qubits: distancia ordenada")
    axes[1, 1].tick_params(labelsize=5)

    sns.heatmap(result_5q.corr_ordered, cmap="coolwarm", vmin=-0.2, vmax=1.0, square=True, cbar=True, ax=axes[1, 2])
    axes[1, 2].set_title("5 qubits: correlacion ordenada")
    axes[1, 2].tick_params(labelsize=5)

    plt.suptitle("Sensibilidad de qubits en simulacion ideal QHRP", fontsize=14)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary_text(
    selected_assets: List[str],
    returns_hw: pd.DataFrame,
    features_hw: np.ndarray,
    drop_feature_index: int,
    selection_note: str,
    result_6q: ScenarioResult,
    result_5q: ScenarioResult,
    cmp_metrics: dict,
    plot_path: Path,
) -> str:
    tri6 = np.triu_indices_from(result_6q.distance, 1)
    tri5 = np.triu_indices_from(result_5q.distance, 1)
    off6 = result_6q.distance[tri6]
    off5 = result_5q.distance[tri5]

    state_dim_6 = 2 ** result_6q.n_qubits
    state_dim_5 = 2 ** result_5q.n_qubits

    dropped_name = FEATURE_NAMES[drop_feature_index] if drop_feature_index < len(FEATURE_NAMES) else f"f{drop_feature_index + 1}"

    lines = [
        "ANALISIS 6 VS 5 QUBITS (SIMULACION IDEAL)",
        f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Contexto del experimento",
        f"- Subconjunto: N={returns_hw.shape[1]} activos, T={returns_hw.shape[0]} dias",
        f"- Tensor de entrada: {features_hw.shape}",
        f"- Criterio de seleccion de activos: {selection_note}",
        f"- Regla 5 qubits: se elimina la feature indice {drop_feature_index} ({dropped_name})",
        "",
        "Escenario 6 qubits",
        f"- Features usadas: {', '.join(result_6q.used_features)}",
        f"- Dimension Hilbert por estado: {state_dim_6}",
        f"- Tiempo simulacion: {result_6q.elapsed_s:.2f} s",
        f"- Rango distancias (off-diagonal): [{off6.min():.4f}, {off6.max():.4f}]",
        f"- Contraste bloques sin ordenar: {result_6q.block_raw:.4f}",
        f"- Contraste bloques ordenado: {result_6q.block_ordered:.4f}",
        f"- Mejora de estructura: {result_6q.block_ordered - result_6q.block_raw:+.4f}",
        "",
        "Escenario 5 qubits",
        f"- Features usadas: {', '.join(result_5q.used_features)}",
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
        "Lectura rapida",
        "- Si las correlaciones de distancia son altas y la mejora de bloques se mantiene,",
        "  entonces 5 qubits conserva estructura util para clustering, aunque con menos informacion.",
        "- Si Kendall/Spearman de ranking bajan mucho, el orden jerarquico cambia de forma relevante.",
        f"- Figura de comparacion guardada en: {plot_path}",
        "",
        "Activos seleccionados",
        "- " + ", ".join(selected_assets),
    ]

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sensibilidad QHRP en simulacion: 6 vs 5 qubits")
    parser.add_argument("--assets", type=int, default=40, help="Numero de activos del subconjunto piloto")
    parser.add_argument("--days", type=int, default=90, help="Numero de dias recientes del subconjunto piloto")
    parser.add_argument("--alpha", type=float, default=2.0, help="Parametro alpha del feature map")
    parser.add_argument(
        "--drop-feature-index",
        type=int,
        default=5,
        help="Indice de feature a eliminar en 5 qubits (por defecto 5 = curtosis)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "results_6_vs_5_qubits",
        help="Directorio de salida para figuras e informes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando y alineando datos...")
    returns, features = load_and_align_returns_features(PROJECT_ROOT)

    # Esta es la prueba solicitada de preseleccion: 40 activos x 90 dias (por defecto).
    # Se ejecuta 6q vs 5q exactamente sobre el mismo subconjunto para comparar de forma justa.
    selected_assets, returns_hw, features_hw, selection_note = select_assets_pilot(
        returns=returns,
        features=features,
        project_root=PROJECT_ROOT,
        target_assets=args.assets,
        target_days=args.days,
    )

    print(f"Subconjunto seleccionado -> activos={returns_hw.shape[1]}, dias={returns_hw.shape[0]}, features={features_hw.shape[2]}")

    tensor_6q, names_6q, _ = scenario_features(features_hw, n_qubits=6, drop_feature_index=args.drop_feature_index)
    tensor_5q, names_5q, _ = scenario_features(features_hw, n_qubits=5, drop_feature_index=args.drop_feature_index)

    result_6q = run_scenario("6q", tensor_6q, returns_hw, alpha=args.alpha)
    result_6q.used_features = names_6q
    result_5q = run_scenario("5q", tensor_5q, returns_hw, alpha=args.alpha)
    result_5q.used_features = names_5q

    cmp_metrics = compare_scenarios(result_6q, result_5q)

    plot_path = out_dir / "qubits_6_vs_5_heatmaps.png"
    save_plot(plot_path, result_6q, result_5q)

    summary_text = build_summary_text(
        selected_assets=selected_assets,
        returns_hw=returns_hw,
        features_hw=features_hw,
        drop_feature_index=args.drop_feature_index,
        selection_note=selection_note,
        result_6q=result_6q,
        result_5q=result_5q,
        cmp_metrics=cmp_metrics,
        plot_path=plot_path,
    )

    summary_path = out_dir / "summary_6_vs_5_qubits.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    metrics_payload = {
        "selected_assets": selected_assets,
        "shape_returns_subset": list(returns_hw.shape),
        "shape_features_subset": list(features_hw.shape),
        "selection_note": selection_note,
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
        "comparison": cmp_metrics,
    }

    json_path = out_dir / "metrics_6_vs_5_qubits.json"
    json_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"- Summary: {summary_path}")
    print(f"- Metrics: {json_path}")
    print(f"- Figure:  {plot_path}")


if __name__ == "__main__":
    main()
