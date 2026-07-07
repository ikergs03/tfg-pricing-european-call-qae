#!/usr/bin/env python3
"""Genera la tabla comparativa de tiempos secuencial vs paralelo.

Lee metrics_6_vs_5_qubits_all_assets.json de dos carpetas de resultados
(una generada por analyze_qubits_6_vs_5_all_assets.py, otra por
analyze_qubits_6_vs_5_all_assets_parallel.py) y compara timing_seconds.

Uso:
    python3 compare_timings.py \
        --sequential-dir results_6_vs_5_qubits_all_assets_sequential_test \
        --parallel-dir results_6_vs_5_qubits_all_assets_parallel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
METRICS_FILENAME = "metrics_6_vs_5_qubits_all_assets.json"


def load_timing(results_dir: Path) -> dict:
    metrics_path = results_dir / METRICS_FILENAME
    if not metrics_path.exists():
        raise FileNotFoundError(f"No se encontro {metrics_path}")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    timing = payload["timing_seconds"]
    return {
        "six_qubits": float(timing["six_qubits"]),
        "five_qubits": float(timing["five_qubits"]),
        "workers": payload.get("pipeline", {}).get("workers"),
        "asset_count": payload.get("asset_count"),
        "days_requested": payload.get("days_requested"),
    }


def format_seconds(seconds: float) -> str:
    if seconds >= 60:
        return f"{seconds:.2f} s (~{seconds / 60:.1f} min)"
    return f"{seconds:.2f} s"


def compute_rows(seq: dict, par: dict) -> list[tuple[str, float, float, float]]:
    pairs = [
        ("6 qubits", seq["six_qubits"], par["six_qubits"]),
        ("5 qubits", seq["five_qubits"], par["five_qubits"]),
    ]
    total_seq = seq["six_qubits"] + seq["five_qubits"]
    total_par = par["six_qubits"] + par["five_qubits"]
    pairs.append(("Total simulacion", total_seq, total_par))

    return [
        (name, t_seq, t_par, (t_seq / t_par if t_par > 0 else float("nan")))
        for name, t_seq, t_par in pairs
    ]


def build_table(rows: list[tuple[str, float, float, float]]) -> tuple[str, str]:
    md_lines = [
        "| Escenario | Secuencial | Paralela | Speedup |",
        "|---|---|---|---|",
    ]
    csv_lines = ["escenario,secuencial_s,paralela_s,speedup"]

    for name, t_seq, t_par, speedup in rows:
        md_lines.append(f"| {name} | {format_seconds(t_seq)} | {format_seconds(t_par)} | {speedup:.1f}x |")
        csv_lines.append(f"{name},{t_seq:.4f},{t_par:.4f},{speedup:.2f}")

    return "\n".join(md_lines), "\n".join(csv_lines)


def save_png_table(path: Path, rows: list[tuple[str, float, float, float]], header: str) -> None:
    """Genera la tabla como figura, igual que el resto de figuras del proyecto (matplotlib)."""
    col_labels = ["Escenario", "Secuencial", "Paralela", "Speedup"]
    cell_text = [
        [name, format_seconds(t_seq), format_seconds(t_par), f"{speedup:.1f}x"]
        for name, t_seq, t_par, speedup in rows
    ]

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.4))
    ax.axis("off")
    ax.set_title(header, fontsize=11, pad=14)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.34, 0.22, 0.22, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    n_cols = len(col_labels)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row_idx == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f2f2f2" if row_idx % 2 == 0 else "#ffffff")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara tiempos secuencial vs paralelo (QHRP 6q vs 5q)")
    parser.add_argument(
        "--sequential-dir",
        type=Path,
        default=SCRIPT_DIR / "results_6_vs_5_qubits_all_assets",
        help="Carpeta de resultados de la version secuencial",
    )
    parser.add_argument(
        "--parallel-dir",
        type=Path,
        default=SCRIPT_DIR / "results_6_vs_5_qubits_all_assets_parallel",
        help="Carpeta de resultados de la version paralela",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=SCRIPT_DIR / "comparacion_tiempos.md",
        help="Ruta de salida para la tabla en Markdown",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=SCRIPT_DIR / "comparacion_tiempos.csv",
        help="Ruta de salida para la tabla en CSV",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=SCRIPT_DIR / "comparacion_tiempos.png",
        help="Ruta de salida para la tabla en PNG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seq = load_timing(args.sequential_dir)
    par = load_timing(args.parallel_dir)

    if seq["asset_count"] != par["asset_count"] or seq["days_requested"] != par["days_requested"]:
        print(
            "[Aviso] Las dos corridas no usan el mismo universo/horizonte "
            f"(secuencial: assets={seq['asset_count']}, days={seq['days_requested']} | "
            f"paralela: assets={par['asset_count']}, days={par['days_requested']}). "
            "La comparacion puede no ser homogenea."
        )

    rows = compute_rows(seq, par)
    md_table, csv_table = build_table(rows)

    header = (
        f"Comparacion de tiempos ({par['asset_count']} activos, "
        f"{par['days_requested']} dias, {par['workers']} workers en la version paralela)"
    )
    print(header)
    print()
    print(md_table)

    args.out_md.write_text(header + "\n\n" + md_table + "\n", encoding="utf-8")
    args.out_csv.write_text(csv_table + "\n", encoding="utf-8")
    save_png_table(args.out_png, rows, header)

    print()
    print(f"Guardado en: {args.out_md}")
    print(f"Guardado en: {args.out_csv}")
    print(f"Guardado en: {args.out_png}")


if __name__ == "__main__":
    main()
