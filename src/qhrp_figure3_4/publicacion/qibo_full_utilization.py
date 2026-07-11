#!/usr/bin/env python3
"""QHRP: aprovechar todos los qubits de un ordenador cuantico real (Qibo).

Version en Qibo del mismo enfoque que qiskit_full_utilization.py: cada activo
usa un feature-map de P qubits (P=6, o P=5) que solo entrelaza qubits dentro
de su propio bloque, asi que varios activos independientes caben en un unico
circuito mas ancho. Dado el numero de qubits fisicos del ordenador objetivo,
este script agrupa los activos de `total_qubits // qubits_por_activo` en
`total_qubits // qubits_por_activo` y ejecuta un circuito conjunto por grupo
y por paso temporal (en vez de un circuito por activo).

Por defecto usa el backend de simulacion local de Qibo ("numpy"), pensado
para validar la logica de agrupacion con pocos qubits: simular N qubits
localmente cuesta ~2^N amplitudes, igual que con cualquier simulador
clasico. Para hardware real, Qibo necesita un backend/plataforma aparte
(p.ej. qibolab para hardware propio, o qibo-cloud-backends para un proveedor
en la nube); ninguno de los dos esta instalado en este proyecto, asi que
--qibo-backend/--platform quedan como punto de enganche para cuando se
configure ese acceso: la logica de agrupacion y construccion de circuitos no
cambia, solo el backend que ejecuta el circuito.

Uso (ejecutar desde esta carpeta, publicacion/):
    python qibo_full_utilization.py --total-qubits 32
    python qibo_full_utilization.py --total-qubits 20 --qibo-backend qibojit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import qibo
    from qibo import Circuit, gates
except Exception:  # pragma: no cover - dependencia opcional
    qibo = None
    Circuit = None
    gates = None

from asset_features import (
    PROJECT_ROOT,
    build_legacy_math_features,
    load_all_assets_returns,
    scenario_features_exact,
)
from quantum_batching import (
    BatchPlan,
    compute_distribution_distance_matrix,
    plan_batches,
    quantum_ordering_from_distance,
    theta_from_observation,
)

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_ENTANGLERS: List[Tuple[int, int]] = [
    (0, 1),
    (0, 5),
    (2, 4),
    (0, 4),
    (2, 3),
    (1, 2),
]

LOCAL_SIMULATOR_BACKENDS = {"numpy", "qibojit", "tensorflow", "pytorch"}


##############################################
# Construccion de circuitos: empaquetar varios activos en un circuito ancho
##############################################
def _append_feature_map_gates(circuit, offset: int, theta: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> None:
    p = len(theta)
    for i in range(p):
        circuit.add(gates.H(offset + i))
    for i in range(p):
        circuit.add(gates.T(offset + i))
        circuit.add(gates.RY(offset + i, theta[i]))
    for i, j in pairs:
        if i < p and j < p:
            circuit.add(gates.iSWAP(offset + i, offset + j))
    for i in range(p):
        circuit.add(gates.T(offset + i))
        circuit.add(gates.RX(offset + i, theta[i]))
    for i, j in pairs:
        if i < p and j < p:
            circuit.add(gates.iSWAP(offset + i, offset + j))
    for i in range(p):
        circuit.add(gates.T(offset + i))
        circuit.add(gates.RY(offset + i, theta[i]))
    for i, j in pairs:
        if i < p and j < p:
            circuit.add(gates.iSWAP(offset + i, offset + j))
    for i in range(p):
        circuit.add(gates.T(offset + i))
        circuit.add(gates.H(offset + i))


def build_grouped_circuit(
    group_observations: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    qubits_per_asset: int,
    alpha: float = 2.0,
    entanglement_pairs: Optional[Sequence[Tuple[int, int]]] = None,
):
    """Un circuito para un paso temporal, con un bloque de qubits disjunto por activo del grupo."""
    if Circuit is None:
        raise ImportError("qibo es necesario. Instalalo con: pip install qibo")

    pairs = entanglement_pairs or DEFAULT_ENTANGLERS
    group_size = len(group_observations)
    n = group_size * qubits_per_asset
    circuit = Circuit(n)
    for k, (x, x_min, x_max) in enumerate(group_observations):
        theta = theta_from_observation(x, x_min, x_max, alpha)
        _append_feature_map_gates(circuit, k * qubits_per_asset, theta, pairs)
    circuit.add(gates.M(*range(n)))
    return circuit


##############################################
# Ejecucion y contabilidad por activo
##############################################
def _samples_to_probs(samples_block: np.ndarray, p: int) -> np.ndarray:
    """Convierte un array (nshots, p) de 0/1 en un vector de probabilidad de longitud 2^p.

    samples() de Qibo devuelve una columna por qubit medido, en el orden en
    que se pasaron a gates.M, asi que el bloque del activo k son simplemente
    las columnas [k*P, (k+1)*P). El qubit local 0 (columna mas a la
    izquierda) se trata como el bit mas significativo; solo hace falta que
    sea consistente dentro de este script.
    """
    weights = 1 << np.arange(p - 1, -1, -1)
    idx = samples_block.astype(np.int64) @ weights
    counts = np.bincount(idx, minlength=2**p).astype(float)
    return counts / max(len(samples_block), 1)


def compute_asset_distributions_full_utilization(
    features_tensor: np.ndarray,
    total_qubits: int,
    shots: int = 2048,
    alpha: float = 2.0,
    qubits_per_asset: Optional[int] = None,
    entanglement_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    backend: Optional[str] = "numpy",
    platform: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], BatchPlan]:
    """Calcula una distribucion de medida promedio por activo, usando todos los `total_qubits`.

    features_tensor con forma: (n_activos, T, P). Devuelve (distribuciones, plan).
    """
    if qibo is None:
        raise ImportError("qibo es necesario. Instalalo con: pip install qibo")

    if backend is not None:
        if platform is not None:
            qibo.set_backend(backend, platform=platform)
        else:
            qibo.set_backend(backend)

    n_assets, t_steps, p_features = features_tensor.shape
    qpa = qubits_per_asset or p_features
    if qpa != p_features:
        raise ValueError(f"qubits_per_asset ({qpa}) no coincide con las features del tensor ({p_features})")

    plan = plan_batches(n_assets, total_qubits, qpa)
    if verbose:
        print(plan.summary())
        print(
            f"Circuitos a ejecutar: {plan.n_batches * t_steps} "
            f"(vs. {n_assets * t_steps} si cada activo fuese suelto)"
        )

    probs_by_asset: dict = {i: [] for i in range(n_assets)}
    for group in plan.groups:
        mins = {i: features_tensor[i].min(axis=0) for i in group}
        maxs = {i: features_tensor[i].max(axis=0) for i in group}
        for t in range(t_steps):
            observations = [(features_tensor[i, t], mins[i], maxs[i]) for i in group]
            circuit = build_grouped_circuit(observations, qpa, alpha=alpha, entanglement_pairs=entanglement_pairs)
            samples = circuit(nshots=shots).samples(binary=True)
            for local_k, asset_idx in enumerate(group):
                block = samples[:, local_k * qpa : (local_k + 1) * qpa]
                probs_by_asset[asset_idx].append(_samples_to_probs(block, qpa))

    distributions = [np.mean(np.stack(probs_by_asset[i], axis=0), axis=0) for i in range(n_assets)]
    return distributions, plan


def _check_local_simulation_size(total_qubits: int, backend_name: str, force: bool) -> None:
    if backend_name in LOCAL_SIMULATOR_BACKENDS and total_qubits > 24 and not force:
        gb = 2**total_qubits * 16 / 1e9
        raise RuntimeError(
            f"Simular localmente {total_qubits} qubits en el backend '{backend_name}' requiere "
            f"~2^{total_qubits} amplitudes (~{gb:.1f} GB); eso solo es gratis en un chip cuantico "
            "real, no en un simulador clasico. Apunta a una plataforma real con --platform, o "
            "pasa --force-local si de verdad quieres forzarlo (a tu riesgo)."
        )


##############################################
# CLI (linea de comandos)
##############################################
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QHRP: agrupa activos para usar todos los qubits de un ordenador cuantico (Qibo)."
    )
    parser.add_argument("--total-qubits", type=int, required=True, help="Qubits fisicos del backend objetivo")
    parser.add_argument("--qubits-per-asset", type=int, default=6, choices=[5, 6])
    parser.add_argument("--days", type=int, default=756)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--drop-feature-index", type=int, default=5)
    parser.add_argument(
        "--qibo-backend",
        type=str,
        default="numpy",
        help="'numpy'/'qibojit' para simulacion local, o el nombre de una plataforma real "
        "(requiere qibolab / qibo-cloud-backends instalado y configurado aparte).",
    )
    parser.add_argument(
        "--platform", type=str, default=None, help="Plataforma de hardware real para qibo.set_backend, si aplica"
    )
    parser.add_argument(
        "--force-local", action="store_true", help="Permite simular localmente mas de 24 qubits (lento/costoso)"
    )
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "results_full_utilization_qibo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _check_local_simulation_size(args.total_qubits, args.qibo_backend, args.force_local)

    print("Cargando activos...")
    asset_names, returns = load_all_assets_returns(PROJECT_ROOT, days=args.days)
    features = build_legacy_math_features(returns)
    tensor, used_names, _ = scenario_features_exact(
        features, n_qubits=args.qubits_per_asset, drop_feature_index=args.drop_feature_index
    )
    print(f"Universo -> activos={tensor.shape[0]}, dias={tensor.shape[1]}, features={used_names}")

    start = time.perf_counter()
    distributions, plan = compute_asset_distributions_full_utilization(
        tensor,
        total_qubits=args.total_qubits,
        shots=args.shots,
        alpha=args.alpha,
        qubits_per_asset=args.qubits_per_asset,
        backend=args.qibo_backend,
        platform=args.platform,
    )
    elapsed = time.perf_counter() - start

    distance = compute_distribution_distance_matrix(distributions, metric="hellinger")
    order = quantum_ordering_from_distance(distance)

    print(f"\nCompletado en {elapsed:.1f}s")
    print(f"Orden resultante: {[asset_names[i] for i in order]}")

    payload = {
        "backend": "qibo",
        "qibo_backend": args.qibo_backend,
        "platform": args.platform,
        "total_qubits": args.total_qubits,
        "qubits_per_asset": args.qubits_per_asset,
        "batch_size": plan.batch_size,
        "n_batches": plan.n_batches,
        "idle_qubits_per_batch": plan.idle_qubits,
        "n_assets": len(asset_names),
        "used_features": used_names,
        "shots": args.shots,
        "elapsed_seconds": elapsed,
        "asset_order_indices": order.tolist(),
        "asset_order_names": [asset_names[i] for i in order],
    }
    out_path = args.out_dir / "summary_full_utilization_qibo.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Resumen guardado en: {out_path}")


if __name__ == "__main__":
    main()
