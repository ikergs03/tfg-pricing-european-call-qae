#!/usr/bin/env python3
"""QHRP: aprovechar todos los qubits de un ordenador cuantico real (Qiskit/IBM).

Cada activo se codifica en un feature-map de P qubits (P=6, o P=5 en la
variante reducida) que solo entrelaza qubits dentro de su propio bloque. Como
los bloques de activos distintos son independientes entre si, varios caben
sin problema en un unico circuito mas ancho: dado el numero de qubits fisicos
del ordenador cuantico objetivo, este script agrupa los activos de
`total_qubits // qubits_por_activo` en `total_qubits // qubits_por_activo`,
construye un circuito conjunto por grupo y por paso temporal, y lo ejecuta
como una unica tanda. Esto reduce el numero de circuitos que hay que enviar
a la cola del backend en un factor igual al tamano de grupo, que en hardware
real (donde el cuello de botella suele ser la cola, no la CPU) importa mucho
mas que el numero de qubits en si.

Ejemplo: con 32 qubits y activos de 6 qubits, 32 // 6 = 5 activos por grupo.
Con 152 qubits, 152 // 6 = 25 activos por grupo (2 qubits quedan ociosos).

Uso (ejecutar desde esta carpeta, publicacion/):
    python qiskit_full_utilization.py --total-qubits 32
    python qiskit_full_utilization.py --total-qubits 127 --backend-name ibm_torino

Sin --backend-name se ejecuta localmente en AerSimulator (solo pensado para
probar la logica de agrupacion con pocos qubits: simular N qubits localmente
cuesta ~2^N amplitudes, asi que --total-qubits grande sin hardware real es
inviable en un portatil).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from qiskit import QuantumCircuit, transpile

try:
    from qiskit_aer import AerSimulator
except Exception:  # pragma: no cover - dependencia opcional
    AerSimulator = None

try:
    from qiskit_ibm_runtime import Batch, Sampler
except Exception:  # pragma: no cover - dependencia opcional
    Batch = None
    Sampler = None

SCRIPT_DIR = Path(__file__).resolve().parent

# Este fichero vive en <TFG>/src/qhrp_figure3_4/publicacion; quantum_hrp_hardware.py
# esta un nivel arriba, en <TFG>/src/qhrp_figure3_4/src.
QHRP_SRC_DIR = SCRIPT_DIR.parent / "src"
if str(QHRP_SRC_DIR) not in sys.path:
    sys.path.append(str(QHRP_SRC_DIR))

from quantum_hrp_hardware import get_ibm_backend  # noqa: E402

from asset_features import (  # noqa: E402
    PROJECT_ROOT,
    build_legacy_math_features,
    load_all_assets_returns,
    scenario_features_exact,
)
from quantum_batching import (  # noqa: E402
    BatchPlan,
    compute_distribution_distance_matrix,
    plan_batches,
    quantum_ordering_from_distance,
    theta_from_observation,
)


DEFAULT_ENTANGLERS: List[Tuple[int, int]] = [
    (0, 1),
    (0, 5),
    (2, 4),
    (0, 4),
    (2, 3),
    (1, 2),
]


##############################################
# Construccion de circuitos: empaquetar varios activos en un circuito ancho
##############################################
def _append_feature_map_gates(
    qc: QuantumCircuit, offset: int, theta: np.ndarray, pairs: Sequence[Tuple[int, int]]
) -> None:
    p = len(theta)
    qc.h(range(offset, offset + p))
    for i in range(p):
        qc.t(offset + i)
        qc.ry(theta[i], offset + i)
    for i, j in pairs:
        if i < p and j < p:
            qc.iswap(offset + i, offset + j)
    for i in range(p):
        qc.t(offset + i)
        qc.rx(theta[i], offset + i)
    for i, j in pairs:
        if i < p and j < p:
            qc.iswap(offset + i, offset + j)
    for i in range(p):
        qc.t(offset + i)
        qc.ry(theta[i], offset + i)
    for i, j in pairs:
        if i < p and j < p:
            qc.iswap(offset + i, offset + j)
    for i in range(p):
        qc.t(offset + i)
        qc.h(offset + i)


def build_grouped_circuit(
    group_observations: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    qubits_per_asset: int,
    alpha: float = 2.0,
    entanglement_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> QuantumCircuit:
    """Un circuito para un paso temporal, con un bloque de qubits disjunto por activo del grupo."""
    pairs = entanglement_pairs or DEFAULT_ENTANGLERS
    group_size = len(group_observations)
    qc = QuantumCircuit(group_size * qubits_per_asset)
    for k, (x, x_min, x_max) in enumerate(group_observations):
        theta = theta_from_observation(x, x_min, x_max, alpha)
        _append_feature_map_gates(qc, k * qubits_per_asset, theta, pairs)
    qc.measure_all()
    return qc


##############################################
# Ejecucion y contabilidad por activo
##############################################
def _extract_counts(result, idx: int) -> Dict[str, int]:
    return result[idx].data.meas.get_counts()


def run_grouped_circuits(
    circuits: Sequence[QuantumCircuit],
    backend,
    shots: int,
    optimization_level: int = 1,
    max_circuits_per_job: int = 300,
    use_batch: bool = True,
) -> List[Dict[str, int]]:
    if Sampler is None:
        raise ImportError(
            "qiskit_ibm_runtime es necesario para ejecutar circuitos. "
            "Instalalo con: pip install qiskit-ibm-runtime"
        )

    circuits_t = transpile(circuits, backend=backend, optimization_level=optimization_level)
    chunks = [
        circuits_t[start : start + max_circuits_per_job]
        for start in range(0, len(circuits_t), max_circuits_per_job)
    ]

    counts_list: List[Dict[str, int]] = []
    if use_batch:
        if Batch is None:
            raise ImportError("qiskit_ibm_runtime.Batch no esta disponible. Instala/actualiza qiskit-ibm-runtime.")
        with Batch(backend=backend) as batch:
            sampler = Sampler(mode=batch)
            jobs = [sampler.run(chunk, shots=shots) for chunk in chunks]
            for job in jobs:
                result = job.result()
                for idx in range(len(result)):
                    counts_list.append(_extract_counts(result, idx))
    else:
        sampler = Sampler(mode=backend)
        for chunk in chunks:
            result = sampler.run(chunk, shots=shots).result()
            for idx in range(len(result)):
                counts_list.append(_extract_counts(result, idx))

    return counts_list


def _split_counts_per_asset(counts: Dict[str, int], group_size: int, qubits_per_asset: int) -> List[Dict[str, int]]:
    """Divide cada bitstring conjunto en un trozo por bloque de activo.

    Los bitstrings de Qiskit son little-endian (el caracter mas a la derecha
    es el qubit 0), y el activo k ocupa los qubits [k*P, (k+1)*P). Eso
    equivale a un trozo *contiguo* por activo, asi que no hace falta
    reordenar bits: basta con recortar desde la derecha. Un circuito suelto
    de P qubits para ese mismo activo produciria un bitstring identico por
    disparo, lo que sirve como comprobacion de esta division.
    """
    length = group_size * qubits_per_asset
    per_asset: List[Dict[str, int]] = [dict() for _ in range(group_size)]
    for bitstring, c in counts.items():
        b = bitstring.replace(" ", "")
        for k in range(group_size):
            chunk = b[length - (k + 1) * qubits_per_asset : length - k * qubits_per_asset]
            per_asset[k][chunk] = per_asset[k].get(chunk, 0) + c
    return per_asset


def _counts_to_probs(counts: Dict[str, int], n_qubits: int, shots: int) -> np.ndarray:
    vec = np.zeros(2**n_qubits, dtype=float)
    if shots <= 0:
        return vec
    for bitstring, c in counts.items():
        vec[int(bitstring, 2)] = c / shots
    return vec


##############################################
# Punto de entrada publico
##############################################
def compute_asset_distributions_full_utilization(
    features_tensor: np.ndarray,
    backend,
    total_qubits: int,
    shots: int = 2048,
    alpha: float = 2.0,
    qubits_per_asset: Optional[int] = None,
    entanglement_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    optimization_level: int = 1,
    max_circuits_per_job: int = 300,
    use_batch: bool = True,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], BatchPlan]:
    """Calcula una distribucion de medida promedio por activo, usando todos los `total_qubits`.

    features_tensor con forma: (n_activos, T, P). Devuelve (distribuciones, plan).
    """
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

    all_circuits: List[QuantumCircuit] = []
    owners: List[List[int]] = []
    for group in plan.groups:
        mins = {i: features_tensor[i].min(axis=0) for i in group}
        maxs = {i: features_tensor[i].max(axis=0) for i in group}
        for t in range(t_steps):
            observations = [(features_tensor[i, t], mins[i], maxs[i]) for i in group]
            qc = build_grouped_circuit(observations, qpa, alpha=alpha, entanglement_pairs=entanglement_pairs)
            all_circuits.append(qc)
            owners.append(group)

    counts_list = run_grouped_circuits(
        all_circuits,
        backend,
        shots=shots,
        optimization_level=optimization_level,
        max_circuits_per_job=max_circuits_per_job,
        use_batch=use_batch,
    )

    probs_by_asset: Dict[int, List[np.ndarray]] = {i: [] for i in range(n_assets)}
    for group, counts in zip(owners, counts_list):
        per_asset_counts = _split_counts_per_asset(counts, len(group), qpa)
        for local_k, asset_idx in enumerate(group):
            probs_by_asset[asset_idx].append(_counts_to_probs(per_asset_counts[local_k], qpa, shots))

    distributions = [np.mean(np.stack(probs_by_asset[i], axis=0), axis=0) for i in range(n_assets)]
    return distributions, plan


def build_local_simulator(total_qubits: int, force: bool = False):
    if AerSimulator is None:
        raise ImportError("qiskit-aer es necesario para el modo local. Instalalo con: pip install qiskit-aer")
    if total_qubits > 24 and not force:
        gb = 2**total_qubits * 16 / 1e9
        raise RuntimeError(
            f"Simular localmente {total_qubits} qubits requiere ~2^{total_qubits} amplitudes "
            f"(~{gb:.1f} GB en complex128); eso solo es gratis en un ordenador cuantico real, "
            "no en un simulador clasico. Usa --backend-name para un backend real de IBM, o "
            "pasa --force-local si de verdad quieres forzarlo (a tu riesgo)."
        )
    return AerSimulator()


##############################################
# CLI (linea de comandos)
##############################################
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QHRP: agrupa activos para usar todos los qubits de un ordenador cuantico (Qiskit/IBM)."
    )
    parser.add_argument("--total-qubits", type=int, required=True, help="Qubits fisicos del backend objetivo")
    parser.add_argument("--qubits-per-asset", type=int, default=6, choices=[5, 6])
    parser.add_argument("--days", type=int, default=756)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--drop-feature-index", type=int, default=5)
    parser.add_argument(
        "--backend-name", type=str, default=None, help="Backend real de IBM; si se omite, se usa AerSimulator local"
    )
    parser.add_argument("--account-name", type=str, default="labqcc-2025")
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--max-circuits-per-job", type=int, default=300)
    parser.add_argument(
        "--force-local", action="store_true", help="Permite simular localmente mas de 24 qubits (lento/costoso)"
    )
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "results_full_utilization_qiskit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando activos...")
    asset_names, returns = load_all_assets_returns(PROJECT_ROOT, days=args.days)
    features = build_legacy_math_features(returns)
    tensor, used_names, _ = scenario_features_exact(
        features, n_qubits=args.qubits_per_asset, drop_feature_index=args.drop_feature_index
    )
    print(f"Universo -> activos={tensor.shape[0]}, dias={tensor.shape[1]}, features={used_names}")

    if args.backend_name:
        backend = get_ibm_backend(
            min_num_qubits=args.total_qubits,
            account_name=args.account_name,
            backend_name=args.backend_name,
        )
        use_batch = True
        print(f"Backend real: {backend.name} ({backend.num_qubits} qubits fisicos)")
    else:
        backend = build_local_simulator(args.total_qubits, force=args.force_local)
        use_batch = False
        print(f"Sin --backend-name: AerSimulator local, circuitos de hasta {args.total_qubits} qubits.")

    start = time.perf_counter()
    distributions, plan = compute_asset_distributions_full_utilization(
        tensor,
        backend,
        total_qubits=args.total_qubits,
        shots=args.shots,
        alpha=args.alpha,
        optimization_level=args.optimization_level,
        max_circuits_per_job=args.max_circuits_per_job,
        use_batch=use_batch,
    )
    elapsed = time.perf_counter() - start

    distance = compute_distribution_distance_matrix(distributions, metric="hellinger")
    order = quantum_ordering_from_distance(distance)

    print(f"\nCompletado en {elapsed:.1f}s")
    print(f"Orden resultante: {[asset_names[i] for i in order]}")

    payload = {
        "backend": "qiskit",
        "total_qubits": args.total_qubits,
        "qubits_per_asset": args.qubits_per_asset,
        "batch_size": plan.batch_size,
        "n_batches": plan.n_batches,
        "idle_qubits_per_batch": plan.idle_qubits,
        "n_assets": len(asset_names),
        "used_features": used_names,
        "shots": args.shots,
        "backend_name": args.backend_name,
        "elapsed_seconds": elapsed,
        "asset_order_indices": order.tolist(),
        "asset_order_names": [asset_names[i] for i in order],
    }
    out_path = args.out_dir / "summary_full_utilization_qiskit.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Resumen guardado en: {out_path}")


if __name__ == "__main__":
    main()
