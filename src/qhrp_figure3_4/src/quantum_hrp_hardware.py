"""Utilities to run a QHRP-style pipeline on real quantum hardware.

This module is intentionally separate from the simulation-based implementation
in quantum_hrp.py. It keeps the same conceptual flow:

1) Encode feature vectors into quantum circuits.
2) Execute circuits on a real backend.
3) Build per-asset average measurement distributions.
4) Compute a distance matrix between assets.
5) Obtain hierarchical ordering from that distance matrix.

Unlike the simulator code, this module does not build exact density matrices.
On real hardware we use measurement distributions as the observable output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
except Exception:  # pragma: no cover - optional dependency in local dev
    QiskitRuntimeService = None
    Sampler = None


DEFAULT_ENTANGLERS: List[Tuple[int, int]] = [
    (0, 1),
    (0, 5),
    (2, 4),
    (0, 4),
    (2, 3),
    (1, 2),
]


@dataclass
class HardwareConfig:
    """Configuration for hardware execution."""

    shots: int = 2048
    alpha: float = 2.0
    optimization_level: int = 1


def build_feature_map_circuit(
    x: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
    alpha: float = 2.0,
    entanglement_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    with_measurements: bool = True,
) -> QuantumCircuit:
    """Build the QHRP feature-map circuit for one observation.

    This mirrors the encoding logic from the simulation version and appends
    measurements so the circuit is executable on real hardware.
    """
    n = len(x)
    pairs = entanglement_pairs or DEFAULT_ENTANGLERS

    theta = np.zeros(n)
    for i in range(n):
        if x_max[i] > x_min[i]:
            theta[i] = alpha * math.pi * (x[i] - x_min[i]) / (x_max[i] - x_min[i])
        else:
            theta[i] = 0.0

    qc = QuantumCircuit(n)
    qc.h(range(n))

    # Round 1
    for i in range(n):
        qc.t(i)
        qc.ry(theta[i], i)
    for i, j in pairs:
        if i < n and j < n:
            qc.iswap(i, j)

    # Round 2
    for i in range(n):
        qc.t(i)
        qc.rx(theta[i], i)
    for i, j in pairs:
        if i < n and j < n:
            qc.iswap(i, j)

    # Round 3
    for i in range(n):
        qc.t(i)
        qc.ry(theta[i], i)
    for i, j in pairs:
        if i < n and j < n:
            qc.iswap(i, j)

    # Final layer
    for i in range(n):
        qc.t(i)
        qc.h(i)

    if with_measurements:
        qc.measure_all()

    return qc


def counts_to_probability_vector(
    counts: Dict[str, int],
    n_qubits: int,
    shots: int,
) -> np.ndarray:
    """Convert shot counts into a 2^n probability vector."""
    vec = np.zeros(2**n_qubits, dtype=float)
    if shots <= 0:
        return vec

    for bitstring, c in counts.items():
        idx = int(bitstring.replace(" ", ""), 2)
        vec[idx] = c / shots
    return vec


def _extract_counts_from_sampler_result(result, idx: int = 0) -> Dict[str, int]:
    """Extract counts from a SamplerV2 result in a robust way."""
    pub = result[idx]
    # Runtime typically exposes pub.data.meas.get_counts().
    return pub.data.meas.get_counts()


def run_circuits_on_backend(
    circuits: Sequence[QuantumCircuit],
    backend,
    shots: int = 2048,
    optimization_level: int = 1,
) -> List[Dict[str, int]]:
    """Transpile and run a list of circuits on a real backend.

    Returns one counts dictionary per circuit.
    """
    if Sampler is None:
        raise ImportError(
            "qiskit_ibm_runtime is required for real-hardware execution. "
            "Install it with: pip install qiskit-ibm-runtime"
        )

    circuits_t = transpile(
        circuits,
        backend=backend,
        optimization_level=optimization_level,
    )

    sampler = Sampler(mode=backend)
    job = sampler.run(circuits_t, shots=shots)
    result = job.result()

    counts_list: List[Dict[str, int]] = []
    for i in range(len(circuits_t)):
        counts_list.append(_extract_counts_from_sampler_result(result, i))
    return counts_list


def average_measurement_distribution(
    asset_data: np.ndarray,
    backend,
    config: HardwareConfig = HardwareConfig(),
    entanglement_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Compute average measurement distribution for one asset.

    asset_data shape: (T, P)
    Returns a vector of size 2^P.
    """
    t_steps, p_features = asset_data.shape
    x_min = np.min(asset_data, axis=0)
    x_max = np.max(asset_data, axis=0)

    circuits = [
        build_feature_map_circuit(
            asset_data[t],
            x_min=x_min,
            x_max=x_max,
            alpha=config.alpha,
            entanglement_pairs=entanglement_pairs,
            with_measurements=True,
        )
        for t in range(t_steps)
    ]

    counts_list = run_circuits_on_backend(
        circuits,
        backend=backend,
        shots=config.shots,
        optimization_level=config.optimization_level,
    )

    probs = [
        counts_to_probability_vector(c, n_qubits=p_features, shots=config.shots)
        for c in counts_list
    ]
    return np.mean(np.stack(probs, axis=0), axis=0)


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two probability distributions."""
    p = np.clip(p, 0.0, 1.0)
    q = np.clip(q, 0.0, 1.0)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon distance between two probability distributions."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.sqrt(max(js, 0.0)))


def compute_distribution_distance_matrix(
    dist_list: Sequence[np.ndarray],
    metric: str = "hellinger",
) -> np.ndarray:
    """Build NxN distance matrix from per-asset distributions."""
    n_assets = len(dist_list)
    d = np.zeros((n_assets, n_assets), dtype=float)

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            if metric == "hellinger":
                val = hellinger_distance(dist_list[i], dist_list[j])
            elif metric in {"js", "jensen-shannon", "jensen_shannon"}:
                val = jensen_shannon_distance(dist_list[i], dist_list[j])
            else:
                raise ValueError("metric must be 'hellinger' or 'js'")
            d[i, j] = val
            d[j, i] = val

    return d


def quantum_ordering_from_distance(D: np.ndarray, method: str = "ward") -> np.ndarray:
    """Get hierarchical ordering from a symmetric distance matrix."""
    condensed = squareform(D)
    z = linkage(condensed, method=method)
    return leaves_list(z)


def compute_asset_distributions_hardware(
    features_tensor: np.ndarray,
    backend,
    config: HardwareConfig = HardwareConfig(),
) -> List[np.ndarray]:
    """Compute one average distribution per asset on real hardware.

    features_tensor shape: (N, T, P)
    """
    n_assets = features_tensor.shape[0]
    out: List[np.ndarray] = []
    for i in range(n_assets):
        out.append(
            average_measurement_distribution(
                features_tensor[i],
                backend=backend,
                config=config,
            )
        )
    return out


def average_density_matrix(
    asset_data: np.ndarray,
    alpha: float = 2.0,
    entanglement_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Compute average density matrix for one asset in ideal simulation.

    asset_data shape: (T, P)
    Returns a (2^P, 2^P) complex matrix.
    """
    t_steps, p_features = asset_data.shape
    x_min = np.min(asset_data, axis=0)
    x_max = np.max(asset_data, axis=0)

    rho_acc = np.zeros((2**p_features, 2**p_features), dtype=complex)
    for t in range(t_steps):
        qc = build_feature_map_circuit(
            asset_data[t],
            x_min=x_min,
            x_max=x_max,
            alpha=alpha,
            entanglement_pairs=entanglement_pairs,
            with_measurements=False,
        )
        sv = Statevector.from_instruction(qc)
        rho_acc += DensityMatrix(sv).data

    return rho_acc / max(t_steps, 1)


def trace_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Compute the trace distance between two density matrices."""
    delta = rho_a - rho_b
    eigvals = np.linalg.eigvalsh(delta)
    return float(0.5 * np.sum(np.abs(eigvals)))


def compute_distance_matrix(rho_list: Sequence[np.ndarray]) -> np.ndarray:
    """Build NxN trace-distance matrix from per-asset density matrices."""
    n_assets = len(rho_list)
    d = np.zeros((n_assets, n_assets), dtype=float)

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            val = trace_distance(rho_list[i], rho_list[j])
            d[i, j] = val
            d[j, i] = val

    return d


def quantum_ordering(D: np.ndarray, method: str = "ward") -> np.ndarray:
    """Get hierarchical ordering from a symmetric distance matrix."""
    condensed = squareform(D)
    z = linkage(condensed, method=method)
    return leaves_list(z)


def get_ibm_backend(
    min_num_qubits: int = 6,
    channel: str = "ibm_quantum_platform",
    account_name: Optional[str] = "labqcc-2025",
    backend_name: Optional[str] = None,
):
    """Return an operational least-busy IBM backend with enough qubits."""
    if QiskitRuntimeService is None:
        raise ImportError(
            "qiskit_ibm_runtime is required. "
            "Install it with: pip install qiskit-ibm-runtime"
        )

    if account_name:
        service = QiskitRuntimeService(name=account_name)
    else:
        service = QiskitRuntimeService(channel=channel)

    if backend_name:
        backend = service.backend(backend_name)
        if backend.num_qubits < min_num_qubits:
            raise ValueError(
                f"Backend {backend.name} has {backend.num_qubits} qubits, "
                f"but {min_num_qubits} are required"
            )
        return backend

    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=min_num_qubits,
    )
    return backend


__all__ = [
    "HardwareConfig",
    "build_feature_map_circuit",
    "counts_to_probability_vector",
    "run_circuits_on_backend",
    "average_measurement_distribution",
    "hellinger_distance",
    "jensen_shannon_distance",
    "compute_distribution_distance_matrix",
    "quantum_ordering_from_distance",
    "compute_asset_distributions_hardware",
    "average_density_matrix",
    "compute_distance_matrix",
    "quantum_ordering",
    "get_ibm_backend",
]
