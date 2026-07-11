"""Piezas compartidas y agnosticas de framework para los scripts de "aprovechamiento total de qubits".

qiskit_full_utilization.py y qibo_full_utilization.py necesitan las mismas
tres piezas independientes del framework cuantico: como agrupar activos en
lotes que llenen un numero determinado de qubits, como convertir una
observacion cruda en los angulos de rotacion del feature-map, y como
convertir las distribuciones de probabilidad por activo en una matriz de
distancias y un orden jerarquico. Mantenerlas aqui hace que los dos scripts
solo difieran en como construyen/ejecutan los circuitos, no en la logica de
agrupacion ni en la metrica de distancia.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


##############################################
# Planificacion de lotes
##############################################
@dataclass
class BatchPlan:
    n_assets: int
    total_qubits: int
    qubits_per_asset: int
    batch_size: int
    groups: List[List[int]]
    idle_qubits: int

    @property
    def n_batches(self) -> int:
        return len(self.groups)

    def summary(self) -> str:
        used = self.batch_size * self.qubits_per_asset
        ratio = self.total_qubits / self.qubits_per_asset
        last = len(self.groups[-1]) if self.groups else 0
        return (
            f"Ordenador cuantico: {self.total_qubits} qubits\n"
            f"Qubits por activo:  {self.qubits_per_asset}\n"
            f"{self.total_qubits}/{self.qubits_per_asset} = {ratio:.2f} -> "
            f"{self.batch_size} activos por lote "
            f"({used} qubits usados por circuito, {self.idle_qubits} ociosos)\n"
            f"{self.n_assets} activos -> {self.n_batches} lotes "
            f"(el ultimo con {last} activos)"
        )


def plan_batches(n_assets: int, total_qubits: int, qubits_per_asset: int) -> BatchPlan:
    """Agrupa `n_assets` activos en lotes de `total_qubits // qubits_per_asset`.

    El feature-map de cada activo solo entrelaza qubits dentro de su propio
    bloque, asi que varios activos independientes pueden compartir un
    circuito mas ancho (un bloque de qubits disjunto cada uno) sin cambiar
    la codificacion de ningun activo individual. Ejemplo: 32 qubits, 6 por
    activo -> 32 // 6 = 5 activos por lote; 152 qubits -> 152 // 6 = 25
    activos por lote (2 qubits quedan ociosos).
    """
    if qubits_per_asset <= 0:
        raise ValueError("qubits_per_asset debe ser positivo")
    if total_qubits < qubits_per_asset:
        raise ValueError(
            f"total_qubits ({total_qubits}) es menor que qubits_per_asset "
            f"({qubits_per_asset}): no cabe ni un activo"
        )

    batch_size = total_qubits // qubits_per_asset
    groups = [
        list(range(start, min(start + batch_size, n_assets)))
        for start in range(0, n_assets, batch_size)
    ]
    idle_qubits = total_qubits - batch_size * qubits_per_asset

    return BatchPlan(
        n_assets=n_assets,
        total_qubits=total_qubits,
        qubits_per_asset=qubits_per_asset,
        batch_size=batch_size,
        groups=groups,
        idle_qubits=idle_qubits,
    )


##############################################
# Angulos del feature-map
##############################################
def theta_from_observation(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, alpha: float) -> np.ndarray:
    """Convierte una observacion cruda en los angulos de rotacion del feature-map."""
    p = len(x)
    theta = np.zeros(p)
    for i in range(p):
        if x_max[i] > x_min[i]:
            theta[i] = alpha * math.pi * (x[i] - x_min[i]) / (x_max[i] - x_min[i])
    return theta


##############################################
# Distancia y ordenamiento (mismas formulas que quantum_hrp_hardware.py)
##############################################
def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 0.0, 1.0)
    q = np.clip(q, 0.0, 1.0)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.sqrt(max(js, 0.0)))


def compute_distribution_distance_matrix(dist_list: Sequence[np.ndarray], metric: str = "hellinger") -> np.ndarray:
    """Construye la matriz de distancias NxN a partir de las distribuciones de probabilidad de cada activo."""
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
    """Obtiene el orden jerarquico a partir de una matriz de distancias simetrica."""
    condensed = squareform(D)
    z = linkage(condensed, method=method)
    return leaves_list(z)


__all__ = [
    "BatchPlan",
    "plan_batches",
    "theta_from_observation",
    "hellinger_distance",
    "jensen_shannon_distance",
    "compute_distribution_distance_matrix",
    "quantum_ordering_from_distance",
]
