"""Nucleo del QHRP de la memoria (Fig. 3/4), vectorizado para series largas.

Reproduce exactamente el metodo de src/qhrp_figure3_4/src/quantum_hrp.py:
  features de ventana movil (5 dias) -> feature-map de 6 qubits ->
  matriz de densidad promedio -> distancia de Frobenius -> enlace de Ward,
y el HRP clasico (distancia sqrt((1 - corr) / 2) con enlace simple).

La unica diferencia es como se simula el circuito: en vez de construir un
QuantumCircuit de Qiskit por dia (lento con miles de dias x cientos de
activos), se aplican las mismas puertas con numpy a todos los dias a la vez.
verificar_contra_qiskit() comprueba que el resultado coincide con Qiskit.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

ENTANGLEMENT_PAIRS = [(0, 1), (0, 5), (2, 4), (0, 4), (2, 3), (1, 2)]
FEATURE_NAMES = ["retorno", "media_movil", "desv_movil", "ret_acumulado", "asimetria", "curtosis"]

_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_T = np.diag([1, np.exp(1j * math.pi / 4)])
_ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)


##############################################
# Features (igual que compute_financial_features, window=5)
##############################################
def rolling_features(r: np.ndarray, window: int = 5) -> np.ndarray:
    """(T,) retornos -> (T, 6) features. Los NaN (p.ej. asimetria de una ventana constante) pasan a 0."""
    t_total = len(r)
    out = np.zeros((t_total, 6))
    # Ventanas completas, de golpe
    if t_total >= window:
        w = np.lib.stride_tricks.sliding_window_view(r, window)
        full = slice(window - 1, None)
        out[full, 0] = r[full]
        out[full, 1] = w.mean(axis=1)
        out[full, 2] = w.std(axis=1, ddof=1)
        out[full, 3] = np.prod(1 + w, axis=1) - 1
        out[full, 4] = stats.skew(w, axis=1)
        out[full, 5] = stats.kurtosis(w, axis=1)
    # Primeros dias, con ventana incompleta
    for t in range(min(window - 1, t_total)):
        w = r[: t + 1]
        out[t, 0] = r[t]
        out[t, 1] = np.mean(w)
        out[t, 2] = np.std(w, ddof=1) if len(w) > 1 else 0.0
        out[t, 3] = np.prod(1 + w) - 1
        out[t, 4] = stats.skew(w) if len(w) > 2 else 0.0
        out[t, 5] = stats.kurtosis(w) if len(w) > 3 else 0.0
    return np.nan_to_num(out, nan=0.0)


##############################################
# Simulacion vectorizada del feature-map (quantum_feature_map_density)
##############################################
def _apply_1q(psi: np.ndarray, gate: np.ndarray, q: int) -> np.ndarray:
    """psi: (T, 2, ..., 2) con el eje q+1 = qubit q. gate: (2, 2) o (T, 2, 2)."""
    psi = np.moveaxis(psi, q + 1, -1)
    if gate.ndim == 2:
        psi = psi @ gate.T
    else:
        psi = np.einsum("t...k,tjk->t...j", psi, gate)
    return np.moveaxis(psi, -1, q + 1)


def _apply_iswap(psi: np.ndarray, i: int, j: int) -> np.ndarray:
    psi = np.moveaxis(psi, (i + 1, j + 1), (-2, -1))
    shape = psi.shape
    psi = (psi.reshape(*shape[:-2], 4) @ _ISWAP.T).reshape(shape)
    return np.moveaxis(psi, (-2, -1), (i + 1, j + 1))


def _ry(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)


def _rx(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    return np.stack([np.stack([c, s], -1), np.stack([s, c], -1)], -2)


def feature_map_states(x: np.ndarray, alpha: float = 2.0, x_min=None, x_max=None) -> np.ndarray:
    """(T, n) observaciones -> (T, 2^n) estados, en el orden de bits de Qiskit.

    x_min/x_max por defecto son los de x; se pueden pasar para escalar un tramo
    de dias con el rango de la serie completa.
    """
    t_total, n = x.shape
    x_min = x.min(axis=0) if x_min is None else x_min
    x_max = x.max(axis=0) if x_max is None else x_max
    span = np.where(x_max > x_min, x_max - x_min, 1.0)
    theta = np.where(x_max > x_min, alpha * math.pi * (x - x_min) / span, 0.0)
    return states_from_theta(theta)


def states_from_theta(theta: np.ndarray) -> np.ndarray:
    """(T, n) angulos -> (T, 2^n) estados del feature-map."""
    t_total, n = theta.shape

    psi = np.zeros((t_total,) + (2,) * n, dtype=complex)
    psi[(slice(None),) + (0,) * n] = 1.0
    pairs = [(i, j) for i, j in ENTANGLEMENT_PAIRS if i < n and j < n]

    for q in range(n):
        psi = _apply_1q(psi, _H, q)
    for rot in (_ry, _rx, _ry):
        for q in range(n):
            psi = _apply_1q(psi, _T, q)
            psi = _apply_1q(psi, rot(theta[:, q]), q)
        for i, j in pairs:
            psi = _apply_iswap(psi, i, j)
    for q in range(n):
        psi = _apply_1q(psi, _T, q)
        psi = _apply_1q(psi, _H, q)

    # Qiskit es little-endian (qubit 0 = bit menos significativo)
    psi = np.transpose(psi, (0,) + tuple(range(n, 0, -1)))
    return psi.reshape(t_total, 2**n)


def average_density_matrix(x: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    psi = feature_map_states(x, alpha)
    return psi.T @ psi.conj() / len(psi)


def frobenius_distance_matrix(rhos: list, scale: float = 1.5) -> np.ndarray:
    """scale * 0.5 * ||rho_i - rho_j||_F para todos los pares, con una sola matriz de Gram.

    ||A - B||_F^2 = ||A||_F^2 + ||B||_F^2 - 2 Re<A, B>, igual que la version par a par de quantum_hrp.py.
    """
    v = np.stack([r.reshape(-1) for r in rhos])
    gram = np.real(v @ v.conj().T)
    sq = np.diag(gram)
    d2 = np.clip(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0, None)
    d = scale * 0.5 * np.sqrt(d2)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


##############################################
# Ordenaciones
##############################################
def quasi_diagonal(linkage_matrix: np.ndarray) -> list:
    linkage_matrix = linkage_matrix.astype(int)
    sorted_items = pd.Series([linkage_matrix[-1, 0], linkage_matrix[-1, 1]])
    num_items = linkage_matrix[-1, 3]
    while sorted_items.max() >= num_items:
        sorted_items.index = range(0, sorted_items.shape[0] * 2, 2)
        dataframe = sorted_items[sorted_items >= num_items]
        i = dataframe.index
        j = dataframe.values - num_items
        sorted_items[i] = linkage_matrix[j, 0]
        dataframe = pd.Series(linkage_matrix[j, 1], index=i + 1)
        sorted_items = pd.concat([sorted_items, dataframe])
        sorted_items = sorted_items.sort_index()
        sorted_items.index = range(sorted_items.shape[0])
    return sorted_items.tolist()


def quantum_linkage(d: np.ndarray):
    z = linkage(squareform(d, checks=False), method="ward")
    return z, leaves_list(z).tolist()


def classical_linkage(corr: np.ndarray):
    d = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="single")
    return z, [int(i) for i in quasi_diagonal(z)], d


##############################################
# Verificacion
##############################################
def verificar_contra_qiskit(n_obs: int = 20, seed: int = 0) -> float:
    """Maxima diferencia entre la matriz de densidad promedio de numpy y la de Qiskit."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_obs, 6))
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    rho_ref = np.zeros((64, 64), dtype=complex)
    for obs in x:
        theta = 2.0 * math.pi * (obs - x_min) / (x_max - x_min)
        qc = QuantumCircuit(6)
        qc.h(range(6))
        for rot in ("ry", "rx", "ry"):
            for i in range(6):
                qc.t(i)
                getattr(qc, rot)(theta[i], i)
            for i, j in ENTANGLEMENT_PAIRS:
                qc.iswap(i, j)
        for i in range(6):
            qc.t(i)
            qc.h(i)
        sv = Statevector.from_instruction(qc).data
        rho_ref += np.outer(sv, sv.conj())
    rho_ref /= n_obs
    return float(np.abs(average_density_matrix(x) - rho_ref).max())
