"""Carga los retornos de los activos y construye las 6 features matematicas clasicas.

Solo usa pandas/numpy, sin importar ningun framework cuantico, para que lo
puedan compartir qiskit_full_utilization.py y qibo_full_utilization.py sin
que ninguno de los dos dependa del framework del otro. (analyze_qubits_6_vs_5_all_assets.py
tiene el mismo codigo de carga, pero importar ese script arrastra
quantum_hrp_hardware y por tanto qiskit; este modulo es el extracto sin
qiskit de la parte de carga/construccion de features.)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
# Este fichero vive en <TFG>/src/qhrp_figure3_4/publicacion, asi que hay que
# subir tres niveles para llegar a la raiz del proyecto (donde esta data/).
PROJECT_ROOT = SCRIPT_DIR.parents[2]

LEGACY_FEATURE_NAMES = [
    "retorno",
    "retorno_cuadrado",
    "abs_ret_1p5",
    "sign_log1p_abs_ret",
    "tanh_ret",
    "sign_sqrt_abs_ret",
]


def load_all_assets_returns(project_root: Path, days: int) -> Tuple[List[str], pd.DataFrame]:
    """Cargar returns limpios/alineados conservando todo el universo disponible."""
    returns_path = project_root / "data" / "returns.csv"
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.dropna(axis=0, how="any")
    returns = returns.dropna(axis=1, how="all")

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
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Selecciona features para 6q o 5q manteniendo el resto del pipeline intacto."""
    p = features_tensor.shape[2]
    if p != 6:
        raise ValueError(f"Se esperaban 6 features base y se encontro p={p}")

    if n_qubits == 6:
        keep_idx = list(range(6))
    elif n_qubits == 5:
        if not (0 <= drop_feature_index < 6):
            raise ValueError(f"drop_feature_index debe estar en [0, 5], recibido {drop_feature_index}")
        keep_idx = [i for i in range(6) if i != drop_feature_index]
    else:
        raise ValueError("Solo se soporta n_qubits=6 o n_qubits=5")

    names = [LEGACY_FEATURE_NAMES[i] for i in keep_idx]
    return features_tensor[:, :, keep_idx], names, keep_idx


__all__ = [
    "SCRIPT_DIR",
    "PROJECT_ROOT",
    "LEGACY_FEATURE_NAMES",
    "load_all_assets_returns",
    "build_legacy_math_features",
    "scenario_features_exact",
]
