import numpy as np
from .quantum_encoding import quantum_state_from_features

def density_matrix(X, n_qubits=6):
    dim = 2 ** n_qubits
    rho = np.zeros((dim, dim))
    for x in X:
        psi = quantum_state_from_features(x, n_qubits)
        rho += np.outer(psi, psi)
    return rho / X.shape[0]

def frobenius_distance(rho1, rho2):
    diff = rho1 - rho2
    return 0.5 * np.sqrt(np.trace(diff @ diff))

def quantum_distance_matrix(features):
    N = features.shape[0]
    rhos = [density_matrix(features[i]) for i in range(N)]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            D[i, j] = D[j, i] = frobenius_distance(rhos[i], rhos[j])
    return D
