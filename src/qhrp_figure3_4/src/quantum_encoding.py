import numpy as np

def quantum_state_from_features(X):
    """
    X: array (T, P) con features estandarizadas de un activo
    return: vector (P,) normalizado
    """
    T, P = X.shape
    psi = np.zeros(P)

    for t in range(T):
        xt = X[t]
        norm = np.linalg.norm(xt)
        if norm > 0:
            psi += xt / norm

    psi = psi / np.sqrt(T)
    psi = psi / np.linalg.norm(psi)

    return psi
