import numpy as np

def quantum_state_from_features(x, n_qubits=6):
    dim = 2 ** n_qubits
    psi = np.zeros(dim)
    psi[:len(x)] = x
    return psi / np.linalg.norm(psi)
