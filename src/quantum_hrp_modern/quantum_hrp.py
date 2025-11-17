import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def create_rotation_circuit(probability: float) -> QuantumCircuit:
    """
    Crea un circuito cuántico que codifica una probabilidad p en la amplitud |1>.
    probability: número entre 0 y 1.
    """
    qc = QuantumCircuit(1)
    theta = 2 * np.arcsin(np.sqrt(probability))
    qc.ry(theta, 0)
    qc.measure_all()
    return qc


def estimate_probability(probability: float, shots: int = 2000) -> float:
    """
    Estima cuántas veces aparece |1> usando un simulador Aer moderno.
    """
    qc = create_rotation_circuit(probability)
    sim = AerSimulator()

    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    count_1 = counts.get("1", 0)
    return count_1 / shots


def quantum_payoff_estimation(S_T: float, K: float) -> float:
    """
    Codifica un payoff sencillo para demostrar el flujo cuántico.
    payoff = max(S_T - K, 0)
    Normalizado a [0, 1] para poder codificarlo como probabilidad.
    """
    payoff = max(S_T - K, 0)

    # Normalizamos a una probabilidad
    prob = min(payoff / 5, 1)

    estimated_prob = estimate_probability(prob)
    return estimated_prob


if __name__ == "__main__":
    K = 1.0
    S_T = 1.4
    est = quantum_payoff_estimation(S_T, K)

    print("Estimación cuántica (normalizada):", est)
