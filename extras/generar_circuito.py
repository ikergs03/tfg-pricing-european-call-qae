import math
import numpy as np
from qiskit import QuantumCircuit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n = 6
theta = [0.5 * math.pi] * n

qc = QuantumCircuit(n)

entanglement_pairs = [(0, 1), (0, 5), (2, 4), (0, 4), (2, 3), (1, 2)]

# Las barreras marcan el INICIO de cada bloque

# B0 - Preparación
qc.barrier(label='B0')
qc.h(range(n))

# B1 - 1er encoding: T + Ry
qc.barrier(label='B1')
for i in range(n):
    qc.t(i)
    qc.ry(theta[i], i)

# B2 - Entrelazamiento
qc.barrier(label='B2')
for (i, j) in entanglement_pairs:
    qc.iswap(i, j)

# B3 - 2do encoding: T + Rx
qc.barrier(label='B3')
for i in range(n):
    qc.t(i)
    qc.rx(theta[i], i)

# B4 - Entrelazamiento
qc.barrier(label='B4')
for (i, j) in entanglement_pairs:
    qc.iswap(i, j)

# B5 - 3er encoding: T + Ry
qc.barrier(label='B5')
for i in range(n):
    qc.t(i)
    qc.ry(theta[i], i)

# B6 - Entrelazamiento
qc.barrier(label='B6')
for (i, j) in entanglement_pairs:
    qc.iswap(i, j)

# B7 - Cierre: T + H
qc.barrier(label='B7')
for i in range(n):
    qc.t(i)
    qc.h(i)

fig = qc.draw(
    output='mpl',
    fold=16,
    style={'fontsize': 11},
    plot_barriers=True,
)

output_path = '/home/iker/TFG/latex/img/circuito_qhrp_implementado.pdf'
fig.savefig(output_path, bbox_inches='tight', dpi=150)
print(f"Guardado en: {output_path}")

output_path_png = '/home/iker/TFG/latex/img/circuito_qhrp_implementado.png'
fig.savefig(output_path_png, bbox_inches='tight', dpi=150)
print(f"Guardado en: {output_path_png}")
