import os
import numpy as np
from qiskit.visualization import plot_bloch_vector
from qiskit import QuantumCircuit

img_dir = os.path.join(os.path.dirname(__file__), '..', 'latex', 'img')

# Esfera de Bloch
fig = plot_bloch_vector([1/np.sqrt(2), 0, 1/np.sqrt(2)])  # estado en superposición
fig.savefig(os.path.join(img_dir, 'bloch-sphere.png'), dpi=200, bbox_inches='tight')
print('bloch-sphere.png guardada')

# Circuito de Bell
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
fig = qc.draw(output='mpl')
fig.savefig(os.path.join(img_dir, 'circuito-bell.png'), dpi=200, bbox_inches='tight')
print('circuito-bell.png guardada')