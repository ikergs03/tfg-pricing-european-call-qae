from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

output_dir = "/home/iker/TFG/extras/"

# --- Puertas de 1 qubit ---
puertas_1q = {
    "puerta_H":  lambda qc: qc.h(0),
    "puerta_X":  lambda qc: qc.x(0),
    "puerta_Y":  lambda qc: qc.y(0),
    "puerta_Z":  lambda qc: qc.z(0),
    "puerta_S":  lambda qc: qc.s(0),
    "puerta_T":  lambda qc: qc.t(0),
    "puerta_RX": lambda qc: qc.rx(0.5, 0),
    "puerta_RY": lambda qc: qc.ry(0.5, 0),
    "puerta_RZ": lambda qc: qc.rz(0.5, 0),
    "puerta_U":  lambda qc: qc.u(0.3, 0.5, 0.7, 0),
}

for nombre, aplicar in puertas_1q.items():
    qc = QuantumCircuit(1)
    aplicar(qc)
    fig = qc.draw('mpl')
    fig.savefig(output_dir + nombre + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardada: {nombre}.png")

# --- Puertas de 2 qubits ---
puertas_2q = {
    "puerta_CNOT": lambda qc: qc.cx(0, 1),
    "puerta_CZ":   lambda qc: qc.cz(0, 1),
    "puerta_CY":   lambda qc: qc.cy(0, 1),
    "puerta_SWAP": lambda qc: qc.swap(0, 1),
    "puerta_iSWAP": lambda qc: qc.iswap(0, 1),
}

for nombre, aplicar in puertas_2q.items():
    qc = QuantumCircuit(2)
    aplicar(qc)
    fig = qc.draw('mpl')
    fig.savefig(output_dir + nombre + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardada: {nombre}.png")

# --- Puertas de 3 qubits ---
puertas_3q = {
    "puerta_Toffoli": lambda qc: qc.ccx(0, 1, 2),
    "puerta_Fredkin": lambda qc: qc.cswap(0, 1, 2),
}

for nombre, aplicar in puertas_3q.items():
    qc = QuantumCircuit(3)
    aplicar(qc)
    fig = qc.draw('mpl')
    fig.savefig(output_dir + nombre + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardada: {nombre}.png")

# --- Grid con todas las puertas de 1 qubit ---
fig, axes = plt.subplots(2, 5, figsize=(20, 6))
axes = axes.flatten()
for i, (nombre, aplicar) in enumerate(puertas_1q.items()):
    qc = QuantumCircuit(1)
    aplicar(qc)
    qc.draw('mpl', ax=axes[i])
    axes[i].set_title(nombre.replace("puerta_", ""), fontsize=14, fontweight='bold')
plt.suptitle("Puertas de 1 qubit", fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(output_dir + "grid_puertas_1q.png", dpi=150, bbox_inches='tight')
plt.close()
print("Guardada: grid_puertas_1q.png")

# --- Grid con todas las puertas de 2 y 3 qubits ---
fig, axes = plt.subplots(1, 7, figsize=(28, 4))
todas_multi = {**puertas_2q, **puertas_3q}
for i, (nombre, aplicar) in enumerate(todas_multi.items()):
    n = 3 if nombre in puertas_3q else 2
    qc = QuantumCircuit(n)
    aplicar(qc)
    qc.draw('mpl', ax=axes[i])
    axes[i].set_title(nombre.replace("puerta_", ""), fontsize=12, fontweight='bold')
plt.suptitle("Puertas de 2 y 3 qubits", fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(output_dir + "grid_puertas_2q3q.png", dpi=150, bbox_inches='tight')
plt.close()
print("Guardada: grid_puertas_2q3q.png")

# --- Grid SOLO con las puertas que se usan en el feature map de QHRP ---
puertas_qhrp = ["puerta_X", "puerta_H", "puerta_T", "puerta_RY", "puerta_RX", "puerta_iSWAP"]
fig, axes = plt.subplots(1, 6, figsize=(24, 4))
for i, nombre in enumerate(puertas_qhrp):
    if nombre in puertas_1q:
        qc = QuantumCircuit(1)
        puertas_1q[nombre](qc)
    else:
        qc = QuantumCircuit(2)
        puertas_2q[nombre](qc)
    qc.draw('mpl', ax=axes[i])
    axes[i].set_title(nombre.replace("puerta_", ""), fontsize=14, fontweight='bold')
plt.suptitle("Puertas usadas en el feature map de QHRP", fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(output_dir + "grid_puertas_qhrp.png", dpi=150, bbox_inches='tight')
plt.close()
print("Guardada: grid_puertas_qhrp.png")

# --- Figuras combinadas: simbolo + matriz + accion sobre |0> ---
# Solo para las puertas "cabecera": H y RY(theta)
preamble = r"\usepackage{amsmath}\newcommand{\ket}[1]{|#1\rangle}"
with plt.rc_context({'text.usetex': True, 'text.latex.preamble': preamble}):

    # Hadamard
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={'width_ratios': [1, 2]})
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.draw('mpl', ax=axes[0])
    axes[0].set_title("Circuito", fontsize=12)
    axes[1].axis('off')
    axes[1].text(0.5, 0.7,
                  r"$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$",
                  fontsize=18, ha='center', va='center')
    axes[1].text(0.5, 0.25,
                  r"$H\ket{0} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{1})$",
                  fontsize=16, ha='center', va='center')
    plt.suptitle("Puerta Hadamard (H)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir + "puerta_H_matriz.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Guardada: puerta_H_matriz.png")

    # RY(theta)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={'width_ratios': [1, 2]})
    qc = QuantumCircuit(1)
    qc.ry(0.5, 0)
    qc.draw('mpl', ax=axes[0])
    axes[0].set_title("Circuito", fontsize=12)
    axes[1].axis('off')
    axes[1].text(0.5, 0.7,
                  r"$R_y(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$",
                  fontsize=18, ha='center', va='center')
    axes[1].text(0.5, 0.25,
                  r"$\theta \propto x_{\mathrm{activo}}$  (codifica el dato)",
                  fontsize=16, ha='center', va='center')
    plt.suptitle(r"Puerta de rotación $R_y(\theta)$", fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir + "puerta_RY_matriz.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Guardada: puerta_RY_matriz.png")
