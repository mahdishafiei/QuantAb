"""PennyLane quantum kernel implementations."""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def _angle_encoding(x: np.ndarray, wires: list) -> None:
    """Encode a feature vector via RY rotations."""
    for i, wire in enumerate(wires):
        qml.RY(x[i] * np.pi, wires=wire)


def _entangling_layer(wires: list) -> None:
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def build_minimal_kernel(n_qubits: int):
    """Angle encoding + single entangling layer (shallow circuit)."""
    dev = qml.device("lightning.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        _angle_encoding(x1, wires)
        _entangling_layer(wires)
        qml.adjoint(_entangling_layer)(wires)
        qml.adjoint(_angle_encoding)(x2, wires)
        return qml.probs(wires=wires)

    def kernel(x1, x2):
        return float(kernel_circuit(x1, x2)[0])

    return kernel


def build_expressive_kernel(n_qubits: int, depth: int = 2):
    """Angle encoding + repeated Ry+CNOT layers (deeper circuit)."""
    dev = qml.device("lightning.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    def ansatz(x):
        for _ in range(depth):
            _angle_encoding(x, wires)
            _entangling_layer(wires)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        ansatz(x1)
        qml.adjoint(ansatz)(x2)
        return qml.probs(wires=wires)

    def kernel(x1, x2):
        return float(kernel_circuit(x1, x2)[0])

    return kernel


def build_kernel_matrix(kernel_fn, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Compute full kernel matrix K[i,j] = kernel(X1[i], X2[j])."""
    K = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K[i, j] = kernel_fn(x1, x2)
    return K


KERNELS = {
    "quantum_minimal": build_minimal_kernel,
    "quantum_expressive": build_expressive_kernel,
}
