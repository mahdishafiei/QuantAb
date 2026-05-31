"""PennyLane quantum kernel implementations and VQC regressor.

Single-register kernels (Mahdi's ablation arm):
  build_minimal_kernel, build_expressive_kernel

Two-register kernels (core QuantAb contribution — H_ab ⊗ H_tgt):
  build_two_register_minimal_kernel, build_two_register_expressive_kernel

Variational regressor:
  VQCRegressor — sklearn-compatible, uses StronglyEntanglingLayers + PauliZ readout.
"""

import copy

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.svm import SVR
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

def _angle_encoding(x: np.ndarray, wires: list) -> None:
    for i, wire in enumerate(wires):
        qml.RY(x[i] * np.pi, wires=wire)


def _entangling_layer(wires: list) -> None:
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


# ---------------------------------------------------------------------------
# Single-register kernels (ablation / Mahdi baseline)
# ---------------------------------------------------------------------------

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
        return float(kernel_circuit(np.asarray(x1), np.asarray(x2))[0])

    return kernel


def build_expressive_kernel(n_qubits: int, depth: int = 2):
    """Angle encoding + repeated RY+CNOT layers (deeper circuit)."""
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
        return float(kernel_circuit(np.asarray(x1), np.asarray(x2))[0])

    return kernel


# ---------------------------------------------------------------------------
# Two-register kernels (core QuantAb contribution)
#
# Input convention: each sample vector x = [x_ab | x_tgt] (concatenated).
# The kernel splits at n_qubits_ab to recover the two modalities.
# Cross-register CNOTs create entanglement in H_ab ⊗ H_tgt, which is the
# exponentially larger space that single-register kernels cannot exploit.
# ---------------------------------------------------------------------------

def _encode_two_registers(
    x_ab: np.ndarray,
    x_tgt: np.ndarray,
    wires_ab: list,
    wires_tgt: list,
) -> None:
    """Encode both modalities with intra- and cross-register entanglement."""
    _angle_encoding(x_ab, wires_ab)
    _entangling_layer(wires_ab)
    _angle_encoding(x_tgt, wires_tgt)
    _entangling_layer(wires_tgt)
    # Cross-register CNOT: last ab qubit → first tgt qubit
    qml.CNOT(wires=[wires_ab[-1], wires_tgt[0]])


def build_two_register_minimal_kernel(n_qubits_ab: int, n_qubits_tgt: int):
    """Single-layer two-register kernel with cross-register entanglement.

    Args:
        n_qubits_ab: Qubits for antibody register.
        n_qubits_tgt: Qubits for antigen register.

    Returns:
        kernel(x1, x2) → float, where each x = [x_ab | x_tgt] concatenated.
    """
    n_total = n_qubits_ab + n_qubits_tgt
    dev = qml.device("lightning.qubit", wires=n_total)
    wires_ab = list(range(n_qubits_ab))
    wires_tgt = list(range(n_qubits_ab, n_total))

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        x1_ab, x1_tgt = x1[:n_qubits_ab], x1[n_qubits_ab:]
        x2_ab, x2_tgt = x2[:n_qubits_ab], x2[n_qubits_ab:]
        _encode_two_registers(x1_ab, x1_tgt, wires_ab, wires_tgt)
        qml.adjoint(_encode_two_registers)(x2_ab, x2_tgt, wires_ab, wires_tgt)
        return qml.probs(wires=list(range(n_total)))

    def kernel(x1, x2):
        return float(kernel_circuit(np.asarray(x1), np.asarray(x2))[0])

    return kernel


def build_two_register_expressive_kernel(
    n_qubits_ab: int, n_qubits_tgt: int, depth: int = 2
):
    """Multi-layer two-register kernel (repeated encoding + cross entanglement).

    Args:
        n_qubits_ab: Qubits for antibody register.
        n_qubits_tgt: Qubits for antigen register.
        depth: Number of encoding repetitions.

    Returns:
        kernel(x1, x2) → float, where each x = [x_ab | x_tgt] concatenated.
    """
    n_total = n_qubits_ab + n_qubits_tgt
    dev = qml.device("lightning.qubit", wires=n_total)
    wires_ab = list(range(n_qubits_ab))
    wires_tgt = list(range(n_qubits_ab, n_total))

    def ansatz(x_ab, x_tgt):
        for _ in range(depth):
            _encode_two_registers(x_ab, x_tgt, wires_ab, wires_tgt)
            # Second cross-register entanglement in the reverse direction
            if len(wires_ab) > 1 and len(wires_tgt) > 1:
                qml.CNOT(wires=[wires_tgt[0], wires_ab[-2]])

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        x1_ab, x1_tgt = x1[:n_qubits_ab], x1[n_qubits_ab:]
        x2_ab, x2_tgt = x2[:n_qubits_ab], x2[n_qubits_ab:]
        ansatz(x1_ab, x1_tgt)
        qml.adjoint(ansatz)(x2_ab, x2_tgt)
        return qml.probs(wires=list(range(n_total)))

    def kernel(x1, x2):
        return float(kernel_circuit(np.asarray(x1), np.asarray(x2))[0])

    return kernel


# ---------------------------------------------------------------------------
# Kernel matrix utility
# ---------------------------------------------------------------------------

def build_kernel_matrix(kernel_fn, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Compute full kernel matrix K[i,j] = kernel(X1[i], X2[j])."""
    K = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K[i, j] = kernel_fn(x1, x2)
    return K


# ---------------------------------------------------------------------------
# VQC Regressor (sklearn-compatible)
# ---------------------------------------------------------------------------

class VQCRegressor:
    """Variational Quantum Circuit regressor.

    Uses angle encoding + StronglyEntanglingLayers with a PauliZ(0) readout.
    Labels are z-scored internally so the [-1, 1] output range is calibrated.
    Sklearn-compatible: fit(X, y) / predict(X).

    Args:
        n_qubits: Number of qubits (= PCA dimension of the input).
        n_layers: Depth of StronglyEntanglingLayers.
        n_epochs: Gradient descent steps.
        lr: Adam learning rate.
        random_state: RNG seed for weight initialization.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 3,
        n_epochs: int = 100,
        lr: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.random_state = random_state
        # Set after fit:
        self.weights_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0

    # sklearn compatibility
    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "VQCRegressor":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _build_circuit(self):
        dev = qml.device("lightning.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="autograd")
        def circuit(x, weights):
            for i in range(self.n_qubits):
                qml.RY(float(x[i]) * np.pi, wires=i)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))

        return circuit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQCRegressor":
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) + 1e-8
        y_norm = (y - self.y_mean_) / self.y_std_

        circuit = self._build_circuit()
        shape = qml.StronglyEntanglingLayers.shape(self.n_layers, self.n_qubits)
        rng = np.random.RandomState(self.random_state)
        weights = pnp.array(
            rng.uniform(-np.pi, np.pi, shape).astype(np.float64),
            requires_grad=True,
        )

        opt = qml.AdamOptimizer(stepsize=self.lr)
        y_pnp = pnp.array(y_norm.astype(np.float64), requires_grad=False)

        def cost(w):
            preds = pnp.stack([circuit(X[i], w) for i in range(len(X))])
            return pnp.mean((preds - y_pnp) ** 2)

        for _ in tqdm(range(self.n_epochs), desc=f"VQC {self.n_qubits}q", leave=False):
            weights, _ = opt.step_and_cost(cost, weights)

        self.weights_ = np.array(weights)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict().")
        circuit = self._build_circuit()
        w = pnp.array(self.weights_, requires_grad=False)
        preds = np.array([float(circuit(X[i], w)) for i in range(len(X))])
        return preds * self.y_std_ + self.y_mean_


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

KERNELS = {
    "quantum_minimal": build_minimal_kernel,
    "quantum_expressive": build_expressive_kernel,
    "two_reg_minimal": build_two_register_minimal_kernel,
    "two_reg_expressive": build_two_register_expressive_kernel,
}
