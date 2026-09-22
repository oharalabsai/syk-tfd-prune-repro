"""Baseline: the fixed hardware-efficient 35-CNOT block with untuned seed-0 angles.

This is deliberately naive but honestly scoreable: it is exactly the sibling
campaign's sealed block topology (6 layers of Rz/Rx on 8 qubits, interleaved with
5 linear CNOT chains of 7 gates each = 35 logical CNOTs) driven by a single
seed-0 random angle draw, with no fit to the instance Hamiltonian at all. It is
instance-independent by construction, so its fidelity is whatever a random block
happens to give. Any real candidate must beat it by actually using `terms`.
"""

import numpy as np

LAYERS = 6
N_CHAINS = 5
BLOCK_QUBITS = 8
ANGLE_SEED = 0


def block_angles():
    """The pinned seed-0 angle array, shape (LAYERS, BLOCK_QUBITS, 2)."""
    rng = np.random.default_rng(ANGLE_SEED)
    return rng.uniform(-np.pi, np.pi, (LAYERS, BLOCK_QUBITS, 2))


def block_gates(theta):
    """Gate list for the 35-CNOT block given an angle array."""
    gates = []
    for layer in range(LAYERS):
        for q in range(BLOCK_QUBITS):
            gates.append({"name": "rz", "qubits": [q], "param": float(theta[layer, q, 0])})
            gates.append({"name": "rx", "qubits": [q], "param": float(theta[layer, q, 1])})
        if layer < N_CHAINS:
            for q in range(BLOCK_QUBITS - 1):
                gates.append({"name": "cx", "qubits": [q, q + 1], "param": 0.0})
    return gates


def prepare(payload):
    """Return the fixed block. The instance is ignored: that is the point."""
    return {"n_qubits": int(payload["n_qubits"]), "gates": block_gates(block_angles())}
