import numpy as np
import torch
from torch import nn

from qulacs import ParametricQuantumCircuit, QuantumState
from qulacs.gate import ParametricRX, ParametricRY, ParametricRZ, CNOT

from qSGAN.utils import *
from qSGAN.optimizer.adam import Adam


class HEA(ParametricQuantumCircuit):
    def __init__(self, num_qubit, depth=3, rotation_gates=None, entanglement_gates=None):
        super(HEA, self).__init__(num_qubit)
        self.depth = depth

        if rotation_gates is None:
            self.rotation_gates = [ParametricRX, ParametricRY, ParametricRZ]

        if entanglement_gates is None:
            self.entanglement_gates = CNOT

        self._build_circuit()

    def _build_circuit(self):
        depth = self.depth

        for d in range(depth):
            self._build_rotation_layer()
            self._build_entanglement_layer()

    def _build_rotation_layer(self):
        n = self.get_qubit_count()

        rotation_gates = (self.rotation_gates, n)
        for i in range(n):
            self.add_parametric_gate(rotation_gates[i](i, 2*np.pi*np.random.rand()))

    def _build_entanglement_layer(self):
        n = self.get_qubit_count()

        entanglement_gates = self.entanglement_gates
        for i in range(n-1):
            self.add_gate(entanglement_gates(i, i+1))


class QuantumGenerator(nn.Module):
    def __init__(self, ansatz: ParametricQuantumCircuit, batch_size=8, optimizer=Adam()):
        super(QuantumGenerator, self).__init__()

        self.ansatz = ansatz
        self.batch_size = batch_size
        self.n_qubit = ansatz.get_qubit_count()
        self.optimizer = optimizer
        self.out = None

    def qunatum_circuit(self, num_samples):
        state = QuantumState(self.n_qubit)
        self.ansatz.update_quantum_state(state)
        out = state.sampling(num_samples)
        return torch.tensor(out)

    def foward(self):
        out = self.qunatum_circuit(self.batch_size)

        return out

    def calculate_x_plus_minus(self):
        n = self.ansatz.get_qubit_count()
        state = QuantumState(n)
        x_plus_list = []
        x_minus_list = []
        for i in range(self.n_qubit):
            ansatz_plus = self.ansatz.copy()
            ansatz_minus = self.ansatz.copy()
            parameter_i = self.ansatz.get_parameter(i)
            ansatz_plus.set_parameter(i, parameter_i + np.pi / 2)
            ansatz_minus.set_parameter(i, parameter_i - np.pi / 2)

            ansatz_plus.update_quantum_state(state)
            x_plus = state.sampling(2)
            state.set_zero_state()

            ansatz_minus.update_quantum_state(state)
            x_minus = state.sampling(2)
            state.set_zero_state()

            x_plus_list.append(x_plus)
            x_minus_list.append(x_minus)
        x_plus_list = torch.tensor(x_plus_list)
        x_minus_list = torch.tensor(x_minus_list)
        return x_plus_list, x_minus_list

    def update_parameter(self, grad):
        params = [self.ansatz.get_parameter(i) for i in range(self.ansatz.get_parameter_count())]
        self.optimizer.update(params, grad)
