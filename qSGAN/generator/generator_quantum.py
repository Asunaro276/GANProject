import numpy as np
import torch
from torch import nn

from qulacs import ParametricQuantumCircuit, QuantumStateGpu
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

    @property
    def parameter(self):
        params = [self.get_parameter(i) for i in range(self.get_parameter_count())]
        return params

    @parameter.setter
    def parameter(self, params_list):
        for i in range(self.get_parameter_count()):
            self.set_parameter(i, params_list[i])

    def _build_circuit(self):
        depth = self.depth

        for d in range(depth):
            self._build_rotation_layer()
            self._build_entanglement_layer()

    def _build_rotation_layer(self):
        n = self.get_qubit_count()

        rotation_gates = np.random.choice(self.rotation_gates, n)
        for i in range(n):
            self.add_parametric_gate(rotation_gates[i](i, 2*np.pi*np.random.rand()))

    def _build_entanglement_layer(self):
        n = self.get_qubit_count()

        entanglement_gates = self.entanglement_gates
        for i in range(n-1):
            self.add_gate(entanglement_gates(i, i+1))


class QuantumGenerator(nn.Module):
    def __init__(self, num_qubit: int, depth: int = 4, batch_size=7, optimizer=Adam(lr=0.005, betas=(0.9, 0.999))):
        super(QuantumGenerator, self).__init__()

        self.batch_size = batch_size
        self.ansatz = HEA(num_qubit, depth)
        self.n_qubit = self.ansatz.get_qubit_count()
        self.optimizer = optimizer
        self.out = None

    def qunatum_circuit(self, num_samples):
        state = QuantumStateGpu(self.n_qubit)
        self.ansatz.update_quantum_state(state)
        out = state.sampling(num_samples)
        return torch.tensor(out)

    def forward(self, batch_size=7):
        out = self.qunatum_circuit(batch_size)
        return out

    def calculate_x_plus_minus(self):
        n = self.ansatz.get_qubit_count()
        state = QuantumStateGpu(n)
        x_plus_list = []
        x_minus_list = []
        for i in range(self.ansatz.get_parameter_count()):
            ansatz_plus = self.ansatz.copy()
            ansatz_minus = self.ansatz.copy()
            parameter_i = self.ansatz.get_parameter(i)
            ansatz_plus.set_parameter(i, parameter_i + np.pi / 2)
            ansatz_minus.set_parameter(i, parameter_i - np.pi / 2)

            ansatz_plus.update_quantum_state(state)
            x_plus = state.sampling(self.batch_size)
            state.set_zero_state()

            ansatz_minus.update_quantum_state(state)
            x_minus = state.sampling(self.batch_size)
            state.set_zero_state()

            x_plus_list.append(x_plus)
            x_minus_list.append(x_minus)
        x_plus_list = torch.tensor(x_plus_list)
        x_minus_list = torch.tensor(x_minus_list)
        return x_plus_list, x_minus_list

    def update_parameter(self, grad):
        device = grad.device
        params = torch.tensor([self.ansatz.get_parameter(i) for i in range(self.ansatz.get_parameter_count())]).to(device)
        params = self.optimizer.update(params, grad)
        self.ansatz.parameter = params
