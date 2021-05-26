import numpy as np
import torch
from torch import nn

from qulacs import ParametricQuantumCircuit, QuantumState
from qulacs.gate import ParametricRX, ParametricRY, ParametricRZ, CNOT


def calculate_grad(output_plus, output_minus):
    out = torch.mean(-torch.log(output_plus) + torch.log(output_minus)) / 2
    return out

def transform_to_8_px(x):
    out = torch.tensor(list(map(int, list(format(x, "b")))))
    return out
