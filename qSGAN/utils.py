import numpy as np
import torch
from torch import nn

from qulacs import ParametricQuantumCircuit, QuantumState
from qulacs.gate import ParametricRX, ParametricRY, ParametricRZ, CNOT


def calculate_grad(output_plus, output_minus):
    out = torch.mean(-torch.log(output_plus) + torch.log(output_minus), axis=1) / 2
    return out.view(out.size()[0])


def transform_to_8_px(x_list):
    out_list = []
    for x in x_list:
        out = list(format(x, "08b"))
        out = list(map(int, out))
        out_list.append(out)
    return torch.tensor(out_list, dtype=torch.float)


def transform_to_8_px_2D(x_list):
    out_list = []
    for xs in x_list:
        x_2D = []
        for x in xs:
            out = list(format(x, "08b"))
            out = list(map(int, out))
            x_2D.append(out)
        out_list.append(x_2D)
    return torch.tensor(out_list, dtype=torch.float)
