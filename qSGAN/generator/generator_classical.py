import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, input_size=1, output_size=8, hidden_size=40):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(inplace=True))

        self.last = nn.Sequential(nn.Linear(hidden_size, output_size),
                                  nn.ReLU(inplace=True),
                                  nn.Sigmoid())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.last(out)

        return out
