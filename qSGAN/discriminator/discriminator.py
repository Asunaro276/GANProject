import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, input_size=8, output_size=8, hidden_size=56, num_classes=2):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(nn.Linear(hidden_size, output_size),
                                    nn.LeakyReLU(0.1, inplace=True))

        self.last_unsupervised = nn.Sequential(nn.Linear(output_size, 1),
                                               nn.Sigmoid())

        self.last_supervised = nn.Sequential(nn.Linear(output_size, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out_unsupervised = self.last_unsupervised(out)
        out_supervised = self.last_supervised(out)

        return out_unsupervised, out_supervised
