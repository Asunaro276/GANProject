import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_size=8, output_size=8, hidden_size=56, num_class=2):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(nn.Linear(hidden_size, output_size),
                                    nn.LeakyReLU(0.1, inplace=True))

        self.discriminate_layer = nn.Sigmoid()

        self.classify_layer = nn.Softmax(dim=1)

    def forward_unsupervised(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.discriminate_layer(out)

        return out

    def forward_supervised(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.classify_layer(out)

        return out
