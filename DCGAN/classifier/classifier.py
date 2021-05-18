from torch import nn
import torch


class Classifier(nn.Module):
    def __init__(self, image_size=64, num_classes=10):
        super(Classifier, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, image_size, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(image_size),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size, image_size, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(image_size),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(image_size, image_size * 4, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(image_size * 4, image_size * 8, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU(0.1, inplace=True))

        self.last= nn.Sequential(
            nn.Conv2d(image_size * 8, num_classes, kernel_size=4, stride=1))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out
