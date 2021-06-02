from torch import nn
import torch


class SemiSupervisedDiscriminator(nn.Module):
    def __init__(self, image_size=64, num_classes=10):
        super(SemiSupervisedDiscriminator, self).__init__()

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
            nn.Conv2d(image_size, image_size * 2, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(image_size * 2, image_size * 2, kernel_size=4,
                      stride=2, padding=1),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.1, inplace=True))

        self.last_supervised = nn.Sequential(
            nn.Conv2d(image_size * 2, num_classes, kernel_size=4, stride=1))

        self.last_unsupervised = nn.Sequential(
            nn.Conv2d(image_size * 2, 1, kernel_size=4, stride=1))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out_unsupervised = self.last_unsupervised(out)
        out_supervised = self.last_supervised(out)

        return out_unsupervised, out_supervised


if __name__ == "__main__":
    from DCGAN.generator import Generator

    D = SemiSupervisedDiscriminator(image_size=64)

    G = Generator()
    input_z = torch.randn(1, 20)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    fake_images = G.forward(input_z)

    d_out = D(fake_images)

    print(d_out)
