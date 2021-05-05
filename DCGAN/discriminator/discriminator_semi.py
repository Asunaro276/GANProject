from torch import nn
import torch


class SemiSupervisedDiscriminator(nn.Module):
    def __init__(self, z_dim=20, output_dim=10, image_size=64):
        super(SemiSupervisedDiscriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, image_size, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size, image_size * 2, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(image_size * 2, image_size * 4, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(image_size * 4, image_size * 8, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.last_unsupervised = nn.Sequential(nn.Conv2d(image_size * 8, 1, kernel_size=4, stride=1))

        self.last_supervised = nn.Sequential(nn.Conv2d(image_size * 8, output_dim, kernel_size=4, stride=1))


    def forward_supervised(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last_supervised(out)

        return out

    def forward_unsupervised(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last_unsupervised(out)

        return out


if __name__ == "__main__":
    from DCGAN.generator.generator_normal import Generator

    D = SemiSupervisedDiscriminator(z_dim=20, image_size=64)

    G = Generator()
    input_z = torch.randn(1, 20)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
    fake_images = G.forward(input_z)


    d_out_unsupervised = D.forward_unsupervised(fake_images)

    print(d_out_unsupervised)

    d_out_supervised = D.forward_supervised(fake_images)

    print(d_out_supervised)