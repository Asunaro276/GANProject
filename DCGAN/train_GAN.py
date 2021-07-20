import torch
from torch import nn
from time import time
import matplotlib.pyplot as plt


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("conv") != -1:
        nn.init.normal(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_model(G, D, dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We use {device}")

    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, (beta1, beta2))
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, (beta1, beta2))

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    z_dim = 20

    G.to(device)
    D.to(device)

    G.train()
    D.train()

    torch.backends.cudnn.benchmark = True

    batch_size = dataloader.batch_size
    d_logs = []
    g_logs = []

    t_start = time()
    for epoch in range(1, num_epochs+1):
        t_epoch_start = time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print("-"*40)
        print(f"Epoch {epoch}/{num_epochs}")
        print("-"*40)
        print("(train)")

        for imges in dataloader:
            if imges.size()[0] == 1:
                continue
            imges = imges.to(device)

            mini_batch_size = imges.size()[0]
            label_real = torch.full((mini_batch_size,), 1, dtype=torch.float).to(device)
            label_fake = torch.full((mini_batch_size,), 0, dtype=torch.float).to(device)

            d_out_real = D.forward(imges)

            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G.forward(input_z)
            d_out_fake = D.forward(fake_images)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G.forward(input_z)
            d_out_fake = D.forward(fake_images)

            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        t_epoch_finish = time()
        print("-"*40)
        print(f"epoch {epoch} || Epoch_D_Loss: {epoch_d_loss/batch_size} || Epoch_G_Loss: {epoch_g_loss/batch_size}")
        print(f"timer: {t_epoch_finish - t_epoch_start} sec.")
        d_logs.append(epoch_d_loss)
        g_logs.append(epoch_g_loss)

    t_finish = time()
    print("-" * 40)
    print(f"Total time: {t_finish - t_start} sec.")

    return G, D, g_logs, d_logs


def visualization(G_update, train_dataloader, g_logs, d_logs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 10
    z_dim = 20
    fixed_z = torch.randn(batch_size, z_dim)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

    fake_images = G_update(fixed_z.to(device))

    batch_iterator = iter(train_dataloader)
    imges = next(batch_iterator)

    plt.figure(figsize=(15, 6))
    plt.suptitle("Result of GAN", fontsize=25)
    for i in range(0, 10):
        plt.subplot(4, 5, i+1)
        plt.imshow(imges[i][0].cpu().detach().numpy(), "gray")

        plt.subplot(4, 5, 10+i+1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), "gray")
    plt.savefig("./figures/image_GAN.png", dpi=500)
    plt.show()

    num_epochs = range(len(d_logs))
    plt.plot(num_epochs, d_logs, label="d_logs")
    plt.plot(num_epochs, g_logs, label="g_logs")
    plt.legend()
    plt.savefig("./figures/loss_GAN.png", dpi=500)
    plt.show()


if __name__ == "__main__":
    from torch.utils import data

    from DCGAN.generator.generator import Generator
    from DCGAN.discriminator.discriminator_semi import Discriminator
    from data.data_loader import UnSupervisedImageDataset, ImageTransform, make_datapath_list

    G = Generator(z_dim=20, image_size=64)
    D = Discriminator(image_size=64)

    G.apply(weights_init)
    D.apply(weights_init)

    print("Finish initialization of the network")

    label_list = list(range(10))
    train_img_list, _ = make_datapath_list(label_list)

    mean = (0.5,)
    std = (0.5,)
    train_dataset = UnSupervisedImageDataset(file_list=train_img_list, transform=ImageTransform(mean, std))
    batch_size = 64
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(train_dataloader)

    num_epochs = 20
    G_update, D_update, g_logs, d_logs = train_model(G, D, dataloader=train_dataloader, num_epochs=num_epochs)

    visualization(G_update, train_dataloader, g_logs, d_logs)
