import torch
from torch import nn
from time import time
import os
import matplotlib.pyplot as plt
from utils import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("conv") != -1:
        nn.init.normal(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_classical_model(G, D, train_dataloader, test_dataloader, num_epochs, save_params=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We use {device}")

    g_lr, d_lr = 0.001, 0.001
    beta1, beta2 = 0.9, 0.999
    g_optimizer = torch.optim.Adam(GC.parameters(), g_lr, (beta1, beta2))
    d_optimizer = torch.optim.Adam(DC.parameters(), d_lr, (beta1, beta2))

    criterion_unsupervised = nn.BCELoss(reduction="mean")
    criterion_supervised = nn.CrossEntropyLoss(reduction="none")

    z_dim = 1

    G.to(device)
    D.to(device)

    torch.backends.cudnn.benchmark = True

    g_logs = []
    d_logs = []

    test_accuracy_list = []

    params_dir = "params/"
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)

    t_start = time()
    for epoch in range(1, num_epochs+1):
        t_epoch_start = time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        G.train()
        D.train()

        g_params_dir = os.path.join(params_dir, "g_params/")
        d_params_dir = os.path.join(params_dir, "d_params/")
        if not os.path.exists(g_params_dir):
            os.makedirs(g_params_dir)
        if not os.path.exists(d_params_dir):
            os.makedirs(d_params_dir)

        print("-"*40)
        print(f"Epoch {epoch}/{num_epochs}")
        print("-"*40)
        print("(train)")

        for imges, labels, labels_mask in train_dataloader:

            if imges.size()[0] != train_batch_size:
                continue

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            imges = imges.to(device)
            labels = labels.to(device)
            labels_mask = labels_mask.to(device)


            # 真偽label作成
            label_real = torch.full((train_batch_size,), 1, dtype=torch.float).to(device)
            label_fake = torch.full((train_batch_size,), 0, dtype=torch.float).to(device)

            # 偽データ生成
            input_z = torch.randn(train_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1))
            fake_images = G(input_z)

            # discriminatorの学習
            imges = imges.view(train_batch_size, -1)
            d_out_real, d_out_cls = D(imges)
            d_out_fake, _ = D(fake_images)

            d_loss_real = criterion_unsupervised(d_out_real.view(-1), label_real)
            d_loss_fake = criterion_unsupervised(d_out_fake.view(-1), label_fake)
            d_loss = torch.mean(d_loss_real + d_loss_fake)

            sum_masked = torch.sum(labels_mask.data) if torch.any(labels_mask) else 1
            d_loss_cls = criterion_supervised(d_out_cls.view(-1, num_classes), labels)
            d_loss_cls = torch.sum(labels_mask * d_loss_cls) / sum_masked
            d_loss += d_loss_cls

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()


            # generatorの学習
            input_z = torch.randn(batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1))
            fake_images = GC(input_z)
            d_out_fake, _ = D(fake_images)

            g_loss = criterion_unsupervised(d_out_fake.view(-1), label_real)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        D.eval()
        test_d_loss = 0
        correct = 0
        num_samples = 0
        with torch.no_grad():
            for imges, labels, _ in test_dataloader:
                imges = imges.to(device)
                labels = labels.to(device)

                _, d_pred = D(imges)
                d_loss = torch.mean(criterion_supervised(d_pred.view(-1, num_classes), labels))
                test_d_loss += d_loss.item()

                d_pred_labels = torch.max(d_pred, 3)[1]
                d_eq = torch.eq(labels, d_pred_labels.view(-1))
                correct += torch.sum(d_eq.float())
                num_samples += len(labels)


        epoch_test_accuracy = correct / num_samples

        test_accuracy_list.append(epoch_test_accuracy.item())

        t_epoch_finish = time()
        print("-"*40)
        print(f"epoch {epoch} || Epoch_DC_Loss: {epoch_d_loss/train_batch_size} || "
              f"Epoch_GC_Loss: {epoch_g_loss/train_batch_size}")
        print(f"timer: {t_epoch_finish - t_epoch_start} sec.")
        g_logs.append(epoch_g_loss/train_batch_size)
        d_logs.append(epoch_d_loss/train_batch_size)

        # パラメータ保存
        if save_params:
            g_params_filename = os.path.join(g_params_dir, f"g_params_{epoch}")
            d_params_filename = os.path.join(d_params_dir, f"d_params_{epoch}")
            torch.save(GC.state_dict(), g_params_filename)
            torch.save(DC.state_dict(), d_params_filename)

    t_finish = time()
    print("-" * 40)
    print(f"Total time: {t_finish - t_start} sec.")

    return GC, DC, g_logs, d_logs, test_accuracy_list


def train_quantum_model(G, D, train_dataloader, test_dataloader, num_epochs, save_params=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We use {device}")

    d_lr = 0.001
    beta1, beta2 = 0.9, 0.999
    d_optimizer = torch.optim.Adam(DQ.parameters(), d_lr, (beta1, beta2))

    criterion_unsupervised = nn.BCELoss(reduction="mean")
    criterion_supervised = nn.CrossEntropyLoss(reduction="none")

    D.to(device)

    torch.backends.cudnn.benchmark = True

    g_logs = []
    d_logs = []

    test_accuracy_list = []

    params_dir = "params/"
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)

    t_start = time()
    for epoch in range(1, num_epochs+1):
        t_epoch_start = time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        D.train()

        g_params_dir = os.path.join(params_dir, "g_params/")
        d_params_dir = os.path.join(params_dir, "d_params/")
        if not os.path.exists(g_params_dir):
            os.makedirs(g_params_dir)
        if not os.path.exists(d_params_dir):
            os.makedirs(d_params_dir)

        print("-"*40)
        print(f"Epoch {epoch}/{num_epochs}")
        print("-"*40)
        print("(train)")

        for imges, labels, labels_mask in train_dataloader:

            if imges.size()[0] != train_batch_size:
                continue

            d_optimizer.zero_grad()

            imges = imges.to(device)
            labels = labels.to(device)
            labels_mask = labels_mask.to(device)


            # 真偽label作成
            label_real = torch.full((train_batch_size,), 1, dtype=torch.float).to(device)
            label_fake = torch.full((train_batch_size,), 0, dtype=torch.float).to(device)

            # 偽データ生成
            fake_images = GQ()
            fake_images = transform_to_8_px(fake_images)
            fake_images = fake_images.to(device)

            # discriminatorの学習
            imges = imges.view(train_batch_size, -1)
            d_out_real, d_out_cls = D(imges)
            d_out_fake, _ = D(fake_images)

            d_loss_real = criterion_unsupervised(d_out_real.view(-1), label_real)
            d_loss_fake = criterion_unsupervised(d_out_fake.view(-1), label_fake)
            d_loss = torch.mean(d_loss_real + d_loss_fake)

            sum_masked = torch.sum(labels_mask.data) if torch.any(labels_mask) else 1
            d_loss_cls = criterion_supervised(d_out_cls.view(-1, num_classes), labels)
            d_loss_cls = torch.sum(labels_mask * d_loss_cls) / sum_masked
            d_loss += d_loss_cls

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # generatorの学習
            fake_images = G()
            fake_images = transform_to_8_px(fake_images)
            fake_images = fake_images.to(device)
            d_out_fake, _ = D(fake_images)
            g_loss = criterion_unsupervised(d_out_fake.view(-1), label_real)
            x_plus, x_minus = G.calculate_x_plus_minus()
            x_plus = transform_to_8_px_2D(x_plus).to(device)
            x_minus = transform_to_8_px_2D(x_minus).to(device)
            x_plus, _ = D(x_plus)
            x_minus, _ = D(x_minus)
            grad = calculate_grad(x_plus, x_minus)
            G.update_parameter(grad)

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        D.eval()
        test_d_loss = 0
        d_correct = 0
        num_samples = 0
        with torch.no_grad():
            for imges, labels, _ in test_dataloader:
                imges = imges.to(device)
                labels = labels.to(device)

                _, d_pred = D(imges)
                d_loss = torch.mean(criterion_supervised(d_pred.view(-1, num_classes), labels))
                test_d_loss += d_loss.item()

                d_pred_labels = torch.max(d_pred, 3)[1]
                d_eq = torch.eq(labels, d_pred_labels.view(-1))
                d_correct += torch.sum(d_eq.float())
                num_samples += len(labels)

        epoch_test_d_accuracy = d_correct / num_samples

        test_accuracy_list.append(epoch_test_d_accuracy.item())

        t_epoch_finish = time()
        print("-"*40)
        print(f"epoch {epoch} || Epoch_DC_Loss: {epoch_d_loss/train_batch_size} || "
              f"Epoch_GC_Loss: {epoch_g_loss/train_batch_size}")
        print(f"timer: {t_epoch_finish - t_epoch_start} sec.")
        g_logs.append(epoch_g_loss/train_batch_size)
        d_logs.append(epoch_d_loss/train_batch_size)

        # パラメータ保存
        if save_params:
            g_params_filename = os.path.join(g_params_dir, f"g_params_{epoch}")
            d_params_filename = os.path.join(d_params_dir, f"d_params_{epoch}")
            torch.save(GC.state_dict(), g_params_filename)
            torch.save(DC.state_dict(), d_params_filename)

    t_finish = time()
    print("-" * 40)
    print(f"Total time: {t_finish - t_start} sec.")

    return G, D, g_logs, d_logs, test_accuracy_list


def visualize_GC_image(G_update, train_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 5
    z_dim = 1
    fixed_z = torch.randn(batch_size, z_dim)

    fake_images = G_update(fixed_z.to(device))
    fake_images = (fake_images > 0.5).view(fake_images.size()[0], fake_images.size()[1], 1)

    batch_iterator = iter(train_dataloader)
    imges = next(batch_iterator)

    plt.figure(figsize=(15, 6))
    plt.suptitle("Result of GC", fontsize=25)
    for i in range(0, 5):
        plt.subplot(2, 5, i+1)
        plt.imshow(imges[0][i][0].cpu().detach().numpy())

        plt.subplot(2, 5, 5+i+1)
        plt.imshow(fake_images[i].view(1, fake_images.size()[1]).cpu().detach().numpy())
    plt.savefig("figures/image_qSGAN_c.png", dpi=500)
    plt.show()

def visualize_GQ_image(G_update, train_dataloader):
    batch_size = 5
    fake_images = G_update(batch_size)
    fake_images = transform_to_8_px(fake_images)

    batch_iterator = iter(train_dataloader)
    imges = next(batch_iterator)

    plt.figure(figsize=(15, 6))
    plt.suptitle("Result of GQ", fontsize=25)
    for i in range(0, 5):
        plt.subplot(2, 5, i+1)
        plt.imshow(imges[0][i][0].cpu().detach().numpy())

        plt.subplot(2, 5, 5+i+1)
        plt.imshow(fake_images[i].view(1, fake_images.size()[1]).cpu().detach().numpy())
    plt.savefig("figures/image_qSGAN_q.png", dpi=500)
    plt.show()

def visualize_logs(num_epochs, g_logs, d_logs, label):
    epochs_list = range(num_epochs)
    plt.plot(epochs_list, d_logs, label=f"d{label}_logs")
    plt.plot(epochs_list, g_logs, label=f"g{label}_logs")
    plt.legend()
    plt.savefig(f"figures/loss_qSGAN_{label}.png", dpi=500)
    plt.show()

def visualize_accuracy(num_epochs, dc_accuracy, dq_accuracy):
    epochs_list = range(num_epochs)
    plt.plot(epochs_list, dc_accuracy, label="dc_accuracy")
    # plt.plot(epochs_list, dq_accuracy, label="dq_accuracy")
    plt.legend()
    plt.savefig("figures/accuracy_qSGAN.png", dpi=500)
    plt.show()


if __name__ == "__main__":
    from torch.utils import data
    from sklearn.model_selection import train_test_split

    from generator.generator_quantum import QuantumGenerator
    from generator.generator_classical import Generator
    from discriminator.discriminator import Discriminator
    from qSGAN.data_loader import ImageDataset, ImageTransform, make_datapath_list

    num_qubit = 8
    input_size = 8
    output_size = 8
    hidden_size = 28
    num_classes = 2
    GQ = QuantumGenerator(num_qubit)
    GC = Generator(1, 8, 256)
    DC = Discriminator(input_size, output_size, hidden_size, num_classes)
    DQ = Discriminator(input_size, output_size, hidden_size, num_classes)

    GC.apply(weights_init)
    DC.apply(weights_init)
    DQ.apply(weights_init)

    print("Finish initialization of the network")

    img_list, label_list = make_datapath_list()
    train_img_list, test_img_list, train_label_list, test_label_list = train_test_split(
        img_list, label_list, test_size=1/4)

    mean = (0.002,)
    std = (0.001,)
    train_dataset = ImageDataset(data_list=train_img_list, transform=ImageTransform(mean, std),
                                 label_list=train_label_list)
    test_dataset = ImageDataset(data_list=test_img_list, transform=ImageTransform(mean, std),
                                label_list=test_label_list)

    batch_size = 7
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(set(train_dataloader.dataset.label))
    train_batch_size = train_dataloader.batch_size

    num_epochs = 300
    GC_update, DC_update, gc_logs, dc_logs, dc_accuracy = train_classical_model(
        GC, DC, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
        num_epochs=num_epochs, save_params=True)

    # GQ_update, DQ_update, gq_logs, dq_logs, dq_accuracy = train_quantum_model(
    #     GQ, DQ, img_list=img_list, label_list=label_list,
    #     num_epochs=num_epochs, save_params=True)
    dq_accuracy = 0

    visualize_GC_image(GC_update, train_dataloader)
    # visualize_GQ_image(GQ_update, train_dataloader)

    visualize_logs(num_epochs, gc_logs, dc_logs, label="c")
    # visualize_logs(num_epochs, gq_logs, dq_logs, label="q")

    visualize_accuracy(num_epochs, dc_accuracy, dq_accuracy)
