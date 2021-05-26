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


def train_model(G, D, C, train_dataloader, test_dataloader, num_epochs, save_params=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We use {device}")

    g_lr, d_lr, c_lr = 0.0001, 0.0004, 0.0002
    beta1, beta2 = 0.0, 0.9
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, (beta1, beta2))
    c_optimizer = torch.optim.Adam(C.parameters(), c_lr, (beta1, beta2))

    criterion_unsupervised = nn.BCELoss(reduction="mean")
    criterion_supervised = nn.CrossEntropyLoss(reduction="none")
    criterion_classifier = nn.CrossEntropyLoss(reduction="none")

    D.to(device)
    C.to(device)

    torch.backends.cudnn.benchmark = True

    num_classes = len(set(train_dataloader.dataset.label))
    train_batch_size = train_dataloader.batch_size
    test_batch_size = test_dataloader.batch_size
    g_logs = []
    d_logs = []
    c_logs = []

    test_c_accuracy_list = []
    test_d_accuracy_list = []

    params_dir = "DCGAN/params/"
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)

    t_start = time()
    for epoch in range(1, num_epochs+1):
        t_epoch_start = time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_c_loss = 0.0

        D.train()
        C.train()

        g_params_dir = os.path.join(params_dir, "g_params/")
        d_params_dir = os.path.join(params_dir, "d_params/")
        c_params_dir = os.path.join(params_dir, "c_params/")
        if not os.path.exists(g_params_dir):
            os.makedirs(g_params_dir)
        if not os.path.exists(d_params_dir):
            os.makedirs(d_params_dir)
        if not os.path.exists(c_params_dir):
            os.makedirs(c_params_dir)

        print("-"*40)
        print(f"Epoch {epoch}/{num_epochs}")
        print("-"*40)
        print("(train)")

        for imges, labels, labels_mask in train_dataloader:
            if imges.size()[0] != batch_size:
                continue

            d_optimizer.zero_grad()
            c_optimizer.zero_grad()

            imges = imges.to(device)
            labels = labels.to(device)
            labels_mask = labels_mask.to(device)


            # 真偽label作成
            label_real = torch.full((batch_size,), 1, dtype=torch.float).to(device)
            label_fake = torch.full((batch_size,), 0, dtype=torch.float).to(device)

            # 偽データ生成
            fake_images = G()
            fake_images = transform_to_8_px(fake_images)

            # discriminatorの学習
            d_out_real, d_out_cls = D(imges)
            d_out_fake, _ = D(fake_images)

            d_loss_real = criterion_unsupervised(d_out_real.view(-1), label_real)
            d_loss_fake = criterion_unsupervised(d_out_fake.view(-1), label_fake)
            d_loss = torch.mean(d_loss_real + d_loss_fake)

            sum_masked = torch.max(torch.Tensor([torch.sum(labels_mask.data), 1.0]))
            d_loss_cls = criterion_supervised(d_out_cls.view(-1, num_classes), labels)
            d_loss_cls = torch.sum(labels_mask * d_loss_cls) / sum_masked
            d_loss += d_loss_cls

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # classifierの学習
            c_out = C(imges)
            c_loss = criterion_classifier(c_out.view(-1, num_classes), labels)
            c_loss = torch.sum(labels_mask * c_loss) / sum_masked
            c_loss.backward()
            c_optimizer.step()

            # generatorの学習
            fake_images = G()
            x_plus, x_minus = G.calculate_x_plus_minus()
            x_plus = transform_to_8_px(x_plus)
            x_minus = transform_to_8_px(x_minus)
            x_plus, _ = D(x_plus)
            x_minus, _ = D(x_minus)
            grad = calculate_grad(x_plus, x_minus)
            G.update_parameter(grad)

            d_out_fake, _ = D(fake_images)

            g_loss = criterion_unsupervised(d_out_fake.view(-1), label_real)
            g_loss.backward()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_c_loss += c_loss.item()

        D.eval()
        C.eval()
        test_c_loss = 0
        test_d_loss = 0
        c_correct = 0
        d_correct = 0
        num_samples = 0
        with torch.no_grad():
            for imges, labels, _ in test_dataloader:
                imges = imges.to(device)
                labels = labels.to(device)

                c_pred = C(imges)
                _, d_pred = D(imges)
                c_loss = torch.mean(criterion_classifier(c_pred.view(-1, num_classes), labels))
                d_loss = torch.mean(criterion_supervised(d_pred.view(-1, num_classes), labels))
                test_c_loss += c_loss.item()
                test_d_loss += d_loss.item()

                c_pred_labels = torch.max(c_pred, 1)[1]
                d_pred_labels = torch.max(d_pred, 1)[1]
                c_eq = torch.eq(labels, c_pred_labels.view(-1))
                d_eq = torch.eq(labels, d_pred_labels.view(-1))
                c_correct += torch.sum(c_eq.float())
                d_correct += torch.sum(d_eq.float())
                num_samples += len(labels)


        epoch_test_c_accuracy = c_correct / num_samples
        epoch_test_d_accuracy = d_correct / num_samples

        test_c_accuracy_list.append(epoch_test_c_accuracy.item())
        test_d_accuracy_list.append(epoch_test_d_accuracy.item())

        t_epoch_finish = time()
        print("-"*40)
        print(f"epoch {epoch} || Epoch_D_Loss: {epoch_d_loss/train_batch_size} || "
              f"Epoch_G_Loss: {epoch_g_loss/train_batch_size}")
        print(f"timer: {t_epoch_finish - t_epoch_start} sec.")
        d_logs.append(epoch_d_loss/train_batch_size)
        g_logs.append(epoch_g_loss/train_batch_size)
        c_logs.append(epoch_c_loss/train_batch_size)

        # パラメータ保存
        if save_params:
            g_params_filename = os.path.join(g_params_dir, f"g_params_{epoch}")
            d_params_filename = os.path.join(d_params_dir, f"d_params_{epoch}")
            c_params_filename = os.path.join(c_params_dir, f"c_params_{epoch}")
            torch.save(G.state_dict(), g_params_filename)
            torch.save(D.state_dict(), d_params_filename)
            torch.save(C.state_dict(), c_params_filename)

    t_finish = time()
    print("-" * 40)
    print(f"Total time: {t_finish - t_start} sec.")

    return G, D, g_logs, d_logs, c_logs, test_d_accuracy_list, test_c_accuracy_list


def visualization(G_update, train_dataloader, g_logs, d_logs, c_logs, d_accuracy, c_accuracy):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 10
    z_dim = 20
    fixed_z = torch.randn(batch_size, z_dim)
    fixed_z = fixed_z.view(fixed_z.size()[0], fixed_z.size()[1], 1, 1)

    fake_images = G_update(fixed_z.to(device))

    batch_iterator = iter(train_dataloader)
    imges = next(batch_iterator)

    plt.figure(figsize=(15, 6))
    plt.suptitle("Result of SGAN", fontsize=25)
    for i in range(0, 10):
        plt.subplot(4, 5, i+1)
        plt.imshow(imges[0][i][0].cpu().detach().numpy(), "gray")

        plt.subplot(4, 5, 10+i+1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), "gray")
    plt.savefig("DCGAN/figures/image_SGAN.png", dpi=500)
    plt.show()

    epochs_list = range(len(d_logs))
    plt.plot(epochs_list, d_logs, label="d_logs")
    plt.plot(epochs_list, g_logs, label="g_logs")
    plt.plot(epochs_list, c_logs, label="c_logs")
    plt.legend()
    plt.savefig("DCGAN/figures/loss_SGAN.png", dpi=500)
    plt.show()

    plt.plot(epochs_list, d_accuracy, label="d_accuracy")
    plt.plot(epochs_list, c_accuracy, label="c_accuracy")
    plt.legend()
    plt.savefig("DCGAN/figures/accuracy_SGAN.png", dpi=500)
    plt.show()


if __name__ == "__main__":
    from torch.utils import data
    from sklearn.model_selection import train_test_split

    from generator.generator_quantum import QuantumGenerator, HEA
    from discriminator.discriminator import Discriminator
    from classifier.classifier import Classifier
    from qSGAN.data_loader import ImageDataset, ImageTransform, make_datapath_list

    num_qubit = 8
    image_size_g = 64
    image_size_d = 12
    num_classes = 2
    ansatz = HEA(num_qubit, 3)
    G = QuantumGenerator(ansatz)
    D = Discriminator(image_size_d, num_classes)
    C = Classifier(image_size_d, num_classes)

    D.apply(weights_init)

    print("Finish initialization of the network")

    label_list = list(range(num_classes))
    img_list, label_list = make_datapath_list(label_list)
    train_img_list, test_img_list, train_label_list, test_label_list = train_test_split(
        img_list, label_list, test_size=0.2)

    mean = (0.5,)
    std = (0.5,)
    train_dataset = ImageDataset(data_list=train_img_list, transform=ImageTransform(mean, std),
                                 label_list=train_label_list)
    test_dataset = ImageDataset(data_list=test_img_list, transform=ImageTransform(mean, std),
                                 label_list=test_label_list)

    batch_size = 64
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 300
    G_update, D_update, g_logs, d_logs, c_logs, d_accuracy, c_accuracy = train_model(G, D, C,
                                                                             train_dataloader=train_dataloader,
                                                                             test_dataloader=test_dataloader,
                                                                             num_epochs=num_epochs, save_params=True)

    visualization(G_update, train_dataloader, g_logs, d_logs, c_logs, d_accuracy, c_accuracy)