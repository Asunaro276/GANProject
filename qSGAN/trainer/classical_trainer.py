import torch
from torch import nn
from time import time
import os
import matplotlib.pyplot as plt
import wandb

from ..utils import *


class ClassicalTrainer:
    def __init__(self, generator, discriminator, train_dataloader, test_dataloader,
                 g_optimizer, d_optimizer, z_dim=1, num_classes=None, save_params=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = generator.to(self.device)
        self.D = discriminator.to(self.device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.criterion_supervised = nn.CrossEntropyLoss(reduction="none")
        self.criterion_unsupervised = nn.BCELoss(reduction="mean")
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.save_params = save_params
        print(f"We use {self.device}")

    def train_one_cycle(self):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        self.G.train()
        self.D.train()

        params_dir = "params/"
        g_params_dir = os.path.join(params_dir, "g_params/")
        d_params_dir = os.path.join(params_dir, "d_params/")
        os.makedirs(g_params_dir, exist_ok=True)
        os.makedirs(d_params_dir, exist_ok=True)

        for imges, labels, labels_mask in self.train_dataloader:
            imges = (imges > 0.5).to(torch.float)
            batch_size = imges.size()[0]

            imges = imges.to(self.device)
            labels = labels.to(self.device)
            labels_mask = labels_mask.to(self.device)

            # 真偽label作成
            label_real = torch.full((batch_size,), 1, dtype=torch.float).to(self.device)
            label_fake = torch.full((batch_size,), 0, dtype=torch.float).to(self.device)

            # 偽データ生成
            input_z = torch.randn(batch_size, self.z_dim).to(self.device)
            fake_images = self.G(input_z)

            # discriminatorの学習
            imges = imges.view(batch_size, -1)
            d_out_real, d_out_cls = self.D(imges)
            d_out_fake, _ = self.D(fake_images)

            d_loss_real = self.criterion_unsupervised(d_out_real.view(-1), label_real)
            d_loss_fake = self.criterion_unsupervised(d_out_fake.view(-1), label_fake)
            d_loss = (d_loss_real + d_loss_fake) / 2

            if self.num_classes is not None:
                sum_masked = torch.sum(labels_mask.data) if torch.any(labels_mask) else 1
                d_loss_cls = self.criterion_supervised(d_out_cls.view(-1, self.num_classes), labels)
                d_loss_cls = torch.sum(labels_mask * d_loss_cls) / sum_masked
                d_loss *= 2 / 3
                d_loss += d_loss_cls / 3

            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # generatorの学習
            input_z = torch.randn(batch_size, self.z_dim).to(self.device)
            fake_images = self.G(input_z)
            d_out_fake, _ = self.D(fake_images)

            g_loss = self.criterion_unsupervised(d_out_fake.view(-1), label_real)

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            epoch_g_loss += g_loss.item() / batch_size
            epoch_d_loss += d_loss.item() / batch_size

        fake_images = wandb.Image(fake_images.cpu().detach().numpy())
        wandb.log({"example": fake_images})

        return epoch_g_loss, epoch_d_loss

    def valid_one_cycle(self):
        self.D.eval()
        test_d_loss = 0
        correct = 0
        num_samples = 0
        with torch.no_grad():
            for imges, labels, _ in self.test_dataloader:
                imges = (imges > 0.5).to(torch.float)

                imges = imges.to(self.device)
                labels = labels.to(self.device)

                _, d_pred = self.D(imges)
                d_loss = torch.mean(self.criterion_supervised(d_pred.view((-1, self.num_classes)), labels))
                test_d_loss += d_loss.item()

                d_pred_labels = torch.max(d_pred, 2)[1]
                d_eq = torch.eq(labels, d_pred_labels.view(-1))
                correct += torch.sum(d_eq.float())
                num_samples += len(labels)
        accuracy = correct / num_samples
        return accuracy.item()

    def train(self, num_epochs, log_freq):
        wandb.watch((self.G, self.D), log_freq=log_freq)
        g_loss_list = []
        d_loss_list = []
        accuracy_list = []
        best_accuracy = 0
        t_start = time()
        for epoch in range(1, num_epochs + 1):
            g_loss, d_loss = self.train_one_cycle()
            g_loss_list.append(g_loss)
            d_loss_list.append(d_loss)

            if self.num_classes is not None:
                accuracy = self.valid_one_cycle()
                accuracy_list.append(accuracy)

                if epoch % log_freq == 0:
                        t_finish = time()
                        print("-"*40)
                        print(f"Epoch: {epoch} ||G_Loss: {g_loss} ||D_Loss: {d_loss} ||Epoch_Accuracy: {accuracy} ||Timer: {t_finish - t_start}")
                        t_start = time()

                wandb.log({"G_Loss": g_loss,
                           "D_Loss": d_loss,
                           "Epoch_Accuracy": accuracy})

                if accuracy > best_accuracy:
                    wandb.run.summary["best_accuracy"] = accuracy
                    best_accuracy = accuracy
            else:
                if epoch % log_freq == 0:
                    t_finish = time()
                    print("-" * 40)
                    print(f"Epoch: {epoch} ||G_Loss: {g_loss} ||D_Loss: {d_loss} ||Timer: {t_finish - t_start}")
                    t_start = time()
                wandb.log({"G_Loss": g_loss,
                           "D_Loss": d_loss})


        return g_loss_list, d_loss_list, accuracy_list

    def visualize_G_image(self):
        batch_size = 20
        fixed_z = torch.randn(batch_size, self.z_dim)

        fake_images = self.G(fixed_z.to(self.device))
        fake_images = (fake_images > 0.5).view(fake_images.size()[0], fake_images.size()[1], 1)

        plt.figure(figsize=(15, 6))
        plt.suptitle("Result of GC", fontsize=25)
        for i in range(0, batch_size):
            plt.subplot(4, 5, i+1)
            plt.imshow(fake_images[i].view(1, fake_images.size()[1]).cpu().detach().numpy())
        plt.savefig("figures/image_qSGAN_c.png", dpi=500)
        plt.show()

    def visualize_logs(self, g_logs, d_logs):
        epochs_list = range(len(g_logs))
        plt.plot(epochs_list, d_logs, label=f"dc_logs")
        plt.plot(epochs_list, g_logs, label=f"gc_logs")
        plt.legend()
        plt.savefig(f"figures/loss_qSGAN_c.png", dpi=500)
        plt.show()
