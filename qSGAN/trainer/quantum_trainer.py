import torch
from torch import nn
from time import time
import os
import matplotlib.pyplot as plt
import wandb
from qulacs import QuantumState

from ..utils import *


class QuantumTrainer:
    def __init__(self, generator, discriminator, train_dataloader, test_dataloader,
                 d_optimizer, z_dim=1, num_classes=None, save_params=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = generator
        self.D = discriminator.to(self.device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.d_optimizer = d_optimizer
        self.criterion_unsupervised = nn.BCELoss(reduction="mean")
        self.criterion_supervised = nn.CrossEntropyLoss(reduction="none")
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.save_params = save_params
        print(f"We use {self.device}")

    def train_one_cycle(self):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        self.D.train()

        for imges, labels, labels_mask in self.train_dataloader:
            imges = (imges > 0.5).to(torch.float)
            batch_size, _, N = imges.size()

            imges = imges.to(self.device)
            labels = labels.to(self.device)
            labels_mask = labels_mask.to(self.device)

            # 真偽label作成
            label_real = torch.full((batch_size,), 1, dtype=torch.float).to(self.device)
            label_fake = torch.full((batch_size,), 0, dtype=torch.float).to(self.device)

            # 偽データ生成
            fake_images = self.G(batch_size)
            fake_images = transform_to_N_px(fake_images, N)
            fake_images = fake_images.to(self.device)

            # discriminatorの学習
            imges = imges.view(batch_size, -1)
            d_out_real, d_out_cls = self.D(imges)
            d_out_fake, _ = self.D(fake_images)

            d_loss_real = self.criterion_unsupervised(d_out_real.view(-1), label_real)
            d_loss_fake = self.criterion_unsupervised(d_out_fake.view(-1), label_fake)
            d_loss = torch.mean(d_loss_real + d_loss_fake)

            if self.num_classes is not None:
                sum_masked = torch.sum(labels_mask.data) if torch.any(labels_mask) else 1
                d_loss_cls = self.criterion_supervised(d_out_cls.view(-1, self.num_classes), labels)
                d_loss_cls = torch.sum(labels_mask * d_loss_cls) / sum_masked
                d_loss += d_loss_cls

            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # generatorの学習
            with torch.no_grad():
                fake_images = self.G(batch_size)
                fake_images = transform_to_N_px(fake_images, N)
                fake_images = fake_images.to(self.device)
                d_out_fake, _ = self.D(fake_images)
                g_loss = self.criterion_unsupervised(d_out_fake.view(-1), label_real)
                x_plus, x_minus = self.G.calculate_x_plus_minus()
                x_plus = transform_to_N_px_2D(x_plus, N).to(self.device)
                x_minus = transform_to_N_px_2D(x_minus, N).to(self.device)
                x_plus, _ = self.D(x_plus)
                x_minus, _ = self.D(x_minus)
                grad = calculate_grad(x_plus, x_minus)
                self.G.update_parameter(grad)
                del x_plus, x_minus

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            del d_loss, g_loss, grad

        fake_images = wandb.Image(fake_images.cpu().detach().numpy())
        wandb.log({"example": fake_images})

        return epoch_g_loss, epoch_d_loss

    def valid_one_cycle(self):
        self.D.eval()
        correct = 0
        num_samples = 0
        with torch.no_grad():
            for imges, labels, _ in self.test_dataloader:
                imges = imges.to(self.device)
                labels = labels.to(self.device)

                _, d_pred = self.D(imges)

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
        for epoch in range(1, num_epochs+1):
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

                wandb.log({"Epoch_DQ_Loss": d_loss,
                           "Epoch_GQ_Loss": g_loss,
                           "Epoch_Accuracy": accuracy})

                if accuracy > best_accuracy:
                    wandb.run.summary["best_accuracy"] = accuracy
                    best_accuracy = accuracy
            else:
                t_finish = time()
                print("-" * 40)
                print(f"Epoch: {epoch} ||G_Loss: {g_loss} ||D_Loss: {d_loss} ||Timer: {t_finish - t_start}")
                t_start = time()

                wandb.log({"Epoch_DQ_Loss": d_loss,
                           "Epoch_GQ_Loss": g_loss,})

        return g_loss_list, d_loss_list, accuracy_list

    def visualize_GQ_image(self, N):
        batch_size = 20
        fake_images = self.G(batch_size)
        fake_images = transform_to_N_px(fake_images, N)

        plt.figure(figsize=(15, 6))
        plt.suptitle("Result of GQ", fontsize=25)
        for i in range(0, batch_size):
            plt.subplot(4, 5, i + 1)
            plt.imshow(fake_images[i].view(1, fake_images.size()[1]).cpu().detach().numpy())
        plt.savefig("figures/image_qSGAN_q.png", dpi=500)
        plt.show()

    def visualize_logs(self, g_logs, d_logs):
        epochs_list = range(len(g_logs))
        plt.plot(epochs_list, d_logs, label=f"dq_logs")
        plt.plot(epochs_list, g_logs, label=f"gq_logs")
        plt.legend()
        plt.savefig(f"figures/loss_qSGAN_q.png", dpi=500)
        plt.show()

    def visualize_probability(self):
        state = QuantumState(self.G.n_qubit)
        self.G.ansatz.update_quantum_state(state)
        measured_list = [list(map(int, list(format(x, f"0{self.G.n_qubit}b")))) for x in range(2**self.G.n_qubit)]
        prob_list = []
        for lis in measured_list:
            prob_list.append(state.get_marginal_probability(lis))
        plt.plot(range(2**self.G.n_qubit), prob_list)
