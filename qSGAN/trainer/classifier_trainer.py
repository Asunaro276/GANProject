from torch import nn
import wandb
from time import time

from ..utils import *


class ClassifierTrainer:
    def __init__(self, classifier, train_dataloader, test_dataloader,
                 c_optimizer, z_dim=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.C = classifier.to(self.device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_classes = len(set(train_dataloader.dataset.label))
        self.c_optimizer = c_optimizer
        self.criterion_supervised = nn.CrossEntropyLoss(reduction="none")
        self.z_dim = z_dim
        self.batch_size = train_dataloader.batch_size
        print(f"We use {self.device}")

    def train_one_cycle(self):
        epoch_loss = 0.0

        self.C.train()

        for imges, labels, labels_mask in self.train_dataloader:
            imges = (imges > 0.5).to(torch.float)

            imges = imges.to(self.device)
            labels = labels.to(self.device)
            labels_mask = labels_mask.to(self.device)

            _, c_out = self.C(imges)

            sum_masked = torch.sum(labels_mask.data) if torch.any(labels_mask) else 1
            loss = self.criterion_supervised(c_out.view(-1, self.num_classes), labels)
            loss = torch.sum(labels_mask * loss) / sum_masked

            self.c_optimizer.zero_grad()
            loss.backward()
            self.c_optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss

    def valid_one_cycle(self):
        self.C.eval()
        correct = 0
        num_samples = 0
        with torch.no_grad():
            for imges, labels, _ in self.test_dataloader:
                imges = (imges > 0.5).to(torch.float)

                imges = imges.to(self.device)
                labels = labels.to(self.device)

                _, c_pred = self.C(imges)

                d_pred_labels = torch.max(c_pred, 2)[1]
                d_eq = torch.eq(labels, d_pred_labels.view(-1))
                correct += torch.sum(d_eq.float()).item()
                num_samples += len(labels)
        accuracy = correct / num_samples
        return accuracy

    def train(self, num_epochs, log_freq):
        wandb.watch(models=self.C, log_freq=100)
        c_loss_list = []
        accuracy_list = []
        best_accuracy = 0
        t_start = time()
        for epoch in range(1, num_epochs+1):
            loss = self.train_one_cycle()
            c_loss_list.append(loss / self.batch_size)

            accuracy = self.valid_one_cycle()
            accuracy_list.append(accuracy)

            if epoch % log_freq == 0:
                t_finish = time()
                print("-"*40)
                print(f"Epoch: {epoch} ||C_Loss: {loss} ||Epoch_Accuracy: {accuracy} ||Timer: {t_finish - t_start}")
                t_start = time()

            wandb.log({"Epoch_C_Loss": loss,
                       "Epoch_Accuracy": accuracy})
            if accuracy > best_accuracy:
                wandb.run.summary["best_accuracy"] = accuracy
                best_accuracy = accuracy

        return c_loss_list, accuracy_list

    def visualize_logs(self, g_logs, d_logs):
        epochs_list = range(len(g_logs))
        plt.plot(epochs_list, d_logs, label=f"dc_logs")
        plt.plot(epochs_list, g_logs, label=f"gc_logs")
        plt.legend()
        plt.savefig(f"figures/loss_qSGAN_c.png", dpi=500)
        plt.show()
