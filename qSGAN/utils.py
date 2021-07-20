import matplotlib.pyplot as plt
import torch


def calculate_grad(output_plus, output_minus):
    out = torch.mean(-torch.log(output_plus) + torch.log(output_minus), dim=1) / 2
    return out.view(out.size()[0])


def transform_to_N_px(x_list, N):
    out_list = []
    for x in x_list:
        out = list(format(x, f"0{N}b"))
        out = list(map(int, out))
        out_list.append(out)
    return torch.tensor(out_list, dtype=torch.float)


def transform_to_N_px_2D(x_list, N):
    out_list = []
    for xs in x_list:
        x_2D = []
        for x in xs:
            out = list(format(x, f"0{N}b"))
            out = list(map(int, out))
            x_2D.append(out)
        out_list.append(x_2D)
    return torch.tensor(out_list, dtype=torch.float)


def visualize_accuracy(num_epochs, dc_accuracy, dq_accuracy):
    epochs_list = range(num_epochs)
    plt.plot(epochs_list, dc_accuracy, label="dc_accuracy")
    plt.plot(epochs_list, dq_accuracy, label="dq_accuracy")
    plt.legend()
    plt.savefig("figures/accuracy_qSGAN.png", dpi=500)
    plt.show()
