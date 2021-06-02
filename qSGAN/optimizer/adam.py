import torch

# Adamの実装
class Adam:
    def __init__(self, lr=0.005, beta1=0.9, beta2=0.999):
        self.lr = lr  # 学習率
        self.beta1 = beta1  # mの減衰率
        self.beta2 = beta2  # vの減衰率
        self.iter = 0  # 試行回数を初期化
        self.m = None  # モーメンタム
        self.v = None  # 適合的な学習係数

    def update(self, params, grads):
        device = grads.device
        if self.m is None:
            self.m = torch.zeros_like(params).to(device)
            self.v = torch.zeros_like(params).to(device)

        # パラメータごとに値を更新
        self.iter += 1  # 更新回数をカウント
        lr_t = self.lr * torch.sqrt(torch.tensor(1.0 - self.beta2 ** self.iter)) / (1.0 - self.beta1 ** self.iter)
        lr_t = lr_t.to(device)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        params -= lr_t * self.m / (torch.sqrt(self.v) + 1e-7)
        return params
