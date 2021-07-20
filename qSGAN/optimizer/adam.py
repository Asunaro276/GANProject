import torch

# Adamの実装
class Adam:
    def __init__(self, lr=0.001, betas=(0.9, 0.999)):
        self.lr = lr  # 学習率
        self.betas = betas  # mの減衰率
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
        lr_t = self.lr * torch.sqrt(torch.tensor(1.0 - self.betas[1] ** self.iter)) / (1.0 - self.betas[0] ** self.iter)
        lr_t = lr_t.to(device)
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * grads
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (grads ** 2)
        params -= lr_t * self.m / (torch.sqrt(self.v) + 1e-7)
        return params
