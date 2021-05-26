import torch

# Adamの実装
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr  # 学習率
        self.beta1 = beta1  # mの減衰率
        self.beta2 = beta2  # vの減衰率
        self.iter = 0  # 試行回数を初期化
        self.m = None  # モーメンタム
        self.v = None  # 適合的な学習係数

    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = torch.zeros_like(val)
                self.v[key] = torch.zeros_like(val)

        # パラメータごとに値を更新
        self.iter += 1  # 更新回数をカウント
        lr_t = self.lr * torch.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            params[key] -= lr_t * self.m[key] / (torch.sqrt(self.v[key]) + 1e-7)
