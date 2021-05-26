import torch
from torch import nn


def BCE_unsupervised(outputs, targets=None):
    '''
    outputs: 予測結果(ネットワークの出力)
　　　　 targets: 正解
    '''
    # 損失の計算
    loss = - torch.mean(torch.log(outputs))
    return loss

def CE_supervised(outputs, targets=None):
    '''
    outputs: 予測結果(ネットワークの出力)
　　　　 targets: 正解
    '''
    # 損失の計算
    loss = - torch.mean(torch.log(outputs) + torch.log())
    return loss

