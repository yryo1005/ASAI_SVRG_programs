"""
確率的勾配降下法（Stochastic Gradient Descent，SGD）の実装．

ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）の式(3)・式(4)に対応する．
`torch.optim.SGD` は使用せず，更新則を陽に実装する．
"""

import torch


class SGD(torch.optim.Optimizer):
    """
    確率的勾配降下法（Stochastic Gradient Descent，SGD）．
    `torch.optim.SGD` は使用せず，(3)/(4)式の更新則
        w_{k+1} = w_k - eta * grad
    を陽に実装したもの．
    """

    def __init__(self, params, lr: float):
        """
        概要: SGDを初期化する．
        引数:
            params (iterable of torch.nn.Parameter)．更新対象のパラメータ．
            lr (float)．学習率 η．
        戻り値: なし
        """
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        概要: 各パラメータの `.grad` に格納された確率的勾配を用いてパラメータを1回更新する．
        引数: closure (callable, optional)．`torch.optim.Optimizer` の慣例に合わせた引数．
            本実装では使用しない．
        戻り値: なし
        """
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=-lr)
