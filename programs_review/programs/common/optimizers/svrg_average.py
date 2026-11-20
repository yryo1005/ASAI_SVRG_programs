"""
SVRG-average：スナップショットを内側ループのパラメータの平均点とし，スナップショット勾配を
全勾配とするSVRG．

ASA-SVRG（`ASAISVRG`）とはスナップショット勾配 $ {\\mathbf g}_{s} $ の定義だけが異なる．
ASA-SVRGは $ {\\mathbf g}_{s} $ を直前の内側ループの確率的勾配の平均とし，本クラスは
新しいスナップショットにおける全勾配 $ \\nabla f({\\mathbf z}_{s}) $ とする．
両者を比べることで，近似誤差（全勾配を平均勾配で近似すること）の効果を介入的に評価する．
スナップショット $ {\\mathbf z}_{s} $ は，`ASAISVRG` と同じく，内側ループの更新前のパラメータ
$ {\\mathbf w}_{s-1}^{0},\\dots,{\\mathbf w}_{s-1}^{K-1} $ の平均である（式(13)の逐次更新）．
全勾配は `SVRG` と同じく外部で計算し，`set_snapshot_gradient()` で設定する．
"""

from typing import List, Sequence

import numpy as np
import torch


class SVRGAverage(torch.optim.Optimizer):
    """
    スナップショットを平均点，スナップショット勾配を全勾配とするSVRG．
    インターフェースは `SVRG` と同じである．
    """

    def __init__(self, params, lr: float, K: int):
        """
        概要: 初期化する．初期スナップショットは $ {\\mathbf z}_{0} = {\\mathbf w}_{0} $ とする．
            $ {\\mathbf g}_{0} $ は外部で全勾配を計算し，`set_snapshot_gradient()` で設定する．
        引数:
            params (iterable of torch.nn.Parameter)．更新対象のパラメータ．
            lr (float)．学習率 $ \\eta $．
            K (int)．内側ループの長さ．
        戻り値: なし
        """
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.K = K
        self._step_count = 0

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["snapshot_gradient"] = torch.zeros_like(p)
                state["snapshot_params"] = p.detach().clone()
                state["running_avg_params"] = torch.zeros_like(p)

    def set_snapshot_gradient(self, snapshot_gradients: Sequence[torch.Tensor]):
        """
        概要: スナップショット勾配 $ {\\mathbf g}_{s} $（全勾配）を設定する．
        引数: snapshot_gradients (Sequence[torch.Tensor])．パラメータと同じ形状のテンソル列．
        戻り値: なし
        """
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["snapshot_gradient"] = snapshot_gradients[idx].clone()
                idx += 1

    def begin_epoch(self, rng: np.random.Generator = None):
        """
        概要: 外側ループの開始時に呼び出す．平均パラメータの蓄積を0にリセットする．
        引数: rng (numpy.random.Generator, optional)．乱数は使わないが，他のクラスと
            呼び出し方を揃えるために受け取る．
        戻り値: なし
        """
        self._step_count = 0
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["running_avg_params"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, grad_at_snapshot: Sequence[torch.Tensor], closure=None):
        """
        概要: 内側ループの1ステップを実行する．更新前のパラメータを平均に加えたのち，
            $ {\\mathbf v} = \\nabla f_{n}({\\mathbf w}) - \\nabla f_{n}({\\mathbf z}_{s}) + {\\mathbf g}_{s} $
            でパラメータを更新する．
        引数:
            grad_at_snapshot (Sequence[torch.Tensor])．スナップショットにおける同一ミニバッチの
                確率的勾配．各パラメータの `.grad` には現在のパラメータにおける確率的勾配が
                設定されている必要がある．
            closure (callable, optional)．使用しない．
        戻り値: なし
        """
        k = self._step_count
        idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                # 式(13)：更新前のパラメータを平均に加える（ASAISVRG と同じ演算）．
                state["running_avg_params"].mul_(k / (k + 1)).add_(p, alpha=1.0 / (k + 1))
                v = p.grad - grad_at_snapshot[idx] + state["snapshot_gradient"]
                p.add_(v, alpha=-lr)
                idx += 1
        self._step_count += 1

    def end_epoch(self):
        """
        概要: 内側ループの終了後に呼び出し，平均パラメータを次のスナップショットとして確定する．
            スナップショット勾配は外部で全勾配を計算して設定する．
        引数: なし
        戻り値: なし
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["snapshot_params"] = state["running_avg_params"].clone()

    def get_snapshot_params(self) -> List[torch.Tensor]:
        """
        概要: スナップショット $ {\\mathbf z}_{s} $ を取得する．
        引数: なし
        戻り値: params (list of torch.Tensor)．
        """
        return [self.state[p]["snapshot_params"].clone() for group in self.param_groups for p in group["params"]]

    def get_snapshot_gradient(self) -> List[torch.Tensor]:
        """
        概要: スナップショット勾配 $ {\\mathbf g}_{s} $ を取得する．
        引数: なし
        戻り値: grads (list of torch.Tensor)．
        """
        return [self.state[p]["snapshot_gradient"].clone() for group in self.param_groups for p in group["params"]]
