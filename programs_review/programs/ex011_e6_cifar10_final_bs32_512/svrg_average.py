"""
SVRG-average（E2で新規に導入する手法）の実装．

E2（`実験計画.md` 4章）の2×3の要因配置で，「スナップショット：平均点」と「$ {\\mathbf g}_{s} $：全勾配」の
組に当たる手法である．

- スナップショット：`ASAISVRG` と同じく，内側ループの更新前のパラメータ
  $ {\\mathbf w}_{s-1}^{0},\\dots,{\\mathbf w}_{s-1}^{K-1} $ の平均とする．逐次更新の式も `ASAISVRG` の
  `running_avg_params` と同じ式(13) $ \\bar{\\mathbf z}^{k+1} = \\frac{k}{k+1}\\bar{\\mathbf z}^{k} + \\frac{1}{k+1}{\\mathbf w}^{k} $ を用いる．
- $ {\\mathbf g}_{s} $：`SVRG` と同じく，新しいスナップショットにおける全勾配 $ \\nabla f({\\mathbf z}_{s}) $ とし，
  学習ループが外部で計算して `set_snapshot_gradient()` で設定する．平均勾配（`running_avg_grad`）は持たない．

インターフェース（`begin_epoch(rng)`，`step(grad_at_snapshot)`，`end_epoch()`，`get_snapshot_params()`，
`get_snapshot_gradient()`，`set_snapshot_gradient()`）は `SVRG` と揃えている．
"""

from typing import List, Sequence

import numpy as np
import torch


class SVRGAverage(torch.optim.Optimizer):
    """
    SVRG-average．スナップショットを内側ループのパラメータの平均（`ASAISVRG` と同じ）とし，
    スナップショット勾配 $ {\\mathbf g}_{s} $ を全勾配（`SVRG` と同じ）とする手法．
    """

    def __init__(self, params, lr: float, K: int):
        """
        概要: SVRG-averageを初期化する．初期スナップショットは $ {\\mathbf z}_{0} = {\\mathbf w}_{0} $ とする．
            $ {\\mathbf g}_{0} = \\nabla f({\\mathbf z}_{0}) $ は学習ループが `set_snapshot_gradient()` で設定する
            （設定前は $ {\\mathbf 0} $）．
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
        引数: snapshot_gradients (Sequence[torch.Tensor])．パラメータと同じ順序・形状のテンソル列．
        戻り値: なし
        """
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["snapshot_gradient"] = snapshot_gradients[idx].clone()
                idx += 1

    def begin_epoch(self, rng: np.random.Generator = None):
        """
        概要: 外側ループの開始時に呼び出す．平均パラメータの蓄積を $ \\bar{\\mathbf z}^{0} = {\\mathbf 0} $ にリセットする．
        引数: rng (numpy.random.Generator, optional)．乱数による選択を行わないため使用しないが，
            `SVRG` との呼び出しのインターフェースを揃えるために受け取る．
        戻り値: なし
        """
        self._step_count = 0
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["running_avg_params"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, grad_at_snapshot: Sequence[torch.Tensor], closure=None):
        """
        概要: 内側ループの1ステップを実行する．平均パラメータを式(13)で逐次更新した後，補正勾配
            $ {\\mathbf v}_{s}^{k} = \\nabla f_{n_{s}^{k}}({\\mathbf w}_{s}^{k}) - \\nabla f_{n_{s}^{k}}({\\mathbf z}_{s}) + {\\mathbf g}_{s} $
            を計算し，$ {\\mathbf w}_{s}^{k+1} = {\\mathbf w}_{s}^{k} - \\eta{\\mathbf v}_{s}^{k} $ で更新する．
        引数:
            grad_at_snapshot (Sequence[torch.Tensor])．スナップショット $ {\\mathbf z}_{s} $ における同じデータに
                対する確率的勾配．各パラメータの `.grad` には，現在のパラメータ $ {\\mathbf w}_{s}^{k} $ に
                おける確率的勾配があらかじめ設定されている必要がある．
            closure (callable, optional)．使用しない．
        戻り値: なし
        """
        k = self._step_count
        idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                # 式(13)：更新前のパラメータ w_s^k を平均に加える（ASAISVRG と同じ式）．
                state["running_avg_params"].mul_(k / (k + 1)).add_(p, alpha=1.0 / (k + 1))

                g_snap = grad_at_snapshot[idx]
                g_s = state["snapshot_gradient"]
                v = p.grad - g_snap + g_s
                p.add_(v, alpha=-lr)
                idx += 1

        self._step_count += 1

    def end_epoch(self):
        """
        概要: 内側ループの終了後に呼び出す．蓄積した平均パラメータを次のスナップショット $ {\\mathbf z}_{s+1} $ として
            確定する．スナップショット勾配は，学習ループが新しいスナップショットにおける全勾配を計算して
            `set_snapshot_gradient()` で設定する．
        引数: なし
        戻り値: なし
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["snapshot_params"] = state["running_avg_params"].clone()

    def get_snapshot_params(self) -> List[torch.Tensor]:
        """
        概要: スナップショット $ {\\mathbf z}_{s+1} $（内側ループの平均パラメータ）を取得する．
        引数: なし
        戻り値: params (list of torch.Tensor)．パラメータと同じ順序・形状．
        """
        return [
            self.state[p]["snapshot_params"].clone()
            for group in self.param_groups
            for p in group["params"]
        ]

    def get_snapshot_gradient(self) -> List[torch.Tensor]:
        """
        概要: 現在のスナップショット勾配 $ {\\mathbf g}_{s} $ を取得する．
        引数: なし
        戻り値: grads (list of torch.Tensor)．パラメータと同じ順序・形状．
        """
        return [
            self.state[p]["snapshot_gradient"].clone()
            for group in self.param_groups
            for p in group["params"]
        ]
