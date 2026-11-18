"""
SVRG（Stochastic Variance Reduced Gradient）の実装．

ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）のAlgorithm 1に対応する．同論文の理論解析
（Algorithm 4）に合わせ，次の外部ループのスナップショット点 z_{s+1} を内部ループのパラメータ列
{w_s^k}_{k=0}^{K-1} から一様ランダムに選ぶ．内部ループの最終パラメータを採用する実装は
`svrg_final_point.py` の `SVRGFinalPoint` を参照．
"""

from typing import List, Sequence

import numpy as np
import torch


class SVRG(torch.optim.Optimizer):
    """
    SVRG（Stochastic Variance Reduced Gradient）．論文Algorithm 1に対応する．
    スナップショット勾配 g_s はフル勾配（データセット全体）であり，各外部ループの開始時に
    `set_snapshot_gradient()` で外部から設定する．次エポックのスナップショット点 z_{s+1} は，
    内部ループのパラメータ列 {w_s^k}_{k=0}^{K-1} から一様ランダムに選ぶ．
    """

    def __init__(self, params, lr: float, K: int):
        """
        概要: SVRGを初期化する．
        引数:
            params (iterable of torch.nn.Parameter)．更新対象のパラメータ．
            lr (float)．学習率 η．
            K (int)．内部ループ長．
        戻り値: なし
        """
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.K = K
        self._step_count = 0
        self._target_k = 0

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["snapshot_gradient"] = torch.zeros_like(p)
                state["candidate_snapshot_params"] = p.detach().clone()

    def set_snapshot_gradient(self, snapshot_gradients: Sequence[torch.Tensor]):
        """
        概要: スナップショット勾配 g_s を設定する．SVRGでは外部でフル勾配を計算して渡す．
        引数: snapshot_gradients (Sequence[torch.Tensor])．パラメータと同じ形状のテンソル列．
        戻り値: なし
        """
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["snapshot_gradient"] = snapshot_gradients[idx].clone()
                idx += 1

    def begin_epoch(self, rng: np.random.Generator):
        """
        概要: 外部ループ（エポック）の開始時に呼び出す．次エポックのスナップショット点として
            採用する内部ループのステップ番号を一様ランダムに1つ選ぶ．
        引数: rng (numpy.random.Generator)．乱数生成器．
        戻り値: なし
        """
        self._step_count = 0
        self._target_k = int(rng.integers(0, self.K))

    @torch.no_grad()
    def step(self, grad_at_snapshot: Sequence[torch.Tensor], closure=None):
        """
        概要: 内部ループの1ステップを実行する．補正勾配
            v_s^k = ∇f_{n_s^k}(w_s^k) - ∇f_{n_s^k}(z_s) + g_s
            を計算し，w_s^{k+1} = w_s^k - eta * v_s^k によりパラメータを更新する．
            選ばれたステップ番号に到達した場合，更新前のパラメータ値を候補スナップショットとして保持する．
        引数:
            grad_at_snapshot (Sequence[torch.Tensor])．スナップショット z_s における
                同一データに対する確率的勾配 ∇f_{n_s^k}(z_s)．各パラメータの `.grad` には，
                現在のパラメータ w_s^k における確率的勾配 ∇f_{n_s^k}(w_s^k) が
                あらかじめ設定されている必要がある．
            closure (callable, optional)．本実装では使用しない．
        戻り値: なし
        """
        idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                if self._step_count == self._target_k:
                    state["candidate_snapshot_params"] = p.detach().clone()

                g_snap = grad_at_snapshot[idx]
                g_s = state["snapshot_gradient"]
                v = p.grad - g_snap + g_s
                p.add_(v, alpha=-lr)
                idx += 1

        self._step_count += 1

    def end_epoch(self):
        """
        概要: 内部ループ終了後に呼び出す．SVRGではスナップショット勾配は次エポックの開始時に
            `set_snapshot_gradient()` で改めて（フル勾配として）設定されるため，本メソッドは
            インターフェースの統一のために提供するのみで，状態の変更は行わない．
        引数: なし
        戻り値: なし
        """
        pass

    def get_snapshot_params(self) -> List[torch.Tensor]:
        """
        概要: 次エポックのスナップショット点 z_{s+1}（一様ランダムに選ばれたパラメータ値）を取得する．
        引数: なし
        戻り値: params (list of torch.Tensor)．
        """
        return [
            self.state[p]["candidate_snapshot_params"].clone()
            for group in self.param_groups
            for p in group["params"]
        ]

    def get_snapshot_gradient(self) -> List[torch.Tensor]:
        """
        概要: 現在のスナップショット勾配 g_s を取得する．
        引数: なし
        戻り値: grads (list of torch.Tensor)．
        """
        return [
            self.state[p]["snapshot_gradient"].clone()
            for group in self.param_groups
            for p in group["params"]
        ]
