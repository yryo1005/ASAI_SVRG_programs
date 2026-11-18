"""
Averaged Snapshot Approximate Incremental SVRG（ASAI SVRG，提案手法）の実装．

ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）のAlgorithm 3に対応する．NFG SVRGと同様に
フル勾配の計算を回避しつつ，内部ループで得られたパラメータ列の平均（式(13)）を次の外部ループの
スナップショット点 z_{s+1} として用いる点が特徴である．
"""

from typing import List, Sequence

import numpy as np
import torch


class ASAISVRG(torch.optim.Optimizer):
    """
    Averaged Snapshot Approximate Incremental SVRG（ASAI SVRG，提案手法）．
    論文Algorithm 3に対応する．NFG SVRGと同様にフル勾配を回避しつつ，内部ループで得られた
    パラメータ列自体の平均（式(13)）を次エポックのスナップショット点 z_{s+1} として用いる．
    """

    def __init__(self, params, lr: float, K: int):
        """
        概要: ASAI SVRGを初期化する．初期スナップショットは z_0 = w_0，g_0 = 0 とする．
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

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["snapshot_gradient"] = torch.zeros_like(p)
                state["snapshot_params"] = p.detach().clone()
                state["running_avg_grad"] = torch.zeros_like(p)
                state["running_avg_params"] = torch.zeros_like(p)

    def begin_epoch(self, rng: np.random.Generator = None):
        """
        概要: 外部ループ（エポック）の開始時に呼び出す．平均勾配 ḡ および平均パラメータ z̄ の
            蓄積を0にリセットする（論文Algorithm 3の z̄^0 ← 0, ḡ^0 ← 0）．
        引数: rng (numpy.random.Generator, optional)．ASAI SVRGでは乱数選択を行わないため
            使用しないが，SVRG・NFG SVRGとの呼び出しインターフェースを揃えるために受け取る．
        戻り値: なし
        """
        self._step_count = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["running_avg_grad"] = torch.zeros_like(p)
                state["running_avg_params"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, grad_at_snapshot: Sequence[torch.Tensor], closure=None):
        """
        概要: 内部ループの1ステップを実行する．平均パラメータ z̄（式(13)）および平均勾配 ḡ
            （式(14)）を逐次更新しつつ，補正勾配
            v_s^k = ∇f_{n_s^k}(w_s^k) - ∇f_{n_s^k}(z_s) + g_s
            を計算し，パラメータを更新する．
        引数:
            grad_at_snapshot (Sequence[torch.Tensor])．スナップショット z_s における
                同一データに対する確率的勾配．各パラメータの `.grad` には，現在のパラメータ
                w_s^k における確率的勾配があらかじめ設定されている必要がある．
            closure (callable, optional)．本実装では使用しない．
        戻り値: なし
        """
        k = self._step_count
        idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]

                # 式(13): z̄^{k+1} = k/(k+1) z̄^k + 1/(k+1) w_s^k （更新前のパラメータ値を使用）
                state["running_avg_params"].mul_(k / (k + 1)).add_(p, alpha=1.0 / (k + 1))
                # 式(14): ḡ^{k+1} = k/(k+1) ḡ^k + 1/(k+1) ∇f_{n_s^k}(w_s^k)
                state["running_avg_grad"].mul_(k / (k + 1)).add_(p.grad, alpha=1.0 / (k + 1))

                g_snap = grad_at_snapshot[idx]
                g_s = state["snapshot_gradient"]
                v = p.grad - g_snap + g_s
                p.add_(v, alpha=-lr)
                idx += 1

        self._step_count += 1

    def end_epoch(self):
        """
        概要: 内部ループ終了後に呼び出す．蓄積した平均パラメータ z̄^K および平均勾配 ḡ^K を，
            次エポックのスナップショット z_{s+1}，g_{s+1} として確定する．
        引数: なし
        戻り値: なし
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["snapshot_params"] = state["running_avg_params"].clone()
                state["snapshot_gradient"] = state["running_avg_grad"].clone()

    def get_snapshot_params(self) -> List[torch.Tensor]:
        """
        概要: 次エポックのスナップショット点 z_{s+1}（内部ループにおける平均パラメータ）を取得する．
        引数: なし
        戻り値: params (list of torch.Tensor)．
        """
        return [
            self.state[p]["snapshot_params"].clone()
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
