"""
No Full Grad SVRG（NFG SVRG）の実装．

ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）のAlgorithm 2に対応する．同論文の理論解析
（Algorithm 4）に合わせ，次の外部ループのスナップショット点 z_{s+1} を内部ループのパラメータ列
から一様ランダムに選ぶ．NFG SVRG原論文のAlgorithm 1に忠実な（内部ループの最終パラメータを
採用する）実装は `nfg_svrg_final_point.py` の `NFGSVRGFinalPoint` を参照．
"""

from typing import List, Sequence

import numpy as np
import torch


class NFGSVRG(torch.optim.Optimizer):
    """
    No Full Grad SVRG（NFG SVRG）．論文Algorithm 2に対応する．
    フル勾配の計算を回避し，前回の内部ループで得た確率的勾配の平均（式(8)）でスナップショット
    勾配 g_s を近似する．次エポックのスナップショット点 z_{s+1} は，SVRGと同様に内部ループの
    パラメータ列から一様ランダムに選ぶ．
    """

    def __init__(self, params, lr: float, K: int):
        """
        概要: NFG SVRGを初期化する．初期スナップショット勾配は g_0 = 0 とする．
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
                state["running_avg_grad"] = torch.zeros_like(p)
                state["candidate_snapshot_params"] = p.detach().clone()

    def begin_epoch(self, rng: np.random.Generator):
        """
        概要: 外部ループ（エポック）の開始時に呼び出す．平均勾配の蓄積を ḡ^0 = 0 にリセットし，
            次エポックのスナップショット点として採用する内部ループのステップ番号を
            一様ランダムに1つ選ぶ．
        引数: rng (numpy.random.Generator)．乱数生成器．
        戻り値: なし
        """
        self._step_count = 0
        self._target_k = int(rng.integers(0, self.K))
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["running_avg_grad"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, grad_at_snapshot: Sequence[torch.Tensor], closure=None):
        """
        概要: 内部ループの1ステップを実行する．平均勾配 ḡ を式(8)により逐次更新しつつ，
            補正勾配 v_s^k = ∇f_{n_s^k}(w_s^k) - ∇f_{n_s^k}(z_s) + g_s を計算し，
            パラメータを更新する．
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
                if k == self._target_k:
                    state["candidate_snapshot_params"] = p.detach().clone()

                # 式(8): ḡ^{k+1} = k/(k+1) ḡ^k + 1/(k+1) ∇f_{n_s^k}(w_s^k)
                state["running_avg_grad"].mul_(k / (k + 1)).add_(p.grad, alpha=1.0 / (k + 1))

                g_snap = grad_at_snapshot[idx]
                g_s = state["snapshot_gradient"]
                v = p.grad - g_snap + g_s
                p.add_(v, alpha=-lr)
                idx += 1

        self._step_count += 1

    def end_epoch(self):
        """
        概要: 内部ループ終了後に呼び出す．蓄積した平均勾配 ḡ^K を次エポックのスナップショット
            勾配 g_{s+1} として確定する．
        引数: なし
        戻り値: なし
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["snapshot_gradient"] = state["running_avg_grad"].clone()

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
