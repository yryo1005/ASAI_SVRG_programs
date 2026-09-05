"""
No Full Grad SVRG（NFG SVRG）原論文の記述に忠実な実装．

NFG SVRG原論文（Medyakov, Molodtsov, Chezhegov, Rebrikov, Beznosikov,
"Variance Reduction Methods Do Not Need to Compute Full Gradients: Improved Efficiency
through Shuffling", 2025, `references/No_Full_Grad_SVRG.pdf`）のAlgorithm 1に対応する．
`nfg_svrg.py` の `NFGSVRG` との違いは，次の外部ループのスナップショット点として内部ループの
最終パラメータ ω_{s+1} = x_s^n をそのまま採用する点のみである．
"""

from typing import List, Sequence

import numpy as np
import torch


class NFGSVRGFinalPoint(torch.optim.Optimizer):
    """
    No Full Grad SVRG（NFG SVRG）原論文
    （Medyakov, Molodtsov, Chezhegov, Rebrikov, Beznosikov,
    "Variance Reduction Methods Do Not Need to Compute Full Gradients: Improved Efficiency
    through Shuffling", 2025, `references/No_Full_Grad_SVRG.pdf`）のAlgorithm 1に対応する実装．

    `nfg_svrg.py` の `NFGSVRG` クラスは，ASAI SVRG論文が理論解析上の都合（Algorithm 4）により
    採用する「次エポックのスナップショット点 z_{s+1} を内部ループのパラメータ列から一様
    ランダムに選ぶ」実装であるのに対し，本クラスはNFG SVRG原論文のAlgorithm 1（11〜12行目，
    ω_{s+1} = x_s^n）に忠実に，内部ループ終了時点の最終パラメータをそのまま次エポックの
    スナップショット点として採用する．平均勾配 g_s の計算方法（原論文の式(6)，ASAI SVRG論文の
    式(8)に相当）は `NFGSVRG` と同一である．
    """

    def __init__(self, params, lr: float, K: int):
        """
        概要: NFG SVRG（原論文，最終点採用版）を初期化する．初期スナップショット勾配は g_0 = 0
            とする．
        引数:
            params (iterable of torch.nn.Parameter)．更新対象のパラメータ．
            lr (float)．学習率 η（原論文の γ）．
            K (int)．内部ループ長（原論文の n）．
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
                state["running_avg_grad"] = torch.zeros_like(p)
                state["snapshot_params"] = p.detach().clone()

    def begin_epoch(self, rng: np.random.Generator = None):
        """
        概要: 外部ループ（エポック）の開始時に呼び出す．平均勾配の蓄積を ḡ^0 = 0 にリセットする．
        引数: rng (numpy.random.Generator, optional)．本クラスは乱数選択を行わないため
            使用しないが，`NFGSVRG`／`SVRG` との呼び出しインターフェースを揃えるために受け取る．
        戻り値: なし
        """
        self._step_count = 0
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["running_avg_grad"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, grad_at_snapshot: Sequence[torch.Tensor], closure=None):
        """
        概要: 内部ループの1ステップを実行する．平均勾配 ḡ を逐次更新しつつ，
            補正勾配 v_s^t = ∇f_{π_s^t}(x_s^t) - ∇f_{π_s^t}(ω_s) + v_s を計算し，
            パラメータを更新する（原論文Algorithm 1の7〜9行目）．
        引数:
            grad_at_snapshot (Sequence[torch.Tensor])．スナップショット ω_s における
                同一データに対する確率的勾配．各パラメータの `.grad` には，現在のパラメータ
                x_s^t における確率的勾配があらかじめ設定されている必要がある．
            closure (callable, optional)．本実装では使用しない．
        戻り値: なし
        """
        k = self._step_count
        idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]

                # 原論文(6)式: ṽ_s^{t+1} = t/(t+1) ṽ_s^t + 1/(t+1) ∇f_{π_s^t}(x_s^t)
                state["running_avg_grad"].mul_(k / (k + 1)).add_(p.grad, alpha=1.0 / (k + 1))

                g_snap = grad_at_snapshot[idx]
                g_s = state["snapshot_gradient"]
                v = p.grad - g_snap + g_s
                p.add_(v, alpha=-lr)
                idx += 1

        self._step_count += 1

    def end_epoch(self):
        """
        概要: 内部ループ終了後に呼び出す．蓄積した平均勾配 ṽ_s^n を次エポックのスナップショット
            勾配 v_{s+1} として確定する（原論文Algorithm 1の14行目）．また，現在のパラメータ
            （内部ループの最終点 x_s^n）を次エポックのスナップショット ω_{s+1} として確定する
            （原論文Algorithm 1の11〜12行目）．
        引数: なし
        戻り値: なし
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["snapshot_gradient"] = state["running_avg_grad"].clone()
                state["snapshot_params"] = p.detach().clone()

    def get_snapshot_params(self) -> List[torch.Tensor]:
        """
        概要: 次エポックのスナップショット点 ω_{s+1}（内部ループの最終パラメータ）を取得する．
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
        概要: 現在のスナップショット勾配 v_s を取得する．
        引数: なし
        戻り値: grads (list of torch.Tensor)．
        """
        return [
            self.state[p]["snapshot_gradient"].clone()
            for group in self.param_groups
            for p in group["params"]
        ]
