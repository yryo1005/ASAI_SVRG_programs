"""
古典的SVRG（内部ループの最終パラメータをスナップショットとする版）の実装．

NFG SVRG原論文（Medyakov, Molodtsov, Chezhegov, Rebrikov, Beznosikov,
"Variance Reduction Methods Do Not Need to Compute Full Gradients: Improved Efficiency
through Shuffling", 2025, `references/No_Full_Grad_SVRG.pdf`）が，Algorithm 1（No Full Grad
SVRG）とペアで比較対象とする古典的SVRGに対応する．`nfg_svrg_final_point.py` の
`NFGSVRGFinalPoint` との唯一の違いは，スナップショット勾配 g_s がフル勾配であり，外部で計算した
上で `set_snapshot_gradient()` により明示的に設定する点である．
"""

from typing import List, Sequence

import numpy as np
import torch


class SVRGFinalPoint(torch.optim.Optimizer):
    """
    SVRG（Stochastic Variance Reduced Gradient）．NFG SVRG原論文が比較対象とする古典的な
    SVRGに対応する実装．

    `svrg.py` の `SVRG` クラスは，ASAI SVRG論文が理論解析上の都合（Algorithm 4）により
    採用する「次エポックのスナップショット点 z_{s+1} を内部ループのパラメータ列から一様
    ランダムに選ぶ」実装であるのに対し，本クラスは `NFGSVRGFinalPoint` と同様に内部ループ
    終了時点の最終パラメータをそのまま次エポックのスナップショット点として採用する．
    NFG SVRGとの唯一の違いは，スナップショット勾配 g_s がフル勾配（データセット全体）であり，
    外部で計算した上で `set_snapshot_gradient()` により明示的に設定する点である．
    """

    def __init__(self, params, lr: float, K: int):
        """
        概要: SVRG（最終点採用版）を初期化する．
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

    def begin_epoch(self, rng: np.random.Generator = None):
        """
        概要: 外部ループ（エポック）の開始時に呼び出す．
        引数: rng (numpy.random.Generator, optional)．本クラスは乱数選択を行わないため
            使用しないが，他クラスとの呼び出しインターフェースを揃えるために受け取る．
        戻り値: なし
        """
        self._step_count = 0

    @torch.no_grad()
    def step(self, grad_at_snapshot: Sequence[torch.Tensor], closure=None):
        """
        概要: 内部ループの1ステップを実行する．補正勾配
            v_s^k = ∇f_{n_s^k}(w_s^k) - ∇f_{n_s^k}(z_s) + g_s を計算し，パラメータを更新する．
        引数:
            grad_at_snapshot (Sequence[torch.Tensor])．スナップショット z_s における
                同一データに対する確率的勾配．各パラメータの `.grad` には，現在のパラメータ
                w_s^k における確率的勾配があらかじめ設定されている必要がある．
            closure (callable, optional)．本実装では使用しない．
        戻り値: なし
        """
        idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                g_snap = grad_at_snapshot[idx]
                g_s = state["snapshot_gradient"]
                v = p.grad - g_snap + g_s
                p.add_(v, alpha=-lr)
                idx += 1

        self._step_count += 1

    def end_epoch(self):
        """
        概要: 内部ループ終了後に呼び出す．現在のパラメータ（内部ループの最終点）を次エポックの
            スナップショット z_{s+1} として確定する．スナップショット勾配は，次エポックの
            開始時に外部から `set_snapshot_gradient()` で改めて設定される．
        引数: なし
        戻り値: なし
        """
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["snapshot_params"] = p.detach().clone()

    def get_snapshot_params(self) -> List[torch.Tensor]:
        """
        概要: 次エポックのスナップショット点 z_{s+1}（内部ループの最終パラメータ）を取得する．
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
