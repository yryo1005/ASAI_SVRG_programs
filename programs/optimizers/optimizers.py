"""
論文（ASAI SVRG）のAlgorithm 1〜4に対応する，再利用可能な最適化手法クラス群．

`.orders/order_002.md` の指示に基づき，SGD，SVRG，NFG SVRG，ASAI SVRGの4手法を，
`torch.optim.Optimizer` のサブクラスとしてそれぞれ独立に陽実装する．SVRG系3手法（SVRG，
NFG SVRG，ASAI SVRG）は内部ループの更新則やスナップショット構成に共通する部分があるが，
指示に従い共通の基底クラスへ抽出せず，各クラスが自身のアルゴリズムを完結して実装する．

いずれのクラスも `torch.optim.SGD` 等，PyTorch公式の最適化手法実装は使用しない．勾配は
各パラメータの `.grad` 属性から読み取る（`.grad` は自動微分ではなく，`model.py` の解析的勾配
計算関数によって外部で計算・代入される）．これは通常の `torch.optim.Optimizer` の利用規約と
同じインターフェースである．

SVRG系手法は，論文のAlgorithm 1〜3が示す外部ループ・内部ループ構造を持つため，通常の
`step()` だけでは1エポック分の処理を表現できない．そのため，各クラスは `step()` に加えて，
外部ループの境界で呼び出す `begin_epoch()`／`end_epoch()` を提供する．
- `begin_epoch()`: 内部ループ開始前に呼び出す．エポック内で用いる平均勾配・平均パラメータの
  蓄積状態をリセットする．SVRG・NFG SVRGでは，次のスナップショットとして採用する内部ループの
  ステップ番号 `k` をこの時点で一様ランダムに1つ選ぶ（論文Algorithm 4の「ランダム選択」に対応）．
- `step()`: 内部ループの1ステップ分の更新（補正勾配の計算とパラメータ更新）を行う．
  SVRG・NFG SVRGでは，選ばれたステップ番号に到達した時点のパラメータ値を保持する．
  NFG SVRG・ASAI SVRGでは，このタイミングで平均勾配（および平均パラメータ）を逐次更新する．
- `end_epoch()`: 内部ループ終了後に呼び出す．次エポックのスナップショット勾配 `g_{s+1}`
  （NFG SVRG・ASAI SVRGでは蓄積した平均勾配，SVRGでは変更なし）を確定する．
- `get_snapshot_params()` / `get_snapshot_gradient()`: 次エポックのスナップショット
  `z_{s+1}`，`g_{s+1}` を取得する．

SVRGのみ，スナップショット勾配 `g_s` がフル勾配（データセット全体を用いた計算）であるため，
外部（`train.py`）でフル勾配を計算した上で `set_snapshot_gradient()` により明示的に設定する．
"""

from typing import List, Sequence

import numpy as np
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


class NFGSVRGFinalPoint(torch.optim.Optimizer):
    """
    No Full Grad SVRG（NFG SVRG）原論文
    （Medyakov, Molodtsov, Chezhegov, Rebrikov, Beznosikov,
    "Variance Reduction Methods Do Not Need to Compute Full Gradients: Improved Efficiency
    through Shuffling", 2025, `references/No_Full_Grad_SVRG.pdf`）のAlgorithm 1に対応する実装．
    `.orders/order_006.md` の指示に基づき，原論文の実験結果を再現できるかを検証するために追加した．

    本モジュールの `NFGSVRG` クラスは，ASAI SVRG論文が理論解析上の都合（Algorithm 4）により
    採用する「次エポックのスナップショット点 z_{s+1} を内部ループのパラメータ列から一様
    ランダムに選ぶ」実装であるのに対し，本クラスはNFG SVRG原論文のAlgorithm 1（11〜12行目，
    ω_{s+1} = x_s^n）に忠実に，内部ループ終了時点の最終パラメータをそのまま次エポックの
    スナップショット点として採用する．平均勾配 g_s の計算方法（原論文の式(7)，本稿の式(8)に
    相当）は `NFGSVRG` と同一である．
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

                # 原論文(7)式: ṽ_s^{t+1} = t/(t+1) ṽ_s^t + 1/(t+1) ∇f_{π_s^t}(x_s^t)
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


class SVRGFinalPoint(torch.optim.Optimizer):
    """
    SVRG（Stochastic Variance Reduced Gradient）．NFG SVRG原論文
    （Medyakov, Molodtsov, Chezhegov, Rebrikov, Beznosikov,
    "Variance Reduction Methods Do Not Need to Compute Full Gradients: Improved Efficiency
    through Shuffling", 2025, `references/No_Full_Grad_SVRG.pdf`）が比較対象とする古典的な
    SVRGに対応する実装．`.orders/order_007.md`（Ex003）の指示に基づき，原論文の実験結果を
    再現できるかを検証するために追加した．

    本モジュールの `SVRG` クラスは，ASAI SVRG論文が理論解析上の都合（Algorithm 4）により
    採用する「次エポックのスナップショット点 z_{s+1} を内部ループのパラメータ列から一様
    ランダムに選ぶ」実装であるのに対し，本クラスは，NFG SVRG原論文がAlgorithm 1（No Full
    Grad SVRG）とペアで比較対象とする古典的SVRGの実装に合わせ，`NFGSVRGFinalPoint` と同様に
    内部ループ終了時点の最終パラメータをそのまま次エポックのスナップショット点として採用する．
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
