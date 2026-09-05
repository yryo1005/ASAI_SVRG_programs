"""
`references/No_Full_Grad_SVRG.pdf` 付録A.1（LEAST SQUARES REGRESSION）の式(8)に対応する，
非線形最小二乗損失モデルの定義．

式(8): f(x) = (1/n) Σ_i (y_i - h_i)^2，h_i = 1/(1+exp(-z_i))，z_i = A_i・x

線形結合 z_i にシグモイド関数を適用した出力 h_i を，二乗誤差でラベル y_i に近づける非線形最小二乗
回帰である．原論文の式(8)にバイアス項・正則化項は現れないため，本モジュールも切片なしの線形結合
（`nn.Linear(..., bias=False)`）とし，正則化を行わない．
"""

import os
import sys

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed


class LeastSquaresSigmoidModel(nn.Module):
    """
    非線形最小二乗回帰モデル．式(8)における z_i = A_i・x（切片なし）に対応する．
    """

    def __init__(self, input_dim: int = 123):
        """
        概要: 線形層（重み x，切片なし）を初期化する．
        引数: input_dim (int) = 123．入力特徴量の次元数 d．
        戻り値: なし
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False, dtype=torch.float64)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        概要: 入力特徴量から線形結合 z（シグモイド関数適用前の値）を計算する．
        引数: A (torch.Tensor)，形状 (B, d)．入力特徴量．
        戻り値: z (torch.Tensor)，形状 (B,)．
        """
        return self.linear(A).squeeze(-1)


def load_model(
    ModelClass=LeastSquaresSigmoidModel, weight_path: str = None, seed: int = 0
) -> nn.Module:
    """
    概要: モデルをインスタンス化するための関数．
    引数:
        ModelClass (torch.nn.Moduleのクラス) = LeastSquaresSigmoidModel．
        weight_path (str) = None．学習済み重みのパス．指定した場合はこれを読み込む．
        seed (int) = 0．パラメータ初期値を固定する乱数シード．
    戻り値: model (torch.nn.Module)．
    """
    set_seed(seed)
    model = ModelClass()
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model


def _compute_batch_loss(model: nn.Module, A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    概要: 非線形最小二乗損失 f(x)（式(8)）を計算する（勾配計算グラフを構築したまま返す）．
    引数:
        model (torch.nn.Module)．`LeastSquaresSigmoidModel` のインスタンス．
        A (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号．
    戻り値: loss (torch.Tensor)，形状 ()．勾配計算グラフ付きの損失．
    """
    z = model(A)
    h = torch.sigmoid(z)
    return torch.mean((y - h) ** 2)


def compute_gradient(model: nn.Module, A: torch.Tensor, y: torch.Tensor) -> list:
    """
    概要: PyTorchの自動微分（`loss.backward()`）により，非線形最小二乗損失（式(8)）の勾配を
        計算する．バッチサイズ B に対する平均勾配 (1/B)Σ∇(y_i-h_i)^2 を返す．B=1のとき，
        論文における単一サンプルの確率的勾配 ∇f_n(x) と一致する．
    引数:
        model (torch.nn.Module)．勾配を評価するパラメータを保持するモデル．
        A (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号．
    戻り値: grads (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の勾配テンソル列．
    """
    model.zero_grad(set_to_none=True)
    loss = _compute_batch_loss(model, A, y)
    loss.backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def compute_loss(model: nn.Module, A: torch.Tensor, y: torch.Tensor) -> float:
    """
    概要: 非線形最小二乗損失 f(x)（式(8)）を計算する．勾配は計算しない．
    引数:
        model (torch.nn.Module)．評価対象のパラメータを保持するモデル．
        A (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号．
    戻り値: loss (float)．1データあたりの損失の平均値．
    """
    with torch.no_grad():
        loss = _compute_batch_loss(model, A, y)
    return loss.item()


def compute_accuracy(model: nn.Module, A: torch.Tensor, y: torch.Tensor) -> float:
    """
    概要: 分類精度（Accuracy，h_i >= 0.5 を正例予測とみなす）を計算する．原論文の式(8)自体は
        分類精度を評価指標としないが，学習の進行を確認する補助指標として算出する．
    引数:
        model (torch.nn.Module)．評価対象のパラメータを保持するモデル．
        A (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号．
    戻り値: accuracy (float)．1データあたりの正解率の平均値．
    """
    with torch.no_grad():
        z = model(A)
        pred = (torch.sigmoid(z) >= 0.5).to(y.dtype)
        accuracy = (pred == y).to(torch.float64).mean().item()
    return accuracy


def set_model_params(model: nn.Module, param_values: list, source_model: nn.Module = None):
    """
    概要: モデルのパラメータを，指定したテンソル列の値で上書きする．
        スナップショット専用モデルのパラメータを z_{s+1} で更新する際に用いる．
    引数:
        model (torch.nn.Module)．上書き対象のモデル．
        param_values (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の値．
        source_model (torch.nn.Module) = None．`programs/optimizers/optimizers.py` を用いる
            他の実験とインターフェースを揃えるための引数．`LeastSquaresSigmoidModel` は
            Batch Normalization等のバッファを持たないため，指定されても何も行わない．
    戻り値: なし
    """
    with torch.no_grad():
        for p, v in zip(model.parameters(), param_values):
            p.copy_(v)
