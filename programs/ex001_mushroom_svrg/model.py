"""
マッシュルームデータセットの二値分類に用いる線形モデル（ロジスティック回帰）の定義．

論文の(38)式に対応する，L2正則化付き二値交差エントロピー損失を目的関数とする．
`.orders/order_003.md` の指示に基づき，勾配はPyTorchの標準的な自動微分（`torch.autograd`，
`loss.backward()`）により取得する．SVRG系手法はスナップショット z_s における勾配も必要と
するため，スナップショット専用の `LogisticRegressionModel` インスタンス（パラメータのみが
異なる同一構造のモデル）を別途保持し，そのモデルに対して forward／backward を実行することで
勾配を得る（`train.py` 側で運用する）．
"""

import os
import sys

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed


class LogisticRegressionModel(nn.Module):
    """
    L2正則化付きロジスティック回帰モデル．
    論文における ŷ(w) = σ(w^T x + b) （w: 重みベクトル，b: バイアス項）に対応する．
    """

    def __init__(self, input_dim: int = 22):
        """
        概要: 線形層（重み w とバイアス b）を初期化する．
        引数: input_dim (int) = 22．入力特徴量の次元数 d．
        戻り値: なし
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: 入力特徴量からロジット（シグモイド関数適用前の値）を計算する．
        引数: x (torch.Tensor)，形状 (B, d)．入力特徴量．
        戻り値: logits (torch.Tensor)，形状 (B,)．
        """
        return self.linear(x).squeeze(-1)


def load_model(
    ModelClass=LogisticRegressionModel, weight_path: str = None, seed: int = 0
) -> nn.Module:
    """
    概要: モデルをインスタンス化するための関数．
    引数:
        ModelClass (torch.nn.Moduleのクラス) = LogisticRegressionModel．
        weight_path (str) = None．学習済み重みのパス．指定した場合はこれを読み込む．
        seed (int) = 0．パラメータ初期値を固定する乱数シード．
    戻り値: model (torch.nn.Module)．
    """
    set_seed(seed)
    model = ModelClass()
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model


def _compute_batch_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor, reg_lambda: float) -> torch.Tensor:
    """
    概要: L2正則化付き二値交差エントロピー損失 f(w)（(38)式）を計算する（勾配計算グラフを
        構築したまま返す）．
    引数:
        model (torch.nn.Module)．`LogisticRegressionModel` のインスタンス．
        X (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号．
        reg_lambda (float)．L2正則化係数 λ．
    戻り値: loss (torch.Tensor)，形状 ()．勾配計算グラフ付きの損失．
    """
    logits = model(X)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="mean")
    reg = 0.5 * reg_lambda * torch.sum(model.linear.weight ** 2)
    return bce + reg


def compute_gradient(model: nn.Module, X: torch.Tensor, y: torch.Tensor, reg_lambda: float) -> list:
    """
    概要: PyTorchの自動微分（`loss.backward()`）により，L2正則化付き二値交差エントロピー損失の
        勾配を計算する．バッチサイズ B に対する平均勾配 (1/B)Σ∇ℓ_BCE + λw を返す．B=1のとき，
        論文における単一サンプルの確率的勾配 ∇f_n(w) と一致する．
    引数:
        model (torch.nn.Module)．勾配を評価するパラメータを保持するモデル．
        X (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号（0または1）．
        reg_lambda (float)．L2正則化係数 λ．
    戻り値: grads (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の勾配テンソル列．
    """
    model.zero_grad(set_to_none=True)
    loss = _compute_batch_loss(model, X, y, reg_lambda)
    loss.backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def compute_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor, reg_lambda: float) -> float:
    """
    概要: L2正則化付き二値交差エントロピー損失 f(w) を計算する（(38)式）．勾配は計算しない．
    引数:
        model (torch.nn.Module)．評価対象のパラメータを保持するモデル．
        X (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号．
        reg_lambda (float)．L2正則化係数 λ．
    戻り値: loss (float)．1データあたりの損失の平均値．
    """
    with torch.no_grad():
        loss = _compute_batch_loss(model, X, y, reg_lambda)
    return loss.item()


def compute_accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """
    概要: 分類精度（Accuracy）を計算する．
    引数:
        model (torch.nn.Module)．評価対象のパラメータを保持するモデル．
        X (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号．
    戻り値: accuracy (float)．1データあたりの正解率の平均値．
    """
    with torch.no_grad():
        logits = model(X)
        pred = (torch.sigmoid(logits) >= 0.5).to(y.dtype)
        accuracy = (pred == y).to(torch.float64).mean().item()
    return accuracy


def set_model_params(model: nn.Module, param_values: list, source_model: nn.Module = None):
    """
    概要: モデルのパラメータを，指定したテンソル列の値で上書きする．
        スナップショット専用モデルのパラメータを z_{s+1} で更新する際に用いる．
        `source_model` が指定された場合，Batch Normalizationの移動平均統計量
        （running_mean，running_var等）のバッファも `source_model` の現在値で同期する．
        `param_values`（`optimizer.get_snapshot_params()`）はSVRG系手法が追跡する学習可能な
        パラメータのみを対象とし，バッファは含まないため，バッファの同期はこの引数を通じて
        別途行う必要がある．
    引数:
        model (torch.nn.Module)．上書き対象のモデル．
        param_values (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の値．
        source_model (torch.nn.Module) = None．バッファの同期元となるモデル．Noneの場合は
            バッファの同期を行わない．
    戻り値: なし
    """
    with torch.no_grad():
        for p, v in zip(model.parameters(), param_values):
            p.copy_(v)
        if source_model is not None:
            for b, b_src in zip(model.buffers(), source_model.buffers()):
                b.copy_(b_src)
