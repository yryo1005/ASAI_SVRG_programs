"""
実験1（Mushroomデータセットを用いた二値分類問題）のモデルの定義に関するモジュール．

`.orders/order_021.md` 実験1の式(38)相当に対応する，L2正則化付き二値交差エントロピー損失を
目的関数とするロジスティック回帰モデルを定義する．

$$
f(w, b) = \\frac{1}{N}\\sum_{n=1}^N \\ell_{BCE}(y_n, \\hat{y}_n(w, b))
    + \\frac{\\lambda}{2}(\\|w\\|^2 + b^2), \\quad \\hat{y}_n(w, b) = \\sigma(w^\\top x_n + b)
$$

理論解析（Assumption 2(a)，$ \\mu $-強凸性）はモデルの全パラメータ（$ w $ と切片 $ b $ の両方）に
ついて成立する必要がある．正則化項を $ w $ のみに課す場合，切片 $ b $ 方向にはヘッセ行列の
下界が保証されず，厳密には $ \\mu $-強凸性を満たさない．そのため本モジュールは，`.orders/
order_020.md` 実施時の事前実験（`programs_old/ex001_mushroom_svrg/model.py`，正則化を $ w $
のみに課す実装）とは異なり，正則化項に切片 $ b $ を含める．これにより，全パラメータ
$ (w, b) $ に対して一様に $ \\mu = \\lambda $ の強凸性が成立し，Assumption 2(a)を厳密に満たす．
"""

import os
import sys

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed  # noqa: E402


class LogisticRegressionModel(nn.Module):
    """
    L2正則化付きロジスティック回帰モデル（切片項を含む）．
    $ \\hat{y}(w, b) = \\sigma(w^\\top x + b) $ に対応する．
    """

    def __init__(self, input_dim: int = 22):
        """
        概要: 線形層（重み $ w $ とバイアス $ b $）を初期化する．
        引数: input_dim (int) = 22．入力特徴量の次元数 $ d $．
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
    ModelClass=LogisticRegressionModel,
    weight_path: str = None,
    seed: int = 0,
    input_dim: int = 22,
) -> nn.Module:
    """
    概要: モデルをインスタンス化するための関数．
    引数:
        ModelClass (torch.nn.Moduleのクラス) = LogisticRegressionModel．
        weight_path (str) = None．学習済み重みのパス．指定した場合はこれを読み込む．
        seed (int) = 0．パラメータ初期値を固定する乱数シード．
        input_dim (int) = 22．入力特徴量の次元数 $ d $．
    戻り値: model (torch.nn.Module)．
    """
    set_seed(seed)
    model = ModelClass(input_dim)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model


def loss_func(
    logits: torch.Tensor, teacher_signals: torch.Tensor, weight: torch.Tensor,
    bias: torch.Tensor, reg_lambda: float,
) -> torch.Tensor:
    """
    概要: L2正則化付き二値交差エントロピー損失 $ f(w, b) $ を計算する．正則化項は重み $ w $
        と切片 $ b $ の両方に課す（モジュールdocstring参照）．
    引数:
        logits (torch.Tensor)，形状 (B,)．モデルの出力（シグモイド適用前）．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号 $ y \\in \\{0, 1\\} $．
        weight (torch.Tensor)，形状 (1, d)．線形層の重み．
        bias (torch.Tensor)，形状 (1,)．線形層の切片．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: loss (torch.Tensor)，形状 ()．勾配計算グラフ付きの損失．
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, teacher_signals, reduction="mean")
    reg = 0.5 * reg_lambda * (torch.sum(weight ** 2) + torch.sum(bias ** 2))
    return bce + reg


def metrics_func(logits: torch.Tensor, teacher_signals: torch.Tensor) -> dict:
    """
    概要: モデルの出力と教師信号の一致度（分類精度）を評価する関数．
    引数:
        logits (torch.Tensor)，形状 (B,)．モデルの出力（シグモイド適用前）．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号 $ y \\in \\{0, 1\\} $．
    戻り値: metrics_to_value (dict)．{"accuracy": float} の辞書．
    """
    with torch.no_grad():
        pred = (torch.sigmoid(logits) >= 0.5).to(teacher_signals.dtype)
        accuracy = (pred == teacher_signals).to(torch.float64).mean().item()
    return {"accuracy": accuracy}


def compute_gradient(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, reg_lambda: float
) -> list:
    """
    概要: PyTorchの自動微分（`loss.backward()`）により，L2正則化付き二値交差エントロピー損失の
        勾配を計算し，各パラメータの `.grad` に設定する．バッチサイズ $ B $ に対する平均勾配を
        返す．$ B=1 $ のとき，単一サンプルの確率的勾配 $ \\nabla f_n(w, b) $ と一致する．
    引数:
        model (torch.nn.Module)．勾配を評価するパラメータを保持するモデル．
        X (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: grads (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の勾配テンソル列．
    """
    model.zero_grad(set_to_none=True)
    logits = model(X)
    loss = loss_func(logits, y, model.linear.weight, model.linear.bias, reg_lambda)
    loss.backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def compute_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor, reg_lambda: float) -> float:
    """
    概要: L2正則化付き二値交差エントロピー損失 $ f(w, b) $ を計算する．勾配は計算しない．
    引数:
        model (torch.nn.Module)．評価対象のパラメータを保持するモデル．
        X (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: loss (float)．1データあたりの損失の平均値．
    """
    with torch.no_grad():
        logits = model(X)
        loss = loss_func(logits, y, model.linear.weight, model.linear.bias, reg_lambda)
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
        return metrics_func(model(X), y)["accuracy"]


def set_model_params(model: nn.Module, param_values: list, source_model: nn.Module = None):
    """
    概要: モデルのパラメータを，指定したテンソル列の値で上書きする．
        スナップショット専用モデルのパラメータを $ z_{s+1} $ で更新する際に用いる．
    引数:
        model (torch.nn.Module)．上書き対象のモデル．
        param_values (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の値．
        source_model (torch.nn.Module) = None．他の実験とインターフェースを揃えるための引数．
            `LogisticRegressionModel` はBatch Normalization等のバッファを持たないため，
            指定されても何も行わない．
    戻り値: なし
    """
    with torch.no_grad():
        for p, v in zip(model.parameters(), param_values):
            p.copy_(v)
