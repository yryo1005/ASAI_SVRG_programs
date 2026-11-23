"""
E4（付録B.1，a9a の MSE・BCE の再実施）のモデルと目的関数の定義に関するモジュール．

既存の実験用リポジトリの2つのモデル定義をコピーしたものであり，モデルと目的関数の計算は変更していない．
引数 `loss` で2つの実験を切り替える．

| `loss` | 既存のファイル | モデル | 目的関数 |
|---|---|---|---|
| `"mse"` | `programs/ex000_a9a_least_squares/model.py` | $ \\sigma({\\mathbf w}^{\\top}{\\mathbf x}) $，切片なし，float64 | $ f({\\mathbf w}) = \\frac{1}{N}\\sum_{n}(y_{n} - \\sigma({\\mathbf w}^{\\top}{\\mathbf x}_{n}))^{2} $．正則化なし |
| `"bce"` | `programs/ex0001_a9a_logistic/model.py` | $ \\sigma({\\mathbf w}^{\\top}{\\mathbf x} + b) $，切片付き，float64 | BCE $ + \\frac{\\lambda}{2}(\\|{\\mathbf w}\\|^{2} + b^{2}) $ |

MSE は NFG-SVRG の論文の付録A.1の式(8)と同じ非凸の目的関数である．BCE の正則化項は切片 $ b $ にも課し，
全パラメータについて $ \\mu = \\lambda $ の強凸性を成り立たせる．
"""

import os
import sys

import torch
import torch.nn as nn

# programs_review/programs/common を参照する（programs_review の外のファイルは参照しない）．
_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common")
sys.path.insert(0, _COMMON_DIR)

from machine_learning_utils import set_seed  # noqa: E402

LOSSES = ("mse", "bce")


class LeastSquaresSigmoidModel(nn.Module):
    """
    非線形最小二乗回帰のモデル（MSE）．$ z = {\\mathbf w}^{\\top}{\\mathbf x} $（切片なし）を出力する．
    パラメータは倍精度（`torch.float64`）で保持する．
    """

    def __init__(self, input_dim: int = 123):
        """
        概要: 線形層（重み $ {\\mathbf w} $，切片なし）を初期化する．
        引数: input_dim (int) = 123．入力特徴量の次元数 $ d $．
        戻り値: なし
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False, dtype=torch.float64)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        概要: 入力特徴量から線形結合 $ z $（シグモイド関数を適用する前の値）を計算する．
        引数: A (torch.Tensor)，形状 (B, d)．入力特徴量．
        戻り値: z (torch.Tensor)，形状 (B,)．
        """
        return self.linear(A).squeeze(-1)


class LogisticRegressionModel(nn.Module):
    """
    L2正則化付きロジスティック回帰のモデル（BCE）．$ z = {\\mathbf w}^{\\top}{\\mathbf x} + b $ を出力する．
    """

    def __init__(self, input_dim: int = 123):
        """
        概要: 線形層（重み $ {\\mathbf w} $ と切片 $ b $）を初期化する．
        引数: input_dim (int) = 123．入力特徴量の次元数 $ d $．
        戻り値: なし
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: 入力特徴量からロジット（シグモイド関数を適用する前の値）を計算する．
        引数: x (torch.Tensor)，形状 (B, d)．入力特徴量．
        戻り値: logits (torch.Tensor)，形状 (B,)．
        """
        return self.linear(x).squeeze(-1)


MODEL_CLASSES = {"mse": LeastSquaresSigmoidModel, "bce": LogisticRegressionModel}


def load_model(loss: str = "bce", weight_path: str = None, seed: int = 0, input_dim: int = 123) -> nn.Module:
    """
    概要: `loss` に対応するモデルをインスタンス化する．`set_seed(seed)` の直後に生成するため，
        初期値は既存のプログラムの `load_model(seed=seed)` と同一になる．
    引数:
        loss (str) = "bce"．"mse" または "bce"．
        weight_path (str) = None．学習済みの重みのパス．指定した場合はこれを読み込む．
        seed (int) = 0．パラメータの初期値を固定する乱数シード．
        input_dim (int) = 123．入力特徴量の次元数 $ d $．
    戻り値: model (torch.nn.Module)．
    """
    set_seed(seed)
    model = MODEL_CLASSES[loss](input_dim)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model


def loss_func(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss: str, reg_lambda: float) -> torch.Tensor:
    """
    概要: 目的関数（1データあたりの平均）を計算する．計算の順序は既存のプログラムと同じである．
        - "mse"：`torch.mean((y - sigmoid(z)) ** 2)`（`reg_lambda` は用いない）．
        - "bce"：`binary_cross_entropy_with_logits(z, y, reduction="mean") + 0.5 * reg_lambda * (sum(w ** 2) + sum(b ** 2))`．
    引数:
        model (torch.nn.Module)．パラメータを保持するモデル．
        X (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号 $ y \\in \\{0, 1\\} $．
        loss (str)．"mse" または "bce"．
        reg_lambda (float)．L2正則化係数 $ \\lambda $（"bce" のみ）．
    戻り値: value (torch.Tensor)，形状 ()．勾配の計算グラフ付きの目的関数の値．
    """
    outputs = model(X)
    if loss == "mse":
        h = torch.sigmoid(outputs)
        return torch.mean((y - h) ** 2)
    if loss == "bce":
        bce = nn.functional.binary_cross_entropy_with_logits(outputs, y, reduction="mean")
        reg = 0.5 * reg_lambda * (torch.sum(model.linear.weight ** 2) + torch.sum(model.linear.bias ** 2))
        return bce + reg
    raise ValueError(f"未知の損失: {loss}")


def compute_gradient(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss: str, reg_lambda: float) -> list:
    """
    概要: 自動微分により目的関数の勾配（バッチに対する平均）を計算し，各パラメータの `.grad` に設定する．
        $ B = 1 $ のとき，単一サンプルの確率的勾配 $ \\nabla f_{n} $ と一致する．
    引数:
        model (torch.nn.Module)．勾配を評価するパラメータを保持するモデル．
        X (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号．
        loss (str)．"mse" または "bce"．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: grads (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の勾配の列．
    """
    model.zero_grad(set_to_none=True)
    value = loss_func(model, X, y, loss, reg_lambda)
    value.backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def compute_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss: str, reg_lambda: float) -> float:
    """
    概要: 目的関数の値 $ f $ を計算する．勾配は計算しない．
    引数:
        model (torch.nn.Module)．評価するパラメータを保持するモデル．
        X (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号．
        loss (str)．"mse" または "bce"．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: value (float)．1データあたりの目的関数の値．
    """
    with torch.no_grad():
        return loss_func(model, X, y, loss, reg_lambda).item()


def compute_accuracy_and_correct(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> tuple[float, int]:
    """
    概要: 分類精度と正解数を計算する．$ \\sigma(z) \\ge 0.5 $ を正例の予測とみなす（既存の `metrics_func` と同じ）．
    引数:
        model (torch.nn.Module)．評価するパラメータを保持するモデル．
        X (torch.Tensor)，形状 (B, d)．入力特徴量．
        y (torch.Tensor)，形状 (B,)．教師信号．
    戻り値:
        accuracy (float)．1データあたりの正解率．
        correct (int)．正解数．
    """
    with torch.no_grad():
        pred = (torch.sigmoid(model(X)) >= 0.5).to(y.dtype)
        hit = (pred == y)
        accuracy = hit.to(torch.float64).mean().item()
        correct = int(hit.sum().item())
    return accuracy, correct


def set_model_params(model: nn.Module, param_values: list, source_model: nn.Module = None):
    """
    概要: モデルのパラメータを，指定したテンソルの列の値で上書きする．
    引数:
        model (torch.nn.Module)．上書きするモデル．
        param_values (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の値．
        source_model (torch.nn.Module) = None．他の実験とインターフェースを揃えるための引数．
            本モジュールのモデルはバッファを持たないため，指定されても何も行わない．
    戻り値: なし
    """
    with torch.no_grad():
        for p, v in zip(model.parameters(), param_values):
            p.copy_(v)
