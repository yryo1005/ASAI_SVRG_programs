"""
実験2（CIFAR-10を用いた多値分類問題）のモデルの定義に関するモジュール．

`.orders/order_022.md` 3節に対応する，AlexNetをCIFAR-10向けに縮小した構成のCNN
（`AlexNetCIFAR`）を定義する．同2節の制約（SVRG系手法との理論的整合性）に従い，
BatchNormalization・Dropoutは使用しない決定論的なモデルとし，正則化はL2 weight decayを
損失関数に明示的に加える形で実装する（`torch.optim.Optimizer` の `weight_decay` 引数は
用いない．`programs/optimizers/` の各クラスは `weight_decay` を実装しておらず，かつ
SVRG系手法の補正勾配は `.grad` 属性を通じて明示的に扱う必要があるため）．

## 層構成

5つの畳み込み層（各層の後にReLUを適用し，1・2・5層目の後に2×2の最大値プーリングを適用）と
3つの全結合層から構成される．オリジナルのAlexNet（全結合層が各4096ユニット）に対し，本実験の
比較手法にはフル勾配計算を伴うSVRG系手法が含まれ計算コストが大きいため，全結合層のユニット数を
縮小している．

| 層 | 入力形状 | 出力形状 | パラメータ数 |
| :--- | :--- | :--- | ---: |
| Conv1 (5x5, pad2) | (3, 32, 32) | (64, 32, 32) | 4,864 |
| MaxPool | (64, 32, 32) | (64, 16, 16) | 0 |
| Conv2 (5x5, pad2) | (64, 16, 16) | (192, 16, 16) | 307,392 |
| MaxPool | (192, 16, 16) | (192, 8, 8) | 0 |
| Conv3 (3x3, pad1) | (192, 8, 8) | (384, 8, 8) | 663,936 |
| Conv4 (3x3, pad1) | (384, 8, 8) | (256, 8, 8) | 884,992 |
| Conv5 (3x3, pad1) | (256, 8, 8) | (256, 8, 8) | 590,080 |
| MaxPool | (256, 8, 8) | (256, 4, 4) | 0 |
| FC1 | 4096 | 1024 | 4,195,328 |
| FC2 | 1024 | 512 | 524,800 |
| FC3 | 512 | 10 | 5,130 |

総パラメータ数は約717万である．
"""

import os
import sys

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed  # noqa: E402


class AlexNetCIFAR(nn.Module):
    """
    AlexNetをCIFAR-10向けに縮小した，決定論的な（BatchNormalization・Dropoutを含まない）CNN．
    """

    def __init__(self, num_classes: int = 10):
        """
        概要: 5つの畳み込み層と3つの全結合層を初期化する．
        引数: num_classes (int) = 10．分類先のクラス数 $ C $．
        戻り値: なし
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (64, 32, 32) -> (64, 16, 16)
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (192, 16, 16) -> (192, 8, 8)
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (256, 8, 8) -> (256, 4, 4)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: 入力画像からロジットを計算する．
        引数: x (torch.Tensor)，形状 (B, 3, 32, 32)．入力画像．
        戻り値: logits (torch.Tensor)，形状 (B, num_classes)．
        """
        features = self.conv_layers(x)
        features = features.flatten(start_dim=1)
        return self.fc_layers(features)


def load_model(ModelClass=AlexNetCIFAR, weight_path: str = None, seed: int = 0) -> nn.Module:
    """
    概要: モデルをインスタンス化するための関数．
    引数:
        ModelClass (torch.nn.Moduleのクラス) = AlexNetCIFAR．
        weight_path (str) = None．学習済み重みのパス．指定した場合はこれを読み込む．
        seed (int) = 0．パラメータ初期値を固定する乱数シード．
    戻り値: model (torch.nn.Module)．
    """
    set_seed(seed)
    model = ModelClass()
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model


def compute_l2_regularization(model: nn.Module, reg_lambda: float) -> torch.Tensor:
    """
    概要: L2正則化項 $ \\frac{\\lambda}{2}\\|w\\|^2 $ を計算する．畳み込み層・全結合層の重み
        （`weight`）のみを対象とし，切片（`bias`）は正則化しない（一般的なweight decayの
        慣例に従う）．
    引数:
        model (torch.nn.Module)．正則化対象の重みを保持するモデル．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: reg (torch.Tensor)，形状 ()．正則化項．
    """
    reg = model.conv_layers[0].weight.new_zeros(())
    for name, param in model.named_parameters():
        if name.endswith(".weight"):
            reg = reg + torch.sum(param ** 2)
    return 0.5 * reg_lambda * reg


def loss_func(
    outputs: torch.Tensor, teacher_signals: torch.Tensor, model: nn.Module, reg_lambda: float
) -> torch.Tensor:
    """
    概要: L2正則化付き多値交差エントロピー損失（`.orders/order_022.md` 3節の式）を計算する．
    引数:
        outputs (torch.Tensor)，形状 (B, C)．モデルの出力（ロジット）．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号（クラスインデックス）．
        model (torch.nn.Module)．正則化項の計算対象のモデル．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: loss (torch.Tensor)，形状 ()．1データあたりの誤差の平均値＋正則化項．
    """
    cce = nn.functional.cross_entropy(outputs, teacher_signals, reduction="mean")
    return cce + compute_l2_regularization(model, reg_lambda)


def metrics_func(outputs: torch.Tensor, teacher_signals: torch.Tensor) -> dict:
    """
    概要: モデルの出力と教師信号の一致度（分類精度）を評価する関数．
    引数:
        outputs (torch.Tensor)，形状 (B, C)．モデルの出力（ロジット）．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号（クラスインデックス）．
    戻り値: metrics_to_value (dict)．{"accuracy": 1データあたりの正解率の平均値}．
    """
    with torch.no_grad():
        pred = outputs.argmax(dim=1)
        accuracy = (pred == teacher_signals).to(torch.float32).mean().item()
    return {"accuracy": accuracy}


def set_model_params(model: nn.Module, param_values: list, source_model: nn.Module = None):
    """
    概要: モデルのパラメータを，指定したテンソル列の値で上書きする．
        SVRG系手法におけるスナップショット専用モデルのパラメータを $ z_{s+1} $ で更新する
        際に用いる．
    引数:
        model (torch.nn.Module)．上書き対象のモデル．
        param_values (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の値．
        source_model (torch.nn.Module) = None．他の実験とインターフェースを揃えるための
            引数．`AlexNetCIFAR` はBatch Normalization等のバッファを持たないため，
            指定されても何も行わない．
    戻り値: なし
    """
    with torch.no_grad():
        for p, v in zip(model.parameters(), param_values):
            p.copy_(v)
