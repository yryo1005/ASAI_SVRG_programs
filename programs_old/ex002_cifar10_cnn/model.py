"""
CIFAR-10の多値分類に用いる畳み込みニューラルネットワーク（CNN）の定義．

論文4.2節の記載に対応し，3つの畳み込み層と1つの全結合層により構成されるCNNを用いる．
"""

import os
import sys

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed


class CNNModel(nn.Module):
    """
    3つの畳み込み層と1つの全結合層から構成されるCNN．
    入力画像 (3, 32, 32) を3つの畳み込み層（各層の後にReLUと2x2の最大値プーリングを適用）で
    特徴抽出し，全結合層によりクラス数 num_classes 次元のロジットへ写像する．
    """

    def __init__(self, num_classes: int = 10):
        """
        概要: 3つの畳み込み層と1つの全結合層を初期化する．
        引数: num_classes (int) = 10．分類先のクラス数 C．
        戻り値: なし
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 32, 32) -> (32, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (64, 16, 16) -> (64, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (128, 8, 8) -> (128, 4, 4)
        )
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: 入力画像からロジットを計算する．
        引数: x (torch.Tensor)，形状 (B, 3, 32, 32)．入力画像．
        戻り値: logits (torch.Tensor)，形状 (B, num_classes)．
        """
        features = self.conv_layers(x)
        features = features.flatten(start_dim=1)
        return self.fc(features)


def load_model(ModelClass=CNNModel, weight_path: str = None, seed: int = 0) -> nn.Module:
    """
    概要: モデルをインスタンス化するための関数．
    引数:
        ModelClass (torch.nn.Moduleのクラス) = CNNModel．
        weight_path (str) = None．学習済み重みのパス．指定した場合はこれを読み込む．
        seed (int) = 0．パラメータ初期値を固定する乱数シード．
    戻り値: model (torch.nn.Module)．
    """
    set_seed(seed)
    model = ModelClass()
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model


def set_model_params(model: nn.Module, param_values: list, source_model: nn.Module = None):
    """
    概要: モデルのパラメータを，指定したテンソル列の値で上書きする．
        SVRG系手法におけるスナップショット専用モデルのパラメータを z_{s+1} で更新する際に用いる．
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
