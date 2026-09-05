"""
`.orders/order_010.md` の指示に基づく，σ（敵対的摂動）を外した純粋なCIFAR-10多値分類問題の
ResNet-18モデルを定義するモジュール．

Ex003・Ex004（`programs/ex003_cifar10_resnet_minmax/`，`programs/ex004_cifar10_resnet_minmax/`）
が採用するmin-max敵対的ロバスト性の定式化
    min_w max_sigma (1/M) sum_i [ f(w, x_i + sigma, y_i) ] + (lambda1/2)||w||^2 - (lambda2/2)||sigma||^2
から，最大化変数sigma（入力画像全体に一様に加える敵対的摂動）と正則化項 -(lambda2/2)||sigma||^2
を取り除き，通常の多値分類の最小化問題
    min_w (1/M) sum_i [ CE(w, x_i, y_i) ] + (lambda1/2)||w||^2
のみを残す．`.reports/report_010.md` が指摘する「NFG/ASAI SVRGの不安定性はmin-max（敵対的
摂動）構造自体に起因するのではないか」という仮説を，Ex004と条件を可能な限り揃えたまま検証する
ための実験（Ex005）に用いる．
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed


class BasicBlock(nn.Module):
    """ResNetの基本残差ブロック（3x3畳み込み2層 + ショートカット）．"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        概要: 残差ブロックを初期化する．
        引数:
            in_channels (int)．入力チャネル数．
            out_channels (int)．出力チャネル数．
            stride (int) = 1．最初の畳み込みのストライド．
        戻り値: なし
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion, kernel_size=1, stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: 残差ブロックの順伝播を計算する．
        引数: x (torch.Tensor)，形状 (B, in_channels, H, W)．
        戻り値: out (torch.Tensor)，形状 (B, out_channels, H', W')．
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """
    CIFAR-10向けResNet-18（He et al., 2016）．入力画像が32x32と小さいため，最初の畳み込みは
    3x3・ストライド1とし，ImageNet向け実装にある7x7畳み込み・最大値プーリングは用いない
    （CIFAR系データセットに対するResNetの標準的な適用方法）．Ex003・Ex004の`ResNet18`と
    同一の構造である．
    """

    def __init__(self, num_classes: int = 10):
        """
        概要: ResNet-18を初期化する．
        引数: num_classes (int) = 10．分類先のクラス数．
        戻り値: なし
        """
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        概要: 同一チャネル数の残差ブロックを複数連結した層を構築する．
        引数:
            out_channels (int)．出力チャネル数．
            num_blocks (int)．ブロック数．
            stride (int)．最初のブロックのストライド．
        戻り値: layer (nn.Sequential)．
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: 入力画像からロジットを計算する．
        引数: x (torch.Tensor)，形状 (B, 3, 32, 32)．
        戻り値: logits (torch.Tensor)，形状 (B, num_classes)．
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(start_dim=1)
        return self.fc(out)


def load_model(ModelClass=ResNet18, weight_path: str = None, seed: int = 0) -> nn.Module:
    """
    概要: モデルをインスタンス化するための関数．
    引数:
        ModelClass (torch.nn.Moduleのクラス) = ResNet18．
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
        SVRG系手法におけるスナップショット専用モデルのパラメータを更新する際に用いる．
        `source_model` が指定された場合，Batch Normalizationの移動平均統計量
        （running_mean，running_var等）のバッファも `source_model` の現在値で同期する．
        `param_values`（`optimizer.get_snapshot_params()`）はSVRG系手法が追跡する学習可能な
        パラメータ（w）のみを対象とし，バッファは含まないため，バッファの同期はこの引数を通じて
        別途行う必要がある（`.orders/order_010.md` で報告されたバグの修正）．
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
