"""
実験5（Imagewoofを用いたResNet18画像分類）のモデルの定義に関するモジュール．

`.orders/order_033.md` 3節・5節の指示に基づき，標準的なResNet18
（`torchvision.models.resnet18`，$224\\times224$入力，約1118万パラメータ）を基本構造として
用いる．ただし，SVRG系手法の理論的前提（補正勾配 $ v_s^k=\\nabla f_n(w_s^k)-\\nabla
f_n(z_s)+g_s $ が同一の $ n $・同一のパラメータ $ z_s $ に対し再評価しても同じ値になること）
を満たすため，**全てのBatchNorm2d層をLayerNorm2d（ex0021由来のチャネル方向LayerNorm，
`programs/ex0021_cifar10_alexnet_norm/model.py`と同一実装）に置換する**．

置換は，`torchvision.models.resnet18(weights=None, num_classes=10)`をインスタンス化した後，
モジュール木を再帰的に走査し，`nn.BatchNorm2d`を発見するたびに同一のチャネル数を持つ
`LayerNorm2d`に置き換える方式で実装する．この方式は，ResNet18本体の層構成（Residual
ブロック数，チャネル数，Skip connectionの位置）をtorchvision公式実装からそのまま継承する
ため，独自再実装によるバグのリスクを避けられる利点がある．置換後もパラメータ数は変化しない
（`LayerNorm2d`・`BatchNorm2d`はいずれもチャネルあたり2パラメータ（スケール・シフト）の
アフィン変換のみを持つ）．置換前後でDropout層は一切含まれない（標準ResNet18の仕様通り）．

Skip connection（Residual接続）自体は決定論的な操作であるため，SVRG系手法の前提と両立する．
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed  # noqa: E402

NUM_CLASSES = 10


class LayerNorm2d(nn.Module):
    """
    概要: `programs/ex0021_cifar10_alexnet_norm/model.py`の`LayerNorm2d`と同一の実装．
        `(B, C, H, W)`形式の特徴マップに対し，各空間位置においてチャネル方向（$ C $次元）に
        正規化する．バッチ内の他サンプルに依存しない．
    """

    def __init__(self, num_channels: int):
        """
        概要: チャネル方向のLayerNormを初期化する．
        引数: num_channels (int)．正規化対象のチャネル数 $ C $．
        戻り値: なし
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: `(B, C, H, W)`を`(B, H, W, C)`に並べ替えてLayerNormを適用し，元の形状に戻す．
        引数: x (torch.Tensor)，形状 (B, C, H, W)．
        戻り値: y (torch.Tensor)，形状 (B, C, H, W)．
        """
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        return x.permute(0, 3, 1, 2)


def _replace_batchnorm_with_layernorm(module: nn.Module) -> None:
    """
    概要: モジュール木を再帰的に走査し，`nn.BatchNorm2d`を同一チャネル数の`LayerNorm2d`に
        インプレースで置換する．
    引数: module (torch.nn.Module)．
    戻り値: なし
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, LayerNorm2d(child.num_features))
        else:
            _replace_batchnorm_with_layernorm(child)


class ResNet18LayerNorm(nn.Module):
    """
    概要: 全BatchNorm2d層をLayerNorm2dに置換したResNet18（10クラス分類）．
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        """
        概要: LayerNorm化ResNet18を初期化する．
        引数: num_classes (int) = 10．出力クラス数．
        戻り値: なし
        """
        super().__init__()
        self.backbone = models.resnet18(weights=None, num_classes=num_classes)
        _replace_batchnorm_with_layernorm(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: 分類ロジットを計算する．
        引数: x (torch.Tensor)，形状 (B, 3, 224, 224)．
        戻り値: logits (torch.Tensor)，形状 (B, num_classes)．
        """
        return self.backbone(x)


def load_model(ModelClass, weight_path: str = None, seed: int = 0, **kwargs):
    """
    概要: モデルをインスタンス化するための関数．
    引数:
        ModelClass (torch.nn.Moduleのクラス)．
        weight_path (str) = None．
        seed (int) = 0．
        **kwargs．`ModelClass` に渡すその他の引数．
    戻り値: model (torch.nn.Module)．
    """
    set_seed(seed)
    model = ModelClass(**kwargs)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model


def compute_l2_regularization(model: nn.Module, reg_lambda: float) -> torch.Tensor:
    """
    概要: L2正則化項 $ \\frac\\lambda2\\|w\\|^2 $ を計算する．`nn.Conv2d`・`nn.Linear`の
        重み（バイアスを除く）を対象とし，`LayerNorm2d`のアフィンパラメータは対象外とする
        （`.orders/order_033.md` 5節）．
    引数:
        model (torch.nn.Module)．
        reg_lambda (float)．正則化係数 $ \\lambda $．
    戻り値: reg (torch.Tensor)．スカラー．
    """
    reg = next(model.parameters()).new_zeros(())
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            reg = reg + torch.sum(module.weight ** 2)
    return 0.5 * reg_lambda * reg


def loss_func(outputs: torch.Tensor, teacher_signals: torch.Tensor, model: nn.Module, reg_lambda: float) -> torch.Tensor:
    """
    概要: モデルの出力と教師信号のカテゴリカルクロスエントロピー誤差にL2正則化項を加えた
        誤差を計算する．
    引数:
        outputs (torch.Tensor)，形状 (B, num_classes)．モデルの出力（分類ロジット）．
        teacher_signals (torch.Tensor)，形状 (B,)，dtype long．教師ラベル．
        model (torch.nn.Module)．
        reg_lambda (float)．L2正則化係数．
    戻り値: loss (torch.Tensor)．スカラー．
    """
    cce = F.cross_entropy(outputs, teacher_signals)
    return cce + compute_l2_regularization(model, reg_lambda)


def metrics_func(outputs: torch.Tensor, teacher_signals: torch.Tensor) -> dict:
    """
    概要: 分類精度を計算する．
    引数:
        outputs (torch.Tensor)，形状 (B, num_classes)．
        teacher_signals (torch.Tensor)，形状 (B,)．
    戻り値: metrics_to_value (dict)．{"accuracy": float}．
    """
    predictions = torch.argmax(outputs, dim=-1)
    accuracy = (predictions == teacher_signals).float().mean().item()
    return {"accuracy": accuracy}


def set_model_params(model: nn.Module, param_values, source_model: nn.Module = None) -> None:
    """
    概要: モデルのパラメータを指定した値で置き換える．
    引数:
        model (torch.nn.Module)．更新対象のモデル．
        param_values (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の値．
        source_model (torch.nn.Module) = None．本モデルでは未使用（他モデルとの
            インターフェース統一のために受け取る）．
    戻り値: なし
    """
    with torch.no_grad():
        for p, value in zip(model.parameters(), param_values):
            p.copy_(value)
