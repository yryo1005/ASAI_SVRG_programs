"""
実験ex0021（CIFAR-10における正規化層・バッチサイズの構造的変更による安定性の検証）の
データセットの定義に関するモジュール．

`.orders/order_023.md` の指示（「データセット・前処理・目的関数は実験2と同一」）に基づき，
`programs/ex002_cifar10_alexnet/data.py` と同一の前処理（公式の学習・検証分割，チャネル
ごとの標準化，データ拡張なし）を用いる．
"""

import os
import sys

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed  # noqa: E402

EXPERIMENT_NAME = "ex0021_cifar10_alexnet_norm"
RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, "datasets", EXPERIMENT_NAME, "raw")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_dataloader(seed: int = 0, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """
    概要: 学習用および検証用のデータローダーをインスタンス化するための関数．
        CIFAR-10の公式な学習・検証分割（50000枚／10000枚）をそのまま用いる．
        データ拡張は行わず，チャネルごとの標準化のみを適用する．
    引数:
        seed (int) = 0．データの並び（シャッフル）を固定する乱数シード．
        batch_size (int) = 128．データローダーのミニバッチサイズ．
    戻り値:
        train_dataloader (torch.utils.data.DataLoader)．学習用データローダー．
        test_dataloader (torch.utils.data.DataLoader)．検証用データローダー．
    """
    set_seed(seed)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    train_dataset = torchvision.datasets.CIFAR10(
        root=RAW_DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=RAW_DATA_DIR, train=False, download=True, transform=transform
    )

    generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
