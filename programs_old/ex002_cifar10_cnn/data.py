"""
CIFAR-10データセットの読み込みおよび前処理を行うモジュール．

論文4.2節（CNNとCIFAR-10の多値分類問題）に対応するデータセットを構築する．CIFAR-10は
32×32ピクセルのカラー画像からなる10クラスの画像分類データセットであり，訓練データ数
N = 50000，テストデータ数 10000 が公式に定義されている．
"""

import os
import sys

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed

RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, "datasets", "ex002_cifar10_cnn", "raw")

# CIFAR-10の画素値を標準化するためのチャネルごとの平均・標準偏差（一般に広く用いられる値）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_dataloader(seed: int = 0, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """
    概要: 学習用および検証用のデータローダーをインスタンス化するための関数．
        CIFAR-10は学習用データ（50000枚）と検証用データ（10000枚）の公式な分割が
        定義されているため，`sklearn.model_selection.train_test_split` による9:1分割は
        行わず，公式の分割をそのまま用いる．
    引数:
        seed (int) = 0．データの並び（シャッフル）を固定する乱数シード．
        batch_size (int) = 128．データローダーのミニバッチサイズ．
    戻り値:
        train_dataloader (torch.utils.data.DataLoader)
        test_dataloader (torch.utils.data.DataLoader)
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

    # 学習全体をmultiprocessing.Poolで並列化しており，Poolのワーカープロセスはデーモン
    # プロセスであるため，DataLoader側でnum_workers>0を指定した子プロセスを追加で生成する
    # ことはできない（"daemonic processes are not allowed to have children"）．そのため
    # num_workers=0（メインプロセス内での読み込み）とする．
    generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
