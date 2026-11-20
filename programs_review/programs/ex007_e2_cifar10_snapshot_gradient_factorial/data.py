"""
E0（CIFAR-10）のデータセットの定義．

前処理は既存の実験（`ex0024_cifar10_alexnet_layernorm_longrun`）と同一である．すなわち，
CIFAR-10の公式の学習・テスト分割（50000枚，10000枚）をそのまま用い，`ToTensor` と
チャネルごとの `Normalize`（平均 (0.4914, 0.4822, 0.4465)，標準偏差 (0.2470, 0.2435, 0.2616)）
のみを適用し，データ拡張は行わない．

前処理は決定論的であるため，前処理後のテンソルを `datasets/cifar10/cache/` に1回だけ保存し，
学習ではこれをGPUに載せて `CachedTensorLoader` で読み出す（`common/cached_loader.py`）．
ミニバッチの値と順序は既存の `DataLoader` を用いた実装と一致する．
"""

import os
import sys

import torch
import torchvision
from torchvision import transforms

_PROGRAMS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))

from cached_loader import CachedTensorLoader  # noqa: E402
from machine_learning_utils import set_seed  # noqa: E402

RAW_DATA_DIR = os.path.join(_REVIEW_ROOT, "datasets", "cifar10", "raw")
CACHE_DIR = os.path.join(_REVIEW_ROOT, "datasets", "cifar10", "cache")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transform() -> transforms.Compose:
    """
    概要: 既存の実験と同一の前処理（`ToTensor`，`Normalize`）を構築する．
    引数: なし
    戻り値: transform (torchvision.transforms.Compose)．
    """
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def _cache_path(train: bool) -> str:
    """
    概要: キャッシュファイルのパスを返す．
    引数: train (bool)．学習データであれば True．
    戻り値: path (str)．
    """
    return os.path.join(CACHE_DIR, "train.pt" if train else "test.pt")


def _build_split(train: bool) -> dict:
    """
    概要: 1つの分割について，前処理後の全画像と教師ラベルをテンソルにまとめる．
    引数: train (bool)．学習データであれば True．
    戻り値: data (dict)．{"inputs": torch.Tensor (N, 3, 32, 32) float32, "targets": torch.Tensor (N,) int64}．
    """
    dataset = torchvision.datasets.CIFAR10(root=RAW_DATA_DIR, train=train, download=True,
                                           transform=build_transform())
    inputs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    targets = torch.tensor(dataset.targets, dtype=torch.int64)
    return {"inputs": inputs, "targets": targets}


def prepare_cache() -> None:
    """
    概要: 前処理後のテンソルのキャッシュが無ければ作成する．並列に学習を始める前に，
        メインプロセスで1回だけ呼び出す．
    引数: なし
    戻り値: なし
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    for train in (True, False):
        path = _cache_path(train)
        if os.path.exists(path):
            continue
        tmp_path = path + ".tmp"
        torch.save(_build_split(train), tmp_path)
        os.replace(tmp_path, path)


def load_dataloader(seed: int = 0, batch_size: int = 128, device: torch.device = None):
    """
    概要: 学習用とテスト用のローダーをインスタンス化する．学習用のローダーは既存の実装と同じく
        `torch.Generator().manual_seed(seed)` を用いてエポックごとにシャッフルする．
    引数:
        seed (int) = 0．シャッフルの乱数シード．
        batch_size (int) = 128．ミニバッチサイズ．
        device (torch.device) = None．キャッシュを載せるデバイス．
    戻り値:
        train_loader (CachedTensorLoader)．学習用のローダー．
        test_loader (CachedTensorLoader)．テスト用のローダー（シャッフルなし）．
    """
    set_seed(seed)
    prepare_cache()
    train_data = torch.load(_cache_path(True))
    test_data = torch.load(_cache_path(False))
    if device is not None:
        train_data = {key: value.to(device) for key, value in train_data.items()}
        test_data = {key: value.to(device) for key, value in test_data.items()}

    generator = torch.Generator().manual_seed(seed)
    train_loader = CachedTensorLoader(train_data["inputs"], train_data["targets"], batch_size,
                                      shuffle=True, generator=generator)
    test_loader = CachedTensorLoader(test_data["inputs"], test_data["targets"], batch_size, shuffle=False)
    return train_loader, test_loader
