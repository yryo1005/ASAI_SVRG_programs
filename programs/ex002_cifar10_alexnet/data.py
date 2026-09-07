"""
実験2（CIFAR-10を用いた多値分類問題）のデータセットの定義に関するモジュール．

`.orders/order_022.md` の実験2に対応する．CIFAR-10は32×32ピクセルのカラー画像から成る
10クラス画像分類データセットであり，訓練データ数 $ N_{\\text{train}}=50000 $，検証データ数
$ N_{\\text{test}}=10000 $ が公式に定義されている．`.orders/order_022.md` 3節は「学習用データ
からさらに検証用の一部を分割するかは実装時に判断する」としているが，本実験ではCIFAR-10の
公式な学習・検証分割をそのまま用いる（`programs_old/ex002_cifar10_cnn/data.py` と同様の判断）．
理由は，(1) 公式分割は広く再現性の基準として用いられており，恣意的な追加分割を避けられる，
(2) 本実験の主目的は手法比較であり，ハイパーパラメータ選択のための独立した検証集合を別途
必要としない（学習率・バッチサイズはグリッド内の固定値を用い，チューニングを行わない）ため
である．

`.orders/order_022.md` 2節の制約（SVRG系手法との理論的整合性）に従い，データ拡張
（ランダムクロップ・反転等）は用いない．画素値の正規化（チャネルごとの平均・標準偏差による
標準化）のみを行う．
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

EXPERIMENT_NAME = "ex002_cifar10_alexnet"
RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, "datasets", EXPERIMENT_NAME, "raw")

# CIFAR-10の画素値を標準化するためのチャネルごとの平均・標準偏差（一般に広く用いられる値）．
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

    # multiprocessing.Poolのワーカープロセスはデーモンプロセスであり，DataLoader側で
    # num_workers>0を指定した子プロセスをさらに生成することはできない
    # （"daemonic processes are not allowed to have children"）．そのためnum_workers=0とする．
    generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
