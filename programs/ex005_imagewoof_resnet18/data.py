"""
実験5（Imagewoofを用いたResNet18画像分類，`ex005_imagewoof_resnet18`）のデータセットの
定義に関するモジュール．

`.orders/order_033.md` 4節の指示に基づき，Imagewoof（fast.aiが配布する，ImageNet-1kから
抽出した犬種10クラスのサブセット）を用いる．本モジュールでは，最大辺320pxにリサイズ済みの
配布物（`imagewoof2-320.tgz`）を用いる．画像サイズ・解像度の縮小以外，元の配布物（公式
train/val分割，クラスラベル）と同一である．

## 10クラスの内訳（WordNet synset ID順，`torchvision.datasets.ImageFolder`の既定の
クラス順序と一致）

| synset ID | 犬種（英名） |
|---|---|
| n02086240 | Shih-Tzu |
| n02087394 | Rhodesian ridgeback |
| n02088364 | Beagle |
| n02089973 | English foxhound |
| n02093754 | Australian terrier |
| n02096294 | Border terrier |
| n02099601 | Golden retriever |
| n02105641 | Old English sheepdog |
| n02111889 | Samoyed |
| n02115641 | Dingo |

## 前処理（`.orders/order_033.md` 3節・4節）

SVRG系手法の理論的前提（同一の $ n $ に対する勾配評価が常に同じ値になること）を満たすため，
決定論的な前処理のみを用いる．具体的には，短辺を256pxにリサイズした後，中心から
$ 224\times224 $ を切り出す（ImageNet分類タスクで標準的に用いられる評価時の前処理と同一
であり，ランダム性を含まない）．ランダムクロップ・反転等のデータ拡張は一切用いない．

正規化（チャネルごとの標準化）には，Imagewoofの学習用画像から算出したチャネルごとの平均・
標準偏差を用いる（ImageNetの一般的な統計値を流用せず，本データセット自身の統計量を用いる
ことで，他の実験系列（実験3・4で語彙やチャンク分割を自データから構築した方針）との一貫性を
保つ）．算出結果は初回計算後，`datasets/ex005_imagewoof_resnet18/raw/channel_stats.json`に
キャッシュし，全条件・全プロセスで同一の値を再利用する．
"""

import json
import os
import sys

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed  # noqa: E402

EXPERIMENT_NAME = "ex005_imagewoof_resnet18"
RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, "datasets", EXPERIMENT_NAME, "raw")
_ARCHIVE_PATH = os.path.join(RAW_DATA_DIR, "imagewoof2-320.tgz")
_EXTRACT_DIR = os.path.join(RAW_DATA_DIR, "imagewoof2-320")
_TRAIN_DIR = os.path.join(_EXTRACT_DIR, "train")
_VAL_DIR = os.path.join(_EXTRACT_DIR, "val")
_STATS_PATH = os.path.join(RAW_DATA_DIR, "channel_stats.json")
_DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz"

IMAGE_SIZE = 224
RESIZE_SHORT_SIDE = 256
NUM_CLASSES = 10

CLASS_NAMES = [
    "Shih-Tzu", "Rhodesian ridgeback", "Beagle", "English foxhound",
    "Australian terrier", "Border terrier", "Golden retriever",
    "Old English sheepdog", "Samoyed", "Dingo",
]


def _download_and_extract_raw_data() -> None:
    """
    概要: Imagewoof2-320の生データをダウンロード・展開する（初回のみ）．
    引数: なし
    戻り値: なし
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    if not os.path.exists(_TRAIN_DIR) or not os.path.exists(_VAL_DIR):
        if not os.path.exists(_ARCHIVE_PATH):
            import urllib.request

            urllib.request.urlretrieve(_DATA_URL, _ARCHIVE_PATH)
        import tarfile

        with tarfile.open(_ARCHIVE_PATH) as tar:
            tar.extractall(RAW_DATA_DIR)


def _compute_channel_stats() -> tuple:
    """
    概要: 学習用画像から，チャネルごとの平均・標準偏差を計算する（キャッシュがあればそれを
        用いる）．
    引数: なし
    戻り値:
        mean (list of float)．長さ3（RGB）．
        std (list of float)．長さ3（RGB）．
    """
    if os.path.exists(_STATS_PATH):
        with open(_STATS_PATH) as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]

    resize_transform = transforms.Compose([
        transforms.Resize(RESIZE_SHORT_SIDE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(_TRAIN_DIR, transform=resize_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    n_pixels = 0
    for images, _ in loader:
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += (images ** 2).sum(dim=(0, 2, 3))
        n_pixels += images.shape[0] * images.shape[2] * images.shape[3]

    mean = (channel_sum / n_pixels).tolist()
    var = (channel_sq_sum / n_pixels) - torch.tensor(mean) ** 2
    std = torch.sqrt(var).tolist()

    with open(_STATS_PATH, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=4)
    return mean, std


def load_dataloader(seed: int = 0, batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """
    概要: 学習用および検証用のデータローダーをインスタンス化するための関数．Imagewoofの
        公式train/val分割をそのまま用いる．
    引数:
        seed (int) = 0．学習用データの並び（シャッフル）を固定する乱数シード．
        batch_size (int) = 64．データローダーのミニバッチサイズ．
    戻り値:
        train_dataloader (torch.utils.data.DataLoader)．学習用データローダー．
        test_dataloader (torch.utils.data.DataLoader)．検証用データローダー（公式val分割）．
    """
    set_seed(seed)

    _download_and_extract_raw_data()
    mean, std = _compute_channel_stats()

    transform = transforms.Compose([
        transforms.Resize(RESIZE_SHORT_SIDE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = datasets.ImageFolder(_TRAIN_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(_VAL_DIR, transform=transform)

    generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
