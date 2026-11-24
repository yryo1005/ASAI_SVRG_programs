"""
E0（ImageWoof）のデータセットの定義．

前処理は既存の実験（`ex0051_imagewoof_resnet18_bs128_longrun`，`ex005_imagewoof_resnet18` の
`data.py`）と同一である．fast.aiが配布する `imagewoof2-320` の公式のtrain/val分割
（$ N_{\\mathrm{train}} = 9025 $，$ N_{\\mathrm{test}} = 3929 $）を用い，短辺を256にリサイズした後に
中央の $ 224\\times224 $ を切り出し，学習データから計算したチャネルごとの平均・標準偏差で
正規化する．データ拡張は行わない．

前処理は決定論的であるため，前処理後の画像をfloat32の配列として `datasets/imagewoof/cache/` に
1回だけ保存し（学習データ約5.4GB，テストデータ約2.4GB），学習では `numpy.memmap` で
ミニバッチごとに読み出す．複数のプロセスはOSのページキャッシュを共有するため，並列実行しても
メモリの使用量は増えない．
"""

import json
import os
import sys
import tarfile
import urllib.request

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROGRAMS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))

from cached_loader import CachedTensorLoader  # noqa: E402
from machine_learning_utils import set_seed  # noqa: E402

RAW_DATA_DIR = os.path.join(_REVIEW_ROOT, "datasets", "imagewoof", "raw")
CACHE_DIR = os.path.join(_REVIEW_ROOT, "datasets", "imagewoof", "cache")
_ARCHIVE_PATH = os.path.join(RAW_DATA_DIR, "imagewoof2-320.tgz")
_EXTRACT_DIR = os.path.join(RAW_DATA_DIR, "imagewoof2-320")
_TRAIN_DIR = os.path.join(_EXTRACT_DIR, "train")
_VAL_DIR = os.path.join(_EXTRACT_DIR, "val")
_STATS_PATH = os.path.join(CACHE_DIR, "channel_stats.json")
_DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz"

IMAGE_SIZE = 224
RESIZE_SHORT_SIDE = 256
NUM_CLASSES = 10
CACHE_BUILD_WORKERS = 16


def _download_and_extract_raw_data() -> None:
    """
    概要: `imagewoof2-320` をダウンロード・展開する（初回のみ）．
    引数: なし
    戻り値: なし
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    if os.path.exists(_TRAIN_DIR) and os.path.exists(_VAL_DIR):
        return
    if not os.path.exists(_ARCHIVE_PATH):
        urllib.request.urlretrieve(_DATA_URL, _ARCHIVE_PATH)
    with tarfile.open(_ARCHIVE_PATH) as tar:
        tar.extractall(RAW_DATA_DIR)


def _compute_channel_stats() -> tuple:
    """
    概要: 学習用画像のチャネルごとの平均・標準偏差を，既存の実装と同じ手順（ミニバッチサイズ64，
        float32での逐次加算，1スレッド）で計算する．結果は `channel_stats.json` に保存し，
        2回目以降はこれを読み込む．
    引数: なし
    戻り値:
        mean (list of float)．長さ3（RGB）．
        std (list of float)．長さ3（RGB）．
    """
    if os.path.exists(_STATS_PATH):
        with open(_STATS_PATH) as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]

    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
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
    for images, _ in tqdm(loader, desc="channel stats", leave=False):
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += (images ** 2).sum(dim=(0, 2, 3))
        n_pixels += images.shape[0] * images.shape[2] * images.shape[3]
    torch.set_num_threads(num_threads)

    mean = (channel_sum / n_pixels).tolist()
    var = (channel_sq_sum / n_pixels) - torch.tensor(mean) ** 2
    std = torch.sqrt(var).tolist()
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_STATS_PATH, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=4)
    return mean, std


def build_transform(mean, std) -> transforms.Compose:
    """
    概要: 既存の実験と同一の前処理を構築する．
    引数:
        mean (list of float)．チャネルごとの平均．
        std (list of float)．チャネルごとの標準偏差．
    戻り値: transform (torchvision.transforms.Compose)．
    """
    return transforms.Compose([
        transforms.Resize(RESIZE_SHORT_SIDE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _cache_paths(split: str) -> tuple:
    """
    概要: キャッシュファイルのパスを返す．
    引数: split (str)．"train" または "test"．
    戻り値: (inputs_path, targets_path) の組．
    """
    return os.path.join(CACHE_DIR, f"{split}_inputs.npy"), os.path.join(CACHE_DIR, f"{split}_targets.npy")


def _build_split_cache(split: str, image_dir: str, transform) -> None:
    """
    概要: 1つの分割について，前処理後の全画像をfloat32の `.npy` に書き出す．
    引数:
        split (str)．"train" または "test"．
        image_dir (str)．`ImageFolder` 形式の画像ディレクトリ．
        transform (torchvision.transforms.Compose)．前処理．
    戻り値: なし
    """
    inputs_path, targets_path = _cache_paths(split)
    dataset = datasets.ImageFolder(image_dir, transform=transform)
    tmp_path = inputs_path + ".tmp.npy"
    inputs = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32,
                                       shape=(len(dataset), 3, IMAGE_SIZE, IMAGE_SIZE))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=CACHE_BUILD_WORKERS)
    offset = 0
    for images, _ in tqdm(loader, desc=f"cache {split}", leave=False):
        inputs[offset:offset + images.shape[0]] = images.numpy()
        offset += images.shape[0]
    inputs.flush()
    del inputs
    np.save(targets_path, np.asarray(dataset.targets, dtype=np.int64))
    os.replace(tmp_path, inputs_path)


def prepare_cache() -> None:
    """
    概要: データの取得，チャネル統計量の計算，前処理後の画像のキャッシュの作成を（未作成の場合のみ）
        行う．並列に学習を始める前に，メインプロセスで1回だけ呼び出す．
    引数: なし
    戻り値: なし
    """
    _download_and_extract_raw_data()
    os.makedirs(CACHE_DIR, exist_ok=True)
    mean, std = _compute_channel_stats()
    transform = build_transform(mean, std)
    for split, image_dir in (("train", _TRAIN_DIR), ("test", _VAL_DIR)):
        if not os.path.exists(_cache_paths(split)[0]):
            _build_split_cache(split, image_dir, transform)


def load_dataloader(seed: int = 0, batch_size: int = 128, device: torch.device = None):
    """
    概要: 学習用とテスト用のローダーをインスタンス化する．学習用のローダーは既存の実装と同じく
        `torch.Generator().manual_seed(seed)` を用いてエポックごとにシャッフルする．
    引数:
        seed (int) = 0．シャッフルの乱数シード．
        batch_size (int) = 128．ミニバッチサイズ．
        device (torch.device) = None．ミニバッチを転送するデバイス．
    戻り値:
        train_loader (CachedTensorLoader)．学習用のローダー．
        test_loader (CachedTensorLoader)．テスト用のローダー（シャッフルなし）．
    """
    set_seed(seed)
    prepare_cache()
    loaders = []
    for split, shuffle in (("train", True), ("test", False)):
        inputs_path, targets_path = _cache_paths(split)
        inputs = np.load(inputs_path, mmap_mode="r")
        targets = torch.from_numpy(np.load(targets_path))
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        loaders.append(CachedTensorLoader(inputs, targets, batch_size, shuffle=shuffle,
                                          generator=generator, device=device))
    return loaders[0], loaders[1]
