"""
実験0-1（a9aデータセットを用いたL2正則化付きロジスティック回帰）のデータセットの定義に関する
モジュール．

`.orders/order_041.md` に対応する．実験0（`programs/ex000_a9a_least_squares/`）と同一の
a9a（LIBSVM Data，UCI Adult所得予測データセットの二値分類向け前処理版）を，実験0と同一の
前処理で読み込む．a9aは，1レコードあたり123次元のカテゴリ変数（One-Hotエンコード済み，
各成分は0/1の二値）と，二値ラベル（年収が5万ドルを超えるか否か，元データでは-1/+1）から
構成される．

実験0-1は誤差関数とモデルのみを実験1（`programs/ex001_mushroom_logistic/`）に揃えた実験で
あり，データ条件は実験0と同一に保つ．実験1で行った標準化は，Mushroomデータセットの順序符号化
（整数値）に対する前処理であるため，One-Hotエンコード済みで各成分が0/1に揃っているa9aには
適用せず，特徴量を無加工（0/1の二値のまま）で用いる．
"""

import os
import sys

import numpy as np
import requests
import torch
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed  # noqa: E402

EXPERIMENT_NAME = "ex0001_a9a_logistic"
RAW_DATA_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"
RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, "datasets", EXPERIMENT_NAME, "raw")
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "a9a")

# a9a（LIBSVM Data）の特徴量次元数．データファイル中に出現する最大の特徴量インデックスは123．
N_FEATURES = 123


def download_raw_data():
    """
    概要: a9aデータセットの生データ（LIBSVM形式）をLIBSVM Dataのリポジトリからダウンロードし，
        ローカルに保存する．既にファイルが存在する場合はダウンロードを省略する．
    引数: なし
    戻り値: なし
    """
    if os.path.exists(RAW_DATA_PATH):
        return

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    response = requests.get(RAW_DATA_URL, timeout=120)
    response.raise_for_status()
    with open(RAW_DATA_PATH, "wb") as f:
        f.write(response.content)


def build_features_and_labels() -> tuple[np.ndarray, np.ndarray]:
    """
    概要: 生データ（LIBSVM形式）から，特徴量行列とラベルベクトルを構築する．
        二値交差エントロピー損失は教師信号 y_i ∈ {0, 1} を前提とするため，元データのラベル
        （-1: 年収5万ドル以下，+1: 年収5万ドル超）を {0, 1} へ写像する．
    引数: なし
    戻り値:
        A (numpy.ndarray)，形状 (N, d) = (32561, 123)．One-Hotエンコード済みの特徴量（0/1）．
        y (numpy.ndarray)，形状 (N,)．二値ラベル（{0.0, 1.0}）．
    """
    download_raw_data()
    A_sparse, y_raw = load_svmlight_file(RAW_DATA_PATH, n_features=N_FEATURES)
    A = A_sparse.toarray().astype(np.float64)
    y = ((y_raw + 1.0) / 2.0).astype(np.float64)
    return A, y


def load_dataloader(seed: int = 0, batch_size: int = 1) -> tuple[DataLoader, DataLoader]:
    """
    概要: 学習用および検証用のデータローダーをインスタンス化するための関数．
        全データを9:1で学習用・検証用に分割する．実験0とデータ条件を揃えるため，特徴量は
        標準化等を行わず0/1の値をそのまま用いる．
    引数:
        seed (int) = 0．データ分割・シャッフルの再現性を担保する乱数シード．
        batch_size (int) = 1．データローダーのミニバッチサイズ（オンライン学習の設定に
            合わせ1）．
    戻り値:
        train_dataloader (torch.utils.data.DataLoader)．学習用データローダー．
        test_dataloader (torch.utils.data.DataLoader)．検証用データローダー．
    """
    set_seed(seed)

    A, y = build_features_and_labels()
    A_train, A_test, y_train, y_test = train_test_split(
        A, y, test_size=0.1, random_state=seed, stratify=y
    )

    train_dataset = TensorDataset(
        torch.tensor(A_train, dtype=torch.float64),
        torch.tensor(y_train, dtype=torch.float64),
    )
    test_dataset = TensorDataset(
        torch.tensor(A_test, dtype=torch.float64),
        torch.tensor(y_test, dtype=torch.float64),
    )

    generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
