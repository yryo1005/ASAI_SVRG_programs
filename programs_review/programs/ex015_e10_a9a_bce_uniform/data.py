"""
E4（付録B.1，a9a の MSE・BCE の再実施）のデータセットの定義に関するモジュール．

既存の実験用リポジトリの `programs/ex000_a9a_least_squares/data.py`（MSE）と
`programs/ex0001_a9a_logistic/data.py`（BCE）をコピーしたものであり，前処理は変更していない．
2つの既存のファイルは，保存先とモジュールの説明以外は同一である．変更点は次の2点のみである．
- `machine_learning_utils` の参照先を `programs_review/programs/common` にした．
- 生データの保存先を `programs_review/datasets/a9a/raw/a9a` にした（MSE と BCE で同じファイルを使う）．

a9a（LIBSVM Data，UCI Adult の二値分類向けの前処理版）は，1レコードあたり123次元の
One-Hot 符号化済みの特徴量（各成分は0/1）と，二値ラベル（元データでは -1/+1）からなる．
特徴量は標準化せず，0/1 のまま用いる．ラベルは $ \\{-1, +1\\} \\to \\{0, 1\\} $ に写像する．
"""

import os
import sys

import numpy as np
import requests
import torch
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

_PROGRAMS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
# programs_review/programs/common を参照する（programs_review の外のファイルは参照しない）．
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))

from machine_learning_utils import set_seed  # noqa: E402

RAW_DATA_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"
RAW_DATA_DIR = os.path.join(_REVIEW_ROOT, "datasets", "a9a", "raw")
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "a9a")

# a9a の特徴量の次元数．データファイル中に出現する最大の特徴量の添字は123．
N_FEATURES = 123


def download_raw_data():
    """
    概要: a9a の生データ（LIBSVM形式）を LIBSVM Data からダウンロードし，ローカルに保存する．
        既にファイルが存在する場合はダウンロードを省略する．
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
    概要: 生データ（LIBSVM形式）から特徴量行列とラベルベクトルを構築する．
        ラベル（-1：年収5万ドル以下，+1：年収5万ドル超）を {0, 1} へ写像する．
    引数: なし
    戻り値:
        A (numpy.ndarray)，形状 (N, d) = (32561, 123)．One-Hot 符号化済みの特徴量（0/1）．
        y (numpy.ndarray)，形状 (N,)．二値ラベル（{0.0, 1.0}）．
    """
    download_raw_data()
    A_sparse, y_raw = load_svmlight_file(RAW_DATA_PATH, n_features=N_FEATURES)
    A = A_sparse.toarray().astype(np.float64)
    y = ((y_raw + 1.0) / 2.0).astype(np.float64)
    return A, y


def load_dataloader(seed: int = 0, batch_size: int = 1) -> tuple[DataLoader, DataLoader]:
    """
    概要: 学習用および検証用のデータローダーをインスタンス化する．全データを 9:1 に層化分割する．
        特徴量は標準化せず 0/1 の値をそのまま用いる．
    引数:
        seed (int) = 0．データの分割・シャッフルの再現性を担保する乱数シード．
        batch_size (int) = 1．データローダーのミニバッチサイズ．
    戻り値:
        train_dataloader (torch.utils.data.DataLoader)．学習用データローダー（$ N_{\\mathrm{train}} = 29304 $）．
        test_dataloader (torch.utils.data.DataLoader)．検証用データローダー（$ N_{\\mathrm{test}} = 3257 $）．
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
