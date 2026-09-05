"""
実験1（Mushroomデータセットを用いた二値分類問題）のデータセットの定義に関するモジュール．

`.orders/order_021.md` の実験1に対応する．ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）
4.1節はMushroomデータセット（UCI Machine Learning Repository）を $ N=8124 $，$ d=22 $ として
扱っており，これは22種類のカテゴリ特徴量を（One-Hotではなく）各カテゴリを整数値へ変換する
順序符号化によって次元数を保った場合の値と一致する．したがって本モジュールは，`.orders/
order_020.md` 実施時の事前実験（`programs_old/ex001_mushroom_svrg/data.py`）と同一の前処理
（順序符号化 + 標準化）を踏襲する．
"""

import os
import sys

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed  # noqa: E402

EXPERIMENT_NAME = "ex001_mushroom_logistic"
RAW_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
)
RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, "datasets", EXPERIMENT_NAME, "raw")
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "agaricus-lepiota.data")

COLUMN_NAMES = [
    "class",
    "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat",
]

# 順序符号化後の特徴量次元数．ASAI SVRG論文4.1節の記載（d=22）と一致する．
N_FEATURES = 22


def download_raw_data():
    """
    概要: マッシュルームデータセットの生データ（CSV形式）をUCI Machine Learning Repositoryから
        ダウンロードし，ローカルに保存する．既にファイルが存在する場合はダウンロードを省略する．
    引数: なし
    戻り値: なし
    """
    if os.path.exists(RAW_DATA_PATH):
        return

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    response = requests.get(RAW_DATA_URL, timeout=30)
    response.raise_for_status()
    with open(RAW_DATA_PATH, "wb") as f:
        f.write(response.content)


def load_raw_dataframe() -> pd.DataFrame:
    """
    概要: マッシュルームデータセットの生データを読み込み，DataFrameとして返す．
    引数: なし
    戻り値: df (pandas.DataFrame)，形状 (8124, 23)．1列目がクラスラベル，残り22列がカテゴリ特徴量．
    """
    download_raw_data()
    return pd.read_csv(RAW_DATA_PATH, header=None, names=COLUMN_NAMES)


def build_features_and_labels() -> tuple[np.ndarray, np.ndarray]:
    """
    概要: 生データフレームから，順序符号化済みの特徴量行列と二値ラベルベクトルを構築する．
        欠損値「?」（stalk-root属性のみに出現）は独立したカテゴリとして扱う．
    引数: なし
    戻り値:
        X (numpy.ndarray)，形状 (N, d) = (8124, 22)．順序符号化された特徴量．
        y (numpy.ndarray)，形状 (N,)．二値ラベル（有毒 = 1，食用 = 0）．
    """
    df = load_raw_dataframe()
    y = (df["class"] == "p").astype(np.float64).to_numpy()
    X = OrdinalEncoder(dtype=np.float64).fit_transform(df[COLUMN_NAMES[1:]])
    return X, y


def load_dataloader(seed: int = 0, batch_size: int = 1) -> tuple[DataLoader, DataLoader]:
    """
    概要: 学習用および検証用のデータローダーをインスタンス化するための関数．
        全データを9:1で学習用・検証用に分割し，学習用データの統計量で標準化した後，
        `torch.utils.data.DataLoader` を構築する．平滑性定数 $ L $ は，この標準化後の
        特徴量に対して計算する必要がある（`train.py` の `compute_smoothness_constant` 参照）．
    引数:
        seed (int) = 0．データ分割・シャッフルの再現性を担保する乱数シード．
        batch_size (int) = 1．データローダーのミニバッチサイズ．
    戻り値:
        train_dataloader (torch.utils.data.DataLoader)．学習用データローダー．
        test_dataloader (torch.utils.data.DataLoader)．検証用データローダー．
    """
    set_seed(seed)

    X, y = build_features_and_labels()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float64),
        torch.tensor(y_train, dtype=torch.float64),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float64),
        torch.tensor(y_test, dtype=torch.float64),
    )

    generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
