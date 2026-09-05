"""
マッシュルームデータセット（UCI Machine Learning Repository）の読み込みおよび前処理を行うモジュール．

論文の4.1節（マッシュルームデータセットの二値分類問題）に対応するデータセットを構築する．
生データは22種類のカテゴリ変数（キノコの形状・色・匂いなどの離散的特徴量）とクラスラベル（食用/有毒）
から構成される．各カテゴリ変数を整数値へ変換（順序符号化）することで，特徴量次元数 d = 22 の
数値データセットを得る．
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

from machine_learning_utils import set_seed

RAW_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
)
RAW_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "datasets",
    "ex001_mushroom_svrg",
    "raw",
)
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
    df = pd.read_csv(RAW_DATA_PATH, header=None, names=COLUMN_NAMES)
    return df


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

    feature_columns = COLUMN_NAMES[1:]
    encoder = OrdinalEncoder(dtype=np.float64)
    X = encoder.fit_transform(df[feature_columns])

    return X, y


def load_dataloader(
    seed: int = 0, batch_size: int = 1
) -> tuple[DataLoader, DataLoader]:
    """
    概要: 学習用および検証用のデータローダーをインスタンス化するための関数．
        全データを9:1で学習用・検証用に分割し，学習用データの統計量で標準化した後，
        torch.utils.data.DataLoader を構築する．
    引数:
        seed (int) = 0．データ分割・シャッフルの再現性を担保する乱数シード．
        batch_size (int) = 1．データローダーのミニバッチサイズ．
    戻り値:
        train_dataloader (torch.utils.data.DataLoader)
        test_dataloader (torch.utils.data.DataLoader)
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
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, test_dataloader
