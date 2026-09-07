"""
実験3 Stage C（長期学習によるASAI SVRGの効率性優位性の持続性検証）のデータセットの定義に
関するモジュール．

`.orders/order_029.md` の指示に基づき，`programs/ex003_tinyshakespeare_transformer/data.py`
と同一の前処理（文字レベルのトークナイズ，固定長 $ T+1=129 $ の非重複チャンクへの分割，
コーパスの前方90%を学習用・後方10%を検証用とする分割）を用いる．
"""

import os
import sys
import urllib.request

import torch
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed  # noqa: E402

EXPERIMENT_NAME = "ex0031_tinyshakespeare_transformer_longrun"
RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, "datasets", EXPERIMENT_NAME, "raw")
RAW_TEXT_PATH = os.path.join(RAW_DATA_DIR, "input.txt")
_DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

SEQUENCE_LENGTH = 128  # 1サンプルあたりの入力トークン数 T
TRAIN_FRACTION = 0.9


def _download_raw_text() -> str:
    """
    概要: Tiny Shakespeareの生テキストをダウンロードする（初回のみ）．
    引数: なし
    戻り値: text (str)．コーパス全文．
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_TEXT_PATH):
        urllib.request.urlretrieve(_DATA_URL, RAW_TEXT_PATH)
    with open(RAW_TEXT_PATH, encoding="utf-8") as f:
        return f.read()


def build_vocabulary(text: str) -> tuple[dict, list]:
    """
    概要: コーパスに出現する文字集合から，文字→ID・ID→文字の対応を構築する．
    引数: text (str)．コーパス全文．
    戻り値:
        char_to_id (dict)．{文字: ID}．
        id_to_char (list)．ID順の文字リスト．
    """
    id_to_char = sorted(set(text))
    char_to_id = {ch: i for i, ch in enumerate(id_to_char)}
    return char_to_id, id_to_char


class _ChunkedTextDataset(Dataset):
    """
    概要: トークンID化した文字列を，固定長 $ T+1 $ の非重複チャンクへ分割し，各チャンクを
        1サンプルとして扱うデータセット．入力は各チャンクの先頭 $ T $ トークン，教師信号は
        1つずらした $ T $ トークン（次文字予測）である．
    """

    def __init__(self, token_ids: list, sequence_length: int):
        self.sequence_length = sequence_length
        chunk_length = sequence_length + 1
        n_chunks = len(token_ids) // chunk_length
        usable_length = n_chunks * chunk_length
        tokens = torch.tensor(token_ids[:usable_length], dtype=torch.long)
        self.chunks = tokens.view(n_chunks, chunk_length)

    def __len__(self):
        return self.chunks.shape[0]

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        inputs = chunk[: self.sequence_length]
        targets = chunk[1:]
        return inputs, targets


def load_dataloader(seed: int = 0, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """
    概要: 学習用および検証用のデータローダーをインスタンス化するための関数．コーパスの
        前方90%を学習用，後方10%を検証用とし，それぞれ独立に固定長 $ T+1 $ の非重複
        チャンクへ分割する．
    引数:
        seed (int) = 0．チャンクの並び（シャッフル）を固定する乱数シード．
        batch_size (int) = 128．データローダーのミニバッチサイズ．
    戻り値:
        train_dataloader (torch.utils.data.DataLoader)．学習用データローダー．
        test_dataloader (torch.utils.data.DataLoader)．検証用データローダー．
    """
    set_seed(seed)

    text = _download_raw_text()
    char_to_id, _ = build_vocabulary(text)
    token_ids = [char_to_id[ch] for ch in text]

    split_index = int(len(token_ids) * TRAIN_FRACTION)
    train_ids = token_ids[:split_index]
    test_ids = token_ids[split_index:]

    train_dataset = _ChunkedTextDataset(train_ids, SEQUENCE_LENGTH)
    test_dataset = _ChunkedTextDataset(test_ids, SEQUENCE_LENGTH)

    generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
