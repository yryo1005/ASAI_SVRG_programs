"""
実験4（WikiText-2を用いた単語レベル言語モデリング，Decoder-only Transformer）のデータセット
の定義に関するモジュール．

`.orders/order_030.md` 4節の指示に基づき，単語レベルの言語モデリングとして実装する．
WikiText-2は，PyTorch公式のword language modelサンプル
（https://github.com/pytorch/examples/tree/main/word_language_model）が配布する，単語分割
済み・レア語を`<unk>`に置換済みの前処理済みテキスト（`train.txt`，`valid.txt`，`test.txt`）
をそのまま利用する．このテキストは元のWikiText-2論文（Merity et al., 2016）の配布物と同一の
前処理（頻度3未満の単語を`<unk>`に置換）が既に適用されているため，本モジュールでは空白区切りの
トークナイズのみを行う．

語彙は学習用テキスト（`train.txt`）のみから構築する．検証・テスト用テキストに学習用語彙へ
存在しない単語は含まれない（本モジュールの検証で確認済み）ため，追加の未知語処理は不要である．

WikiText-2は元々train/valid/testに分割された形で配布されているため，これに従う．本プロジェクト
の`load_dataloader`インターフェース（学習用・検証用の2つのDataLoaderを返す）に合わせ，公式の
`valid.txt`を検証用として用いる（`test.txt`は本Stage Aでは使用せず，将来の最終評価用に温存する）．

コーパス全体（改行を含む全トークン）を単純に連結した1本のトークン列とみなし，固定長 $ T $ の
非重複チャンクに分割する．各チャンクを1サンプル $ n $ として扱う点は実験3
（`programs/ex003_tinyshakespeare_transformer/data.py`）と同一である．
"""

import os
import sys
import urllib.request

import torch
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed  # noqa: E402

EXPERIMENT_NAME = "ex004_wikitext2_transformer"
RAW_DATA_DIR = os.path.join(_PROJECT_ROOT, "datasets", EXPERIMENT_NAME, "raw")
_TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.txt")
_VALID_PATH = os.path.join(RAW_DATA_DIR, "valid.txt")
_TEST_PATH = os.path.join(RAW_DATA_DIR, "test.txt")
_DATA_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2/{split}.txt"
)

SEQUENCE_LENGTH = 64  # 1サンプルあたりの入力トークン数 T（5節の計算コスト見積もりに基づき決定）
UNK_TOKEN = "<unk>"


def _download_raw_text(split: str) -> str:
    """
    概要: WikiText-2の指定split（train/valid/test）の生テキストをダウンロードする（初回のみ）．
    引数: split (str)．"train"，"valid"，"test"のいずれか．
    戻り値: text (str)．該当splitの全文．
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    path = {"train": _TRAIN_PATH, "valid": _VALID_PATH, "test": _TEST_PATH}[split]
    if not os.path.exists(path):
        urllib.request.urlretrieve(_DATA_URL_TEMPLATE.format(split=split), path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_vocabulary(text: str) -> tuple[dict, list]:
    """
    概要: コーパスに出現する単語集合から，単語→ID・ID→単語の対応を構築する．
    引数: text (str)．コーパス全文（空白区切りで単語をトークナイズ済み）．
    戻り値:
        word_to_id (dict)．{単語: ID}．
        id_to_word (list)．ID順の単語リスト．
    """
    id_to_word = sorted(set(text.split()))
    word_to_id = {w: i for i, w in enumerate(id_to_word)}
    return word_to_id, id_to_word


class _ChunkedTextDataset(Dataset):
    """
    概要: トークンID化した単語列を，固定長 $ T+1 $ の非重複チャンクへ分割し，各チャンクを
        1サンプルとして扱うデータセット．入力は各チャンクの先頭 $ T $ トークン，教師信号は
        1つずらした $ T $ トークン（次単語予測）である．
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
    概要: 学習用および検証用のデータローダーをインスタンス化するための関数．WikiText-2の
        公式train/valid分割をそのまま用い，それぞれ独立に固定長 $ T+1 $ の非重複チャンクへ
        分割する．
    引数:
        seed (int) = 0．チャンクの並び（シャッフル）を固定する乱数シード．
        batch_size (int) = 128．データローダーのミニバッチサイズ．
    戻り値:
        train_dataloader (torch.utils.data.DataLoader)．学習用データローダー．
        test_dataloader (torch.utils.data.DataLoader)．検証用データローダー（WikiText-2の
            公式valid分割）．
    """
    set_seed(seed)

    train_text = _download_raw_text("train")
    valid_text = _download_raw_text("valid")
    word_to_id, _ = build_vocabulary(train_text)

    train_ids = [word_to_id[w] for w in train_text.split()]
    valid_ids = [word_to_id[w] for w in valid_text.split()]

    train_dataset = _ChunkedTextDataset(train_ids, SEQUENCE_LENGTH)
    test_dataset = _ChunkedTextDataset(valid_ids, SEQUENCE_LENGTH)

    generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
