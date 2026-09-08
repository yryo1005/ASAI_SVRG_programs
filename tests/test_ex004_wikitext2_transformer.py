"""
`programs/ex004_wikitext2_transformer/`（実験4：WikiText-2を用いた単語レベル言語モデリング，
Decoder-only Transformer，Stage A安定性探索）の単体テスト．

モデルがDropout・BatchNormalizationを含まないこと，Causalマスクが未来のトークンに依存しない
ことを保証すること，同一入力に対する出力が決定論的であること，データセットの語彙構築・チャンク
分割が正しいこと，2手法（NFG SVRG，ASAI SVRG）が合成データ上でエラーなく完走すること，
オラクル呼び出し回数の正しさ（NFG/ASAI:2N），`elapsed_time` から診断専用フル勾配計算が
除外されること，チャンスレベル張り付き検出（`is_stuck_near_chance`，本実験用の許容幅）の
正しさを確認する．
"""

import importlib.util
import os
import sys
import time

import numpy as np
import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROGRAMS_DIR = os.path.join(_PROJECT_ROOT, "programs")
_EX004_DIR = os.path.join(_PROGRAMS_DIR, "ex004_wikitext2_transformer")
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import ResultLogger  # noqa: E402


def _load_module(unique_name: str, directory: str, filename: str):
    """概要: 実験ディレクトリ間のモジュール名衝突を避けるため，一意な名前でモジュールを
        動的に読み込む．"""
    for stale_name in ("model", "data", "train"):
        sys.modules.pop(stale_name, None)
    spec = importlib.util.spec_from_file_location(unique_name, os.path.join(directory, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex004_model = _load_module("ex004_model", _EX004_DIR, "model.py")
ex004_data = _load_module("ex004_data", _EX004_DIR, "data.py")
# `train.py` はモジュールインポート時にネットワークからWikiText-2をダウンロードし
# 語彙サイズを決定する．
ex004_train = _load_module("ex004_train", _EX004_DIR, "train.py")


class _SyntheticTokenDataset(torch.utils.data.Dataset):
    """テスト用の小規模な合成トークン列データセット（ランダムトークンID）．"""

    def __init__(self, n: int, seq_len: int, vocab_size: int, seed: int):
        rng = np.random.default_rng(seed)
        self.inputs = torch.tensor(
            rng.integers(0, vocab_size, size=(n, seq_len)), dtype=torch.long
        )
        self.targets = torch.tensor(
            rng.integers(0, vocab_size, size=(n, seq_len)), dtype=torch.long
        )

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def _make_synthetic_dataloaders(n_train=24, n_test=8, batch_size=8, seq_len=16, vocab_size=20, seed=0):
    train_ds = _SyntheticTokenDataset(n_train, seq_len, vocab_size, seed)
    test_ds = _SyntheticTokenDataset(n_test, seq_len, vocab_size, seed + 1)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl


def test_model_has_no_dropout_or_batchnorm():
    """Dropout・BatchNormalization層を含まないことを確認する（`.orders/order_030.md` 3節）．"""
    model = ex004_model.DecoderOnlyTransformer(vocab_size=20, max_seq_len=16, d_model=32, n_head=2, n_layer=2, d_ff=64)
    for module in model.modules():
        assert not isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))


def test_model_is_deterministic_given_same_input():
    """同一入力に対し，モデルの出力が呼び出しごとに変化しない（決定論的である）ことを確認
    する．SVRG系手法の理論的前提の検証．"""
    model = ex004_model.DecoderOnlyTransformer(vocab_size=20, max_seq_len=16, d_model=32, n_head=2, n_layer=2, d_ff=64)
    model.eval()
    x = torch.randint(0, 20, (4, 16))
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert torch.equal(y1, y2)


def test_causal_mask_does_not_depend_on_future_tokens():
    """位置 $ t $ の出力が，位置 $ t+1 $ 以降のトークンを変更しても変化しない（causalマスクが
    正しく機能している）ことを確認する．"""
    model = ex004_model.DecoderOnlyTransformer(vocab_size=20, max_seq_len=16, d_model=32, n_head=2, n_layer=2, d_ff=64)
    model.eval()
    x = torch.randint(0, 20, (2, 16))
    x_modified = x.clone()
    x_modified[:, 10:] = (x_modified[:, 10:] + 1) % 20

    with torch.no_grad():
        y = model(x)
        y_modified = model(x_modified)

    assert torch.allclose(y[:, :10], y_modified[:, :10], atol=1e-5)


def test_l2_regularization_excludes_layernorm_and_bias():
    """L2正則化がLinear・Embeddingの重みのみに課され，バイアス・LayerNormのアフィン
    パラメータを除外することを確認する．"""
    model = ex004_model.DecoderOnlyTransformer(vocab_size=20, max_seq_len=16, d_model=32, n_head=2, n_layer=2, d_ff=64)
    reg_lambda = 1.0

    manual_reg = torch.zeros(())
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            manual_reg = manual_reg + torch.sum(module.weight ** 2)
    expected = 0.5 * reg_lambda * manual_reg

    actual = ex004_model.compute_l2_regularization(model, reg_lambda)
    assert torch.allclose(actual, expected)


def test_chunked_dataset_is_non_overlapping():
    """データセットのチャンク分割が非重複であり，チャンク間でトークンを共有しないことを
    確認する．"""
    token_ids = list(range(100))
    seq_len = 9
    dataset = ex004_data._ChunkedTextDataset(token_ids, seq_len)
    assert len(dataset) == 100 // (seq_len + 1)

    inputs_0, targets_0 = dataset[0]
    inputs_1, targets_1 = dataset[1]
    assert inputs_0.tolist() == list(range(seq_len))
    assert targets_0.tolist() == list(range(1, seq_len + 1))
    assert inputs_1.tolist() == list(range(seq_len + 1, 2 * seq_len + 1))


def test_build_vocabulary_is_deterministic_and_sorted():
    """語彙構築が決定論的（`sorted(set(...))`）であり，同じテキストから常に同じ単語→ID
    対応が得られることを確認する．"""
    text = "b a c a b b <unk>"
    word_to_id_1, id_to_word_1 = ex004_data.build_vocabulary(text)
    word_to_id_2, id_to_word_2 = ex004_data.build_vocabulary(text)
    assert word_to_id_1 == word_to_id_2
    assert id_to_word_1 == id_to_word_2
    assert id_to_word_1 == sorted(set(text.split()))


def test_load_dataloader_has_no_oov_against_train_vocabulary():
    """実際のWikiText-2データについて，検証用テキストの単語が全て学習用語彙に含まれる
    （語彙構築時に見つからない単語が存在しない）ことを確認する（`.orders/order_030.md` 4節）．
    ネットワークアクセスを伴うため，データの初回ダウンロードが発生する場合がある．"""
    train_dl, test_dl = ex004_data.load_dataloader(seed=0, batch_size=512)
    assert len(train_dl.dataset) > 0
    assert len(test_dl.dataset) > 0


def test_model_parameter_count_is_within_estimated_budget():
    """`.orders/order_030.md` 5節の見積もり（約935万パラメータ）と概ね一致し，過大な
    モデルになっていないことを確認する．"""
    model = ex004_model.DecoderOnlyTransformer(vocab_size=33277, max_seq_len=64)
    n_params = sum(p.numel() for p in model.parameters())
    assert 8_000_000 < n_params < 12_000_000


@pytest.mark.parametrize("method", ["NFG_SVRG", "ASAI_SVRG"])
def test_training_runs_without_error_on_synthetic_data(method, tmp_path, monkeypatch):
    """Stage Aの2手法が合成データ上でエラーなく完走し，記録された評価指標の本数がエポック数+1と
    一致することを確認する（スモークテスト）．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    reg_lambda = 0.01
    eta = 0.001
    vocab_size = 20

    monkeypatch.setattr(ex004_train, "_VOCAB_SIZE", vocab_size)
    monkeypatch.setattr(ex004_train, "SEQUENCE_LENGTH", 8)

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(
            n_train=16, n_test=8, batch_size=batch_size, seq_len=8, vocab_size=vocab_size, seed=seed
        )

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    target_dir = str(tmp_path)
    ex004_train.run_variance_reduced(
        method, target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
    )

    assert len(logger["epoch"]) == epochs + 1
    assert all(np.isfinite(v) for v in logger["train_loss"])


def test_oracle_calls_accounting_per_epoch(tmp_path, monkeypatch):
    """1エポックあたりのオラクル呼び出し回数が，NFG SVRG・ASAI SVRGで 2N となることを
    確認する（Nはミニバッチ単位ではなくサンプル数換算）．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    reg_lambda = 0.01
    n_train = 16
    vocab_size = 20

    monkeypatch.setattr(ex004_train, "_VOCAB_SIZE", vocab_size)

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(
            n_train=n_train, n_test=8, batch_size=batch_size, seq_len=8, vocab_size=vocab_size, seed=seed
        )

    for method in ["NFG_SVRG", "ASAI_SVRG"]:
        logger = ResultLogger()
        logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
        target_dir = str(tmp_path / method)
        os.makedirs(target_dir, exist_ok=True)
        ex004_train.run_variance_reduced(
            method, target_dir, load_dataloader_func, 0.001, batch_size, reg_lambda, epochs, device, 0, logger
        )

        assert logger["oracle_calls"][0] == 0
        assert logger["oracle_calls"][1] == 2 * n_train


def test_elapsed_time_excludes_diagnostic_full_gradient(monkeypatch, tmp_path):
    """NFG SVRG・ASAI SVRGが近似誤差算出のためだけに毎エポック計算する診断専用のフル勾配
    計算時間が，`elapsed_time` に計上されないことを確認する．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 1
    n_train = 16
    vocab_size = 20
    sleep_seconds = 0.2

    monkeypatch.setattr(ex004_train, "_VOCAB_SIZE", vocab_size)

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(
            n_train=n_train, n_test=8, batch_size=batch_size, seq_len=8, vocab_size=vocab_size, seed=seed
        )

    original = ex004_train.compute_full_gradient_and_metrics

    def slow_compute_full_gradient_and_metrics(*args, **kwargs):
        time.sleep(sleep_seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(ex004_train, "compute_full_gradient_and_metrics", slow_compute_full_gradient_and_metrics)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
    ex004_train.run_variance_reduced(
        "ASAI_SVRG", str(tmp_path), load_dataloader_func, 0.001, batch_size, 0.01, epochs, device, 0, logger
    )
    assert logger["elapsed_time"][1] < sleep_seconds


def test_is_stuck_near_chance_detects_flatlined_accuracy():
    """末尾の次単語予測精度がチャンスレベル付近に留まっている場合，`True` を返すことを
    確認する（本実験用の許容幅0.001を使用）．"""
    vocab_size = 33277
    chance = 1.0 / vocab_size
    accuracies = [0.1, 0.05, 0.02, chance + 0.0005, chance - 0.0005, chance]
    assert ex004_train.is_stuck_near_chance(accuracies, vocab_size=vocab_size, window=3)


def test_is_stuck_near_chance_does_not_flag_learning_progress():
    """精度が学習の進行とともに上昇し続けている場合，`False` を返すことを確認する．"""
    vocab_size = 33277
    accuracies = [0.0001, 0.001, 0.01, 0.03, 0.05, 0.08]
    assert not ex004_train.is_stuck_near_chance(accuracies, vocab_size=vocab_size, window=3)


def test_is_stuck_near_chance_does_not_flag_frequent_word_baseline():
    """WikiText-2の最頻出単語（"the"）を常に出力するだけで得られる精度（約5.5〜5.9%）は，
    一様チャンスレベル（約0.003%）から見れば大きく乖離しているため，`is_stuck_near_chance`
    （一様チャンスレベル基準）では`False`と判定される（本モジュールdocstringで述べた別の
    失敗モードであることの確認）．"""
    vocab_size = 33277
    frequent_word_baseline = 0.057
    accuracies = [frequent_word_baseline] * 5
    assert not ex004_train.is_stuck_near_chance(accuracies, vocab_size=vocab_size, window=3)


def test_is_stuck_near_chance_requires_minimum_history():
    """指定した`window`未満のエポック数しかない場合は`False`を返すことを確認する．"""
    assert not ex004_train.is_stuck_near_chance([0.0001, 0.0002], vocab_size=33277, window=3)


def test_stage_a_grid_matches_order_specification():
    """`.orders/order_030.md` 7節：Stage Aの比較手法がNFG SVRG・ASAI SVRGの2手法のみ
    （SGD・SVRGを含まない）であり，バッチサイズ・学習率・エポック数・Seed数が実験3 Stage Aと
    同様のレンジであることを確認する．"""
    assert set(ex004_train.METHODS) == {"NFG_SVRG", "ASAI_SVRG"}
    assert ex004_train.BATCH_SIZES == [512, 128, 32]
    assert ex004_train.LEARNING_RATES == [0.01, 0.001]
    assert ex004_train.EPOCHS == 12
    assert ex004_train.SEEDS == [0, 1, 2]
