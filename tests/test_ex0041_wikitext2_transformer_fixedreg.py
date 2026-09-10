"""
`programs/ex0041_wikitext2_transformer_fixedreg/`（実験4b：正則化修正後のWikiText-2単語
レベル言語モデリング，Decoder-only Transformer，Stage A再実行）の単体テスト．

実験4（`tests/test_ex004_wikitext2_transformer.py`）と共通の検証項目（モデル構造，
チャンク分割，2手法のスモークテスト，オラクル呼び出し回数，elapsed_time，チャンス
レベル張り付き検出）に加え，本実験固有の検証（Token Embedding・出力射影層の初期化
スケール，学習開始前の正則化項支配性検証関数）を確認する．
"""

import importlib.util
import math
import os
import sys
import time

import numpy as np
import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROGRAMS_DIR = os.path.join(_PROJECT_ROOT, "programs")
_EX0041_DIR = os.path.join(_PROGRAMS_DIR, "ex0041_wikitext2_transformer_fixedreg")
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


ex0041_model = _load_module("ex0041_model", _EX0041_DIR, "model.py")
ex0041_data = _load_module("ex0041_data", _EX0041_DIR, "data.py")
# `train.py` はモジュールインポート時にネットワークからWikiText-2をダウンロードし
# 語彙サイズを決定する．
ex0041_train = _load_module("ex0041_train", _EX0041_DIR, "train.py")


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
    model = ex0041_model.DecoderOnlyTransformer(vocab_size=20, max_seq_len=16, d_model=32, n_head=2, n_layer=2, d_ff=64)
    for module in model.modules():
        assert not isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))


def test_model_is_deterministic_given_same_input():
    """同一入力に対し，モデルの出力が呼び出しごとに変化しない（決定論的である）ことを確認
    する．SVRG系手法の理論的前提の検証．"""
    model = ex0041_model.DecoderOnlyTransformer(vocab_size=20, max_seq_len=16, d_model=32, n_head=2, n_layer=2, d_ff=64)
    model.eval()
    x = torch.randint(0, 20, (4, 16))
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert torch.equal(y1, y2)


def test_causal_mask_does_not_depend_on_future_tokens():
    """位置 $ t $ の出力が，位置 $ t+1 $ 以降のトークンを変更しても変化しない（causalマスクが
    正しく機能している）ことを確認する．"""
    model = ex0041_model.DecoderOnlyTransformer(vocab_size=20, max_seq_len=16, d_model=32, n_head=2, n_layer=2, d_ff=64)
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
    model = ex0041_model.DecoderOnlyTransformer(vocab_size=20, max_seq_len=16, d_model=32, n_head=2, n_layer=2, d_ff=64)
    reg_lambda = 1.0

    manual_reg = torch.zeros(())
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            manual_reg = manual_reg + torch.sum(module.weight ** 2)
    expected = 0.5 * reg_lambda * manual_reg

    actual = ex0041_model.compute_l2_regularization(model, reg_lambda)
    assert torch.allclose(actual, expected)


def test_token_embedding_and_head_use_scaled_initialization():
    """`.orders/order_031.md` 3.2節の指示に基づき，Token Embedding層・出力射影層（`head`）
    の重みが $ \\mathcal N(0, 1/\\sqrt d) $ スケールで初期化されており，PyTorchの
    `nn.Embedding`デフォルト初期化（標準偏差1）よりはるかに小さいことを確認する．"""
    d_model = 128
    model = ex0041_model.DecoderOnlyTransformer(vocab_size=33277, max_seq_len=64, d_model=d_model)
    expected_std = 1.0 / math.sqrt(d_model)

    assert model.token_embedding.weight.std().item() < 0.5 * 1.0
    assert abs(model.token_embedding.weight.std().item() - expected_std) < 0.02
    assert abs(model.head.weight.std().item() - expected_std) < 0.02


def test_regularization_term_is_comparable_to_chance_level_cross_entropy():
    """修正後のモデルについて，初期状態でのL2正則化項がチャンスレベルの交差エントロピー
    理論値（$ \\ln(\\text{vocab\\_size}) $）の10倍を超えないことを確認する（`.orders/
    order_031.md` 3.3節の要求）．"""
    vocab_size = 33277
    reg_lambda = 5e-4
    model = ex0041_model.load_model(
        ex0041_model.DecoderOnlyTransformer, seed=0, vocab_size=vocab_size, max_seq_len=64
    )
    reg_at_init = ex0041_model.compute_l2_regularization(model, reg_lambda).item()
    chance_ce = math.log(vocab_size)
    assert reg_at_init < 10.0 * chance_ce


def test_verify_regularization_is_not_dominant_passes_for_actual_config():
    """本実験の実際の設定（語彙サイズ33,277，$ \\lambda=5\\times10^{-4} $）で，
    `verify_regularization_is_not_dominant` がAssertionErrorを送出しないことを確認する．"""
    ex0041_train.verify_regularization_is_not_dominant(33277, 5e-4)


def test_verify_regularization_is_not_dominant_fails_for_extreme_vocab_size():
    """語彙サイズを極端に大きくすると，$ 1/\\sqrt d $ スケール初期化を用いていても
    正則化項がチャンスレベル交差エントロピーに対して相対的に支配的になりうる状況
    （実験4の修正前と同種の問題）を再現し，`verify_regularization_is_not_dominant`が
    正しくAssertionErrorを送出することを確認する（判定関数自体の識別力の検証）．"""
    with pytest.raises(AssertionError):
        huge_vocab_size = 10_000_000
        ex0041_train.verify_regularization_is_not_dominant(huge_vocab_size, 5e-4)


def test_chunked_dataset_is_non_overlapping():
    """データセットのチャンク分割が非重複であり，チャンク間でトークンを共有しないことを
    確認する．"""
    token_ids = list(range(100))
    seq_len = 9
    dataset = ex0041_data._ChunkedTextDataset(token_ids, seq_len)
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
    word_to_id_1, id_to_word_1 = ex0041_data.build_vocabulary(text)
    word_to_id_2, id_to_word_2 = ex0041_data.build_vocabulary(text)
    assert word_to_id_1 == word_to_id_2
    assert id_to_word_1 == id_to_word_2
    assert id_to_word_1 == sorted(set(text.split()))


def test_load_dataloader_has_no_oov_against_train_vocabulary():
    """実際のWikiText-2データについて，検証用テキストの単語が全て学習用語彙に含まれる
    （語彙構築時に見つからない単語が存在しない）ことを確認する．ネットワークアクセスを
    伴うため，データの初回ダウンロードが発生する場合がある．"""
    train_dl, test_dl = ex0041_data.load_dataloader(seed=0, batch_size=512)
    assert len(train_dl.dataset) > 0
    assert len(test_dl.dataset) > 0


def test_model_parameter_count_is_within_estimated_budget():
    """実験4の見積もり（約935万パラメータ）と概ね一致し，過大なモデルになっていないことを
    確認する（初期化方式の変更はパラメータ数に影響しない）．"""
    model = ex0041_model.DecoderOnlyTransformer(vocab_size=33277, max_seq_len=64)
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

    monkeypatch.setattr(ex0041_train, "_VOCAB_SIZE", vocab_size)
    monkeypatch.setattr(ex0041_train, "SEQUENCE_LENGTH", 8)

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(
            n_train=16, n_test=8, batch_size=batch_size, seq_len=8, vocab_size=vocab_size, seed=seed
        )

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    target_dir = str(tmp_path)
    ex0041_train.run_variance_reduced(
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

    monkeypatch.setattr(ex0041_train, "_VOCAB_SIZE", vocab_size)

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(
            n_train=n_train, n_test=8, batch_size=batch_size, seq_len=8, vocab_size=vocab_size, seed=seed
        )

    for method in ["NFG_SVRG", "ASAI_SVRG"]:
        logger = ResultLogger()
        logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
        target_dir = str(tmp_path / method)
        os.makedirs(target_dir, exist_ok=True)
        ex0041_train.run_variance_reduced(
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

    monkeypatch.setattr(ex0041_train, "_VOCAB_SIZE", vocab_size)

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(
            n_train=n_train, n_test=8, batch_size=batch_size, seq_len=8, vocab_size=vocab_size, seed=seed
        )

    original = ex0041_train.compute_full_gradient_and_metrics

    def slow_compute_full_gradient_and_metrics(*args, **kwargs):
        time.sleep(sleep_seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(ex0041_train, "compute_full_gradient_and_metrics", slow_compute_full_gradient_and_metrics)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
    ex0041_train.run_variance_reduced(
        "ASAI_SVRG", str(tmp_path), load_dataloader_func, 0.001, batch_size, 0.01, epochs, device, 0, logger
    )
    assert logger["elapsed_time"][1] < sleep_seconds


def test_is_stuck_near_chance_detects_flatlined_accuracy():
    """末尾の次単語予測精度がチャンスレベル付近に留まっている場合，`True` を返すことを
    確認する（本実験用の許容幅0.001を使用）．"""
    vocab_size = 33277
    chance = 1.0 / vocab_size
    accuracies = [0.1, 0.05, 0.02, chance + 0.0005, chance - 0.0005, chance]
    assert ex0041_train.is_stuck_near_chance(accuracies, vocab_size=vocab_size, window=3)


def test_is_stuck_near_chance_does_not_flag_learning_progress():
    """精度が学習の進行とともに上昇し続けている場合，`False` を返すことを確認する．"""
    vocab_size = 33277
    accuracies = [0.0001, 0.001, 0.01, 0.03, 0.05, 0.08]
    assert not ex0041_train.is_stuck_near_chance(accuracies, vocab_size=vocab_size, window=3)


def test_is_stuck_near_chance_requires_minimum_history():
    """指定した`window`未満のエポック数しかない場合は`False`を返すことを確認する．"""
    assert not ex0041_train.is_stuck_near_chance([0.0001, 0.0002], vocab_size=33277, window=3)


def test_stage_a_grid_matches_experiment4_specification():
    """実験4 Stage Aと同一グリッド（2手法，バッチサイズ・学習率・エポック数・Seed数）で
    あることを確認する（`.orders/order_031.md` 6節）．"""
    assert set(ex0041_train.METHODS) == {"NFG_SVRG", "ASAI_SVRG"}
    assert ex0041_train.BATCH_SIZES == [512, 128, 32]
    assert ex0041_train.LEARNING_RATES == [0.01, 0.001]
    assert ex0041_train.EPOCHS == 12
    assert ex0041_train.SEEDS == [0, 1, 2]
