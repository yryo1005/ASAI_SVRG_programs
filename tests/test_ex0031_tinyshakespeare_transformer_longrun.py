"""
`programs/ex0031_tinyshakespeare_transformer_longrun/`（実験3 Stage C：長期学習による
ASAI SVRGの効率性優位性の持続性検証）の単体テスト．

モデルがDropout・BatchNormalizationを含まないこと，Causalマスクが未来のトークンに依存しない
ことを保証すること，同一入力に対する出力が決定論的であること，データセットのチャンク分割が
非重複であること，4手法（SGD，SVRG，NFG SVRG，ASAI SVRG）が合成データ上でエラーなく完走
すること，オラクル呼び出し回数の正しさ，`elapsed_time` から診断専用フル勾配計算が除外される
こと，チャンスレベル張り付き検出（`is_stuck_near_chance`），プラトー判定
（`compute_trailing_relative_change`）の正しさ，長期学習中に崩壊した場合に学習が打ち切られ
`is_run_completed` がこれを完了済みとして扱うことを確認する．
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
_EX0031_DIR = os.path.join(_PROGRAMS_DIR, "ex0031_tinyshakespeare_transformer_longrun")
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


ex0031_model = _load_module("ex0031_model", _EX0031_DIR, "model.py")
ex0031_data = _load_module("ex0031_data", _EX0031_DIR, "data.py")
# `train.py` はモジュールインポート時にネットワークからTiny Shakespeareをダウンロードし
# 語彙サイズを決定する．
ex0031_train = _load_module("ex0031_train", _EX0031_DIR, "train.py")


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
    """Dropout・BatchNormalization層を含まないことを確認する．"""
    model = ex0031_model.DecoderOnlyTransformer(vocab_size=20, max_seq_len=16, d_model=32, n_head=2, n_layer=2, d_ff=64)
    for module in model.modules():
        assert not isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))


def test_model_is_deterministic_given_same_input():
    """同一入力に対し，モデルの出力が呼び出しごとに変化しないことを確認する．"""
    model = ex0031_model.DecoderOnlyTransformer(vocab_size=20, max_seq_len=16, d_model=32, n_head=2, n_layer=2, d_ff=64)
    model.eval()
    x = torch.randint(0, 20, (4, 16))
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert torch.equal(y1, y2)


def test_causal_mask_does_not_depend_on_future_tokens():
    """位置 $ t $ の出力が，位置 $ t+1 $ 以降のトークンを変更しても変化しないことを確認する．"""
    model = ex0031_model.DecoderOnlyTransformer(vocab_size=20, max_seq_len=16, d_model=32, n_head=2, n_layer=2, d_ff=64)
    model.eval()
    x = torch.randint(0, 20, (2, 16))
    x_modified = x.clone()
    x_modified[:, 10:] = (x_modified[:, 10:] + 1) % 20

    with torch.no_grad():
        y = model(x)
        y_modified = model(x_modified)

    assert torch.allclose(y[:, :10], y_modified[:, :10], atol=1e-5)


def test_chunked_dataset_is_non_overlapping():
    """データセットのチャンク分割が非重複であることを確認する．"""
    token_ids = list(range(100))
    seq_len = 9
    dataset = ex0031_data._ChunkedTextDataset(token_ids, seq_len)
    assert len(dataset) == 100 // (seq_len + 1)

    inputs_0, targets_0 = dataset[0]
    inputs_1, targets_1 = dataset[1]
    assert inputs_0.tolist() == list(range(seq_len))
    assert targets_0.tolist() == list(range(1, seq_len + 1))
    assert inputs_1.tolist() == list(range(seq_len + 1, 2 * seq_len + 1))


@pytest.mark.parametrize("method", ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"])
def test_training_runs_without_error_on_synthetic_data(method, tmp_path, monkeypatch):
    """4手法が合成データ上でエラーなく完走し，記録された評価指標の本数がエポック数+1と
    一致することを確認する（スモークテスト）．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    reg_lambda = 0.01
    eta = 0.01
    vocab_size = 20

    monkeypatch.setattr(ex0031_train, "_VOCAB_SIZE", vocab_size)

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(
            n_train=16, n_test=8, batch_size=batch_size, seq_len=8, vocab_size=vocab_size, seed=seed
        )

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    target_dir = str(tmp_path)
    if method == "SGD":
        ex0031_train.run_sgd(
            target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
        )
    else:
        ex0031_train.run_variance_reduced(
            method, target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
        )

    assert len(logger["epoch"]) == epochs + 1
    assert all(np.isfinite(v) for v in logger["train_loss"])


def test_oracle_calls_accounting_per_epoch(tmp_path, monkeypatch):
    """1エポックあたりのオラクル呼び出し回数が，SGDで N，NFG SVRG・ASAI SVRGで 2N，SVRGで 3N
    となることを確認する．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    reg_lambda = 0.01
    n_train = 16
    vocab_size = 20

    monkeypatch.setattr(ex0031_train, "_VOCAB_SIZE", vocab_size)

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(
            n_train=n_train, n_test=8, batch_size=batch_size, seq_len=8, vocab_size=vocab_size, seed=seed
        )

    expected_per_epoch = {"SGD": n_train, "NFG_SVRG": 2 * n_train, "ASAI_SVRG": 2 * n_train, "SVRG": 3 * n_train}
    for method, per_epoch in expected_per_epoch.items():
        logger = ResultLogger()
        logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
        target_dir = str(tmp_path / method)
        os.makedirs(target_dir, exist_ok=True)
        if method == "SGD":
            ex0031_train.run_sgd(
                target_dir, load_dataloader_func, 0.01, batch_size, reg_lambda, epochs, device, 0, logger
            )
        else:
            ex0031_train.run_variance_reduced(
                method, target_dir, load_dataloader_func, 0.01, batch_size, reg_lambda, epochs, device, 0, logger
            )

        initial_full_grad = n_train if method == "SVRG" else 0
        assert logger["oracle_calls"][0] == 0
        assert logger["oracle_calls"][1] == initial_full_grad + per_epoch


def test_elapsed_time_excludes_diagnostic_full_gradient(monkeypatch, tmp_path):
    """NFG SVRG・ASAI SVRGが近似誤差算出のためだけに毎エポック計算する診断専用のフル勾配
    計算時間が，`elapsed_time` に計上されないことを確認する．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 1
    n_train = 16
    vocab_size = 20
    sleep_seconds = 0.2

    monkeypatch.setattr(ex0031_train, "_VOCAB_SIZE", vocab_size)

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(
            n_train=n_train, n_test=8, batch_size=batch_size, seq_len=8, vocab_size=vocab_size, seed=seed
        )

    original = ex0031_train.compute_full_gradient_and_metrics

    def slow_compute_full_gradient_and_metrics(*args, **kwargs):
        time.sleep(sleep_seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(ex0031_train, "compute_full_gradient_and_metrics", slow_compute_full_gradient_and_metrics)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
    ex0031_train.run_variance_reduced(
        "ASAI_SVRG", str(tmp_path), load_dataloader_func, 0.01, batch_size, 0.01, epochs, device, 0, logger
    )
    assert logger["elapsed_time"][1] < sleep_seconds


def test_is_stuck_near_chance_detects_flatlined_accuracy():
    """末尾の次文字予測精度がチャンスレベル付近に留まっている場合，`True` を返すことを
    確認する．"""
    chance = 1.0 / 65
    accuracies = [0.1, 0.3, 0.2, chance + 0.001, chance - 0.001, chance]
    assert ex0031_train.is_stuck_near_chance(accuracies, vocab_size=65, window=3)


def test_is_stuck_near_chance_does_not_flag_learning_progress():
    """精度が学習の進行とともに上昇し続けている場合，`False` を返すことを確認する．"""
    accuracies = [0.015, 0.05, 0.1, 0.2, 0.3, 0.4]
    assert not ex0031_train.is_stuck_near_chance(accuracies, vocab_size=65, window=3)


def test_compute_trailing_relative_change_detects_plateau():
    """末尾の値がほぼ一定であれば相対変化が小さく判定されることを確認する．"""
    plateaued = [1.0, 0.5, 0.2, 0.101, 0.1005, 0.1002, 0.1001]
    assert ex0031_train.compute_trailing_relative_change(plateaued, window=3) < 1e-2


def test_compute_trailing_relative_change_detects_ongoing_change():
    """末尾の値が単調に大きく変化し続けていれば相対変化が大きく判定されることを確認する．"""
    still_changing = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
    assert ex0031_train.compute_trailing_relative_change(still_changing, window=3) > 0.1


def test_compute_trailing_relative_change_returns_inf_for_nonfinite_tail():
    """末尾にNaN・Infが含まれる場合は無限大を返すことを確認する．"""
    values = [1.0, 0.5, 0.3, float("nan")]
    assert ex0031_train.compute_trailing_relative_change(values, window=2) == float("inf")


def test_run_variance_reduced_stops_early_on_collapse(tmp_path):
    """訓練損失が非有限値化（崩壊）した場合，指定したエポック数まで学習を継続せず打ち切る
    ことを確認する．学習率を極端に大きくすることで人為的に崩壊させる．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 20
    n_train = 16

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=n_train, n_test=8, batch_size=batch_size, seed=seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
    target_dir = str(tmp_path)
    os.makedirs(target_dir, exist_ok=True)

    ex0031_train.run_variance_reduced(
        "ASAI_SVRG", target_dir, load_dataloader_func, eta=1e6, batch_size=batch_size,
        reg_lambda=0.01, epochs=epochs, device=device, seed=0, logger=logger,
    )

    assert len(logger["epoch"]) < epochs + 1
    assert not np.isfinite(logger["train_loss"][-1])


def test_is_run_completed_treats_collapsed_log_as_completed(tmp_path):
    """崩壊により打ち切られたログを，`is_run_completed`が完了済みとして扱うことを確認する．"""
    target_dir = str(tmp_path)
    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
    logger(0, 0, 0.0, 1.0, 0.1, float("nan"))
    logger(1, 16, 1.0, float("nan"), 0.1, float("nan"))
    logger.save(os.path.join(target_dir, "log.json"))

    assert ex0031_train.is_run_completed(target_dir, epochs=20)


def test_epochs_is_four_times_stage_b():
    """`.orders/order_029.md` 2節：エポック数がStage B（12エポック）の4倍（48エポック）で
    あることを確認する．"""
    assert ex0031_train.EPOCHS == 48
    assert ex0031_train.LEARNING_RATE == 0.01
    assert ex0031_train.BATCH_SIZES == [512, 128, 32]
    assert set(ex0031_train.METHODS) == {"SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"}
