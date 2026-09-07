"""
`programs/ex0023_cifar10_alexnet_groupnorm_longrun/`（実験ex0023：GroupNorm・学習率0.001
固定での長期エポック学習，誤差床の収束観察）の単体テスト．

4手法（SGD，SVRG，NFG SVRG，ASAI SVRG）が合成データ上でエラーなく完走すること，オラクル
呼び出し回数の正しさ，`elapsed_time` の計上からNFG SVRG・ASAI SVRGの診断専用フル勾配計算が
正しく除外されること（`.reports/report_025.md` 2.2節の修正の踏襲確認），プラトー判定に
用いる `compute_trailing_relative_change` の正しさ，および長期学習中に崩壊（訓練損失が
非有限値化）した場合に学習を打ち切り，`is_run_completed` がこれを完了済みとして扱うこと
（`.orders/order_026.md` 6節）を確認する．
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
_EX0023_DIR = os.path.join(_PROGRAMS_DIR, "ex0023_cifar10_alexnet_groupnorm_longrun")
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import ResultLogger  # noqa: E402


def _load_module(unique_name: str, directory: str, filename: str):
    """概要: 実験ディレクトリ間のモジュール名衝突（`model`／`data`／`train`）を避けるため，
        一意な名前でモジュールを動的に読み込む．"""
    for stale_name in ("model", "data", "train"):
        sys.modules.pop(stale_name, None)
    spec = importlib.util.spec_from_file_location(unique_name, os.path.join(directory, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex0023_model = _load_module("ex0023_model", _EX0023_DIR, "model.py")
ex0023_train = _load_module("ex0023_train", _EX0023_DIR, "train.py")


class _SyntheticDataset(torch.utils.data.Dataset):
    """テスト用の小規模な合成CIFAR-10形式データセット（ランダム画像・ランダムラベル）．"""

    def __init__(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        self.images = torch.tensor(rng.normal(size=(n, 3, 32, 32)), dtype=torch.float32)
        self.labels = torch.tensor(rng.integers(0, 10, size=n), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def _make_synthetic_dataloaders(n_train=24, n_test=8, batch_size=8, seed=0):
    train_ds = _SyntheticDataset(n_train, seed)
    test_ds = _SyntheticDataset(n_test, seed + 1)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl


def test_model_has_no_batchnorm_or_dropout():
    """BatchNormalization・Dropout層を含まないことを確認する．"""
    model = ex0023_model.AlexNetCIFARNorm(norm_type="groupnorm")
    for module in model.modules():
        assert not isinstance(module, (torch.nn.BatchNorm2d, torch.nn.Dropout))


@pytest.mark.parametrize("method", ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"])
def test_training_runs_without_error_on_synthetic_data(method, tmp_path):
    """4手法が合成データ上でエラーなく完走し，記録された評価指標の本数がエポック数+1と
    一致することを確認する（スモークテスト）．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    reg_lambda = 0.01
    eta = 0.01

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=16, n_test=8, batch_size=batch_size, seed=seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    target_dir = str(tmp_path)
    if method == "SGD":
        ex0023_train.run_sgd(
            target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
        )
    else:
        ex0023_train.run_variance_reduced(
            method, target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
        )

    assert len(logger["epoch"]) == epochs + 1
    assert all(np.isfinite(v) for v in logger["train_loss"])


def test_oracle_calls_accounting_per_epoch(tmp_path):
    """1エポックあたりのオラクル呼び出し回数が，SGDで N，NFG SVRG・ASAI SVRGで 2N，SVRGで 3N
    となることを確認する（Nはミニバッチ単位ではなくサンプル数換算）．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    reg_lambda = 0.01
    n_train = 16

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=n_train, n_test=8, batch_size=batch_size, seed=seed)

    expected_per_epoch = {"SGD": n_train, "NFG_SVRG": 2 * n_train, "ASAI_SVRG": 2 * n_train, "SVRG": 3 * n_train}
    for method, per_epoch in expected_per_epoch.items():
        logger = ResultLogger()
        logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
        target_dir = str(tmp_path / method)
        os.makedirs(target_dir, exist_ok=True)
        if method == "SGD":
            ex0023_train.run_sgd(
                target_dir, load_dataloader_func, 0.01, batch_size, reg_lambda, epochs, device, 0, logger
            )
        else:
            ex0023_train.run_variance_reduced(
                method, target_dir, load_dataloader_func, 0.01, batch_size, reg_lambda, epochs, device, 0, logger
            )

        initial_full_grad = n_train if method == "SVRG" else 0
        assert logger["oracle_calls"][0] == 0
        assert logger["oracle_calls"][1] == initial_full_grad + per_epoch


def test_elapsed_time_excludes_diagnostic_full_gradient_for_nfg_and_asai(monkeypatch, tmp_path):
    """NFG SVRG・ASAI SVRGが近似誤差算出のためだけに毎エポック計算する診断専用のフル勾配
    計算時間が，`elapsed_time` に計上されないことを確認する（`.reports/report_025.md` 2.2節
    の修正の踏襲確認）．SVRGは同じ計算がアルゴリズム自体に必要なため，計上されることも確認．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 1
    n_train = 16
    sleep_seconds = 0.2

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=n_train, n_test=8, batch_size=batch_size, seed=seed)

    original = ex0023_train.compute_full_gradient_and_metrics

    def slow_compute_full_gradient_and_metrics(*args, **kwargs):
        time.sleep(sleep_seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ex0023_train, "compute_full_gradient_and_metrics", slow_compute_full_gradient_and_metrics
    )

    elapsed_time_by_method = {}
    for method in ["SVRG", "NFG_SVRG", "ASAI_SVRG"]:
        logger = ResultLogger()
        logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
        target_dir = str(tmp_path / method)
        os.makedirs(target_dir, exist_ok=True)
        ex0023_train.run_variance_reduced(
            method, target_dir, load_dataloader_func, 0.01, batch_size, 0.01, epochs, device, 0, logger
        )
        elapsed_time_by_method[method] = logger["elapsed_time"][1]

    assert elapsed_time_by_method["SVRG"] >= sleep_seconds
    assert elapsed_time_by_method["NFG_SVRG"] < sleep_seconds
    assert elapsed_time_by_method["ASAI_SVRG"] < sleep_seconds


def test_compute_trailing_relative_change_detects_plateau():
    """末尾の値がほぼ一定であれば相対変化が小さく判定されることを確認する．"""
    plateaued = [1.0, 0.5, 0.2, 0.101, 0.1005, 0.1002, 0.1001]
    assert ex0023_train.compute_trailing_relative_change(plateaued, window=3) < 1e-2


def test_compute_trailing_relative_change_detects_ongoing_change():
    """末尾の値が単調に大きく変化し続けていれば相対変化が大きく判定されることを確認する．"""
    still_changing = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
    assert ex0023_train.compute_trailing_relative_change(still_changing, window=3) > 0.1


def test_compute_trailing_relative_change_returns_inf_for_nonfinite_tail():
    """末尾にNaN・Infが含まれる場合は無限大を返す（プラトーとは判定しない）ことを確認する．"""
    values = [1.0, 0.5, 0.3, float("nan")]
    assert ex0023_train.compute_trailing_relative_change(values, window=2) == float("inf")


def test_run_variance_reduced_stops_early_on_collapse(tmp_path):
    """`.orders/order_026.md` 6節：訓練損失が非有限値化（崩壊）した場合，指定したエポック数
    まで学習を継続せず打ち切ることを確認する．学習率を極端に大きくすることで人為的に崩壊
    させる．"""
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

    ex0023_train.run_variance_reduced(
        "ASAI_SVRG", target_dir, load_dataloader_func, eta=1e6, batch_size=batch_size,
        reg_lambda=0.01, epochs=epochs, device=device, seed=0, logger=logger,
    )

    assert len(logger["epoch"]) < epochs + 1
    assert not np.isfinite(logger["train_loss"][-1])


def test_is_run_completed_treats_collapsed_log_as_completed(tmp_path):
    """崩壊により打ち切られたログ（記録数がepochs+1未満で末尾がNaN）を，`is_run_completed`が
    完了済みとして扱うことを確認する（再学習しても同じ崩壊を再現するだけであるため）．"""
    target_dir = str(tmp_path)
    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
    logger(0, 0, 0.0, 1.0, 0.1, float("nan"))
    logger(1, 16, 1.0, float("nan"), 0.1, float("nan"))
    logger.save(os.path.join(target_dir, "log.json"))

    assert ex0023_train.is_run_completed(target_dir, epochs=20)


def test_is_run_completed_does_not_treat_incomplete_finite_log_as_completed(tmp_path):
    """崩壊しておらず，単に指定エポック数に達していないログは未完了として扱われることを
    確認する．"""
    target_dir = str(tmp_path)
    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
    logger(0, 0, 0.0, 1.0, 0.1, float("nan"))
    logger(1, 16, 1.0, 0.9, 0.15, 0.5)
    logger.save(os.path.join(target_dir, "log.json"))

    assert not ex0023_train.is_run_completed(target_dir, epochs=20)


def test_epochs_by_batch_size_are_roughly_four_times_ex0022():
    """`.orders/order_026.md` 5節の初期値（ex0022比で約4倍の総イテレーション数）を確認する．"""
    ex0022_epochs = {512: 48, 128: 12, 32: 3}
    for batch_size, epochs in ex0023_train.EPOCHS_BY_BATCH_SIZE.items():
        ratio = epochs / ex0022_epochs[batch_size]
        assert 3.5 <= ratio <= 4.5
