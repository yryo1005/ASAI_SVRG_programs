"""
`programs/ex0022_cifar10_alexnet_groupnorm/`（実験ex0022：GroupNorm下での4手法比較，
分散削減の効率性検証）の単体テスト．

`.orders/order_025.md` 3節が要求する「コピー後いずれのプログラムの初期値も同じになることを
テストする」検証（`test_model_initial_parameters_match_ex0021`，
`test_dataloader_initial_order_matches_ex0021`）を必須項目として含む．また，4手法
（SGD，SVRG，NFG SVRG，ASAI SVRG）が合成データ上でエラーなく完走すること，オラクル呼び出し
回数の正しさ，`elapsed_time` の計上からNFG SVRG・ASAI SVRGの診断専用フル勾配計算が正しく
除外されること，バッチサイズごとのエポック数の対応が総イテレーション数をおおむね揃えることを
確認する．
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
_EX0022_DIR = os.path.join(_PROGRAMS_DIR, "ex0022_cifar10_alexnet_groupnorm")
_EX0021_DIR = os.path.join(_PROGRAMS_DIR, "ex0021_cifar10_alexnet_norm")
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import ResultLogger  # noqa: E402


def _load_module(unique_name: str, directory: str, filename: str):
    """概要: 実験ディレクトリ間のモジュール名衝突（`model`／`data`／`train`）を避けるため，
        一意な名前でモジュールを動的に読み込む．
    引数:
        unique_name (str)．`sys.modules` に登録する一意な名前．
        directory (str)．モジュールファイルが存在するディレクトリ．
        filename (str)．モジュールファイル名（例: "model.py"）．
    戻り値: module (module)．
    """
    for stale_name in ("model", "data", "train"):
        sys.modules.pop(stale_name, None)
    spec = importlib.util.spec_from_file_location(unique_name, os.path.join(directory, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex0022_model = _load_module("ex0022_model", _EX0022_DIR, "model.py")
ex0022_data = _load_module("ex0022_data", _EX0022_DIR, "data.py")
ex0022_train = _load_module("ex0022_train", _EX0022_DIR, "train.py")

ex0021_model = _load_module("ex0021_model", _EX0021_DIR, "model.py")
ex0021_data = _load_module("ex0021_data", _EX0021_DIR, "data.py")


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


def test_hp_name_matches_ex0021_for_reused_cells():
    """学習率0.001・GroupNorm・バッチサイズ512, 128の条件で，ex0022の`hp_name`がex0021の
    命名規則（`lr{eta}_bs{bs}_norm{norm_type}_lambda{reg}_epochs{epochs}`）と完全に同一の
    文字列になることを確認する（`.orders/order_025.md` 3節のコピーによる再利用の前提）．"""
    for batch_size, epochs in [(512, 48), (128, 12)]:
        ex0022_name = ex0022_train.hp_name(0.001, batch_size, epochs)
        ex0021_name = f"lr0.001_bs{batch_size}_normgroupnorm_lambda0.0005_epochs{epochs}"
        assert ex0022_name == ex0021_name


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_model_initial_parameters_match_ex0021(seed):
    """`.orders/order_025.md` 3節の必須検証：同一Seedで初期化したモデルのパラメータが，
    ex0021とex0022で完全に一致する（`torch.equal`）ことを確認する．"""
    model_ex0021 = ex0021_model.load_model(
        ex0021_model.AlexNetCIFARNorm, seed=seed, norm_type="groupnorm"
    )
    model_ex0022 = ex0022_model.load_model(
        ex0022_model.AlexNetCIFARNorm, seed=seed, norm_type="groupnorm"
    )
    for p_ex0021, p_ex0022 in zip(model_ex0021.parameters(), model_ex0022.parameters()):
        assert torch.equal(p_ex0021, p_ex0022)


@pytest.mark.parametrize("seed,batch_size", [(0, 512), (1, 512), (0, 128), (2, 128)])
def test_dataloader_initial_order_matches_ex0021(seed, batch_size):
    """`.orders/order_025.md` 3節の必須検証：同一Seed・バッチサイズで構築したデータローダーの
    先頭バッチが，ex0021とex0022で完全に一致することを確認する．"""
    train_dl_ex0021, test_dl_ex0021 = ex0021_data.load_dataloader(seed=seed, batch_size=batch_size)
    train_dl_ex0022, test_dl_ex0022 = ex0022_data.load_dataloader(seed=seed, batch_size=batch_size)

    images_ex0021, labels_ex0021 = next(iter(train_dl_ex0021))
    images_ex0022, labels_ex0022 = next(iter(train_dl_ex0022))
    assert torch.equal(images_ex0021, images_ex0022)
    assert torch.equal(labels_ex0021, labels_ex0022)

    test_images_ex0021, test_labels_ex0021 = next(iter(test_dl_ex0021))
    test_images_ex0022, test_labels_ex0022 = next(iter(test_dl_ex0022))
    assert torch.equal(test_images_ex0021, test_images_ex0022)
    assert torch.equal(test_labels_ex0021, test_labels_ex0022)


def test_model_has_no_batchnorm_or_dropout():
    """BatchNormalization・Dropout層を含まないことを確認する．"""
    model = ex0022_model.AlexNetCIFARNorm(norm_type="groupnorm")
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
        ex0022_train.run_sgd(
            target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
        )
    else:
        ex0022_train.run_variance_reduced(
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
            ex0022_train.run_sgd(
                target_dir, load_dataloader_func, 0.01, batch_size, reg_lambda, epochs, device, 0, logger
            )
        else:
            ex0022_train.run_variance_reduced(
                method, target_dir, load_dataloader_func, 0.01, batch_size, reg_lambda, epochs, device, 0, logger
            )

        initial_full_grad = n_train if method == "SVRG" else 0
        assert logger["oracle_calls"][0] == 0
        assert logger["oracle_calls"][1] == initial_full_grad + per_epoch


def test_elapsed_time_excludes_diagnostic_full_gradient_for_nfg_and_asai(monkeypatch, tmp_path):
    """NFG SVRG・ASAI SVRGが近似誤差算出のためだけに毎エポック計算する診断専用のフル勾配
    計算時間が，`elapsed_time` に計上されないことを確認する．SVRGは同じ計算がアルゴリズム
    自体に必要（次エポックのスナップショット勾配の設定）なため，計上されることも確認する．
    `compute_full_gradient_and_metrics` を人為的に遅延させ，その遅延分が計上されるか否かで
    判定する．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 1
    n_train = 16
    sleep_seconds = 0.2

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=n_train, n_test=8, batch_size=batch_size, seed=seed)

    original = ex0022_train.compute_full_gradient_and_metrics

    def slow_compute_full_gradient_and_metrics(*args, **kwargs):
        time.sleep(sleep_seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ex0022_train, "compute_full_gradient_and_metrics", slow_compute_full_gradient_and_metrics
    )

    elapsed_time_by_method = {}
    for method in ["SVRG", "NFG_SVRG", "ASAI_SVRG"]:
        logger = ResultLogger()
        logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
        target_dir = str(tmp_path / method)
        os.makedirs(target_dir, exist_ok=True)
        ex0022_train.run_variance_reduced(
            method, target_dir, load_dataloader_func, 0.01, batch_size, 0.01, epochs, device, 0, logger
        )
        elapsed_time_by_method[method] = logger["elapsed_time"][1]

    assert elapsed_time_by_method["SVRG"] >= sleep_seconds
    assert elapsed_time_by_method["NFG_SVRG"] < sleep_seconds
    assert elapsed_time_by_method["ASAI_SVRG"] < sleep_seconds


def test_epochs_by_batch_size_keep_total_iterations_roughly_constant():
    """バッチサイズごとに設定したエポック数が，総イテレーション数（epoch×K）をおおむね
    一定に保つことを確認する（`.orders/order_025.md` 5節の要求）．"""
    n_train = 50000
    totals = []
    for batch_size, epochs in ex0022_train.EPOCHS_BY_BATCH_SIZE.items():
        k = -(-n_train // batch_size)  # ceil
        totals.append(epochs * k)

    baseline = totals[0]
    for total in totals:
        assert abs(total - baseline) / baseline < 0.02


def test_reuse_cells_cover_only_nfg_and_asai_at_lr_0001():
    """再利用対象（`.orders/order_025.md` 3節）が，学習率0.001・バッチサイズ512, 128の
    NFG SVRG・ASAI SVRG，計4セルであることを確認する．"""
    assert set(ex0022_train._REUSE_CELLS) == {
        ("NFG_SVRG", 512), ("NFG_SVRG", 128), ("ASAI_SVRG", 512), ("ASAI_SVRG", 128),
    }
    assert ex0022_train._REUSE_LEARNING_RATE == 0.001
