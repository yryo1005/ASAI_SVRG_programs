"""
`programs/ex0024_cifar10_alexnet_layernorm_longrun/`（実験ex0024：LayerNorm・学習率0.001
固定での長期エポック学習，ex0023との比較）の単体テスト．

`model.py`がex0023（GroupNorm版）と完全に同一（バイト単位）であること，`NORM_TYPE`が
"layernorm"であること，バッチサイズごとのエポック数がex0023と同一であること，4手法（SGD，
SVRG，NFG SVRG，ASAI SVRG）が合成データ上でエラーなく完走すること，オラクル呼び出し回数の
正しさ，`elapsed_time` の計上からNFG SVRG・ASAI SVRGの診断専用フル勾配計算が正しく除外される
こと，プラトー判定に用いる `compute_trailing_relative_change` の正しさ，および長期学習中に
崩壊（訓練損失が非有限値化）した場合に学習を打ち切り，`is_run_completed` がこれを完了済みと
して扱うことを確認する．
"""

import filecmp
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROGRAMS_DIR = os.path.join(_PROJECT_ROOT, "programs")
_EX0023_DIR = os.path.join(_PROGRAMS_DIR, "ex0023_cifar10_alexnet_groupnorm_longrun")
_EX0024_DIR = os.path.join(_PROGRAMS_DIR, "ex0024_cifar10_alexnet_layernorm_longrun")
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


ex0024_model = _load_module("ex0024_model", _EX0024_DIR, "model.py")
ex0024_train = _load_module("ex0024_train", _EX0024_DIR, "train.py")


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


def test_model_is_byte_identical_copy_of_ex0023():
    """`.orders/order_037.md` 2節：`model.py`は`AlexNetCIFARNorm`が既にLayerNormを
    `norm_type`引数で選択できるため，ex0023から一切変更せず複製したことを確認する．"""
    assert filecmp.cmp(
        os.path.join(_EX0023_DIR, "model.py"), os.path.join(_EX0024_DIR, "model.py"), shallow=False
    )


def test_norm_type_is_layernorm():
    """`.orders/order_037.md`の指示（GroupNorm -> LayerNorm）通り，`NORM_TYPE`が"layernorm"
    であることを確認する．"""
    assert ex0024_train.NORM_TYPE == "layernorm"


def test_experiment_conditions_match_ex0023_except_norm_type():
    """`.orders/order_037.md`（「ほかの実験条件はex0023と同様とします」）の通り，正規化層
    以外の実験条件（手法，バッチサイズ，学習率，正則化係数，Seed数，バッチサイズごとの
    エポック数）がex0023と完全に一致することを確認する．"""
    ex0023_train = _load_module("ex0023_train_ref", _EX0023_DIR, "train.py")
    assert ex0024_train.METHODS == ex0023_train.METHODS
    assert ex0024_train.BATCH_SIZES == ex0023_train.BATCH_SIZES
    assert ex0024_train.LEARNING_RATE == ex0023_train.LEARNING_RATE
    assert ex0024_train.REG_LAMBDA == ex0023_train.REG_LAMBDA
    assert ex0024_train.SEEDS == ex0023_train.SEEDS
    assert ex0024_train.EPOCHS_BY_BATCH_SIZE == ex0023_train.EPOCHS_BY_BATCH_SIZE
    assert ex0024_train.NORM_TYPE != ex0023_train.NORM_TYPE


def test_model_has_no_batchnorm_or_dropout():
    """BatchNormalization・Dropout層を含まないことを確認する．"""
    model = ex0024_model.AlexNetCIFARNorm(norm_type="layernorm")
    for module in model.modules():
        assert not isinstance(module, (torch.nn.BatchNorm2d, torch.nn.Dropout))
        assert not isinstance(module, torch.nn.GroupNorm)


def test_model_uses_layernorm2d():
    """`norm_type="layernorm"`のモデルが`LayerNorm2d`層を含むことを確認する．"""
    model = ex0024_model.AlexNetCIFARNorm(norm_type="layernorm")
    assert any(isinstance(module, ex0024_model.LayerNorm2d) for module in model.modules())


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
        ex0024_train.run_sgd(
            target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
        )
    else:
        ex0024_train.run_variance_reduced(
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
            ex0024_train.run_sgd(
                target_dir, load_dataloader_func, 0.01, batch_size, reg_lambda, epochs, device, 0, logger
            )
        else:
            ex0024_train.run_variance_reduced(
                method, target_dir, load_dataloader_func, 0.01, batch_size, reg_lambda, epochs, device, 0, logger
            )

        initial_full_grad = n_train if method == "SVRG" else 0
        assert logger["oracle_calls"][0] == 0
        assert logger["oracle_calls"][1] == initial_full_grad + per_epoch


def test_elapsed_time_excludes_diagnostic_full_gradient_for_nfg_and_asai(monkeypatch, tmp_path):
    """NFG SVRG・ASAI SVRGが近似誤差算出のためだけに毎エポック計算する診断専用のフル勾配
    計算時間が，`elapsed_time` に計上されないことを確認する．SVRGは同じ計算がアルゴリズム
    自体に必要なため，計上されることも確認．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 1
    n_train = 16
    sleep_seconds = 0.2

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=n_train, n_test=8, batch_size=batch_size, seed=seed)

    original = ex0024_train.compute_full_gradient_and_metrics

    def slow_compute_full_gradient_and_metrics(*args, **kwargs):
        time.sleep(sleep_seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ex0024_train, "compute_full_gradient_and_metrics", slow_compute_full_gradient_and_metrics
    )

    elapsed_time_by_method = {}
    for method in ["SVRG", "NFG_SVRG", "ASAI_SVRG"]:
        logger = ResultLogger()
        logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
        target_dir = str(tmp_path / method)
        os.makedirs(target_dir, exist_ok=True)
        ex0024_train.run_variance_reduced(
            method, target_dir, load_dataloader_func, 0.01, batch_size, 0.01, epochs, device, 0, logger
        )
        elapsed_time_by_method[method] = logger["elapsed_time"][1]

    assert elapsed_time_by_method["SVRG"] >= sleep_seconds
    assert elapsed_time_by_method["NFG_SVRG"] < sleep_seconds
    assert elapsed_time_by_method["ASAI_SVRG"] < sleep_seconds


def test_compute_trailing_relative_change_detects_plateau():
    """末尾の値がほぼ一定であれば相対変化が小さく判定されることを確認する．"""
    plateaued = [1.0, 0.5, 0.2, 0.101, 0.1005, 0.1002, 0.1001]
    assert ex0024_train.compute_trailing_relative_change(plateaued, window=3) < 1e-2


def test_compute_trailing_relative_change_detects_ongoing_change():
    """末尾の値が単調に大きく変化し続けていれば相対変化が大きく判定されることを確認する．"""
    still_changing = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
    assert ex0024_train.compute_trailing_relative_change(still_changing, window=3) > 0.1


def test_compute_trailing_relative_change_returns_inf_for_nonfinite_tail():
    """末尾にNaN・Infが含まれる場合は無限大を返す（プラトーとは判定しない）ことを確認する．"""
    values = [1.0, 0.5, 0.3, float("nan")]
    assert ex0024_train.compute_trailing_relative_change(values, window=2) == float("inf")


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

    ex0024_train.run_variance_reduced(
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

    assert ex0024_train.is_run_completed(target_dir, epochs=20)


def test_is_run_completed_does_not_treat_incomplete_finite_log_as_completed(tmp_path):
    """崩壊しておらず，単に指定エポック数に達していないログは未完了として扱われることを
    確認する．"""
    target_dir = str(tmp_path)
    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
    logger(0, 0, 0.0, 1.0, 0.1, float("nan"))
    logger(1, 16, 1.0, 0.9, 0.15, 0.5)
    logger.save(os.path.join(target_dir, "log.json"))

    assert not ex0024_train.is_run_completed(target_dir, epochs=20)


def test_hp_name_includes_layernorm():
    """ディレクトリ名にNORM_TYPE（"layernorm"）が含まれ，ex0023（"groupnorm"）と衝突しない
    ことを確認する．"""
    name = ex0024_train.hp_name(128, 48)
    assert "layernorm" in name
    assert "groupnorm" not in name


def test_sgd_double_epochs_are_exactly_double():
    """`.orders/order_038.md`：SGD倍エポックのエポック数が，通常のグリッドのちょうど2倍で
    あることを確認する．"""
    for bs, epochs in ex0024_train.EPOCHS_BY_BATCH_SIZE.items():
        assert ex0024_train.SGD_DOUBLE_EPOCHS_BY_BATCH_SIZE[bs] == 2 * epochs


def test_build_main_tasks_covers_full_grid():
    """基本グリッドのタスク数が4手法×3バッチサイズ×5Seed=60であり，各タスクのエポック数が
    `EPOCHS_BY_BATCH_SIZE`と一致することを確認する．"""
    tasks = ex0024_train._build_main_tasks()
    assert len(tasks) == 60
    for method, batch_size, seed, epochs in tasks:
        assert epochs == ex0024_train.EPOCHS_BY_BATCH_SIZE[batch_size]


def test_build_sgd_double_epoch_tasks_covers_sgd_only():
    """SGD倍エポック追加学習のタスク数が3バッチサイズ×5Seed=15であり，全て手法がSGD，
    エポック数が`SGD_DOUBLE_EPOCHS_BY_BATCH_SIZE`と一致することを確認する．"""
    tasks = ex0024_train._build_sgd_double_epoch_tasks()
    assert len(tasks) == 15
    for method, batch_size, seed, epochs in tasks:
        assert method == "SGD"
        assert epochs == ex0024_train.SGD_DOUBLE_EPOCHS_BY_BATCH_SIZE[batch_size]


def test_main_and_sgd_double_tasks_produce_distinct_directory_names():
    """基本グリッドのSGDタスクと倍エポックタスクとで，`hp_name`（ディレクトリ名）が衝突
    しないことを確認する（エポック数が異なるため自動的に分離される）．"""
    main_names = {
        ex0024_train.hp_name(bs, ex0024_train.EPOCHS_BY_BATCH_SIZE[bs])
        for bs in ex0024_train.BATCH_SIZES
    }
    double_names = {
        ex0024_train.hp_name(bs, ex0024_train.SGD_DOUBLE_EPOCHS_BY_BATCH_SIZE[bs])
        for bs in ex0024_train.BATCH_SIZES
    }
    assert main_names.isdisjoint(double_names)


def test_main_entry_point_runs_sgd_double_epoch_phase_only():
    """`.orders/order_038.md`：基本グリッドは完了済みのため，`if __name__ == "__main__":`
    ブロックが`main(run_grid=False, run_sgd_double=True)`を呼び出すことをASTで検証する．"""
    import ast
    import inspect

    source = inspect.getsource(ex0024_train)
    tree = ast.parse(source)
    main_block = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            main_block = node
            break
    assert main_block is not None
    call_sources = [ast.unparse(stmt) for stmt in main_block.body]
    assert any("main(run_grid=False, run_sgd_double=True)" in c for c in call_sources)

    sig = inspect.signature(ex0024_train.main)
    assert sig.parameters["run_grid"].default is True
    assert sig.parameters["run_sgd_double"].default is True


def test_run_single_experiment_accepts_explicit_epochs(tmp_path, monkeypatch):
    """`run_single_experiment`が (method, batch_size, seed, epochs) の4要素タプルを受け取り，
    指定したエポック数で学習・保存することを確認する（合成データで軽量に検証）．"""
    monkeypatch.setattr(ex0024_train, "OUTPUT_ROOT", str(tmp_path))

    def fake_load_dataloader(seed=0, batch_size=8):
        return _make_synthetic_dataloaders(n_train=16, n_test=8, batch_size=batch_size, seed=seed)

    monkeypatch.setattr(ex0024_train, "load_dataloader", fake_load_dataloader)

    epochs = 2
    ex0024_train.run_single_experiment(("SGD", 8, 0, epochs))

    target_dir = os.path.join(str(tmp_path), "SGD", ex0024_train.hp_name(8, epochs), "0")
    with open(os.path.join(target_dir, "config.json")) as f:
        config = json.load(f)
    assert config["epochs"] == epochs
    logger = ResultLogger(os.path.join(target_dir, "log.json"))
    assert len(logger["epoch"]) == epochs + 1
