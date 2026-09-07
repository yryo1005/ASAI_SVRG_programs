"""
`programs/ex002_cifar10_alexnet/`（実験2：CIFAR-10を用いた多値分類問題）の単体テスト．

モデルがBatchNormalization・Dropoutを含まないこと，L2正則化項が重みのみ（切片を除く）に
課されること，4手法が合成データ上でエラーなく完走すること，オラクル呼び出し回数の正しさを
確認する．CIFAR-10の実データ・ダウンロードを伴わないよう，合成の画像テンソルを用いる．
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROGRAMS_DIR = os.path.join(_PROJECT_ROOT, "programs")
_EXPERIMENT_DIR = os.path.join(_PROGRAMS_DIR, "ex002_cifar10_alexnet")
sys.path.insert(0, _EXPERIMENT_DIR)
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import ResultLogger  # noqa: E402

# 他の実験（ex000，ex001）のテストが同名モジュール（model, data, train）をsys.modulesに
# キャッシュしている場合，train.py内部の `from model import ...` 等の素朴なimportが誤った
# モジュールを解決してしまうため，明示的にキャッシュを破棄してから読み込む．
for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_spec_model = importlib.util.spec_from_file_location(
    "ex002_model", os.path.join(_EXPERIMENT_DIR, "model.py")
)
ex002_model = importlib.util.module_from_spec(_spec_model)
_spec_model.loader.exec_module(ex002_model)

for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_spec_train = importlib.util.spec_from_file_location(
    "ex002_train", os.path.join(_EXPERIMENT_DIR, "train.py")
)
ex002_train = importlib.util.module_from_spec(_spec_train)
_spec_train.loader.exec_module(ex002_train)


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
    """概要: テスト用の合成データローダーの組を生成する．"""
    train_ds = _SyntheticDataset(n_train, seed)
    test_ds = _SyntheticDataset(n_test, seed + 1)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl


def test_model_has_no_batchnorm_or_dropout():
    """`.orders/order_022.md` 2節の制約通り，モデルがBatchNormalization・Dropout層を
    含まないことを確認する．"""
    model = ex002_model.AlexNetCIFAR()
    for module in model.modules():
        assert not isinstance(module, (torch.nn.BatchNorm2d, torch.nn.Dropout))


def test_model_output_shape():
    """モデルの出力形状が (B, num_classes) であることを確認する．"""
    model = ex002_model.AlexNetCIFAR(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_l2_regularization_excludes_bias():
    """L2正則化項が畳み込み層・全結合層の重み（`weight`）のみに課され，切片（`bias`）を
    含まないことを確認する．"""
    model = ex002_model.AlexNetCIFAR()
    reg_lambda = 0.1

    expected = model.conv_layers[0].weight.new_zeros(())
    for name, param in model.named_parameters():
        if name.endswith(".weight"):
            expected = expected + torch.sum(param ** 2)
    expected = 0.5 * reg_lambda * expected

    actual = ex002_model.compute_l2_regularization(model, reg_lambda)
    assert torch.allclose(actual, expected)


def test_loss_func_adds_regularization_to_cross_entropy():
    """`loss_func` が多値交差エントロピー損失にL2正則化項を加算することを確認する．"""
    model = ex002_model.AlexNetCIFAR()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    reg_lambda = 0.1

    logits = model(x)
    loss = ex002_model.loss_func(logits, y, model, reg_lambda)
    cce = torch.nn.functional.cross_entropy(logits, y)
    reg = ex002_model.compute_l2_regularization(model, reg_lambda)

    assert torch.allclose(loss, cce + reg)


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
        ex002_train.run_sgd(
            target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
        )
    else:
        ex002_train.run_variance_reduced(
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
            ex002_train.run_sgd(
                target_dir, load_dataloader_func, 0.01, batch_size, reg_lambda, epochs, device, 0, logger
            )
        else:
            ex002_train.run_variance_reduced(
                method, target_dir, load_dataloader_func, 0.01, batch_size, reg_lambda, epochs, device, 0, logger
            )

        initial_full_grad = n_train if method == "SVRG" else 0
        assert logger["oracle_calls"][0] == 0
        assert logger["oracle_calls"][1] == initial_full_grad + per_epoch
        assert logger["oracle_calls"][2] - logger["oracle_calls"][1] == per_epoch
