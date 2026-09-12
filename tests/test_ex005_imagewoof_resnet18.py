"""
`programs/ex005_imagewoof_resnet18/`（実験5：Imagewoofを用いたResNet18画像分類，Stage A
安定性探索）の単体テスト．

モデルがDropout・BatchNormalizationを含まないこと（全てLayerNorm2dに置換済み），同一入力に
対する出力が決定論的であること，パラメータ数が標準ResNet18と一致すること，L2正則化がConv2d・
Linearの重みのみに課されること，2手法（NFG SVRG，ASAI SVRG）が合成データ上でエラーなく完走
すること，オラクル呼び出し回数の正しさ（NFG/ASAI:2N），`elapsed_time` から診断専用フル勾配
計算が除外されること，チャンスレベル張り付き検出（`is_stuck_near_chance`）の正しさを確認する．
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
_EX005_DIR = os.path.join(_PROGRAMS_DIR, "ex005_imagewoof_resnet18")
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


ex005_model = _load_module("ex005_model", _EX005_DIR, "model.py")
ex005_data = _load_module("ex005_data", _EX005_DIR, "data.py")
ex005_train = _load_module("ex005_train", _EX005_DIR, "train.py")


class _SyntheticImageDataset(torch.utils.data.Dataset):
    """テスト用の小規模な合成画像データセット（ランダム画像・ランダムラベル）．画像サイズは
    実際の224x224より小さい64x64を用い（`AdaptiveAvgPool2d`により任意の空間サイズで動作
    するResNet18の性質を利用），テスト実行を高速化する．"""

    def __init__(self, n: int, seed: int, image_size: int = 64, num_classes: int = 10):
        rng = np.random.default_rng(seed)
        self.images = torch.tensor(
            rng.normal(size=(n, 3, image_size, image_size)), dtype=torch.float32
        )
        self.labels = torch.tensor(rng.integers(0, num_classes, size=n), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def _make_synthetic_dataloaders(n_train=16, n_test=8, batch_size=4, seed=0):
    train_ds = _SyntheticImageDataset(n_train, seed)
    test_ds = _SyntheticImageDataset(n_test, seed + 1)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl


def test_model_has_no_dropout_or_batchnorm():
    """Dropout・BatchNormalization層を含まないことを確認する（`.orders/order_033.md` 3節）．
    全てのBatchNorm2d層がLayerNorm2dに置換されていることも確認する．"""
    model = ex005_model.ResNet18LayerNorm(num_classes=10)
    n_layernorm2d = 0
    for module in model.modules():
        assert not isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
        if isinstance(module, ex005_model.LayerNorm2d):
            n_layernorm2d += 1
    assert n_layernorm2d == 20  # 標準ResNet18のBatchNorm2d層数（stem 1 + 各BasicBlock 2 x 8
    # + downsample 3 = 1+16+3=20）


def test_model_is_deterministic_given_same_input():
    """同一入力に対し，モデルの出力が呼び出しごとに変化しない（決定論的である）ことを確認
    する．SVRG系手法の理論的前提の検証．"""
    model = ex005_model.ResNet18LayerNorm(num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert torch.equal(y1, y2)


def test_model_parameter_count_matches_standard_resnet18():
    """LayerNorm化ResNet18のパラメータ数が，標準ResNet18（BatchNorm版）と一致することを
    確認する（`.orders/order_033.md` 5節，正規化層の置換はパラメータ数に影響しない）．"""
    import torchvision.models as models

    model = ex005_model.ResNet18LayerNorm(num_classes=10)
    n_params_ln = sum(p.numel() for p in model.parameters())

    reference = models.resnet18(weights=None, num_classes=10)
    n_params_bn = sum(p.numel() for p in reference.parameters())

    assert n_params_ln == n_params_bn
    assert 11_000_000 < n_params_ln < 11_300_000


def test_l2_regularization_excludes_layernorm_and_bias():
    """L2正則化がConv2d・Linearの重みのみに課され，バイアス・LayerNorm2dのアフィン
    パラメータを除外することを確認する．"""
    model = ex005_model.ResNet18LayerNorm(num_classes=10)
    reg_lambda = 1e-6  # 大きなモデルのため小さい係数で計算量を抑える

    manual_reg = torch.zeros(())
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            manual_reg = manual_reg + torch.sum(module.weight ** 2)
    expected = 0.5 * reg_lambda * manual_reg

    actual = ex005_model.compute_l2_regularization(model, reg_lambda)
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("method", ["NFG_SVRG", "ASAI_SVRG"])
def test_training_runs_without_error_on_synthetic_data(method, tmp_path, monkeypatch):
    """Stage A・Bで用いるNFG SVRG・ASAI SVRGが合成データ上でエラーなく完走し，記録された
    評価指標の本数がエポック数+1と一致することを確認する（スモークテスト）．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    reg_lambda = 1e-6
    eta = 0.001

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=16, n_test=8, batch_size=batch_size, seed=seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    target_dir = str(tmp_path)
    ex005_train.run_variance_reduced(
        method, target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
    )

    assert len(logger["epoch"]) == epochs + 1
    assert all(np.isfinite(v) for v in logger["train_loss"])


def test_oracle_calls_accounting_per_epoch(tmp_path):
    """1エポックあたりのオラクル呼び出し回数が，NFG SVRG・ASAI SVRGで 2N となることを
    確認する（Nはミニバッチ単位ではなくサンプル数換算）．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    reg_lambda = 1e-6
    n_train = 16

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=n_train, n_test=8, batch_size=batch_size, seed=seed)

    for method in ["NFG_SVRG", "ASAI_SVRG"]:
        logger = ResultLogger()
        logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
        target_dir = str(tmp_path / method)
        os.makedirs(target_dir, exist_ok=True)
        ex005_train.run_variance_reduced(
            method, target_dir, load_dataloader_func, 0.001, batch_size, reg_lambda, epochs, device, 0, logger
        )

        assert logger["oracle_calls"][0] == 0
        assert logger["oracle_calls"][1] == 2 * n_train


def test_sgd_runs_without_error_on_synthetic_data(tmp_path):
    """Stage B（`.orders/order_034.md`）で追加したSGDが合成データ上でエラーなく完走し，
    記録された評価指標の本数がエポック数+1と一致すること，オラクル呼び出し回数が1エポック
    あたり N（Nはサンプル数換算）となることを確認する．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    reg_lambda = 1e-6
    eta = 0.001
    n_train = 16

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=n_train, n_test=8, batch_size=batch_size, seed=seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    ex005_train.run_sgd(
        str(tmp_path), load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
    )

    assert len(logger["epoch"]) == epochs + 1
    assert all(np.isfinite(v) for v in logger["train_loss"])
    assert logger["oracle_calls"][0] == 0
    assert logger["oracle_calls"][1] == n_train


def test_svrg_runs_without_error_on_synthetic_data(tmp_path):
    """Stage Bで追加したSVRGが合成データ上でエラーなく完走し，記録された評価指標の本数が
    エポック数+1と一致すること，オラクル呼び出し回数の推移が期待通りであることを確認する．
    エポック0はz_0のフル勾配計算を行う前に記録するため0，エポック1はエポック0後のz_1用
    フル勾配（N）＋内部ループ（2N）＋エポック1後のz_2用フル勾配（N）＝4N，エポック2以降は
    定常状態の3N（内部ループ2N＋次スナップショット用フル勾配N）ずつ増加する．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    reg_lambda = 1e-6
    eta = 0.001
    n_train = 16

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=n_train, n_test=8, batch_size=batch_size, seed=seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    ex005_train.run_variance_reduced(
        "SVRG", str(tmp_path), load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
    )

    assert len(logger["epoch"]) == epochs + 1
    assert all(np.isfinite(v) for v in logger["train_loss"])
    assert logger["oracle_calls"][0] == 0
    assert logger["oracle_calls"][1] == 4 * n_train
    assert logger["oracle_calls"][2] == 4 * n_train + 3 * n_train


def test_elapsed_time_excludes_diagnostic_full_gradient(tmp_path, monkeypatch):
    """NFG SVRG・ASAI SVRGが近似誤差算出のためだけに毎エポック計算する診断専用のフル勾配
    計算時間が，`elapsed_time` に計上されないことを確認する．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 1
    n_train = 16
    sleep_seconds = 3.0  # ResNet18はCPU上でも1エポックあたりの自然な計算時間が長いため，
    # 診断専用フル勾配計算の人為的遅延がその基準時間を明確に上回るよう大きめの値を用いる

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=n_train, n_test=8, batch_size=batch_size, seed=seed)

    original = ex005_train.compute_full_gradient_and_metrics

    def slow_compute_full_gradient_and_metrics(*args, **kwargs):
        time.sleep(sleep_seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(ex005_train, "compute_full_gradient_and_metrics", slow_compute_full_gradient_and_metrics)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
    ex005_train.run_variance_reduced(
        "ASAI_SVRG", str(tmp_path), load_dataloader_func, 0.001, batch_size, 1e-6, epochs, device, 0, logger
    )
    assert logger["elapsed_time"][1] < sleep_seconds


def test_is_stuck_near_chance_detects_flatlined_accuracy():
    """末尾の分類精度がチャンスレベル付近に留まっている場合，`True` を返すことを確認する．"""
    accuracies = [0.3, 0.2, 0.15, 0.11, 0.09, 0.10]
    assert ex005_train.is_stuck_near_chance(accuracies, num_classes=10, window=3)


def test_is_stuck_near_chance_does_not_flag_learning_progress():
    """精度が学習の進行とともに上昇し続けている場合，`False` を返すことを確認する．"""
    accuracies = [0.1, 0.2, 0.35, 0.5, 0.6, 0.7]
    assert not ex005_train.is_stuck_near_chance(accuracies, num_classes=10, window=3)


def test_is_stuck_near_chance_requires_minimum_history():
    """指定した`window`未満のエポック数しかない場合は`False`を返すことを確認する．"""
    assert not ex005_train.is_stuck_near_chance([0.1, 0.11], num_classes=10, window=3)


def test_stage_a_grid_matches_order_specification():
    """`.orders/order_033.md` 8節：Stage Aの比較手法がNFG SVRG・ASAI SVRGの2手法のみ
    （SGD・SVRGを含まない）であり，バッチサイズ・学習率・エポック数・Seed数が実験条件と
    一致することを確認する．"""
    assert set(ex005_train.METHODS) == {"NFG_SVRG", "ASAI_SVRG"}
    assert ex005_train.BATCH_SIZES == [128, 64, 32]
    assert ex005_train.LEARNING_RATES == [0.01, 0.001]
    assert ex005_train.EPOCHS == 12
    assert ex005_train.SEEDS == [0, 1, 2]


def test_class_names_length_matches_num_classes():
    """`data.py`のクラス名リストがクラス数10と一致することを確認する．"""
    assert len(ex005_data.CLASS_NAMES) == ex005_data.NUM_CLASSES == 10


def test_stage_b_grid_matches_order_specification():
    """`.orders/order_034.md` 2節：Stage Bの比較手法がSGD・SVRG・NFG SVRG・ASAI SVRGの
    4手法であり，バッチサイズ（128, 64のみ，32は対象外），学習率（0.001固定），
    バッチサイズごとのエポック数，Seed数（5）が実験条件と一致することを確認する．"""
    assert set(ex005_train.STAGE_B_METHODS) == {"SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"}
    assert ex005_train.STAGE_B_BATCH_SIZES == [128, 64]
    assert ex005_train.STAGE_B_LEARNING_RATE == 0.001
    assert ex005_train.STAGE_B_EPOCHS_BY_BATCH_SIZE == {128: 64, 64: 32}
    assert ex005_train.STAGE_B_SEEDS == [0, 1, 2, 3, 4]


def test_stage_a_and_stage_b_tasks_do_not_collide():
    """Stage AとStage Bのタスクリストを結合した際，`hp_name`によるディレクトリ名が
    衝突しない（Stage Aのエポック数12とStage Bのエポック数64/32が異なるため，同一の
    (method, batch_size, eta, seed) が存在してもディレクトリが分離されること）を確認する．"""
    stage_a_tasks = ex005_train._build_stage_a_tasks()
    stage_b_tasks = ex005_train._build_stage_b_tasks()

    assert len(stage_a_tasks) == 2 * 3 * 2 * 3  # 2手法 x 3バッチサイズ x 2学習率 x 3Seed
    assert len(stage_b_tasks) == 4 * 2 * 5  # 4手法 x 2バッチサイズ x 5Seed

    def target_dirs(tasks):
        return {
            (method, ex005_train.hp_name(eta, batch_size, epochs), seed)
            for method, batch_size, eta, epochs, seed in tasks
        }

    stage_a_dirs = target_dirs(stage_a_tasks)
    stage_b_dirs = target_dirs(stage_b_tasks)
    assert stage_a_dirs.isdisjoint(stage_b_dirs)


def test_stage_b_epochs_give_comparable_total_iterations():
    """Stage Bのバッチサイズごとのエポック数が，総イテレーション数（epochs * K）をおおむね
    揃えるよう設定されていることを確認する（`.orders/order_034.md` 2節）．"""
    k_by_batch_size = {128: 71, 64: 142}  # N_train=9025のときのK=ceil(N/bs)
    total_iterations = {
        bs: ex005_train.STAGE_B_EPOCHS_BY_BATCH_SIZE[bs] * k_by_batch_size[bs]
        for bs in ex005_train.STAGE_B_BATCH_SIZES
    }
    values = list(total_iterations.values())
    assert max(values) / min(values) < 1.1
