"""
`programs/ex0051_imagewoof_resnet18_bs128_longrun/`（実験5b：バッチサイズ128長期学習と
プラトー判定方法の改善）の単体テスト．

モデルがDropout・BatchNormalizationを含まないこと（全てLayerNorm2dに置換済み），4手法
（SGD, SVRG, NFG SVRG, ASAI SVRG）が合成データ上でエラーなく完走すること，オラクル呼び出し
回数の正しさ，`elapsed_time` から診断専用フル勾配計算が除外されること，チャンスレベル張り付き
検出の正しさ，および本実験で新規実装したプラトー判定の3指標（隣接相対変化，正味変化量，
線形回帰の相対傾き）とそれらを統合する `is_plateaued` 関数の正しさを確認する．
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
_EX0051_DIR = os.path.join(_PROGRAMS_DIR, "ex0051_imagewoof_resnet18_bs128_longrun")
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


ex0051_model = _load_module("ex0051_model", _EX0051_DIR, "model.py")
ex0051_data = _load_module("ex0051_data", _EX0051_DIR, "data.py")
ex0051_train = _load_module("ex0051_train", _EX0051_DIR, "train.py")


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


def test_data_and_model_are_identical_copies_of_ex005():
    """`.orders/order_035.md` 2節：`data.py`・`model.py`がStage A/B（`ex005_imagewoof_
    resnet18/`）と完全に同一（`EXPERIMENT_NAME`も含め無変更）であり，データセットの
    キャッシュが再利用されることを確認する。"""
    assert ex0051_data.EXPERIMENT_NAME == "ex005_imagewoof_resnet18"
    ex005_data_path = os.path.join(_PROGRAMS_DIR, "ex005_imagewoof_resnet18", "data.py")
    ex005_model_path = os.path.join(_PROGRAMS_DIR, "ex005_imagewoof_resnet18", "model.py")
    ex0051_data_path = os.path.join(_EX0051_DIR, "data.py")
    ex0051_model_path = os.path.join(_EX0051_DIR, "model.py")
    with open(ex005_data_path) as f:
        ex005_data_src = f.read()
    with open(ex0051_data_path) as f:
        ex0051_data_src = f.read()
    with open(ex005_model_path) as f:
        ex005_model_src = f.read()
    with open(ex0051_model_path) as f:
        ex0051_model_src = f.read()
    assert ex005_data_src == ex0051_data_src
    assert ex005_model_src == ex0051_model_src


def test_model_has_no_dropout_or_batchnorm():
    """Dropout・BatchNormalization層を含まないことを確認する．全てのBatchNorm2d層が
    LayerNorm2dに置換されていることも確認する．"""
    model = ex0051_model.ResNet18LayerNorm(num_classes=10)
    n_layernorm2d = 0
    for module in model.modules():
        assert not isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
        if isinstance(module, ex0051_model.LayerNorm2d):
            n_layernorm2d += 1
    assert n_layernorm2d == 20


def test_model_is_deterministic_given_same_input():
    """同一入力に対し，モデルの出力が呼び出しごとに変化しない（決定論的である）ことを確認
    する．SVRG系手法の理論的前提の検証．"""
    model = ex0051_model.ResNet18LayerNorm(num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert torch.equal(y1, y2)


@pytest.mark.parametrize("method", ["NFG_SVRG", "ASAI_SVRG"])
def test_variance_reduced_runs_without_error_on_synthetic_data(method, tmp_path):
    """NFG SVRG・ASAI SVRGが合成データ上でエラーなく完走し，記録された評価指標の本数が
    エポック数+1と一致することを確認する（スモークテスト）．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    reg_lambda = 1e-6
    eta = 0.001

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=16, n_test=8, batch_size=batch_size, seed=seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    ex0051_train.run_variance_reduced(
        method, str(tmp_path), load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
    )

    assert len(logger["epoch"]) == epochs + 1
    assert all(np.isfinite(v) for v in logger["train_loss"])


def test_sgd_runs_without_error_on_synthetic_data(tmp_path):
    """SGDが合成データ上でエラーなく完走し，オラクル呼び出し回数が1エポックあたりNと
    なることを確認する。"""
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

    ex0051_train.run_sgd(
        str(tmp_path), load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
    )

    assert len(logger["epoch"]) == epochs + 1
    assert all(np.isfinite(v) for v in logger["train_loss"])
    assert logger["oracle_calls"][0] == 0
    assert logger["oracle_calls"][1] == n_train


def test_svrg_runs_without_error_on_synthetic_data(tmp_path):
    """SVRGが合成データ上でエラーなく完走することを確認する（オラクル呼び出し回数の詳細な
    検証はex005のテストで実施済みのため，ここでは完走のみを確認する）。"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    reg_lambda = 1e-6
    eta = 0.001

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=16, n_test=8, batch_size=batch_size, seed=seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    ex0051_train.run_variance_reduced(
        "SVRG", str(tmp_path), load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, 0, logger
    )

    assert len(logger["epoch"]) == epochs + 1
    assert all(np.isfinite(v) for v in logger["train_loss"])


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

    original = ex0051_train.compute_full_gradient_and_metrics

    def slow_compute_full_gradient_and_metrics(*args, **kwargs):
        time.sleep(sleep_seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(ex0051_train, "compute_full_gradient_and_metrics", slow_compute_full_gradient_and_metrics)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
    ex0051_train.run_variance_reduced(
        "ASAI_SVRG", str(tmp_path), load_dataloader_func, 0.001, batch_size, 1e-6, epochs, device, 0, logger
    )
    assert logger["elapsed_time"][1] < sleep_seconds


def test_is_stuck_near_chance_detects_flatlined_accuracy():
    """末尾の分類精度がチャンスレベル付近に留まっている場合，`True` を返すことを確認する．"""
    accuracies = [0.3, 0.2, 0.15, 0.11, 0.09, 0.10]
    assert ex0051_train.is_stuck_near_chance(accuracies, num_classes=10, window=3)


def test_is_stuck_near_chance_does_not_flag_learning_progress():
    """精度が学習の進行とともに上昇し続けている場合，`False` を返すことを確認する．"""
    accuracies = [0.1, 0.2, 0.35, 0.5, 0.6, 0.7]
    assert not ex0051_train.is_stuck_near_chance(accuracies, num_classes=10, window=3)


def test_experiment_grid_matches_order_specification():
    """`.orders/order_035.md` 4節：4手法×5Seed=20条件，バッチサイズ128・学習率0.001固定，
    128エポックであることを確認する．"""
    assert set(ex0051_train.METHODS) == {"SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"}
    assert ex0051_train.BATCH_SIZE == 128
    assert ex0051_train.LEARNING_RATE == 0.001
    assert ex0051_train.EPOCHS == 128
    assert ex0051_train.SEEDS == [0, 1, 2, 3, 4]


# --- プラトー判定の3指標（`.orders/order_035.md` 3節）のテスト ---


def test_compute_trailing_relative_change_matches_stage_b_reference_value():
    """Stage B（`report_034.md`）のASAI SVRG・バッチサイズ128（Seed平均）の末尾6エポックの
    分類精度軌跡に対し，隣接相対変化の最大値が約0.68%であったという既存の実測値と整合する
    小さい値を返すことを確認する（既存指標の回帰確認）．"""
    # Stage Bのログから抜粋した，プラトーに近いとされた末尾の分類精度軌跡（概略値）
    values = [0.395, 0.398, 0.400, 0.401, 0.402, 0.403]
    rel_change = ex0051_train.compute_trailing_relative_change(values, window=5)
    assert rel_change < 0.02


def test_compute_trailing_net_change_detects_gradual_monotonic_increase():
    """隣接ステップの変化は小さいが，区間全体では無視できない正味の増加がある数値列に対し，
    正味変化量が閾値を超える大きな値を返すことを確認する（`.orders/order_035.md` 1節が
    指摘する，隣接相対変化のみでは検出できない「緩やかな単調増加」の検出）．"""
    # 1ステップあたり約1%ずつ緩やかに増加し続ける数値列（隣接相対変化は小さいまま）
    values = [0.30 * (1.01 ** i) for i in range(11)]
    rel_change = ex0051_train.compute_trailing_relative_change(values, window=10)
    net_change = ex0051_train.compute_trailing_net_change(values, window=10)
    assert rel_change < 0.02  # 隣接相対変化は小さい（プラトーに見える）
    assert net_change > 0.02  # しかし正味変化量は閾値を超える（実際は未プラトー）


def test_compute_trailing_net_change_is_small_for_flat_sequence():
    """値がほぼ一定の数値列に対し，正味変化量が小さい値を返すことを確認する．"""
    values = [0.40, 0.401, 0.399, 0.402, 0.400, 0.401]
    net_change = ex0051_train.compute_trailing_net_change(values, window=5)
    assert net_change < 0.02


def test_compute_trailing_regression_slope_detects_ongoing_trend():
    """明確な上昇トレンドを持つ数値列に対し，回帰の相対傾きが閾値を超える値を返すことを
    確認する．"""
    values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    slope = ex0051_train.compute_trailing_regression_slope(values, window=5)
    assert abs(slope) > 0.001


def test_compute_trailing_regression_slope_is_near_zero_for_flat_sequence():
    """値がほぼ一定の数値列に対し，回帰の相対傾きがほぼゼロであることを確認する．"""
    values = [0.40, 0.401, 0.399, 0.402, 0.400, 0.401]
    slope = ex0051_train.compute_trailing_regression_slope(values, window=5)
    assert abs(slope) < 0.001


def test_compute_trailing_metrics_return_inf_for_nonfinite_values():
    """3指標のいずれも，数値列に非有限値（NaN・Inf）が含まれる場合`float("inf")`を返す
    ことを確認する．"""
    values = [0.1, 0.2, float("nan"), 0.4, 0.5, 0.6]
    assert ex0051_train.compute_trailing_relative_change(values, window=5) == float("inf")
    assert ex0051_train.compute_trailing_net_change(values, window=5) == float("inf")
    assert ex0051_train.compute_trailing_regression_slope(values, window=5) == float("inf")


def test_is_plateaued_flags_gradual_monotonic_increase_as_not_plateaued():
    """`.orders/order_035.md` 1節が指摘する「隣接相対変化は小さいが緩やかに単調増加を
    続ける」数値列に対し，`is_plateaued`が3指標のうち少なくとも1つ（正味変化量または
    回帰傾き）により「未プラトー」と判定することを確認する（3指標統合の主目的）．"""
    values = [0.30 * (1.01 ** i) for i in range(11)]
    result = ex0051_train.is_plateaued(values, window=10)
    assert result["relative_change"] < ex0051_train.PLATEAU_THRESHOLD_RELATIVE_CHANGE
    assert not result["plateaued"]


def test_is_plateaued_flags_flat_sequence_as_plateaued():
    """全ての指標が小さい，明確に横ばいの数値列に対し，`is_plateaued`が「プラトー」と
    判定することを確認する．"""
    values = [0.400 + 0.0001 * ((-1) ** i) for i in range(11)]
    result = ex0051_train.is_plateaued(values, window=10)
    assert result["plateaued"]


def test_is_plateaued_requires_minimum_history():
    """`window + 1`個未満の履歴しかない場合，`plateaued=False`を返すことを確認する．"""
    result = ex0051_train.is_plateaued([0.1, 0.2, 0.3], window=10)
    assert not result["plateaued"]
    assert np.isnan(result["relative_change"])
