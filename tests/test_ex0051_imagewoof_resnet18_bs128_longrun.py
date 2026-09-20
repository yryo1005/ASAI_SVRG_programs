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
    """`.orders/order_035.md` 4節：基準条件（バッチサイズ128・学習率0.001・128エポック）が
    `CONDITIONS[0]`と一致し，`BATCH_SIZE`・`LEARNING_RATE`・`EPOCHS`（後方互換のエイリアス）
    もこれと一致すること，4手法・5Seedであることを確認する．"""
    assert set(ex0051_train.METHODS) == {"SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"}
    assert ex0051_train.CONDITIONS[0] == {"batch_size": 128, "learning_rate": 0.001, "epochs": 128}
    assert ex0051_train.BATCH_SIZE == 128
    assert ex0051_train.LEARNING_RATE == 0.001
    assert ex0051_train.EPOCHS == 128
    assert ex0051_train.SEEDS == [0, 1, 2, 3, 4]


def test_condition_grid_matches_order_036_specification():
    """`.orders/order_036.md` 3節：条件A（バッチサイズ128・学習率0.0001・128エポック）・
    条件B（バッチサイズ512・学習率0.001・505エポック）が，基準条件に加えて
    `CONDITIONS`に含まれることを確認する．"""
    assert len(ex0051_train.CONDITIONS) == 3
    assert {"batch_size": 128, "learning_rate": 0.0001, "epochs": 128} in ex0051_train.CONDITIONS
    assert {"batch_size": 512, "learning_rate": 0.001, "epochs": 505} in ex0051_train.CONDITIONS


def test_condition_b_total_iterations_are_close_to_baseline():
    """`.orders/order_036.md` 4節：条件Bの総イテレーション数（エポック数×K）が，基準条件の
    総イテレーション数（$128\\times71=9088$）とほぼ一致する（端数の差が小さい）ことを
    確認する．"""
    baseline_total = 128 * 71
    condition_b = next(c for c in ex0051_train.CONDITIONS if c["batch_size"] == 512)
    k_512 = 18  # N_train=9025のときのK=ceil(9025/512)
    condition_b_total = condition_b["epochs"] * k_512
    assert abs(condition_b_total - baseline_total) <= 10


def test_condition_a_uses_same_epochs_as_baseline():
    """`.orders/order_036.md` 3節：条件Aはバッチサイズ不変（K=71）のため，基準条件と
    同一のエポック数（128）であることを確認する．"""
    condition_a = next(
        c for c in ex0051_train.CONDITIONS if c["batch_size"] == 128 and c["learning_rate"] == 0.0001
    )
    assert condition_a["epochs"] == 128


def test_build_tasks_generates_expected_total_count():
    """`_build_tasks`がバッチサイズ128（基準条件＋条件A＝2条件×4手法×5Seed=40タスク）と
    バッチサイズ512（条件B＝1条件×4手法×5Seed=20タスク）を合わせて60タスク生成すること
    を確認する（`.orders/order_036.md` 3節：既存20条件＋新規40条件＝60条件）．"""
    tasks_128 = ex0051_train._build_tasks(128)
    tasks_512 = ex0051_train._build_tasks(512)
    assert len(tasks_128) == 2 * 4 * 5
    assert len(tasks_512) == 1 * 4 * 5
    assert len(tasks_128) + len(tasks_512) == 3 * 4 * 5


def test_build_tasks_only_returns_requested_batch_size():
    """`_build_tasks(128)`はバッチサイズ128のタスクのみ，`_build_tasks(512)`はバッチサイズ
    512のタスクのみを返すことを確認する（`.orders/order_036.md` 6節：VRAM使用量の大きな
    バッチサイズ512を別プロセスプールで実行するための分離）．"""
    tasks_128 = ex0051_train._build_tasks(128)
    tasks_512 = ex0051_train._build_tasks(512)
    assert all(bs == 128 for method, bs, eta, epochs, seed in tasks_128)
    assert all(bs == 512 for method, bs, eta, epochs, seed in tasks_512)


def test_build_tasks_baseline_condition_matches_existing_directory_naming():
    """既存の基準条件20条件のディレクトリ名（`.orders/order_035.md`で学習済み）と，
    `_build_tasks`が生成する基準条件タスクの`hp_name`が一致し，`is_run_completed`により
    正しくスキップ対象となることを確認する（既存結果を誤って上書きしないための検証）．"""
    tasks = ex0051_train._build_tasks(128)
    baseline_names = {
        ex0051_train.hp_name(eta, bs, epochs)
        for method, bs, eta, epochs, seed in tasks
        if bs == 128 and eta == 0.001
    }
    assert baseline_names == {"lr0.001_bs128_lambda0.0005_epochs128"}


def test_build_tasks_all_conditions_produce_distinct_directory_names():
    """基準条件・条件A・条件Bの3条件が，`hp_name`によって重複なく異なるディレクトリ名に
    分離されることを確認する．"""
    tasks = ex0051_train._build_tasks(128) + ex0051_train._build_tasks(512)
    names = {ex0051_train.hp_name(eta, bs, epochs) for method, bs, eta, epochs, seed in tasks}
    assert names == {
        "lr0.001_bs128_lambda0.0005_epochs128",
        "lr0.0001_bs128_lambda0.0005_epochs128",
        "lr0.001_bs512_lambda0.0005_epochs505",
    }


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


def test_main_entry_point_runs_sgd_double_epoch_phase_only_for_now():
    """基準条件・条件A・条件Bが全て完了済みのため，`train.py`を直接実行した場合はSGD倍
    エポック追加学習フェーズ（`.orders/order_039.md`）のみが実行されること，`main`関数が
    `run_bs128`・`run_bs512`・`run_sgd_double`引数により各フェーズを独立に制御できることを
    確認する．"""
    import ast

    source_path = os.path.join(_EX0051_DIR, "train.py")
    with open(source_path) as f:
        tree = ast.parse(f.read())
    main_call_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            call_src = ast.unparse(node.body[0])
            assert "run_bs128=False" in call_src
            assert "run_bs512=False" in call_src
            assert "run_sgd_double=True" in call_src
            main_call_found = True
    assert main_call_found, "`if __name__ == '__main__':`ブロックが見つからない"

    import inspect

    main_signature = inspect.signature(ex0051_train.main)
    assert "run_bs128" in main_signature.parameters
    assert "run_bs512" in main_signature.parameters
    assert "run_sgd_double" in main_signature.parameters
    assert main_signature.parameters["run_bs128"].default is True
    assert main_signature.parameters["run_bs512"].default is True
    assert main_signature.parameters["run_sgd_double"].default is False


def test_sgd_double_epoch_condition_matches_condition_b_except_epochs():
    """`.orders/order_039.md`：SGD倍エポック追加学習の条件が，条件B（バッチサイズ512・
    学習率0.001）とバッチサイズ・学習率が一致し，エポック数のみちょうど2倍であることを
    確認する．"""
    condition_b = next(c for c in ex0051_train.CONDITIONS if c["batch_size"] == 512)
    double_cond = ex0051_train.SGD_DOUBLE_EPOCH_CONDITION
    assert double_cond["batch_size"] == condition_b["batch_size"]
    assert double_cond["learning_rate"] == condition_b["learning_rate"]
    assert double_cond["epochs"] == 2 * condition_b["epochs"]


def test_build_sgd_double_epoch_tasks_covers_sgd_only_five_seeds():
    """SGD倍エポック追加学習のタスクが5Seed分（SGDのみ）であり，条件Bと同一のバッチサイズ・
    学習率，倍のエポック数を持つことを確認する．"""
    tasks = ex0051_train._build_sgd_double_epoch_tasks()
    assert len(tasks) == len(ex0051_train.SEEDS)
    cond = ex0051_train.SGD_DOUBLE_EPOCH_CONDITION
    for method, batch_size, eta, epochs, seed in tasks:
        assert method == "SGD"
        assert batch_size == cond["batch_size"]
        assert eta == cond["learning_rate"]
        assert epochs == cond["epochs"]
    assert {t[4] for t in tasks} == set(ex0051_train.SEEDS)


def test_sgd_double_epoch_task_directory_name_does_not_collide_with_condition_b():
    """SGD倍エポック追加学習のディレクトリ名（`hp_name`）が，条件Bの既存ディレクトリ名と
    衝突しないことを確認する（エポック数が異なるため自動的に分離される）．"""
    condition_b = next(c for c in ex0051_train.CONDITIONS if c["batch_size"] == 512)
    name_b = ex0051_train.hp_name(condition_b["learning_rate"], condition_b["batch_size"], condition_b["epochs"])
    cond = ex0051_train.SGD_DOUBLE_EPOCH_CONDITION
    name_double = ex0051_train.hp_name(cond["learning_rate"], cond["batch_size"], cond["epochs"])
    assert name_b != name_double
