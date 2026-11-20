"""
E1（Mushroomにおける $ \\rho $ の異なる学習率での収束挙動とサンプリング方式の影響）のプログラムのテスト．

`実験指示E1.md` 2.6節に対応する．
1. 2.2節の $ \\eta $ が `実験計画.md` 3.3節の表の値と有効数字7桁で一致すること．
2. `random_reshuffle` の添字列と乱数生成器の状態が，E0の実装（`rng.permutation`）と同一であること．
3. `uniform_with_replacement` の添字が重複を含み得て，値域が $ [0, N_{\\mathrm{train}}) $ であること．
4. `inner_start="snapshot"` で，ASA-SVRGの各内側ループが $ {\\mathbf z}_s $ から始まること．
5. SGDの1エポックの `oracle_calls` の増分が $ N_{\\mathrm{train}} $ であること．

実験計画 3.4節（ランダム点の SVRG・NFG-SVRG）に対応する確認．
6. ランダム点の手法と最終点の手法で，各エポックの添字の列が一致すること．
7. 既存の `SVRG`・`NFG_SVRG`・`ASAI_SVRG` の結果が変更前（既存のログ）と完全に一致すること
   （シード0，$ \\rho = 298.4 $，2エポック）．
8. ランダム点の手法の `oracle_calls` と `config.json` の追加項目．
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

_REVIEW_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EX006_DIR = os.path.join(_REVIEW_ROOT, "programs", "ex006_e1_mushroom_rho_sampling")
sys.path.insert(0, os.path.join(_REVIEW_ROOT, "programs", "common"))
sys.path.insert(0, _EX006_DIR)

from machine_learning_utils import ResultLogger  # noqa: E402
from model import load_model, set_model_params  # noqa: E402
from optimizers import ASAISVRG  # noqa: E402
import train as e1_train  # noqa: E402

# 実験計画.md 3.3節の表（有効数字7桁）．
EXPECTED_ETA_BY_RHO = {
    0.5: 1.105364e-3,
    0.64: 1.590556e-3,
    0.8: 2.026671e-3,
    1.0: 2.492011e-3,
    2.0: 4.165425e-3,
    10.0: 8.141732e-3,
    298.4: 1.049767e-2,
}


def _relative_error(a: float, b: float) -> float:
    """
    概要: 2つの値の相対誤差を計算する．
    引数: a, b (float)．
    戻り値: error (float)．
    """
    return abs(a - b) / abs(b)


@pytest.mark.parametrize("target_rho", list(EXPECTED_ETA_BY_RHO))
def test_eta_matches_expected_table(target_rho):
    """
    概要: `build_hyperparameters` が計算する $ \\eta $ が，実験計画の表の値と有効数字7桁
        （相対誤差 $ 5\\times10^{-7} $ 未満）で一致することを確認する．
    引数: target_rho (float)．
    戻り値: なし
    """
    hyperparameters = e1_train.build_hyperparameters([target_rho])
    eta = hyperparameters[target_rho]["eta"]
    expected = EXPECTED_ETA_BY_RHO[target_rho]
    assert _relative_error(eta, expected) < 5e-7


def test_rho_298_4_uses_direct_formula():
    """
    概要: $ \\rho = 298.4 $ の $ \\eta $ が，二分法ではなく $ \\eta = 0.99/(3L) $ で直接求められる
        ことを確認する．
    引数: なし
    戻り値: なし
    """
    hyperparameters = e1_train.build_hyperparameters([298.4])
    L = hyperparameters[298.4]["L"]
    expected = e1_train.ETA_A_MARGIN / (3.0 * L)
    assert hyperparameters[298.4]["eta"] == pytest.approx(expected, rel=1e-15)


def test_random_reshuffle_matches_rng_permutation():
    """
    概要: `get_epoch_indices` の `"random_reshuffle"` が `rng.permutation` と同一の値・同一の
        乱数生成器の状態になることを確認する（E0の実装と同じ乱数消費）．
    引数: なし
    戻り値: なし
    """
    N_train = 123
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    indices = e1_train.get_epoch_indices(rng_a, N_train, "random_reshuffle")
    expected = rng_b.permutation(N_train)
    assert np.array_equal(indices, expected)
    assert np.array_equal(rng_a.bit_generator.state["state"]["state"], rng_b.bit_generator.state["state"]["state"])


def test_uniform_with_replacement_allows_duplicates_and_range():
    """
    概要: `"uniform_with_replacement"` の添字が重複を含み得て，値域が $ [0, N_{\\mathrm{train}}) $ で
        あることを確認する．
    引数: なし
    戻り値: なし
    """
    N_train = 50
    rng = np.random.default_rng(0)
    indices = e1_train.get_epoch_indices(rng, N_train, "uniform_with_replacement")
    assert indices.shape == (N_train,)
    assert indices.min() >= 0 and indices.max() < N_train
    assert len(set(indices.tolist())) < N_train  # 重複を含む（確率的にほぼ必ず成立する）


def test_get_epoch_indices_is_the_single_call_site_for_both_samplings():
    """
    概要: `run_variance_reduced` と `run_sgd` が，いずれも `get_epoch_indices` を介して添字列を
        生成しており，サンプリング方式ごとに異なる専用のコードパスを別途持たないことを確認する．
        これにより，`get_epoch_indices` 内の分岐以外では乱数消費の位置が2方式で揃う．
    引数: なし
    戻り値: なし
    """
    import inspect

    assert "get_epoch_indices" in inspect.getsource(e1_train.run_variance_reduced)
    assert "get_epoch_indices" in inspect.getsource(e1_train.run_sgd)


def _make_small_data(n=12, d=3, seed=0):
    """
    概要: テスト用の小さな二値分類データを作る．
    引数: n (int)．d (int)．seed (int)．
    戻り値: (X_train, y_train, X_test, y_test) の組．torch.float64テンソル．
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = (X[:, 0] > 0).double()
    return X, y, X[:4], y[:4]


def test_snapshot_inner_start_starts_each_inner_loop_from_z_s():
    """
    概要: `inner_start="snapshot"` で，ASA-SVRGの各外側ループの内側ループが，直前の外側ループ
        終了時のスナップショット $ {\\mathbf z}_s $ から始まることを確認する．
    引数: なし
    戻り値: なし
    """
    data = _make_small_data()
    X_train, y_train, _, _ = data
    N_train, input_dim = X_train.shape

    recorded_start_points = []
    original_run_inner_loop = e1_train.run_inner_loop

    def recording_run_inner_loop(model, snapshot_model, optimizer, X, y, reg_lambda, indices):
        recorded_start_points.append([p.detach().clone() for p in model.parameters()])
        original_run_inner_loop(model, snapshot_model, optimizer, X, y, reg_lambda, indices)

    e1_train.run_inner_loop = recording_run_inner_loop
    try:
        logger = ResultLogger()
        logger.set_names(*e1_train.METRIC_NAMES)
        e1_train.run_variance_reduced("ASAI_SVRG", "snapshot", "random_reshuffle", 0, data, eta=1e-2,
                                      reg_lambda=1e-3, f_star=0.0, epochs=3, logger=logger)
    finally:
        e1_train.run_inner_loop = original_run_inner_loop

    # 2エポック目以降の開始点は，前の外側ループが記録したスナップショット（get_snapshot_params）と
    # 一致するはずである．これを独立に再現し比較する．
    model = load_model(seed=0, input_dim=input_dim)
    snapshot_model = load_model(seed=0, input_dim=input_dim)
    optimizer = ASAISVRG(model.parameters(), lr=1e-2, K=N_train)
    rng = np.random.default_rng(0)
    for epoch in range(1, 3):
        indices = e1_train.get_epoch_indices(rng, N_train, "random_reshuffle")
        current_params = [p.detach().clone() for p in model.parameters()]
        assert all(torch.allclose(a, b) for a, b in zip(recorded_start_points[epoch - 1], current_params))
        optimizer.begin_epoch(rng)
        e1_train.run_inner_loop(model, snapshot_model, optimizer, X_train, y_train, 1e-3, indices)
        optimizer.end_epoch()
        set_model_params(snapshot_model, optimizer.get_snapshot_params(), source_model=model)
        set_model_params(model, optimizer.get_snapshot_params(), source_model=model)


def test_sgd_oracle_calls_increment_equals_n_train():
    """
    概要: SGDの1エポックあたりの `oracle_calls` の増分が $ N_{\\mathrm{train}} $ であることを確認する．
    引数: なし
    戻り値: なし
    """
    data = _make_small_data(n=16)
    X_train = data[0]
    N_train = X_train.shape[0]
    logger = ResultLogger()
    logger.set_names(*e1_train.METRIC_NAMES)
    e1_train.run_sgd("random_reshuffle", 0, data, eta=1e-2, reg_lambda=1e-3, f_star=0.0, epochs=3, logger=logger)
    oracle_calls = logger["oracle_calls"]
    assert len(oracle_calls) == 4  # 0エポック目 + 3エポック
    diffs = np.diff(oracle_calls)
    assert np.all(diffs == N_train)


def test_sgd_svrg_specific_fields_are_nan():
    """
    概要: SGDのログで，SVRG系固有の項目（`approx_error`，`approx_cos`）がNaNであることを確認する．
    引数: なし
    戻り値: なし
    """
    data = _make_small_data(n=16)
    logger = ResultLogger()
    logger.set_names(*e1_train.METRIC_NAMES)
    e1_train.run_sgd("random_reshuffle", 0, data, eta=1e-2, reg_lambda=1e-3, f_star=0.0, epochs=2, logger=logger)
    assert all(np.isnan(v) for v in logger["approx_error"])
    assert all(np.isnan(v) for v in logger["approx_cos"])


def test_build_tasks_restricts_uniform_with_replacement_to_three_rhos():
    """
    概要: `uniform_with_replacement` が $ \\rho \\in \\{0.5, 1.0, 298.4\\} $ のみに限定され，
        `random_reshuffle` は全ての $ \\rho $ で実施されることを確認する．
    引数: なし
    戻り値: なし
    """
    tasks = e1_train.build_tasks(["SVRG"], list(e1_train.ALL_SAMPLINGS), e1_train.RHO_CONDITIONS, ["snapshot"],
                                 [0], 1, "/out")
    by_sampling = {}
    for task in tasks:
        by_sampling.setdefault(task["sampling"], set()).add(task["hyperparams"]["target_rho"])
    assert by_sampling["random_reshuffle"] == set(e1_train.RHO_CONDITIONS)
    assert by_sampling["uniform_with_replacement"] == set(e1_train.UNIFORM_WITH_REPLACEMENT_RHOS)


def test_build_tasks_sgd_uses_inner_none_in_directory_name():
    """
    概要: SGDのタスクの保存先ディレクトリ名が `inner_none_...` となることを確認する．
    引数: なし
    戻り値: なし
    """
    tasks = e1_train.build_tasks(["SGD"], ["random_reshuffle"], [0.5], ["snapshot"], [0], 1, "/out")
    assert len(tasks) == 1
    assert "/inner_none_random_reshuffle_" in tasks[0]["target_dir"]


def _record_inner_loop_indices(method: str, sampling: str, epochs: int = 3) -> list:
    """
    概要: `run_variance_reduced` を小さなデータで実行し，各内側ループに渡された添字の列を記録する．
    引数:
        method (str)．手法名．
        sampling (str)．サンプリング方式．
        epochs (int)．外側ループ数．
    戻り値: recorded (list of numpy.ndarray)．各要素の形状は (N_train,)．
    """
    data = _make_small_data(n=16)
    recorded = []
    original_run_inner_loop = e1_train.run_inner_loop

    def recording_run_inner_loop(model, snapshot_model, optimizer, X, y, reg_lambda, indices):
        recorded.append(np.array(indices, copy=True))
        original_run_inner_loop(model, snapshot_model, optimizer, X, y, reg_lambda, indices)

    e1_train.run_inner_loop = recording_run_inner_loop
    try:
        logger = ResultLogger()
        logger.set_names(*e1_train.METRIC_NAMES)
        e1_train.run_variance_reduced(method, "snapshot", sampling, 0, data, eta=1e-2, reg_lambda=1e-3,
                                      f_star=0.0, epochs=epochs, logger=logger)
    finally:
        e1_train.run_inner_loop = original_run_inner_loop
    return recorded


@pytest.mark.parametrize("sampling", list(e1_train.ALL_SAMPLINGS))
@pytest.mark.parametrize("random_method,reference_method", [("SVRG_random", "SVRG"),
                                                            ("NFG_SVRG_random", "NFG_SVRG"),
                                                            ("SVRG_random", "ASAI_SVRG")])
def test_random_point_methods_keep_index_sequence(sampling, random_method, reference_method):
    """
    概要: ランダム点の手法の各エポックの添字の列が，最終点の手法・ASA-SVRG と一致することを確認する
        （スナップショットの選択用の乱数生成器を分けた効果）．
    引数:
        sampling (str)．サンプリング方式．
        random_method (str)．ランダム点の手法名．
        reference_method (str)．比較する手法名．
    戻り値: なし
    """
    random_indices = _record_inner_loop_indices(random_method, sampling)
    reference_indices = _record_inner_loop_indices(reference_method, sampling)
    assert len(random_indices) == len(reference_indices) == 3
    assert all(np.array_equal(a, b) for a, b in zip(random_indices, reference_indices))


def test_random_point_snapshot_rng_is_separate_and_seeded():
    """
    概要: スナップショットの選択用の乱数生成器のシードが `[seed, 1]` であり，その最初の選択が
        `default_rng([seed, 1]).integers(0, K)` と一致することを確認する．
    引数: なし
    戻り値: なし
    """
    assert e1_train.snapshot_rng_seed(3) == [3, 1]
    data = _make_small_data(n=16)
    X_train = data[0]
    N_train, input_dim = X_train.shape
    model = load_model(seed=0, input_dim=input_dim)
    optimizer = e1_train._VARIANCE_REDUCED_OPTIMIZER_CLASSES["SVRG_random"](model.parameters(), lr=1e-2, K=N_train)
    optimizer.begin_epoch(np.random.default_rng(e1_train.snapshot_rng_seed(0)))
    expected = int(np.random.default_rng([0, 1]).integers(0, N_train))
    assert optimizer._target_k == expected


def test_random_point_oracle_calls():
    """
    概要: `SVRG_random` は初期の全勾配 $ N $ と1エポックあたり $ 3N $，`NFG_SVRG_random` は1エポック
        あたり $ 2N $ の `oracle_calls` を記録することを確認する．
    引数: なし
    戻り値: なし
    """
    data = _make_small_data(n=16)
    N_train = data[0].shape[0]
    for method, initial, per_epoch in (("SVRG_random", N_train, 3 * N_train), ("NFG_SVRG_random", 0, 2 * N_train)):
        logger = ResultLogger()
        logger.set_names(*e1_train.METRIC_NAMES)
        e1_train.run_variance_reduced(method, "snapshot", "random_reshuffle", 0, data, eta=1e-2, reg_lambda=1e-3,
                                      f_star=0.0, epochs=2, logger=logger)
        oracle_calls = logger["oracle_calls"]
        assert oracle_calls[1] == initial + per_epoch
        assert oracle_calls[2] - oracle_calls[1] == per_epoch


_EXISTING_E1_ROOT = os.path.join(_REVIEW_ROOT, "outputs", "ex006_e1_mushroom_rho_sampling")
_COMPARED_METRICS = [name for name in e1_train.METRIC_NAMES if name != "elapsed_time"]


def _existing_log_path(method: str, task: dict) -> str:
    """
    概要: 2エポックのタスクに対応する，既存の100エポックの試行のログのパスを返す．
    引数:
        method (str)．手法名．
        task (dict)．`build_tasks` が返すタスク（2エポック）．
    戻り値: path (str)．
    """
    name = os.path.basename(os.path.dirname(task["target_dir"])).replace("_epochs2", "_epochs100")
    return os.path.join(_EXISTING_E1_ROOT, method, name, str(task["seed"]), "log.json")


_HEAD_REFERENCE_ROOT = os.path.join(_REVIEW_ROOT, "outputs", "_verify", "ex006_head_check")
_INTEGER_METRICS = ("epoch", "oracle_calls", "test_correct", "terminated")


def _assert_logs_match(new_log: ResultLogger, reference_log: ResultLogger, rel_tol: float) -> None:
    """
    概要: 2つのログの先頭の点（新しいログの長さ分）を項目ごとに比べる．整数の項目と精度は完全一致，
        実数の項目は相対誤差 `rel_tol` 以下または絶対誤差 $ 10^{-15} $ 以下（$ f^{*} \\approx 0.55 $ の数 ulp．
        `rel_tol = 0` は完全一致）を要求する．
    引数:
        new_log (ResultLogger)．新しく実行したログ．
        reference_log (ResultLogger)．比較対象のログ．
        rel_tol (float)．実数の項目の相対誤差の許容値．
    戻り値: なし
    """
    for name in _COMPARED_METRICS:
        new_values = list(new_log[name])
        reference_values = list(reference_log[name])[:len(new_values)]
        if rel_tol == 0.0 or name in _INTEGER_METRICS or name == "accuracy":
            assert np.array_equal(new_values, reference_values, equal_nan=True), name
        else:
            np.testing.assert_allclose(new_values, reference_values, rtol=rel_tol, atol=1e-15, err_msg=name)


@pytest.mark.parametrize("method", ["SVRG", "NFG_SVRG", "ASAI_SVRG"])
def test_existing_methods_unchanged_against_existing_logs(method, tmp_path):
    """
    概要: 変更後の `train.py` で既存の手法（シード0，$ \\rho = 298.4 $，random reshuffling，
        `inner_start="snapshot"`，2エポック）を実行し，既存の結果と比べる．
        1. 既存のログ（別のPCで実行）とは，整数の項目と精度が完全一致，実数の項目が相対 $ 10^{-12} $ 以下（または絶対 $ 10^{-15} $ 以下）で
           一致すること．別のPCとの差は $ f^{*} $ の1 ulp（$ 1.1\\times10^{-16} $）等の丸め誤差である
           （report_071 2.2節）．
        2. 変更前のコードを本PCで実行した結果（`outputs/_verify/ex006_head_check`）がある場合は，
           それと `elapsed_time` 以外の全項目が完全に一致すること．
    引数:
        method (str)．手法名．
        tmp_path (pathlib.Path)．pytest の一時ディレクトリ．
    戻り値: なし
    """
    task = e1_train.build_tasks([method], ["random_reshuffle"], [298.4], ["snapshot"], [0], 2, str(tmp_path))[0]
    existing_path = _existing_log_path(method, task)
    if not os.path.exists(existing_path):
        pytest.skip(f"既存のログが無い: {existing_path}")
    e1_train.run_single_experiment(task)
    new_log = ResultLogger(os.path.join(task["target_dir"], "log.json"))
    _assert_logs_match(new_log, ResultLogger(existing_path), rel_tol=1e-12)
    head_path = os.path.join(_HEAD_REFERENCE_ROOT, method, os.path.basename(os.path.dirname(task["target_dir"])),
                             "0", "log.json")
    if os.path.exists(head_path):
        _assert_logs_match(new_log, ResultLogger(head_path), rel_tol=0.0)
    with open(os.path.join(task["target_dir"], "config.json")) as f:
        assert "snapshot_rule" not in json.load(f)


def test_random_point_config_records_snapshot_rule(tmp_path):
    """
    概要: ランダム点の手法の `config.json` に `snapshot_rule` と `snapshot_rng_seed` が記録され，
        保存先が手法名のディレクトリになることを確認する（シード0，$ \\rho = 298.4 $，1エポック）．
    引数: tmp_path (pathlib.Path)．pytest の一時ディレクトリ．
    戻り値: なし
    """
    task = e1_train.build_tasks(["NFG_SVRG_random"], ["random_reshuffle"], [298.4], ["snapshot"], [0], 1,
                                str(tmp_path))[0]
    assert os.sep + "NFG_SVRG_random" + os.sep in task["target_dir"]
    e1_train.run_single_experiment(task)
    with open(os.path.join(task["target_dir"], "config.json")) as f:
        config = json.load(f)
    assert config["snapshot_rule"] == "uniform_random"
    assert config["snapshot_rng_seed"] == [0, 1]
