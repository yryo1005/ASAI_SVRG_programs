"""
E1（Mushroomにおける $ \\rho $ の異なる学習率での収束挙動とサンプリング方式の影響）のプログラムのテスト．

`実験指示E1.md` 2.6節に対応する．
1. 2.2節の $ \\eta $ が `実験計画.md` 3.3節の表の値と有効数字7桁で一致すること．
2. `random_reshuffle` の添字列と乱数生成器の状態が，E0の実装（`rng.permutation`）と同一であること．
3. `uniform_with_replacement` の添字が重複を含み得て，値域が $ [0, N_{\\mathrm{train}}) $ であること．
4. `inner_start="snapshot"` で，ASA-SVRGの各内側ループが $ {\\mathbf z}_s $ から始まること．
5. SGDの1エポックの `oracle_calls` の増分が $ N_{\\mathrm{train}} $ であること．
"""

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
