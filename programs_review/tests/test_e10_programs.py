"""
E10（付録B.1，a9a の BCE の独立一様サンプリング．`ex015_e10_a9a_bce_uniform`）のプログラムのテスト．

`実験指示E10.md` 4.2節に対応する．小さな乱数のデータを使い，CPU で数秒で終わる．
1. `data.py`，`model.py` が `ex014_e8_a9a_supplement/` からの変更のないコピーであること．
2. `get_epoch_indices` が，同じシードの乱数生成器から E1（`ex006`）の `get_epoch_indices` と同じ添字の列を返すこと．
3. `uniform_with_replacement` の添字が $ [0, N) $ の範囲にあり，重複を含みうること．
4. 保存先の名前が `inner_{inner_start}_{sampling}_{hp_name}` の形式であり，既定のタスクが20個であること．
5. `random_reshuffle` では，学習の結果が E8 の実装と一致すること（追加の確認）．
6. `analysis_e10.py` の誤差床（最後の20点の中央値）と $ \\sigma^{2} $ が手計算と一致すること．
"""

import filecmp
import importlib
import os
import sys

import numpy as np
import pytest
import torch

_REVIEW_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROGRAMS = os.path.join(_REVIEW_ROOT, "programs")
_EX006_DIR = os.path.join(_PROGRAMS, "ex006_e1_mushroom_rho_sampling")
_EX014_DIR = os.path.join(_PROGRAMS, "ex014_e8_a9a_supplement")
_EX015_DIR = os.path.join(_PROGRAMS, "ex015_e10_a9a_bce_uniform")
sys.path.insert(0, os.path.join(_PROGRAMS, "common"))
sys.path.insert(0, _PROGRAMS)

from machine_learning_utils import ResultLogger  # noqa: E402

import analysis_e10  # noqa: E402


def _import_modules(directory: str):
    """
    概要: 実験のディレクトリの `data`，`model`，`train` を読み込む．他の実験のテストが同じ名前のモジュールを
        `sys.modules` に登録しているため，一時的に退避してから読み込み，読み込んだ後に元に戻す
        （`test_e8_programs.py` と同じ）．
    引数: directory (str)．実験のディレクトリ．
    戻り値: (data, model, train) のモジュールの組．
    """
    names = ("data", "model", "train")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    sys.path.insert(0, directory)
    try:
        modules = tuple(importlib.import_module(name) for name in names)
    finally:
        sys.path.remove(directory)
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return modules


e1_data, e1_model, e1_train = _import_modules(_EX006_DIR)
e8_data, e8_model, e8_train = _import_modules(_EX014_DIR)
e10_data, e10_model, e10_train = _import_modules(_EX015_DIR)

requires_data = pytest.mark.skipif(not os.path.exists(e10_data.RAW_DATA_PATH),
                                   reason="a9a のデータ（datasets/a9a/raw/a9a）が無い")


def _make_small_data(n=16, d=3, seed=0):
    """
    概要: テスト用の小さな二値分類のデータを作る（0/1 の特徴量．`test_e8_programs.py` と同じ作り方）．
    引数: n (int)．d (int)．seed (int)．
    戻り値: (X_train, y_train, X_test, y_test) の組．torch.float64 のテンソル．
    """
    g = torch.Generator().manual_seed(seed)
    X = (torch.rand(n, d, generator=g, dtype=torch.float64) > 0.5).double()
    y = (torch.rand(n, generator=g, dtype=torch.float64) > 0.5).double()
    return X, y, X[:4], y[:4]


def test_copied_files_unchanged():
    """`data.py`，`model.py` は `ex014` のものと同一である．"""
    for name in ("data.py", "model.py"):
        assert filecmp.cmp(os.path.join(_EX014_DIR, name), os.path.join(_EX015_DIR, name), shallow=False), name


@pytest.mark.parametrize("sampling", ["random_reshuffle", "uniform_with_replacement"])
def test_epoch_indices_match_e1(sampling):
    """同じシードの生成器から，E1 と同じ添字の列を（外側ループ3回分続けて）返す．"""
    rng_e1, rng_e10 = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(3):
        expected = e1_train.get_epoch_indices(rng_e1, 50, sampling)
        actual = e10_train.get_epoch_indices(rng_e10, 50, sampling)
        np.testing.assert_array_equal(actual, expected)


def test_random_reshuffle_indices_match_e8():
    """`random_reshuffle` の添字の列は E8 の `get_epoch_indices`（random reshuffling のみ）と同じである．"""
    rng_e8, rng_e10 = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(3):
        np.testing.assert_array_equal(e10_train.get_epoch_indices(rng_e10, 40, "random_reshuffle"),
                                      e8_train.get_epoch_indices(rng_e8, 40))


def test_uniform_indices_range_and_duplicates():
    """一様サンプリングの添字は [0, N) にあり，重複を含む（N = 1000 で重複が無い確率は無視できる）．"""
    indices = e10_train.get_epoch_indices(np.random.default_rng(0), 1000, "uniform_with_replacement")
    assert indices.shape == (1000,)
    assert indices.min() >= 0 and indices.max() < 1000
    assert np.unique(indices).size < 1000
    with pytest.raises(ValueError):
        e10_train.get_epoch_indices(np.random.default_rng(0), 10, "unknown")


@requires_data
def test_default_tasks_and_directory_names():
    """既定の引数で20試行（4手法 × 5シード，BCE，一様サンプリング）を作り，保存先が2節の3の形式である．"""
    args = e10_train.parse_args([])
    assert args.loss == ["bce"] and args.samplings == ["uniform_with_replacement"]
    assert args.methods == ["SGD", "SVRG_random", "NFG_SVRG_random", "ASAI_SVRG"]
    tasks = e10_train.build_tasks(args.loss, args.methods, args.inner_starts, args.samplings, args.seeds, args.epochs,
                                  "/out")
    assert len(tasks) == 20
    hp_name = "eta8.543689e-02_lambda1.125000e-01_K29304_epochs100"
    names = {os.path.relpath(t["target_dir"], "/out") for t in tasks}
    assert f"bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_{hp_name}/0" in names
    assert f"bce/SGD/inner_none_uniform_with_replacement_{hp_name}/4" in names
    assert all(t["sampling"] == "uniform_with_replacement" for t in tasks)


@pytest.mark.parametrize("method", ["SVRG_random", "NFG_SVRG_random", "ASAI_SVRG"])
def test_random_reshuffle_matches_e8_training(method):
    """`random_reshuffle` では，小さなデータでの学習の記録が E8 の実装とビット単位で一致する．"""
    data = _make_small_data()
    logs = []
    for module, extra in ((e8_train, ()), (e10_train, ("random_reshuffle",))):
        logger = ResultLogger()
        logger.set_names(*module.metric_names("bce"))
        module.run_variance_reduced(method, "snapshot", *extra, 0, data, "bce", 0.05, 1e-2, 0.0, 4, logger)
        logs.append(logger)
    for key in ("oracle_calls", "objective_gap", "grad_norm_sq", "approx_error"):
        np.testing.assert_array_equal(np.asarray(logs[1][key], dtype=float), np.asarray(logs[0][key], dtype=float))


def test_uniform_sampling_changes_training():
    """`uniform_with_replacement` では，SGD の2エポック目以降の記録が random reshuffling と異なる．"""
    data = _make_small_data(n=32)
    gaps = {}
    for sampling in ("random_reshuffle", "uniform_with_replacement"):
        logger = ResultLogger()
        logger.set_names(*e10_train.metric_names("bce"))
        e10_train.run_sgd(sampling, 0, data, "bce", 0.05, 1e-2, 0.0, 3, logger)
        gaps[sampling] = np.asarray(logger["objective_gap"], dtype=float)
    assert gaps["random_reshuffle"][0] == gaps["uniform_with_replacement"][0]
    assert np.all(gaps["random_reshuffle"][1:] != gaps["uniform_with_replacement"][1:])


def test_tail_median_matches_hand_calculation():
    """最後の20点の中央値．101点の系列 0, 1, ..., 100 では 81〜100 の中央値 90.5．非有限の値は除いてから数える．"""
    values = list(range(101))
    assert analysis_e10.tail_median(values) == pytest.approx(90.5)
    with_nan = list(range(101)) + [float("nan"), None]
    assert analysis_e10.tail_median(with_nan) == pytest.approx(90.5)
    assert analysis_e10.tail_median([3.0, 1.0, 2.0], window=20) == pytest.approx(2.0)
    assert np.isnan(analysis_e10.tail_median([None, float("inf")]))


def test_summarize_run_and_mean_standard_error():
    """1試行の床と打ち切り，平均 ± 標準誤差（ddof=1）．"""
    log = {"epoch": list(range(25)), "objective_gap": [float(v) for v in range(25)],
           "approx_error": [None] + [2.0 * v for v in range(1, 25)], "accuracy": [0.5] * 24 + [0.8],
           "terminated": [False] * 25}
    row = analysis_e10.summarize_run({"log": log, "config": {}}, "ASAI_SVRG")
    assert row["objective_floor"] == pytest.approx(14.5)       # 5..24 の中央値
    assert row["approx_floor"] == pytest.approx(29.0)          # 2*(5..24) の中央値
    assert row["final_accuracy"] == pytest.approx(0.8) and row["terminated"] is False
    assert np.isnan(analysis_e10.summarize_run({"log": log, "config": {}}, "SGD")["approx_floor"])
    mean, se = analysis_e10.mean_and_standard_error([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0) and se == pytest.approx(1.0 / np.sqrt(3.0))


def test_gradient_variance_matches_naive_computation():
    """σ² は，サンプルごとの勾配（重みと切片）を素朴に計算して求めた値と一致する．"""
    rng = np.random.default_rng(0)
    X = (rng.random((30, 4)) > 0.5).astype(float)
    y = (rng.random(30) > 0.5).astype(float)
    w, b, reg_lambda = rng.normal(size=4), 0.3, 0.1
    grads = []
    for n in range(30):
        p = 1.0 / (1.0 + np.exp(-(X[n] @ w + b)))
        # 正則化項を含めた1サンプルの勾配（差では打ち消し合う）．
        grads.append(np.concatenate([(p - y[n]) * X[n] + reg_lambda * w, [(p - y[n]) + reg_lambda * b]]))
    grads = np.array(grads)
    full = grads.mean(axis=0)
    naive = np.mean([np.sum((g - full) ** 2) for g in grads])
    assert analysis_e10.gradient_variance(X, y, w, b) == pytest.approx(naive, rel=1e-12)
