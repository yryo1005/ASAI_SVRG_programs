"""
E8（付録B.1，a9a の追加実験．`ex014_e8_a9a_supplement`）のプログラムのテスト．

`実験指示E8.md` 5節に対応する．小さな乱数のデータを使い，CPU で数秒で終わる．
1. `--part A` のタスクが10個，`--part B` が5個，`--part all` が15個であり，保存先が2節の形式であること．
   `--loss`・`--methods`・`--inner-starts` を明示した場合は `--part` を使わないこと．
2. `SVRG_random`・`NFG_SVRG_random` のスナップショットが内側ループの点の1つと一致し，その添字が
   `default_rng([seed, 1])` で選ばれること．reset のとき次の内側ループがスナップショットから始まること．
3. 最終点の手法と ASA-SVRG で，`snapshot_rng` を渡しても結果が E4 の実装（`ex008_e4_a9a/train.py`）と一致すること．
4. `oracle_calls` の増分が，`SVRG_random` で初期 $ N $・1外側ループ $ 3N $，`NFG_SVRG_random` で $ 2N $ であること．
5. ランダム点の手法の `config.json` に `snapshot_rule` と `snapshot_rng_seed` が記録されること．
6. `data.py`，`model.py`，`compare_e4.py` が `ex008_e4_a9a/` からの変更のないコピーであること．
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
_EX008_DIR = os.path.join(_PROGRAMS, "ex008_e4_a9a")
_EX014_DIR = os.path.join(_PROGRAMS, "ex014_e8_a9a_supplement")
sys.path.insert(0, os.path.join(_PROGRAMS, "common"))

from machine_learning_utils import ResultLogger  # noqa: E402


def _import_modules(directory: str):
    """
    概要: 実験のディレクトリの `data`，`model`，`train` を読み込む．他の実験のテストが同じ名前のモジュールを
        `sys.modules` に登録しているため，一時的に退避してから読み込み，読み込んだ後に元に戻す．
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


e4_data, e4_model, e4_train = _import_modules(_EX008_DIR)
e8_data, e8_model, e8_train = _import_modules(_EX014_DIR)

requires_data = pytest.mark.skipif(not os.path.exists(e8_data.RAW_DATA_PATH),
                                   reason="a9a のデータ（datasets/a9a/raw/a9a）が無い")


def _make_small_data(n=16, d=3, seed=0):
    """
    概要: テスト用の小さな二値分類のデータを作る（0/1 の特徴量．`test_e4_programs.py` と同じ作り方）．
    引数: n (int)．d (int)．seed (int)．
    戻り値: (X_train, y_train, X_test, y_test) の組．torch.float64 のテンソル．
    """
    g = torch.Generator().manual_seed(seed)
    X = (torch.rand(n, d, generator=g, dtype=torch.float64) > 0.5).double()
    y = (torch.rand(n, generator=g, dtype=torch.float64) > 0.5).double()
    return X, y, X[:4], y[:4]


def _run_vr(train_module, method, inner_start, loss, epochs=4, seed=0):
    """
    概要: 小さなデータで SVRG 系の手法を実行し，ログを返す．
    引数: train_module (module)．method (str)．inner_start (str)．loss (str)．epochs (int)．seed (int)．
    戻り値: logger (ResultLogger)．
    """
    logger = ResultLogger()
    logger.set_names(*train_module.metric_names(loss))
    reg_lambda = 1e-2 if loss == "bce" else 0.0
    train_module.run_variance_reduced(method, inner_start, seed, _make_small_data(), loss, eta=5e-2,
                                      reg_lambda=reg_lambda, f_star=0.0, epochs=epochs, logger=logger)
    return logger


# ---- 1. タスクの構築 ----

@requires_data
@pytest.mark.parametrize("argv,count", [(["--part", "A"], 10), (["--part", "B"], 5), (["--part", "all"], 15), ([], 15)])
def test_part_task_counts_and_paths(argv, count):
    """
    概要: `--part` ごとのタスク数と保存先の形式を確認する．
    """
    args = e8_train.parse_args(argv)
    tasks = e8_train.build_selected_tasks(e8_train.selections_from_args(args), args.seeds, args.epochs, "/root")
    assert len(tasks) == count
    names = {os.path.relpath(t["target_dir"], "/root") for t in tasks}
    expected = set()
    if "B" not in argv:
        for method in ("SVRG_random", "NFG_SVRG_random"):
            expected |= {f"bce/{method}/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/{s}"
                         for s in range(5)}
    if "A" not in argv:
        expected |= {f"mse/ASAI_SVRG/inner_last_iterate_eta1.530030e-01_K29304_epochs100/{s}" for s in range(5)}
    assert names == expected


def test_explicit_selection_ignores_part():
    """
    概要: `--loss`・`--methods`・`--inner-starts` を明示した場合は `--part` を使わないことを確認する．
    """
    args = e8_train.parse_args(["--part", "A", "--loss", "mse", "bce", "--methods", "SGD", "SVRG",
                                "--inner-starts", "snapshot"])
    assert e8_train.selections_from_args(args) == [
        {"losses": ["mse", "bce"], "methods": ["SGD", "SVRG"], "inner_starts": ["snapshot"]}]
    args = e8_train.parse_args(["--methods", "ASAI_SVRG"])
    assert e8_train.selections_from_args(args) == [
        {"losses": ["mse", "bce"], "methods": ["ASAI_SVRG"], "inner_starts": ["snapshot"]}]
    assert e8_train.selections_from_args(e8_train.parse_args(["--part", "B"])) == [e8_train.PARTS["B"]]


# ---- 2. ランダム点のスナップショット ----

def _record_inner_points(method, loss, epochs=3):
    """
    概要: `run_variance_reduced` を実行し，各内側ループの各ステップの直前の点 $ {\\mathbf w}_{s}^{k} $ を記録する．
    引数: method (str)．loss (str)．epochs (int)．
    戻り値: points (list of list of torch.Tensor)．points[s][k] は s+1 回目の内側ループの k 番目の点（パラメータを連結したもの）．
    """
    points = []

    def recording_run_inner_loop(model, snapshot_model, optimizer, X_train, y_train, loss_, reg_lambda, indices):
        epoch_points = []
        for index in indices:
            i = int(index)
            epoch_points.append(torch.cat([p.detach().flatten().clone() for p in model.parameters()]))
            e8_model.compute_gradient(model, X_train[i:i + 1], y_train[i:i + 1], loss_, reg_lambda)
            grad_at_snapshot = e8_model.compute_gradient(snapshot_model, X_train[i:i + 1], y_train[i:i + 1], loss_,
                                                         reg_lambda)
            optimizer.step(grad_at_snapshot)
        points.append(epoch_points)

    original = e8_train.run_inner_loop
    e8_train.run_inner_loop = recording_run_inner_loop
    try:
        _run_vr(e8_train, method, "snapshot", loss, epochs=epochs)
    finally:
        e8_train.run_inner_loop = original
    return points


@pytest.mark.parametrize("loss", ["mse", "bce"])
@pytest.mark.parametrize("method", ["SVRG_random", "NFG_SVRG_random"])
def test_random_snapshot_is_inner_point_selected_by_separate_rng(method, loss):
    """
    概要: s 回目の内側ループのスナップショット（reset のため s+1 回目の内側ループの開始点）が，
        添字 `default_rng([0, 1]).integers(0, K)`（s 回目の値）の内側ループの点と一致することを確認する．
    """
    points = _record_inner_points(method, loss)
    K = len(points[0])
    snapshot_rng = np.random.default_rng([0, 1])
    targets = [int(snapshot_rng.integers(0, K)) for _ in range(len(points))]
    assert len(set(targets)) > 1 or K == 1  # 乱数で選ばれていること（同じ添字が続くだけではない）
    for s in range(len(points) - 1):
        assert torch.equal(points[s + 1][0], points[s][targets[s]]), s


def test_random_point_keeps_index_sequence():
    """
    概要: ランダム点の手法の各内側ループの添字の列が，E4 の最終点の手法と同一であることを確認する．
    """
    recorded = {}
    for method in ("SVRG_random", "SVRG"):
        recorded[method] = []

        def recording(model, snapshot_model, optimizer, X, y, loss_, reg_lambda, indices, _m=method):
            recorded[_m].append(np.array(indices, copy=True))
            original(model, snapshot_model, optimizer, X, y, loss_, reg_lambda, indices)

        original = e8_train.run_inner_loop
        e8_train.run_inner_loop = recording
        try:
            _run_vr(e8_train, method, "snapshot", "bce", epochs=3)
        finally:
            e8_train.run_inner_loop = original
    assert all(np.array_equal(a, b) for a, b in zip(recorded["SVRG_random"], recorded["SVRG"]))
    assert len(recorded["SVRG"]) == 3


# ---- 3. 既存の手法が E4 の実装と一致すること ----

@pytest.mark.parametrize("loss", ["mse", "bce"])
@pytest.mark.parametrize("method,inner_start", [("SVRG", "snapshot"), ("NFG_SVRG", "snapshot"),
                                                ("ASAI_SVRG", "snapshot"), ("ASAI_SVRG", "last_iterate")])
def test_existing_methods_match_e4(method, inner_start, loss):
    """
    概要: 最終点の手法と ASA-SVRG で，`snapshot_rng` を渡しても結果（`elapsed_time` を除く全項目）が
        E4 の `run_variance_reduced` とビット単位で一致することを確認する．
    """
    a = _run_vr(e8_train, method, inner_start, loss)
    b = _run_vr(e4_train, method, inner_start, loss)
    for key in e8_train.metric_names(loss):
        if key == "elapsed_time":
            continue
        assert np.array_equal(np.asarray(a[key], dtype=np.float64), np.asarray(b[key], dtype=np.float64),
                              equal_nan=True), key


# ---- 4. oracle_calls ----

@pytest.mark.parametrize("method,initial,per_epoch,reference", [("SVRG_random", 1, 3, "SVRG"),
                                                                ("NFG_SVRG_random", 0, 2, "NFG_SVRG")])
def test_random_point_oracle_calls(method, initial, per_epoch, reference):
    """
    概要: `oracle_calls`・`full_grad_computations` の増分が規則どおりであり，E4 の最終点版と同じであることを確認する．
        近似誤差は `SVRG_random` で0である．0外側ループ目の値は最終点版と一致し，1外側ループ目以降は異なる．
    """
    N = _make_small_data()[0].shape[0]
    log = _run_vr(e8_train, method, "snapshot", "bce", epochs=3)
    ref = _run_vr(e4_train, reference, "snapshot", "bce", epochs=3)
    calls = log["oracle_calls"]
    assert calls[1] == (initial + per_epoch) * N
    assert all(calls[s + 1] - calls[s] == per_epoch * N for s in range(1, 3))
    assert calls == ref["oracle_calls"]
    assert log["full_grad_computations"] == ref["full_grad_computations"]
    if method == "SVRG_random":
        assert all(v == 0.0 for v in log["approx_error"])
    assert log["objective_gap"][0] == ref["objective_gap"][0]
    assert log["objective_gap"][2:] != ref["objective_gap"][2:]


# ---- 5. config.json ----

def test_config_records_snapshot_rule():
    """
    概要: ランダム点の手法の `config.json` に規則と乱数のシードが記録され，他の手法には記録されないことを確認する．
    """
    data = _make_small_data()
    hp = {"eta": 0.1, "L": 1.0, "reg_lambda": 0.1, "mu": 0.1, "L0": 0.9, "rho": 2.0}
    base = {"loss": "bce", "inner_start": "snapshot", "seed": 3, "epochs": 2, "hyperparams": hp}
    config = e8_train.build_config({**base, "method": "NFG_SVRG_random"}, data, 0.5, False)
    assert config["snapshot_rule"] == "uniform_random"
    assert config["snapshot_rng_seed"] == [3, 1]
    assert config["experiment"] == "ex014_e8_a9a_supplement"
    config = e8_train.build_config({**base, "method": "SVRG"}, data, 0.5, False)
    assert "snapshot_rule" not in config


# ---- 6. コピーしたファイル ----

@pytest.mark.parametrize("name", ["data.py", "model.py", "compare_e4.py"])
def test_copied_files_unchanged(name):
    """
    概要: 変更せずにコピーしたファイルが `ex008_e4_a9a/` と同一であることを確認する．
    """
    assert filecmp.cmp(os.path.join(_EX014_DIR, name), os.path.join(_EX008_DIR, name), shallow=False)
