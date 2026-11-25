"""
E11（`実験指示E11.md`．Mushroom における全てのスナップショットと内側ループの開始点の構成，ex006 の追加）のテスト．

指示書3節に対応する．
1. `SVRG_average` の1外側ループの後のスナップショットが，内側ループの更新前のパラメータ
   $ {\\mathbf w}_{s-1}^{0},\\dots,{\\mathbf w}_{s-1}^{K-1} $ の平均と一致すること．
2. `SVRG_average` の `approx_error` が0（丸め誤差の範囲）であること．
3. `inner_start="last_iterate"` で，次の内側ループの最初のパラメータが前の内側ループの最終点と一致すること．
4. `SVRG_average` の `oracle_calls` が $ N + 3Ns $ であること，手法の一覧と全勾配の手法の一覧．
5. 指示書5節のコマンドのタスク数（continue の4構成：140 + 60，SVRG（average）の reset：35 + 15）．
"""

import importlib
import os
import sys

import numpy as np
import pytest
import torch

_REVIEW_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EX006_DIR = os.path.join(_REVIEW_ROOT, "programs", "ex006_e1_mushroom_rho_sampling")
sys.path.insert(0, os.path.join(_REVIEW_ROOT, "programs", "common"))

from machine_learning_utils import ResultLogger  # noqa: E402


def _import_ex006_train():
    """
    概要: ex006 の `train` を読み込む．他のテストが同じ名前のモジュール（`data`，`model`，`train`）を
        登録している場合があるため，一時的に退避してから読み込み，終了後に元に戻す．
    引数: なし
    戻り値: train (module)．
    """
    names = ("data", "model", "train")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    sys.path.insert(0, _EX006_DIR)
    try:
        module = importlib.import_module("train")
    finally:
        sys.path.remove(_EX006_DIR)
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return module


e1_train = _import_ex006_train()

VR_METHODS = ["SVRG", "NFG_SVRG", "ASAI_SVRG", "SVRG_random", "NFG_SVRG_random", "SVRG_average"]


def _make_small_data(n=12, d=3, seed=0):
    """
    概要: テスト用の小さな二値分類データを作る（`test_e1_programs.py` と同じ）．
    引数: n (int)．d (int)．seed (int)．
    戻り値: (X_train, y_train, X_test, y_test) の組．torch.float64 テンソル．
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = (X[:, 0] > 0).double()
    return X, y, X[:4], y[:4]


def _params(model) -> list:
    """
    概要: モデルのパラメータの複製を返す．
    引数: model (torch.nn.Module)．
    戻り値: params (list of torch.Tensor)．
    """
    return [p.detach().clone() for p in model.parameters()]


def _run_recording(method, inner_start, epochs, monkeypatch, data=None):
    """
    概要: `run_variance_reduced` を実行し，各内側ループの開始点，終了点，スナップショット，
        および各ステップの更新前のパラメータを記録する．
    引数:
        method (str)．手法名．
        inner_start (str)．"snapshot" または "last_iterate"．
        epochs (int)．外側ループ数．
        monkeypatch (pytest.MonkeyPatch)．
        data (tuple, optional)．`_make_small_data` の戻り値．
    戻り値: record (dict)．`start`，`end`，`snapshot`（各内側ループの開始時の $ {\\mathbf z} $），
        `iterates`（外側ループごとの更新前のパラメータの列），`logger` を持つ．
    """
    data = _make_small_data() if data is None else data
    record = {"start": [], "end": [], "snapshot": [], "iterates": []}

    def recording_run_inner_loop(model, snapshot_model, optimizer, X, y, reg_lambda, indices):
        record["start"].append(_params(model))
        record["snapshot"].append(_params(snapshot_model))
        iterates = []
        for index in indices:
            i = int(index)
            iterates.append(_params(model))
            e1_train.compute_gradient(model, X[i:i + 1], y[i:i + 1], reg_lambda)
            grad_at_snapshot = e1_train.compute_gradient(snapshot_model, X[i:i + 1], y[i:i + 1], reg_lambda)
            optimizer.step(grad_at_snapshot)
        record["iterates"].append(iterates)
        record["end"].append(_params(model))

    monkeypatch.setattr(e1_train, "run_inner_loop", recording_run_inner_loop)
    logger = ResultLogger()
    logger.set_names(*e1_train.METRIC_NAMES)
    e1_train.run_variance_reduced(method, inner_start, "random_reshuffle", 0, data, eta=1e-2, reg_lambda=1e-3,
                                  f_star=0.0, epochs=epochs, logger=logger)
    record["logger"] = logger
    return record


def test_recording_inner_loop_matches_original():
    """
    概要: 記録用の内側ループ（`_run_recording`）が元の `run_inner_loop` と同じ計算であることを，
        3外側ループの全ての記録項目の完全一致で確認する．
    引数: なし
    戻り値: なし
    """
    logs = []
    for patch in (False, True):
        mp = pytest.MonkeyPatch()
        try:
            if patch:
                logs.append(_run_recording("SVRG_average", "last_iterate", 3, mp)["logger"])
            else:
                logger = ResultLogger()
                logger.set_names(*e1_train.METRIC_NAMES)
                e1_train.run_variance_reduced("SVRG_average", "last_iterate", "random_reshuffle", 0,
                                              _make_small_data(), eta=1e-2, reg_lambda=1e-3, f_star=0.0,
                                              epochs=3, logger=logger)
                logs.append(logger)
        finally:
            mp.undo()
    for name in e1_train.METRIC_NAMES:
        if name != "elapsed_time":
            assert np.array_equal(logs[0][name], logs[1][name], equal_nan=True), name


@pytest.mark.parametrize("inner_start", ["snapshot", "last_iterate"])
def test_svrg_average_snapshot_is_mean_of_pre_update_iterates(inner_start, monkeypatch):
    """
    概要: `SVRG_average` の外側ループ $ s $ の後のスナップショット（次の内側ループの開始時の $ {\\mathbf z}_{s} $）が，
        外側ループ $ s $ の更新前のパラメータ $ {\\mathbf w}_{s-1}^{0},\\dots,{\\mathbf w}_{s-1}^{K-1} $ の平均と
        一致することを確認する（丸め誤差の範囲）．
    引数: inner_start (str)．monkeypatch．
    戻り値: なし
    """
    record = _run_recording("SVRG_average", inner_start, 3, monkeypatch)
    for s in range(2):
        iterates = record["iterates"][s]
        assert len(iterates) == _make_small_data()[0].shape[0]
        for j, z in enumerate(record["snapshot"][s + 1]):
            mean = torch.stack([w[j] for w in iterates]).mean(dim=0)
            torch.testing.assert_close(z, mean, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("inner_start", ["snapshot", "last_iterate"])
def test_svrg_average_approx_error_is_zero(inner_start):
    """
    概要: `SVRG_average` のスナップショット勾配はスナップショットにおける全勾配であるため，
        全ての外側ループで `approx_error` が丸め誤差の水準（$ 10^{-28} $ 未満）であることを確認する．
    引数: inner_start (str)．
    戻り値: なし
    """
    logger = ResultLogger()
    logger.set_names(*e1_train.METRIC_NAMES)
    e1_train.run_variance_reduced("SVRG_average", inner_start, "random_reshuffle", 0, _make_small_data(), eta=1e-2,
                                  reg_lambda=1e-3, f_star=0.0, epochs=4, logger=logger)
    approx_error = np.asarray(logger["approx_error"], dtype=np.float64)
    assert approx_error.shape == (5,)
    assert np.all(approx_error < 1e-28)


@pytest.mark.parametrize("method", VR_METHODS)
def test_last_iterate_starts_from_previous_final_point(method, monkeypatch):
    """
    概要: `inner_start="last_iterate"` で，外側ループ $ s \\ge 2 $ の内側ループの最初のパラメータが，前の内側ループの
        最終点 $ {\\mathbf w}_{s-1}^{K} $ と完全に一致することを確認する．最終点以外の手法では，開始点が
        スナップショットと異なる（continue が reset と別の計算である）ことも確認する．
    引数: method (str)．monkeypatch．
    戻り値: なし
    """
    record = _run_recording(method, "last_iterate", 4, monkeypatch)
    assert len(record["start"]) == 4
    for s in range(1, 4):
        assert all(torch.equal(a, b) for a, b in zip(record["start"][s], record["end"][s - 1]))
    # ランダム点の手法は，偶然に最終点がスナップショットに選ばれる外側ループがある（K = 12 のテストデータ）．
    # そのため，少なくとも1つの外側ループで開始点とスナップショットが異なることを確認する．
    same = [all(torch.equal(a, b) for a, b in zip(record["start"][s], record["snapshot"][s])) for s in range(1, 4)]
    if method in ("SVRG", "NFG_SVRG"):
        assert all(same)
    else:
        assert not all(same)


@pytest.mark.parametrize("method", VR_METHODS)
def test_snapshot_inner_start_unchanged(method, monkeypatch):
    """
    概要: `inner_start="snapshot"` では，従来どおり外側ループ $ s \\ge 2 $ の内側ループが $ {\\mathbf z}_{s-1} $ から始まることを確認する．
    引数: method (str)．monkeypatch．
    戻り値: なし
    """
    record = _run_recording(method, "snapshot", 3, monkeypatch)
    for s in range(1, 3):
        assert all(torch.equal(a, b) for a, b in zip(record["start"][s], record["snapshot"][s]))


def test_svrg_average_oracle_calls_and_method_lists():
    """
    概要: `SVRG_average` の `oracle_calls` が $ N + 3Ns $ であり，`full_grad_computations` が $ 1 + 3s $ であることと，
        手法の一覧・全勾配の手法の一覧・Optimizer のクラスを確認する．
    引数: なし
    戻り値: なし
    """
    data = _make_small_data(n=16)
    N = data[0].shape[0]
    logger = ResultLogger()
    logger.set_names(*e1_train.METRIC_NAMES)
    e1_train.run_variance_reduced("SVRG_average", "snapshot", "random_reshuffle", 0, data, eta=1e-2, reg_lambda=1e-3,
                                  f_star=0.0, epochs=3, logger=logger)
    assert list(logger["oracle_calls"]) == [0] + [N + 3 * N * s for s in range(1, 4)]
    assert list(logger["full_grad_computations"])[1:] == [1.0 + 3.0 * s for s in range(1, 4)]
    assert "SVRG_average" in e1_train.ALL_METHODS
    assert set(e1_train.FULL_GRADIENT_METHODS) == {"SVRG", "SVRG_random", "SVRG_average"}
    assert e1_train._VARIANCE_REDUCED_OPTIMIZER_CLASSES["SVRG_average"].__name__ == "SVRGAverage"
    assert "SVRG_average" not in e1_train.RANDOM_SNAPSHOT_METHODS


def test_build_tasks_for_e11_commands(tmp_path):
    """
    概要: 指示書5節の2つのコマンドのタスク数と保存先の名前を確認する．
        continue の4構成は $ 4 \\times (7 + 3) \\times 5 = 200 $，SVRG（average）の reset は $ (7 + 3) \\times 5 = 50 $ である．
    引数: tmp_path (pathlib.Path)．
    戻り値: なし
    """
    samplings = list(e1_train.ALL_SAMPLINGS)
    continue_tasks = e1_train.build_tasks(["SVRG_random", "NFG_SVRG_random", "SVRG_average", "ASAI_SVRG"], samplings,
                                          e1_train.RHO_CONDITIONS, ["last_iterate"], e1_train.SEEDS, 100,
                                          str(tmp_path))
    reset_tasks = e1_train.build_tasks(["SVRG_average"], samplings, e1_train.RHO_CONDITIONS, ["snapshot"],
                                       e1_train.SEEDS, 100, str(tmp_path))
    assert len(continue_tasks) == 200 and len(reset_tasks) == 50
    assert sum(t["sampling"] == "uniform_with_replacement" for t in continue_tasks) == 60
    assert all("/inner_last_iterate_" in t["target_dir"] for t in continue_tasks)
    names = {os.path.basename(os.path.dirname(t["target_dir"])) for t in reset_tasks}
    assert "inner_snapshot_random_reshuffle_rho298.4_eta1.049767e-02_lambda9.155984e-01_K7311_epochs100" in names


# ---------------------------------------------------------------------------
# 集計（analysis_e11.py）
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(_REVIEW_ROOT, "programs"))
import analysis_e11  # noqa: E402


def _synthetic_run(gaps, method="ASAI_SVRG", inner_start="last_iterate", seed=0, n_train=10):
    """
    概要: 集計のテスト用の合成の試行を作る．$ \\#\\mathrm{grad}/N $ は外側ループごとに2ずつ増える．
    引数: gaps (list of float)．目的関数誤差の列．method (str)．inner_start (str)．seed (int)．n_train (int)．
    戻り値: run (dict)．
    """
    n = len(gaps)
    log = {"objective_gap": list(gaps), "approx_error": list(range(n)), "oracle_calls": [2 * n_train * s for s in range(n)],
           "terminated": [False] * n, "epoch": list(range(n))}
    return {"method": method, "inner_start": inner_start, "sampling": "random_reshuffle", "target_rho": 1.0,
            "seed": seed, "log": log, "x": np.arange(n) * 2.0, "config": {"epochs": n - 1}}


def test_hat_r_lsq_matches_hand_calculation():
    """$ \\Delta_s = 0.5^{s} $ なら傾きは $ \\log 0.5 $，$ \\hat{r} = 0.5 $．$ 10^{-12} $ 以下の点と $ s = 0 $ は使わない．"""
    gaps = [1.0] + [0.5 ** s for s in range(1, 6)] + [1e-13, 0.5]
    assert analysis_e11.hat_r_lsq(_synthetic_run(gaps)) == pytest.approx(0.5, rel=1e-12)
    assert analysis_e11.hat_r_lsq(_synthetic_run([1.0, 0.1, 1e-13, 1e-14])) is None


def test_reach_tail_and_select_by_inner_start():
    """到達の $ \\#\\mathrm{grad}/N $，最後の20の中央値，`inner_start` による絞り込み，平均 ± 標準誤差．"""
    run = _synthetic_run([1.0, 1e-5, 1e-11, 1e-12])
    assert analysis_e11.reach_grad_n(run) == pytest.approx(4.0)
    assert analysis_e11.reach_grad_n(_synthetic_run([1.0, 1e-5])) is None
    long_run = _synthetic_run([1.0] * 30)
    assert analysis_e11.tail_median(long_run, "approx_error") == pytest.approx(np.median(np.arange(10, 30)))
    runs = [_synthetic_run([1.0], inner_start="snapshot", seed=1), _synthetic_run([1.0], seed=0)]
    assert [r["inner_start"] for r in analysis_e11.select_runs(runs, "ASAI_SVRG", "last_iterate",
                                                               "random_reshuffle", 1.0)] == ["last_iterate"]
    mean, se, n = analysis_e11.mean_se([1.0, None, 3.0, float("nan")])
    assert (mean, n) == (2.0, 2) and se == pytest.approx(1.0)
    assert analysis_e11.fmt([1.0, None], digits=1, total=2) == "1.0（1/2）"
