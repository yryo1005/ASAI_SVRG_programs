"""
`ex016_imagewoof_bs128_walltime`（ImageWoof・バッチサイズ128 の5手法の実時間の取り直し）のテスト．CPU で数秒で終わる．

1. タスクの順序が，シードごとに SGD → SVRG（最終点）→ NFG-SVRG（最終点）→ ASA-SVRG（continue）→ ASA-SVRG（reset）であること．
2. エポック数，内側ループの開始点，保存先の名前．
3. `data.py`，`model.py`，`timed_training.py` が ex009 のものと同一であること．
4. 集計（1エポックの時間，予算での値，揺らぎ，実行の順序）が手計算と一致すること．
"""

import filecmp
import importlib
import os
import sys

import numpy as np
import pytest

_REVIEW_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROGRAMS = os.path.join(_REVIEW_ROOT, "programs")
_EX009_DIR = os.path.join(_PROGRAMS, "ex009_e3_imagewoof_walltime")
_EX016_DIR = os.path.join(_PROGRAMS, "ex016_imagewoof_bs128_walltime")
sys.path.insert(0, os.path.join(_PROGRAMS, "common"))
sys.path.insert(0, _PROGRAMS)

import analysis_ex016 as analysis  # noqa: E402


def _import_train():
    """
    概要: ex016 の `train` を読み込む．他のテストが同じ名前のモジュールを登録しているため，一時的に退避する．
    引数: なし
    戻り値: train (module)．
    """
    names = ("data", "model", "train", "timed_training")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    sys.path.insert(0, _EX016_DIR)
    try:
        module = importlib.import_module("train")
    finally:
        sys.path.remove(_EX016_DIR)
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return module


train = _import_train()


def test_task_order_and_conditions():
    """5周（シード0〜4）× 5手法 = 25試行が，シードごとに指定の順で並ぶ．"""
    tasks = train.build_tasks(list(train.CONDITIONS), [0, 1, 2, 3, 4], "/out")
    assert len(tasks) == 25
    expected = ["SGD", "SVRG_final", "NFG_SVRG_final", "ASAI_SVRG_continue", "ASAI_SVRG_reset"]
    for seed in range(5):
        block = tasks[5 * seed:5 * seed + 5]
        assert [t["condition"] for t in block] == expected
        assert all(t["seed"] == seed for t in block)
    by_condition = {t["condition"]: t for t in tasks[:5]}
    assert by_condition["SGD"]["inner_start"] is None
    assert by_condition["ASAI_SVRG_continue"]["inner_start"] == "last_iterate"
    assert by_condition["ASAI_SVRG_reset"]["inner_start"] == "snapshot"
    assert {c: t["run_config"]["epochs"] for c, t in by_condition.items()} == {
        "SGD": 200, "SVRG_final": 67, "NFG_SVRG_final": 100, "ASAI_SVRG_continue": 100, "ASAI_SVRG_reset": 100}
    assert all(t["run_config"]["record_omega"] is False and t["run_config"]["batch_size"] == 128 for t in tasks)
    assert os.path.relpath(by_condition["ASAI_SVRG_reset"]["target_dir"], "/out") == \
        "ASAI_SVRG/inner_snapshot_lr0.001_bs128_lambda0.0005_epochs100/0"
    assert os.path.relpath(by_condition["SGD"]["target_dir"], "/out") == \
        "SGD/inner_none_lr0.001_bs128_lambda0.0005_epochs200/0"


def test_epochs_override():
    """`--epochs` を指定すると全ての条件のエポック数と保存先が変わる．"""
    tasks = train.build_tasks(["SVRG_final"], [0], "/out", epochs_override=2)
    assert tasks[0]["run_config"]["epochs"] == 2 and tasks[0]["target_dir"].endswith("epochs2/0")


def test_copied_files_unchanged():
    """`data.py`，`model.py`，`timed_training.py` は ex009 と同一である．"""
    for name in ("data.py", "model.py", "timed_training.py"):
        assert filecmp.cmp(os.path.join(_EX009_DIR, name), os.path.join(_EX016_DIR, name), shallow=False), name


def _log(elapsed, oracle_per_epoch=2 * 9025, initial=0):
    """
    概要: 合成のログを作る．
    引数: elapsed (list of float)．oracle_per_epoch (int)．initial (int)．初期の oracle_calls．
    戻り値: log (dict)．
    """
    n = len(elapsed)
    return {"epoch": list(range(n)), "elapsed_time": elapsed,
            "oracle_calls": [initial + oracle_per_epoch * e if e > 0 else initial for e in range(n)],
            "test_accuracy": [0.1 * e for e in range(n)], "train_loss": [5.0 - e for e in range(n)]}


def test_epoch_time_and_budget_values():
    """1エポック目を除いた増分の中央値，予算以下の最後の点の値（予算を超える点は使わない）．"""
    log = _log([0.0, 12.0, 22.0, 33.0, 43.0])          # 増分 12, 10, 11, 10 → 2エポック目以降（10, 11, 10）の中央値 10
    assert analysis.epoch_time(log) == pytest.approx(10.0)
    # #grad/N は 0, 2, 4, 6, 8．予算を 5 とすると 4（添字2）までを使う．
    assert analysis.budget_index(log, budget=5.0) == 2
    row = analysis.summarize_run(log)
    assert row["time_to_budget"] == pytest.approx(43.0) and row["epochs"] == 4


def test_spread_mean_se_and_order():
    """揺らぎ，平均 ± 標準誤差，実行の順序の表（連続の数）．"""
    assert analysis.spread([9.0, 10.0, 11.0]) == pytest.approx(0.2)
    mean, se = analysis.mean_and_standard_error([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0) and se == pytest.approx(1.0 / np.sqrt(3.0))
    runs = {"SGD": {0: {"config": {"started_at": "t1", "finished_at": "t2"}}},
            "SVRG_final": {0: {"config": {"started_at": "t2", "finished_at": "t3"}}}}
    lines = analysis.order_table(runs)
    assert lines[0].startswith("同じ手法が連続した箇所：0")
    assert lines[-1].endswith("SVRG（最終点） |")
