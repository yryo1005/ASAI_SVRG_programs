"""
E7（`ex012_e7_imagewoof_final`，`ex013_e7_epoch_time`，`analysis_e7.py`）のテスト．

小さな線形分類器と乱数のデータ（`test_e0_programs._make_problem`）を用い，CPU で数秒で終わる．
ex012・ex013 の `train.py` は他の実験の `train.py` と同じモジュール名であるため，ファイルのパスから一意の名前で
読み込む．
"""

import filecmp
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REVIEW_ROOT = os.path.dirname(_TESTS_DIR)
_PROGRAMS = os.path.join(_REVIEW_ROOT, "programs")
_A_DIR = os.path.join(_PROGRAMS, "ex012_e7_imagewoof_final")
_B_DIR = os.path.join(_PROGRAMS, "ex013_e7_epoch_time")
sys.path.insert(0, os.path.join(_PROGRAMS, "common"))
sys.path.insert(0, _TESTS_DIR)

from machine_learning_utils import ResultLogger  # noqa: E402
import vr_image_training as vr  # noqa: E402
from test_e0_programs import _make_problem  # noqa: E402

N_TRAIN = 50  # `test_e0_programs._make_data` のデータ数
RUN_CONFIG = {"learning_rate": 0.1, "batch_size": 8, "epochs": 3}


def _load(path: str, name: str):
    """
    概要: ファイルのパスからモジュールを一意の名前で読み込む．
    引数: path (str)．name (str)．モジュール名．
    戻り値: module．
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_a():
    """
    概要: A の `train.py` を読み込む．
    戻り値: module．
    """
    return _load(os.path.join(_A_DIR, "train.py"), "ex012_train")


@pytest.fixture(scope="module")
def train_b():
    """
    概要: B の `train.py` を読み込む．
    戻り値: module．
    """
    return _load(os.path.join(_B_DIR, "train.py"), "ex013_train")


@pytest.fixture(scope="module")
def analysis():
    """
    概要: `analysis_e7.py` を読み込む．
    戻り値: module．
    """
    return _load(os.path.join(_PROGRAMS, "analysis_e7.py"), "analysis_e7_under_test")


@pytest.fixture(scope="module")
def timed():
    """
    概要: B の `timed_training.py` を読み込む．
    戻り値: module．
    """
    return _load(os.path.join(_B_DIR, "timed_training.py"), "ex013_timed_training")


# ---- A：タスクの構築 ----

def test_a_default_tasks(train_a):
    """
    概要: 既定の引数で30個のタスクが，手法とバッチサイズごとのエポック数と条件名で作られることを確認する．
    """
    args = train_a.parse_args([])
    tasks = train_a.build_e7_tasks(args.methods, args.batch_sizes, args.seeds, "/out")
    assert len(tasks) == 30
    expected = {("SVRG_final", 32): 11, ("NFG_SVRG_final", 32): 16, ("SVRG_final", 128): 67,
                ("NFG_SVRG_final", 128): 100, ("SVRG_final", 512): 334, ("NFG_SVRG_final", 512): 500}
    for task in tasks:
        bs, method = task["run_config"]["batch_size"], task["method"]
        epochs = expected[(method, bs)]
        assert task["run_config"]["epochs"] == epochs
        assert task["inner_start"] == "snapshot"
        assert task["target_dir"] == (f"/out/{method}/inner_snapshot_lr0.001_bs{bs}_lambda0.0005_epochs{epochs}/"
                                      f"{task['seed']}")
    assert len({t["target_dir"] for t in tasks}) == 30
    assert sorted({t["seed"] for t in tasks}) == [0, 1, 2, 3, 4]


def test_a_selection_and_epochs_override(train_a):
    """
    概要: `--batch-sizes 32` で10個，`--epochs 2` で全てのエポック数が2になること，SVRG は `--epochs` なしでは
        エポック数が定義されていないことを確認する．
    """
    args = train_a.parse_args(["--batch-sizes", "32"])
    assert len(train_a.build_e7_tasks(args.methods, args.batch_sizes, args.seeds, "/out")) == 10
    args = train_a.parse_args(["--epochs", "2"])
    tasks = train_a.build_e7_tasks(args.methods, args.batch_sizes, args.seeds, "/out", args.epochs)
    assert len(tasks) == 30 and all(t["run_config"]["epochs"] == 2 for t in tasks)
    args = train_a.parse_args(["--methods", "SVRG", "--seeds", "0", "--epochs", "2"])
    tasks = train_a.build_e7_tasks(args.methods, args.batch_sizes, args.seeds, "/out", args.epochs)
    assert [t["target_dir"] for t in tasks] == [
        f"/out/SVRG/inner_snapshot_lr0.001_bs{bs}_lambda0.0005_epochs2/0" for bs in (32, 128, 512)]
    with pytest.raises(ValueError):
        train_a.epochs_for("SVRG", 128)
    with pytest.raises(SystemExit):
        train_a.parse_args(["--methods", "SGD"])


def test_a_longest_first(train_a):
    """
    概要: 並列実行の順序がバッチサイズの大きい順であり，タスクの集合は変わらないことを確認する．
    """
    tasks = train_a.build_e7_tasks(["SVRG_final", "NFG_SVRG_final"], [32, 128, 512], [0, 1], "/out")
    ordered = train_a.order_longest_first(tasks)
    sizes = [t["run_config"]["batch_size"] for t in ordered]
    assert sizes == sorted(sizes, reverse=True)
    assert sorted(t["target_dir"] for t in ordered) == sorted(t["target_dir"] for t in tasks)


# ---- 最終点の手法の学習ループ（A と同じ `common/vr_image_training`） ----

def _run_recording(method, inner_start, tmp_path, monkeypatch):
    """
    概要: 小さな問題で3エポック学習し，各内側ループの開始点・終了点とスナップショットを記録する．
    引数: method (str)，inner_start (str)，tmp_path，monkeypatch．
    戻り値: (start_points, end_points, snapshots, logger) の組．
    """
    start_points, end_points, snapshots = [], [], []
    original = vr.train_epoch

    def recording(model, dataloader, loss_func, reg_lambda, optimizer, snapshot_model, desc, omega=None):
        start_points.append([p.detach().clone() for p in model.parameters()])
        snapshots.append([p.detach().clone() for p in snapshot_model.parameters()])
        original(model, dataloader, loss_func, reg_lambda, optimizer, snapshot_model, desc, omega)
        end_points.append([p.detach().clone() for p in model.parameters()])

    monkeypatch.setattr(vr, "train_epoch", recording)
    logger = ResultLogger()
    logger.set_names(*vr.METRIC_NAMES)
    os.makedirs(tmp_path, exist_ok=True)
    vr.run_variance_reduced(method, inner_start, str(tmp_path), _make_problem([]), RUN_CONFIG,
                            torch.device("cpu"), 0, logger)
    return start_points, end_points, snapshots, logger


def _equal(a, b) -> bool:
    """
    概要: パラメータのリストが完全に一致するかを返す．
    引数: a, b (list of torch.Tensor)．
    戻り値: equal (bool)．
    """
    return all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("method", ["SVRG_final", "NFG_SVRG_final"])
def test_final_point_snapshot(method, tmp_path, monkeypatch):
    """
    概要: スナップショットが内側ループの最終点と一致し，次の内側ループがそこから始まることを確認する．
        また，最終点では `snapshot` と `last_iterate` の記録が一致することを確認する．
    """
    starts, ends, snaps, logger = _run_recording(method, "snapshot", tmp_path / "s", monkeypatch)
    assert len(starts) == 3
    for s in range(1, 3):
        assert _equal(snaps[s], ends[s - 1])
        assert _equal(starts[s], ends[s - 1])
    _, _, _, logger_last = _run_recording(method, "last_iterate", tmp_path / "l", monkeypatch)
    for key in ("oracle_calls", "train_loss", "test_accuracy", "approx_error"):
        assert logger[key] == pytest.approx(logger_last[key], nan_ok=True), key


@pytest.mark.parametrize("method,expected", [
    ("SVRG_final", [0, 4 * N_TRAIN, 7 * N_TRAIN, 10 * N_TRAIN]),
    ("NFG_SVRG_final", [0, 2 * N_TRAIN, 4 * N_TRAIN, 6 * N_TRAIN]),
])
def test_final_point_oracle_calls(method, expected, tmp_path, monkeypatch):
    """
    概要: `oracle_calls` の増分が `SVRG_final` で初期 N・1エポック 3N，`NFG_SVRG_final` で1エポック 2N であることを確認する．
    """
    _, _, _, logger = _run_recording(method, "snapshot", tmp_path, monkeypatch)
    assert logger["oracle_calls"] == expected


# ---- B ----

def test_b_default_tasks_order(train_b):
    """
    概要: 既定のタスクが72個であり，組ごとに ASAI_SVRG → NFG_SVRG_final → SVRG_final → SGD を3周する順序で並ぶこと，
        各手法の `inner_start` と保存先を確認する．
    """
    args = train_b.parse_args([])
    tasks = train_b.build_tasks(args.datasets, args.batch_sizes, args.methods, args.repeats, args.epochs, "/out")
    assert len(tasks) == 72
    order = ["ASAI_SVRG", "NFG_SVRG_final", "SVRG_final", "SGD"]
    assert [t["method"] for t in tasks] == order * 18
    combos = [(t["dataset"], t["run_config"]["batch_size"]) for t in tasks[::12]]
    assert combos == [("imagewoof", 32), ("imagewoof", 128), ("imagewoof", 512),
                      ("cifar10", 32), ("cifar10", 128), ("cifar10", 512)]
    assert [t["repeat"] for t in tasks[:12]] == [0] * 4 + [1] * 4 + [2] * 4
    expected_inner = {"ASAI_SVRG": "last_iterate", "NFG_SVRG_final": "snapshot", "SVRG_final": "snapshot", "SGD": None}
    assert all(t["inner_start"] == expected_inner[t["method"]] for t in tasks)
    assert all(t["seed"] == 0 and t["run_config"]["epochs"] == 4 and t["run_config"]["record_omega"] is False
               for t in tasks)
    assert tasks[5]["target_dir"] == "/out/imagewoof/bs32/NFG_SVRG_final/rep1"
    assert tasks[-1]["target_dir"] == "/out/cifar10/bs512/SGD/rep2"
    assert len({t["target_dir"] for t in tasks}) == 72
    for a, b in zip(tasks, tasks[1:]):  # 同じ組の中で同じ手法が続かない
        assert a["method"] != b["method"]


def test_b_groups(train_b):
    """
    概要: 組ごとの分割が順序を保ち，6組 × 12試行になることを確認する．
    """
    tasks = train_b.build_tasks(["imagewoof", "cifar10"], [32, 128, 512], train_b.METHODS, 3, 4, "/out")
    groups = train_b.group_tasks(tasks)
    assert [len(g) for g in groups] == [12] * 6
    assert [t for g in groups for t in g] == tasks


def test_b_verify_selection(train_b):
    """
    概要: 動作の確認（`--datasets imagewoof --batch-sizes 128 --repeats 1`）で4試行になることを確認する．
    """
    args = train_b.parse_args(["--datasets", "imagewoof", "--batch-sizes", "128", "--repeats", "1"])
    tasks = train_b.build_tasks(args.datasets, args.batch_sizes, args.methods, args.repeats, args.epochs, "/out")
    assert [t["target_dir"] for t in tasks] == [f"/out/imagewoof/bs128/{m}/rep0" for m in train_b.METHODS]


def test_b_finalize_run(train_b, tmp_path):
    """
    概要: 試行の終了後に `config.json` へ周回の番号と時刻が加わり，`best_model.pth` が削除されることを確認する．
    """
    task = {"target_dir": str(tmp_path), "dataset": "cifar10", "repeat": 2}
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"method": "SGD"}, f)
    (tmp_path / "best_model.pth").write_bytes(b"x")
    train_b.finalize_run(task, "2026-11-21T23:00:00", "2026-11-21T23:01:00")
    config = json.load(open(tmp_path / "config.json"))
    assert config["repeat"] == 2 and config["started_at"] == "2026-11-21T23:00:00" and config["method"] == "SGD"
    assert not (tmp_path / "best_model.pth").exists()


def _fake_log(per_epoch, initial_full=0.0, first_epoch_extra=0.0, full_per_epoch=0.0):
    """
    概要: 1エポックの時間の計算を確かめるための `log.json` の内容を作る．
    引数: per_epoch (float)．1エポックの time_optimizer．initial_full (float)．初期の全勾配の時間．
        first_epoch_extra (float)．1エポック目（準備運転）だけに加える時間．full_per_epoch (float)．
        1エポックあたりの time_svrg_full_grad．
    戻り値: log (dict)．
    """
    opt, full = [0.0], [initial_full]
    for e in range(1, 5):
        opt.append(opt[-1] + per_epoch + (first_epoch_extra if e == 1 else 0.0))
        full.append(full[-1] + full_per_epoch)
    return {"time_optimizer": opt, "time_svrg_full_grad": full}


def test_epoch_time_definition(analysis):
    """
    概要: 1エポックの時間が累積値の差分であり，1エポック目と初期の全勾配を除き，3周の平均・最小・最大を返すことを確認する．
    """
    log = _fake_log(10.0, initial_full=7.0, first_epoch_extra=5.0, full_per_epoch=2.0)
    assert analysis.epoch_time_of_run(log) == pytest.approx(12.0)
    assert analysis.initial_full_grad_time(log) == pytest.approx(7.0)
    summary = analysis.summarize_repeats([analysis.epoch_time_of_run(_fake_log(v)) for v in (9.0, 10.0, 11.0)])
    assert summary["mean"] == pytest.approx(10.0)
    assert summary["min"] == pytest.approx(9.0) and summary["max"] == pytest.approx(11.0)
    assert summary["spread"] == pytest.approx(0.2)


def test_budget_epochs(analysis):
    """
    概要: 予算に届く最小のエポック数が A のエポック数の表（SVRG は初期 N を含む）と一致することを確認する．
    """
    assert analysis.epochs_to_budget("SVRG_final", 200) == 67
    assert analysis.epochs_to_budget("NFG_SVRG_final", 200) == 100
    assert analysis.epochs_to_budget("SVRG_final", 30) == 10
    assert analysis.epochs_to_budget("SGD", 90) == 90
    assert analysis.time_to_budget("SVRG_final", 200, 18.0, 7.0) == pytest.approx(67 * 18.0 + 7.0)


def test_b_summary_from_files(analysis, tmp_path):
    """
    概要: 保存先の形式で置いた `log.json`・`config.json` から，表と実行の順序の確認が作られることを確認する．
    """
    root = tmp_path / "b"
    t = 0
    for rep in range(3):
        for method, sec in (("ASAI_SVRG", 11.0), ("NFG_SVRG_final", 10.0), ("SVRG_final", 15.0), ("SGD", 5.0)):
            run_dir = root / "imagewoof" / "bs128" / method / f"rep{rep}"
            run_dir.mkdir(parents=True)
            with open(run_dir / "log.json", "w") as f:
                json.dump(_fake_log(sec, initial_full=3.0 if method == "SVRG_final" else 0.0), f)
            with open(run_dir / "config.json", "w") as f:
                json.dump({"started_at": f"2026-11-22T00:{t:02d}:00"}, f)
            t += 1
    runs = analysis.load_b_runs(str(root))
    table = analysis.epoch_time_table(runs)
    assert table[("imagewoof", 128, "SVRG_final")]["epoch"]["mean"] == pytest.approx(15.0)
    assert table[("imagewoof", 128, "SVRG_final")]["initial"]["mean"] == pytest.approx(3.0)
    assert analysis.check_alternation(runs) == []
    out = analysis.write_summary(str(tmp_path / "summary.md"), str(root))
    text = open(out).read()
    assert "ImageWoof | 128 | SVRG（最終点） | 3 | 15.000" in text
    assert "連続した箇所：なし" in text


# ---- A の集計 ----

def _a_log(grads, accs, terminated_at=None, n=9025):
    """
    概要: A の集計のテスト用の log を作る（横軸 #grad/N = grads）．
    """
    k = len(grads)
    return {"epoch": list(range(k)), "oracle_calls": [g * n for g in grads],
            "train_loss": [1.0 / (i + 1) for i in range(k)], "test_accuracy": list(accs),
            "approx_error": [0.1 * (i + 1) for i in range(k)],
            "terminated": [terminated_at is not None and i == terminated_at for i in range(k)]}


def test_a_value_at_budget(analysis):
    """
    概要: 予算での値が予算以下の最後の評価点の値であり，打ち切られて予算に届かない試行は NaN になることを確認する．
    """
    log = _a_log([0, 4, 7, 10, 13], [0.1, 0.2, 0.3, 0.4, 0.5])
    assert analysis.value_at_budget(log, "test_accuracy", 12) == pytest.approx(0.4)
    assert analysis.value_at_budget(log, "test_accuracy", 13) == pytest.approx(0.5)
    stopped = _a_log([0, 4, 7], [0.1, 0.2, 0.3], terminated_at=2)
    assert np.isnan(analysis.value_at_budget(stopped, "test_accuracy", 12))
    assert analysis.terminated_epoch(stopped) == 2
    assert analysis.terminated_epoch(log) is None


def test_a_latter_half(analysis):
    """
    概要: 後半の区間（予算の1/2以上・予算以下）の中央値と揺らぎを確認する．
    """
    log = _a_log([0, 2, 4, 6, 8, 10, 12], [0.0, 0.1, 0.2, 0.4, 0.3, 0.5, 0.9])
    # 予算10：#grad/N = 6, 8, 10 → 精度 0.4, 0.3, 0.5
    assert analysis.latter_half_mask(log, 10).tolist() == [False, False, False, True, True, True, False]
    assert analysis.latter_half_fluctuation(log, 10) == pytest.approx((0.1 + 0.2) / 2)
    assert analysis.latter_half_median(log, "approx_error", 10) == pytest.approx(0.5)
    mean, se, n = analysis.mean_se([1.0, 3.0, float("nan")])
    assert (mean, n) == (2.0, 2) and se == pytest.approx(1.0)


@pytest.mark.parametrize("method,zero_section", [("SVRG_final", "time_diagnostic"),
                                                 ("NFG_SVRG_final", "time_svrg_full_grad")])
def test_timed_final_point_sections(timed, method, zero_section, tmp_path):
    """
    概要: 計測付きの学習ループが最終点の手法で動き，記録が計測なしの学習ループと一致し，全勾配の区間の
        割り当てが正しいことを確認する．
    """
    logger = ResultLogger()
    logger.set_names(*timed.METRIC_NAMES)
    run_config = dict(RUN_CONFIG, record_omega=False)
    timed.run_timed_variance_reduced(method, "snapshot", str(tmp_path), _make_problem([]), run_config,
                                     torch.device("cpu"), 0, logger)
    plain = ResultLogger()
    plain.set_names(*vr.METRIC_NAMES)
    vr.run_variance_reduced(method, "snapshot", str(tmp_path), _make_problem([]), run_config,
                            torch.device("cpu"), 0, plain)
    for key in ("epoch", "oracle_calls", "train_loss", "test_accuracy", "test_correct", "approx_error"):
        assert logger[key] == pytest.approx(plain[key], nan_ok=True), key
    assert all(v == 0.0 for v in logger[zero_section])
    assert logger["time_optimizer"][-1] > 0.0


# ---- コピーしたファイル ----

@pytest.mark.parametrize("copied,source", [
    ("ex012_e7_imagewoof_final/data.py", "ex003_e0_imagewoof_resnet18_layernorm/data.py"),
    ("ex012_e7_imagewoof_final/model.py", "ex003_e0_imagewoof_resnet18_layernorm/model.py"),
    ("ex013_e7_epoch_time/data.py", "ex009_e3_imagewoof_walltime/data.py"),
    ("ex013_e7_epoch_time/model.py", "ex009_e3_imagewoof_walltime/model.py"),
    ("ex013_e7_epoch_time/timed_training.py", "ex009_e3_imagewoof_walltime/timed_training.py"),
    ("ex013_e7_epoch_time/data_cifar10.py", "ex007_e2_cifar10_snapshot_gradient_factorial/data.py"),
    ("ex013_e7_epoch_time/model_cifar10.py", "ex007_e2_cifar10_snapshot_gradient_factorial/model.py"),
])
def test_copied_files_unchanged(copied, source):
    """
    概要: 変更せずにコピーしたファイルがコピー元と同一であることを確認する．
    """
    assert filecmp.cmp(os.path.join(_PROGRAMS, copied), os.path.join(_PROGRAMS, source), shallow=False)
