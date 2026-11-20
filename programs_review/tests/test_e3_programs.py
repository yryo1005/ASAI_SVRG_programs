"""
E3（実時間の計測）のプログラムのテスト．

- 計測付きの学習ループが，計測なしの学習ループ（`vr_image_training.run_variance_reduced`）と
  同じ値を記録すること（計測が学習の演算を変えないこと）．
- 区間の時間の割り当て：全勾配の手法は `time_diagnostic` が0，平均勾配の手法は
  `time_svrg_full_grad` が0，`elapsed_time` が `time_optimizer + time_svrg_full_grad` であること．
- SGD の勾配評価回数が1エポックあたり $ N $ であること．
- タスクの構築（手法ごとのエポック数，順序，保存先の名前）．
- `--inner-start`（実験計画5.6節）：省略したときの条件名・タスク・既定値が変更前と同じであること，
  `snapshot` を指定したときに条件名と `inner_start` に反映され，計測付きの学習ループが計測なしと一致すること．
"""

import importlib.util
import os
import sys

import pytest
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REVIEW_ROOT = os.path.dirname(_TESTS_DIR)
_E3_DIR = os.path.join(_REVIEW_ROOT, "programs", "ex009_e3_imagewoof_walltime")
sys.path.insert(0, os.path.join(_REVIEW_ROOT, "programs", "common"))
sys.path.insert(0, _TESTS_DIR)
sys.path.insert(0, _E3_DIR)

from machine_learning_utils import ResultLogger  # noqa: E402
import timed_training as tt  # noqa: E402
import vr_image_training as vr  # noqa: E402
from test_e0_programs import _make_problem  # noqa: E402

N_TRAIN = 50
RUN_CONFIG = {"learning_rate": 0.1, "batch_size": 8, "epochs": 3, "record_omega": False}


def _load_e3_train_module():
    """
    概要: E3 の `train.py` をモジュールとして読み込む．
    引数: なし
    戻り値: module．
    """
    spec = importlib.util.spec_from_file_location("e3_train", os.path.join(_E3_DIR, "train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timed(method, tmp_path, inner_start="last_iterate"):
    """
    概要: 小さな問題で計測付きの学習を3エポック行う．
    引数: method (str)，tmp_path，inner_start (str) = "last_iterate"．
    戻り値: logger (ResultLogger)．
    """
    os.makedirs(tmp_path, exist_ok=True)
    logger = ResultLogger()
    logger.set_names(*tt.METRIC_NAMES)
    device = torch.device("cpu")
    if method == "SGD":
        tt.run_timed_sgd(str(tmp_path), _make_problem([]), RUN_CONFIG, device, 0, logger)
    else:
        tt.run_timed_variance_reduced(method, inner_start, str(tmp_path), _make_problem([]), RUN_CONFIG,
                                      device, 0, logger)
    return logger


@pytest.mark.parametrize("inner_start", ["last_iterate", "snapshot"])
@pytest.mark.parametrize("method", ["SVRG", "NFG_SVRG", "ASAI_SVRG"])
def test_timed_loop_matches_untimed_loop(method, inner_start, tmp_path):
    """
    概要: 計測付きの学習ループと計測なしの学習ループの記録が一致することを確認する（両方の内側ループの開始点）．
    引数: method (str)，inner_start (str)，tmp_path．
    戻り値: なし
    """
    timed = _timed(method, tmp_path / "timed", inner_start)
    os.makedirs(tmp_path / "plain", exist_ok=True)
    plain = ResultLogger()
    plain.set_names(*vr.METRIC_NAMES)
    vr.run_variance_reduced(method, inner_start, str(tmp_path / "plain"), _make_problem([]), RUN_CONFIG,
                            torch.device("cpu"), 0, plain)
    for key in ("epoch", "oracle_calls", "train_loss", "test_accuracy", "test_correct", "approx_error"):
        assert timed[key] == plain[key], key


@pytest.mark.parametrize("method,zero_section", [("SVRG", "time_diagnostic"), ("NFG_SVRG", "time_svrg_full_grad"),
                                                 ("ASAI_SVRG", "time_svrg_full_grad"), ("SGD", "time_svrg_full_grad")])
def test_time_sections(method, zero_section, tmp_path):
    """
    概要: 区間の時間の割り当てと，実時間の横軸の定義を確認する．
    引数: method (str)，zero_section (str)．常に0となるべき区間．tmp_path．
    戻り値: なし
    """
    logger = _timed(method, tmp_path)
    assert all(v == 0.0 for v in logger[zero_section])
    for i in range(len(logger["epoch"])):
        assert logger["elapsed_time"][i] == pytest.approx(logger["time_optimizer"][i] + logger["time_svrg_full_grad"][i])
    for key in ("time_optimizer", "time_evaluation"):
        values = logger[key]
        assert all(b >= a for a, b in zip(values, values[1:]))
    assert logger["time_optimizer"][-1] > 0.0
    if method == "SGD":
        assert logger["test_accuracy"][0] == logger["test_accuracy"][0]
        assert all(v != v for v in logger["approx_error"])  # SGD の近似誤差は NaN


def test_sgd_oracle_calls(tmp_path):
    """
    概要: SGD の勾配評価回数の増分が1エポックあたり $ N $ であることを確認する．
    引数: tmp_path．
    戻り値: なし
    """
    logger = _timed("SGD", tmp_path)
    assert logger["oracle_calls"] == [0, N_TRAIN, 2 * N_TRAIN, 3 * N_TRAIN]


def test_build_tasks():
    """
    概要: タスクの数，順序，手法ごとのエポック数，保存先の名前を確認する．
    引数: なし
    戻り値: なし
    """
    train = _load_e3_train_module()
    tasks = train.build_tasks(train.METHODS, [0, 1, 2], "/tmp/out")
    assert len(tasks) == 12
    assert [t["method"] for t in tasks[:4]] == ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
    assert [t["seed"] for t in tasks] == [0] * 4 + [1] * 4 + [2] * 4
    epochs = {t["method"]: t["run_config"]["epochs"] for t in tasks}
    assert epochs == {"SGD": 200, "SVRG": 67, "NFG_SVRG": 100, "ASAI_SVRG": 100}
    assert all(t["run_config"]["record_omega"] is False for t in tasks)
    assert tasks[0]["target_dir"] == "/tmp/out/SGD/inner_none_lr0.001_bs128_lambda0.0005_epochs200/0"
    assert tasks[3]["target_dir"] == "/tmp/out/ASAI_SVRG/inner_last_iterate_lr0.001_bs128_lambda0.0005_epochs100/0"


def test_build_tasks_default_inner_start_unchanged():
    """
    概要: `inner_start` を省略したときのタスクが，`last_iterate` を明示したときと全く同じであり，
        既存の試行の条件名（`inner_last_iterate_...`）と一致することを確認する．
    引数: なし
    戻り値: なし
    """
    train = _load_e3_train_module()
    default = train.build_tasks(train.METHODS, [0, 1, 2, 3, 4], "/tmp/out")
    explicit = train.build_tasks(train.METHODS, [0, 1, 2, 3, 4], "/tmp/out", None, "last_iterate")
    assert default == explicit
    assert train.condition_name("NFG_SVRG", 100) == "inner_last_iterate_lr0.001_bs128_lambda0.0005_epochs100"
    assert all(t["inner_start"] == ("last_iterate" if t["method"] != "SGD" else None) for t in default)
    existing = os.path.join(_REVIEW_ROOT, "outputs", "ex009_e3_imagewoof_walltime")
    for task in default:
        rel = os.path.relpath(task["target_dir"], "/tmp/out")
        if os.path.isdir(os.path.join(existing, os.path.dirname(rel))):
            assert os.path.exists(os.path.join(existing, rel, "config.json")), rel


def test_parse_args_default_inner_start(monkeypatch):
    """
    概要: `--inner-start` を省略したときの既定値が `last_iterate` であり，`snapshot` を指定できることを確認する．
    引数: monkeypatch．
    戻り値: なし
    """
    train = _load_e3_train_module()
    monkeypatch.setattr(sys, "argv", ["train.py"])
    assert train.parse_args().inner_start == "last_iterate"
    monkeypatch.setattr(sys, "argv", ["train.py", "--methods", "NFG_SVRG", "--inner-start", "snapshot"])
    assert train.parse_args().inner_start == "snapshot"


def test_build_tasks_snapshot_inner_start():
    """
    概要: `inner_start="snapshot"` のときに，条件名と `inner_start` に反映され，SGD には適用されないことを確認する．
    引数: なし
    戻り値: なし
    """
    train = _load_e3_train_module()
    tasks = train.build_tasks(["SGD", "NFG_SVRG"], [0], "/tmp/out", None, "snapshot")
    assert tasks[0]["target_dir"] == "/tmp/out/SGD/inner_none_lr0.001_bs128_lambda0.0005_epochs200/0"
    assert tasks[0]["inner_start"] is None
    assert tasks[1]["target_dir"] == "/tmp/out/NFG_SVRG/inner_snapshot_lr0.001_bs128_lambda0.0005_epochs100/0"
    assert tasks[1]["inner_start"] == "snapshot"
