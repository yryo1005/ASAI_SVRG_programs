"""
E6（`ex011_e6_cifar10_final_bs32_512`）のプログラムのテスト．小さな線形モデルと乱数のデータを使い，CPU で数秒で終わる．

1. タスクの構築で，2手法 × 2バッチサイズ × 5シード = 20個の保存先が，指示書2.4節のディレクトリ名とエポック数
   （32 → 12，512 → 192）で作られること．`--batch-sizes`，`--methods`，`--epochs` の指定が反映されること．
2. `SVRG_final` と `NFG_SVRG_final` のスナップショットが内側ループの最終点と一致し，次の内側ループがそこから始まること．
3. `oracle_calls` の増分が，`SVRG_final` で初期 $ N $・1エポック $ 3N $，`NFG_SVRG_final` で1エポック $ 2N $ であること．
4. ex007 からコピーした4つのファイルが ex007 と同一であること．
5. 集計（`analysis_e6.py`）の打ち切りのエポックと予算の後半の中央値．

ex007 と ex011 には同じ名前のモジュール（`e2_runner`，`e2_training` 等）があるため，ex011 のモジュールはファイルの
パスから一意の名前で読み込む．
"""

import filecmp
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REVIEW_ROOT = os.path.dirname(_TESTS_DIR)
_PROGRAMS_DIR = os.path.join(_REVIEW_ROOT, "programs")
_EX011_DIR = os.path.join(_PROGRAMS_DIR, "ex011_e6_cifar10_final_bs32_512")
_EX007_DIR = os.path.join(_PROGRAMS_DIR, "ex007_e2_cifar10_snapshot_gradient_factorial")
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, _TESTS_DIR)

from machine_learning_utils import ResultLogger  # noqa: E402

E6_METHODS = ["SVRG_final", "NFG_SVRG_final"]
N_DATA = 40
BATCH_SIZE = 8
EPOCHS = 3


def _load_ex011(name: str):
    """
    概要: ex011 のモジュールを，ファイルのパスから一意の名前（`ex011_{name}`）で読み込む．
        読み込み中は `sys.modules` の同名のモジュール（`e2_runner`，`e2_training`）を ex011 のものに差し替える．
    引数: name (str)．`e2_training`，`e2_runner`，`train` のいずれか．
    戻り値: module．
    """
    unique = f"ex011_{name}"
    if unique in sys.modules:
        return sys.modules[unique]
    dependencies = {"e2_runner": ["e2_training"], "train": ["e2_training", "e2_runner"]}.get(name, [])
    saved = {dep: sys.modules.get(dep) for dep in dependencies}
    try:
        for dep in dependencies:
            sys.modules[dep] = _load_ex011(dep)
        spec = importlib.util.spec_from_file_location(unique, os.path.join(_EX011_DIR, f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique] = module
        spec.loader.exec_module(module)
    finally:
        for dep, original in saved.items():
            if original is None:
                sys.modules.pop(dep, None)
            else:
                sys.modules[dep] = original
    return module


@pytest.fixture(scope="module")
def e2():
    """
    概要: ex011 の `e2_training` を返す．
    引数: なし
    戻り値: module．
    """
    return _load_ex011("e2_training")


@pytest.fixture(scope="module")
def runner():
    """
    概要: ex011 の `e2_runner` を返す．
    引数: なし
    戻り値: module．
    """
    return _load_ex011("e2_runner")


@pytest.fixture(scope="module")
def train():
    """
    概要: ex011 の `train` を返す．
    引数: なし
    戻り値: module．
    """
    return _load_ex011("train")


def _make_problem():
    """
    概要: 小さな分類問題（E2 のテストと同じ）の定義を作る．
    引数: なし
    戻り値: problem (dict)．
    """
    from test_e2_factorial_programs import _make_problem as make_e2_problem

    return make_e2_problem()


def _run(e2, method, tmp_path):
    """
    概要: 小さな問題で `EPOCHS` エポック学習し，ロガーを返す．
    引数: e2 (module)，method (str)，tmp_path．
    戻り値: logger (ResultLogger)．
    """
    os.makedirs(tmp_path, exist_ok=True)
    logger = ResultLogger()
    logger.set_names(*e2.METRIC_NAMES)
    run_config = {"learning_rate": 0.1, "batch_size": BATCH_SIZE, "epochs": EPOCHS}
    e2.run_variance_reduced(method, "snapshot", str(tmp_path), _make_problem(), run_config,
                            torch.device("cpu"), 0, logger, record_omega=True)
    return logger


# ---------------------------------------------------------------------------
# 1. タスクの構築
# ---------------------------------------------------------------------------

def test_build_tasks_creates_twenty_target_dirs(runner, train):
    """
    概要: 既定の引数で，2手法 × 2バッチサイズ × 5シード = 20個のタスクが2.4節の保存先とエポック数で作られることを確認する．
    引数: runner (module)，train (module)．
    戻り値: なし
    """
    spec = train.SPEC
    args = runner.parse_args(spec, "", argv=[])
    tasks = runner.build_e2_tasks(spec, args.methods, args.seeds, args.epochs, "/out",
                                  inner_starts=tuple(args.inner_starts), batch_sizes=args.batch_sizes)
    assert len(tasks) == 20
    expected = {f"/out/{m}/inner_snapshot_lr0.001_bs{b}_lambda0.0005_epochs{e}/{s}"
                for m in E6_METHODS for b, e in [(32, 12), (512, 192)] for s in range(5)}
    assert {t["target_dir"] for t in tasks} == expected
    for task in tasks:
        epochs = {32: 12, 512: 192}[task["run_config"]["batch_size"]]
        assert task["run_config"] == {"learning_rate": 0.001, "batch_size": task["run_config"]["batch_size"],
                                      "epochs": epochs}
        assert task["inner_start"] == "snapshot" and task["record_omega"] is True
        assert task["experiment"] == "ex011_e6_cifar10_final_bs32_512"


@pytest.mark.parametrize("argv, expected_count", [
    (["--batch-sizes", "32"], 10),
    (["--methods", "SVRG", "--batch-sizes", "32", "--seeds", "0"], 1),
])
def test_command_line_selection(runner, train, argv, expected_count):
    """
    概要: `--batch-sizes 32` で10個，`--methods SVRG --batch-sizes 32 --seeds 0` で1個のタスクになることを確認する．
    引数: runner (module)，train (module)，argv (list of str)，expected_count (int)．
    戻り値: なし
    """
    args = runner.parse_args(train.SPEC, "", argv=argv)
    tasks = runner.build_e2_tasks(train.SPEC, args.methods, args.seeds, args.epochs, "/out",
                                  inner_starts=tuple(args.inner_starts), batch_sizes=args.batch_sizes)
    assert len(tasks) == expected_count
    assert all(t["run_config"]["batch_size"] == 32 for t in tasks)


def test_batch_size_128_with_epochs_for_e2_check(runner, train):
    """
    概要: `--batch-sizes 128 --epochs 3` で E2 との一致の確認用の条件 `inner_snapshot_lr0.001_bs128_lambda0.0005_epochs3`
        が作られることを確認する（バッチサイズをエポック数の表のキーに制限しないこと）．
    引数: runner (module)，train (module)．
    戻り値: なし
    """
    argv = ["--methods", "SVRG_final", "NFG_SVRG_final", "--seeds", "0", "--epochs", "3", "--batch-sizes", "128"]
    args = runner.parse_args(train.SPEC, "", argv=argv)
    tasks = runner.build_e2_tasks(train.SPEC, args.methods, args.seeds, args.epochs, "/out",
                                  inner_starts=tuple(args.inner_starts), batch_sizes=args.batch_sizes)
    assert sorted(t["target_dir"] for t in tasks) == [
        f"/out/{m}/inner_snapshot_lr0.001_bs128_lambda0.0005_epochs3/0" for m in sorted(E6_METHODS)]


def test_methods_choices_are_all_six(runner, train, e2):
    """
    概要: `--methods` の既定値が E6 の2手法であり，選択肢が `e2_training.METHODS` の6手法であることを確認する．
    引数: runner (module)，train (module)，e2 (module)．
    戻り値: なし
    """
    assert runner.parse_args(train.SPEC, "", argv=[]).methods == E6_METHODS
    for method in e2.METHODS:
        assert runner.parse_args(train.SPEC, "", argv=["--methods", method]).methods == [method]
    with pytest.raises(SystemExit):
        runner.parse_args(train.SPEC, "", argv=["--methods", "SGD"])


# ---------------------------------------------------------------------------
# 2. スナップショットが最終点であること
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", E6_METHODS)
def test_snapshot_is_final_point_and_next_loop_starts_there(e2, method, tmp_path, monkeypatch):
    """
    概要: 各外側ループの後のスナップショットが内側ループの最終点 $ {\\mathbf w}_{s-1}^{K} $ と一致し，次の内側ループが
        そこから始まることを確認する．
    引数: e2 (module)，method (str)，tmp_path，monkeypatch．
    戻り値: なし
    """
    start_points, end_points, snapshots = [], [], []
    original_train_epoch = e2.train_epoch

    def recording_train_epoch(model, dataloader, loss_func, reg_lambda, optimizer, snapshot_model, desc,
                              omega_acc=None):
        start_points.append([p.detach().clone() for p in model.parameters()])
        snapshots.append([p.detach().clone() for p in snapshot_model.parameters()])
        result = original_train_epoch(model, dataloader, loss_func, reg_lambda, optimizer, snapshot_model, desc,
                                      omega_acc)
        end_points.append([p.detach().clone() for p in model.parameters()])
        return result

    monkeypatch.setattr(e2, "train_epoch", recording_train_epoch)
    _run(e2, method, tmp_path)
    assert len(start_points) == EPOCHS
    for s in range(EPOCHS - 1):
        assert all(torch.equal(a, b) for a, b in zip(snapshots[s + 1], end_points[s]))
        assert all(torch.equal(a, b) for a, b in zip(start_points[s + 1], end_points[s]))
    assert not all(torch.equal(a, b) for a, b in zip(end_points[0], end_points[-1]))


# ---------------------------------------------------------------------------
# 3. oracle_calls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", E6_METHODS)
def test_oracle_calls_increments(e2, method, tmp_path):
    """
    概要: `oracle_calls` の増分が，`SVRG_final` で初期 $ N $・1エポック $ 3N $，`NFG_SVRG_final` で1エポック $ 2N $ である
        ことを確認する．
    引数: e2 (module)，method (str)，tmp_path．
    戻り値: なし
    """
    calls = np.asarray(_run(e2, method, tmp_path)["oracle_calls"])
    assert calls[0] == 0
    if method == "SVRG_final":
        assert calls[1] == N_DATA + 3 * N_DATA
        assert np.all(np.diff(calls)[1:] == 3 * N_DATA)
    else:
        assert np.all(np.diff(calls) == 2 * N_DATA)


# ---------------------------------------------------------------------------
# 4. コピーしたファイル
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["data.py", "model.py", "e2_training.py", "svrg_average.py"])
def test_copied_files_are_identical_to_ex007(name):
    """
    概要: ex007 から変更せずにコピーしたファイルが ex007 と同一であることを確認する．
    引数: name (str)．
    戻り値: なし
    """
    assert filecmp.cmp(os.path.join(_EX011_DIR, name), os.path.join(_EX007_DIR, name), shallow=False)


# ---------------------------------------------------------------------------
# 5. 集計
# ---------------------------------------------------------------------------

def test_analysis_terminated_epoch_and_latter_half_median():
    """
    概要: `analysis_e6` の打ち切りのエポックと，予算の後半の中央値（非有限の値を除く）を確認する．
    引数: なし
    戻り値: なし
    """
    import analysis_e6 as a

    log = {"epoch": [0, 1, 2, 3], "oracle_calls": [0, 100000, 200000, 300000],
           "terminated": [False, False, False, True], "approx_error": [1.0, 2.0, 4.0, float("nan")],
           "train_loss": [2.0, 1.5, 1.0, float("nan")], "test_accuracy": [0.1, 0.2, 0.3, 0.1]}
    run = {"method": "NFG_SVRG_final", "seed": 0, "log": log, "x": np.asarray(log["oracle_calls"]) / a.N_TRAIN}
    assert a.terminated_epoch(run) == 3
    assert a.latter_half_median(run, "approx_error", 6) == 4.0  # x = 6 の NaN を除く
    assert a.latter_half_median(run, "approx_error", 2.5) == 2.0  # x = 2 のみ（x = 0 は前半）
    assert a.condition_name(32) == "inner_snapshot_lr0.001_bs32_lambda0.0005_epochs12"
    cells = a.row_values([run], "NFG_SVRG_final", 6)
    assert cells["stopped"] == "1（seed 0: 3）"
    assert cells["acc_budget"].startswith("0.3000")
