"""
E5（エポック内の平均点で評価した SGD，実験計画8章）のプログラムのテスト．

- 平均点の添字と式が ASA-SVRG の `running_avg_params` と同じであること（$ {\\mathbf g}_{s} = {\\mathbf 0} $ の
  ASA-SVRG は SGD と同じ更新になることを用いて，同じ軌跡の上でビット単位で比べる）．
- 平均点が更新前のパラメータ $ {\\mathbf w}^{0},\\dots,{\\mathbf w}^{K-1} $ の算術平均であること．
- 平均点の評価の前後で，学習中のパラメータ，学習用のローダーの乱数，CPU の乱数の状態が変わらないこと．
- 平均点も評価する学習ループの最終点の記録が，E3 の SGD の学習ループ（`timed_training.run_timed_sgd`）と
  完全に一致すること．$ s = 0 $ の平均点の記録が初期パラメータの値であること．
- タスクの構築（優先度の順，エポック数，保存先の名前）と，2つのデータセットのモジュールを同じプロセスで読み込めること．
"""

import importlib.util
import os
import sys

import pytest
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REVIEW_ROOT = os.path.dirname(_TESTS_DIR)
_E5_DIR = os.path.join(_REVIEW_ROOT, "programs", "ex010_e5_sgd_epoch_average")
_E3_DIR = os.path.join(_REVIEW_ROOT, "programs", "ex009_e3_imagewoof_walltime")
sys.path.insert(0, os.path.join(_REVIEW_ROOT, "programs", "common"))
sys.path.insert(0, _TESTS_DIR)
sys.path.insert(0, _E5_DIR)
sys.path.insert(0, _E3_DIR)

from machine_learning_utils import ResultLogger  # noqa: E402
from optimizers import ASAISVRG, SGD  # noqa: E402
import e5_training as e5  # noqa: E402
import timed_training as tt  # noqa: E402
from test_e0_programs import _build_model, _make_problem  # noqa: E402

RUN_CONFIG = {"learning_rate": 0.1, "batch_size": 8, "epochs": 3}


def _load_e5_train_module():
    """
    概要: E5 の `train.py` をモジュールとして読み込む．
    引数: なし
    戻り値: module．
    """
    spec = importlib.util.spec_from_file_location("e5_train", os.path.join(_E5_DIR, "train.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _one_epoch_batches(problem, seed=0, batch_size=8):
    """
    概要: 小さな問題の1エポック分のミニバッチを返す．
    引数: problem (dict)，seed (int) = 0，batch_size (int) = 8．
    戻り値: batches (list of tuple)．
    """
    train_loader, _ = problem["load_dataloader"](seed=seed, batch_size=batch_size, device=None)
    return list(train_loader)


def test_epoch_average_matches_asa_running_average():
    """
    概要: 同じ軌跡（g_s = 0 の ASA-SVRG は SGD と同じ更新）の上で，`EpochAverage` が ASA-SVRG の
        `running_avg_params` とビット単位で一致することを確認する．
    引数: なし
    戻り値: なし
    """
    problem = _make_problem([])
    batches = _one_epoch_batches(problem)
    sgd_model, asa_model = _build_model(0), _build_model(0)
    sgd = SGD(sgd_model.parameters(), lr=0.1)
    asa = ASAISVRG(asa_model.parameters(), lr=0.1, K=len(batches))
    average = e5.EpochAverage(sgd_model.parameters())
    asa.begin_epoch()
    for inputs, targets in batches:
        for model, opt in ((sgd_model, sgd), (asa_model, asa)):
            opt.zero_grad()
            problem["loss_func"](model(inputs), targets, model, problem["reg_lambda"]).backward()
        average.add(sgd_model.parameters())
        sgd.step()
        asa.step([torch.zeros_like(p) for p in asa_model.parameters()])
    asa.end_epoch()
    assert all(torch.equal(a, b) for a, b in zip(sgd_model.parameters(), asa_model.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(average.values(), asa.get_snapshot_params()))


def test_epoch_average_is_mean_of_pre_update_iterates():
    """
    概要: 平均点が更新前のパラメータ w^0, ..., w^{K-1} の算術平均であり，`reset` で0に戻ることを確認する．
    引数: なし
    戻り値: なし
    """
    problem = _make_problem([])
    model = _build_model(0)
    optimizer = SGD(model.parameters(), lr=0.1)
    average = e5.EpochAverage(model.parameters())
    iterates = []
    for inputs, targets in _one_epoch_batches(problem):
        optimizer.zero_grad()
        problem["loss_func"](model(inputs), targets, model, problem["reg_lambda"]).backward()
        iterates.append([p.detach().clone() for p in model.parameters()])
        average.add(model.parameters())
        optimizer.step()
    expected = [torch.stack(ws).mean(dim=0) for ws in zip(*iterates)]
    assert all(torch.allclose(a, b, rtol=1e-6, atol=1e-7) for a, b in zip(average.values(), expected))
    assert not all(torch.equal(a, b) for a, b in zip(average.values(), model.parameters()))
    average.reset()
    assert average.count == 0 and all(torch.count_nonzero(a) == 0 for a in average.values())


def test_average_evaluation_does_not_change_training_state():
    """
    概要: 平均点の評価の前後で，学習中のモデルのパラメータ，学習用のローダーの乱数生成器，CPU の乱数の状態が
        変わらないことを確認する．
    引数: なし
    戻り値: なし
    """
    problem = _make_problem([])
    train_loader, test_loader = problem["load_dataloader"](seed=0, batch_size=8, device=None)
    model = _build_model(0)
    avg_model = _build_model(1)
    params_before = [p.detach().clone() for p in model.parameters()]
    generator_state = train_loader.index_loader.generator.get_state().clone()
    rng_state = torch.get_rng_state().clone()
    eval_loader = e5.make_eval_loader(train_loader, 8)
    e5.evaluate_average_point(avg_model, [p.detach() + 1.0 for p in model.parameters()], eval_loader, test_loader,
                              problem, torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(params_before, model.parameters()))
    assert torch.equal(generator_state, train_loader.index_loader.generator.get_state())
    assert torch.equal(rng_state, torch.get_rng_state())
    assert eval_loader.index_loader.generator is None


def _run_e5(tmp_path):
    """
    概要: 小さな問題で平均点も評価する SGD を3エポック学習する．
    引数: tmp_path．
    戻り値: logger (ResultLogger)．
    """
    os.makedirs(tmp_path, exist_ok=True)
    logger = ResultLogger()
    logger.set_names(*e5.METRIC_NAMES)
    e5.run_sgd_with_average(str(tmp_path), _make_problem([]), RUN_CONFIG, torch.device("cpu"), 0, logger)
    return logger


def test_final_point_matches_e3_sgd_loop(tmp_path):
    """
    概要: 最終点の記録が E3 の SGD の学習ループ（`run_timed_sgd`）とビット単位で一致することを確認する．
        平均点の評価が学習の軌跡を変えないことの確認でもある．
    引数: tmp_path．
    戻り値: なし
    """
    e5_log = _run_e5(tmp_path / "e5")
    os.makedirs(tmp_path / "e3", exist_ok=True)
    e3_log = ResultLogger()
    e3_log.set_names(*tt.METRIC_NAMES)
    tt.run_timed_sgd(str(tmp_path / "e3"), _make_problem([]), RUN_CONFIG, torch.device("cpu"), 0, e3_log)
    for key in ("epoch", "oracle_calls", "train_loss", "test_accuracy", "test_correct", "terminated"):
        assert e5_log[key] == e3_log[key], key
    assert e5_log["oracle_calls"] == [0, 50, 100, 150]


def test_initial_average_point_is_initial_parameters(tmp_path):
    """
    概要: s = 0 の平均点の記録が初期パラメータ（最終点）の値であり，s >= 1 では最終点と異なることを確認する．
    引数: tmp_path．
    戻り値: なし
    """
    log = _run_e5(tmp_path)
    for key in ("train_loss", "train_accuracy", "test_loss", "test_accuracy", "test_correct"):
        assert log[f"avg_{key}"][0] == log[key][0]
    assert log["avg_train_loss"][1:] != log["train_loss"][1:]
    assert len(log["epoch"]) == RUN_CONFIG["epochs"] + 1


def test_build_tasks_priority_order_and_names():
    """
    概要: タスクが優先度の順（ImageWoof・128，CIFAR-10・128，CIFAR-10・512，ImageWoof・512）に並び，
        エポック数と保存先の名前が実験計画8.2節・8.3節と一致することを確認する．
    引数: なし
    戻り値: なし
    """
    train = _load_e5_train_module()
    tasks = train.build_tasks(list(train.SETTINGS), [0, 1], "/out")
    order = [(t["dataset"], t["run_config"]["batch_size"], t["run_config"]["epochs"]) for t in tasks[::2]]
    assert order == [("imagewoof", 128, 200), ("cifar10", 128, 90), ("cifar10", 512, 350), ("imagewoof", 512, 1000)]
    assert tasks[0]["target_dir"] == "/out/imagewoof/SGD/lr0.001_bs128_lambda0.0005_epochs200/0"
    assert tasks[3]["target_dir"] == "/out/cifar10/SGD/lr0.001_bs128_lambda0.0005_epochs90/1"
    short = train.build_tasks([("cifar10", 128)], [0], "/out", epochs_override=5)
    assert short[0]["run_config"] == {"learning_rate": 0.001, "batch_size": 128, "epochs": 5}


def test_parse_setting_rejects_unknown():
    """
    概要: 未定義の設定（バッチサイズ32等）を拒否することを確認する．
    引数: なし
    戻り値: なし
    """
    train = _load_e5_train_module()
    assert train.parse_setting("cifar10:512") == ("cifar10", 512)
    with pytest.raises(Exception):
        train.parse_setting("cifar10:32")


def test_both_dataset_modules_load_in_one_process():
    """
    概要: ImageWoof と CIFAR-10 の `data`・`model` を別の名前で同じプロセスに読み込めることを確認する．
    引数: なし
    戻り値: なし
    """
    import problems

    imagewoof_model = problems.load_source_module("imagewoof", "model")
    cifar_model = problems.load_source_module("cifar10", "model")
    assert imagewoof_model is not cifar_model
    assert hasattr(imagewoof_model, "ResNet18LayerNorm") and hasattr(cifar_model, "AlexNetCIFARNorm")
    assert problems.load_source_module("cifar10", "model") is cifar_model
