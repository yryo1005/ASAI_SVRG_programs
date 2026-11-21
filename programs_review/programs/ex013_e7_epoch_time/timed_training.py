"""
E3（実験計画5章）：実時間を計測しながら学習するループ（SGD と SVRG系手法）．

学習の演算は `common/vr_image_training.py` の関数（`train_epoch`，`compute_full_gradient_and_metrics`，
`evaluate_epoch` 等）をそのまま用い，次の4区間の時間を累積して記録する（実験計画5.3節）．

| 区間 | 含める処理 | 実時間 $ t $ に含めるか |
|---|---|---|
| `time_optimizer` | 内側ループの全処理（ミニバッチの転送，現在点とスナップショットの順伝播・逆伝播，`optimizer.step`，`begin_epoch`，`end_epoch`，スナップショットのモデルへのパラメータ設定）．SGD では1エポックの学習 | 含める |
| `time_svrg_full_grad` | 全勾配を用いる手法（SVRG）が次の外側ループで用いる全勾配の計算（初期の全勾配を含む） | 含める |
| `time_diagnostic` | NFG-SVRG・ASA-SVRG の近似誤差の評価のためだけの全勾配の計算（学習損失も同じ走査で得る） | 含めない |
| `time_evaluation` | テスト精度の評価．SGD では学習損失の評価も含む | 含めない |

各区間の開始と終了の直前に `torch.cuda.synchronize()` を呼び，`time.perf_counter()` で計測する．
`elapsed_time` には $ t = \\mathrm{time\\_optimizer} + \\mathrm{time\\_svrg\\_full\\_grad} $ を記録する．
$ \\omega $ の記録は行わない（加算の時間が混入するため）．`best_model.pth` の保存はどの区間にも含めない．
"""

import json
import os
import platform
import time
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm

from machine_learning_utils import ResultLogger, set_seed
from optimizers import SGD
import vr_image_training as vr

SECTIONS = ("optimizer", "svrg_full_grad", "diagnostic", "evaluation")

METRIC_NAMES = (
    "epoch",
    "oracle_calls",
    "elapsed_time",
    "train_loss",
    "test_accuracy",
    "test_correct",
    "approx_error",
    "time_optimizer",
    "time_svrg_full_grad",
    "time_diagnostic",
    "time_evaluation",
    "terminated",
)


class SectionTimer:
    """
    概要: 区間ごとの累積時間を計測する．各区間の開始と終了の直前に CUDA の処理の完了を待つ．
    """

    def __init__(self, device: torch.device):
        """
        概要: 全区間の累積時間を0で初期化する．
        引数: device (torch.device)．CUDA の場合のみ同期する．
        戻り値: なし
        """
        self.device = device
        self.totals = {name: 0.0 for name in SECTIONS}

    def _sync(self) -> None:
        """
        概要: CUDA の処理の完了を待つ．
        引数: なし
        戻り値: なし
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def section(self, name: str):
        """
        概要: `with timer.section("optimizer"):` の形で区間の時間を累積する．
        引数: name (str)．`SECTIONS` のいずれか．
        戻り値: コンテキストマネージャ．
        """
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.totals[name] += time.perf_counter() - start

    def wall_time(self) -> float:
        """
        概要: 実時間の横軸 $ t = \\mathrm{time\\_optimizer} + \\mathrm{time\\_svrg\\_full\\_grad} $ を返す．
        引数: なし
        戻り値: t (float)．[s]．
        """
        return self.totals["optimizer"] + self.totals["svrg_full_grad"]

    def values(self) -> list:
        """
        概要: ログに記録する4区間の累積時間を `SECTIONS` の順に返す．
        引数: なし
        戻り値: values (list of float)．
        """
        return [self.totals[name] for name in SECTIONS]


def _log(logger, timer, epoch, oracle_calls, train_loss, test_metrics, approx_error, terminated) -> None:
    """
    概要: 1つの評価点を記録する．
    引数: logger (ResultLogger)，timer (SectionTimer)，epoch (int)，oracle_calls (int)，train_loss (float)，
        test_metrics (dict)，approx_error (float)，terminated (bool)．
    戻り値: なし
    """
    logger(epoch, oracle_calls, timer.wall_time(), train_loss, test_metrics["accuracy"], test_metrics["correct"],
           approx_error, *timer.values(), terminated)


def run_timed_variance_reduced(method, inner_start, target_dir, problem, run_config, device, seed, logger) -> None:
    """
    概要: SVRG系手法で学習し，実時間を区間ごとに計測して記録する．演算の順序は
        `vr_image_training.run_variance_reduced` と同一である（ω の記録を除く）．
    引数: `vr_image_training.run_variance_reduced` と同じ．`logger` には `METRIC_NAMES` を登録する．
    戻り値: なし
    """
    assert method in vr.VARIANCE_REDUCED_OPTIMIZER_CLASSES and inner_start in vr.INNER_STARTS
    eta, batch_size, epochs = run_config["learning_rate"], run_config["batch_size"], run_config["epochs"]
    loss_func, metrics_func, reg_lambda = problem["loss_func"], problem["metrics_func"], problem["reg_lambda"]
    full_gradient = method in vr.FULL_GRADIENT_METHODS
    first_section = "svrg_full_grad" if full_gradient else "diagnostic"

    train_loader, test_loader = problem["load_dataloader"](seed=seed, batch_size=batch_size, device=device)
    N_train = len(train_loader.dataset)
    K = len(train_loader)
    model = problem["build_model"](seed).to(device)
    snapshot_model = problem["build_model"](seed).to(device)
    optimizer = vr.VARIANCE_REDUCED_OPTIMIZER_CLASSES[method](model.parameters(), lr=eta, K=K)
    rng = np.random.default_rng(seed)
    timer = SectionTimer(device)
    oracle_calls = 0

    with timer.section(first_section):
        snapshot_grad, true_full_grad_0, train_metrics_0 = vr._initial_snapshot_gradient(
            method, optimizer, snapshot_model, train_loader, problem)
        if full_gradient:
            optimizer.set_snapshot_gradient(snapshot_grad)
    approx_error_0 = vr.compute_approx_error(snapshot_grad, true_full_grad_0)
    with timer.section("evaluation"):
        test_metrics = vr.evaluate_epoch(snapshot_model, test_loader, loss_func, metrics_func, reg_lambda)
    _log(logger, timer, 0, oracle_calls, train_metrics_0["loss"], test_metrics, approx_error_0, False)
    best_accuracy = vr._save_if_best(snapshot_model, test_metrics["accuracy"], -1.0, target_dir)
    if full_gradient:
        oracle_calls += N_train

    desc_prefix = f"{method} {inner_start} seed={seed}"
    for epoch_index in tqdm(range(1, epochs + 1), desc=desc_prefix, leave=False):
        with timer.section("optimizer"):
            optimizer.begin_epoch(rng)
            vr.train_epoch(model, train_loader, loss_func, reg_lambda, optimizer, snapshot_model,
                           desc=f"{desc_prefix} ep{epoch_index}")
            optimizer.end_epoch()
            vr.set_model_params(snapshot_model, optimizer.get_snapshot_params())
            if inner_start == "snapshot":
                vr.set_model_params(model, optimizer.get_snapshot_params())
        oracle_calls += 2 * N_train

        with timer.section("svrg_full_grad" if full_gradient else "diagnostic"):
            train_metrics, approx_error, _, _ = vr._end_outer_loop(method, optimizer, snapshot_model,
                                                                   train_loader, problem)
        if full_gradient:
            oracle_calls += N_train

        with timer.section("evaluation"):
            test_metrics = vr.evaluate_epoch(snapshot_model, test_loader, loss_func, metrics_func, reg_lambda)
        terminated = not np.isfinite(train_metrics["loss"])
        _log(logger, timer, epoch_index, oracle_calls, train_metrics["loss"], test_metrics, approx_error, terminated)
        best_accuracy = vr._save_if_best(snapshot_model, test_metrics["accuracy"], best_accuracy, target_dir)
        if terminated:
            break


def sgd_train_epoch(model, dataloader, loss_func, reg_lambda, optimizer, desc) -> None:
    """
    概要: SGD の1エポック分の学習を行う（既存の実装 `ex0051` の `train_epoch(snapshot_model=None)` と同じ演算）．
    引数:
        model (torch.nn.Module)，dataloader (CachedTensorLoader)，loss_func (callable)，reg_lambda (float)，
        optimizer (optimizers.SGD)，desc (str)．進捗バーの説明文．
    戻り値: なし
    """
    for inputs, teacher_signals in tqdm(dataloader, desc=desc, leave=False):
        optimizer.zero_grad()
        loss = loss_func(model(inputs), teacher_signals, model, reg_lambda)
        loss.backward()
        optimizer.step()


def run_timed_sgd(target_dir, problem, run_config, device, seed, logger) -> None:
    """
    概要: SGD で学習し，実時間を区間ごとに計測して記録する．評価点は各エポック終了時の最終点である．
        学習損失とテスト精度の評価は `time_evaluation` に含める．
    引数: target_dir (str)，problem (dict)，run_config (dict)，device (torch.device)，seed (int)，
        logger (ResultLogger)．`run_timed_variance_reduced` を参照．
    戻り値: なし
    """
    eta, batch_size, epochs = run_config["learning_rate"], run_config["batch_size"], run_config["epochs"]
    loss_func, metrics_func, reg_lambda = problem["loss_func"], problem["metrics_func"], problem["reg_lambda"]
    train_loader, test_loader = problem["load_dataloader"](seed=seed, batch_size=batch_size, device=device)
    N_train = len(train_loader.dataset)
    model = problem["build_model"](seed).to(device)
    optimizer = SGD(model.parameters(), lr=eta)
    timer = SectionTimer(device)
    oracle_calls = 0

    with timer.section("evaluation"):
        train_metrics = vr.evaluate_epoch(model, train_loader, loss_func, metrics_func, reg_lambda)
        test_metrics = vr.evaluate_epoch(model, test_loader, loss_func, metrics_func, reg_lambda)
    _log(logger, timer, 0, oracle_calls, train_metrics["loss"], test_metrics, float("nan"), False)
    best_accuracy = vr._save_if_best(model, test_metrics["accuracy"], -1.0, target_dir)

    desc = f"SGD seed={seed}"
    for epoch_index in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        with timer.section("optimizer"):
            sgd_train_epoch(model, train_loader, loss_func, reg_lambda, optimizer, desc=f"{desc} ep{epoch_index}")
        oracle_calls += N_train
        with timer.section("evaluation"):
            train_metrics = vr.evaluate_epoch(model, train_loader, loss_func, metrics_func, reg_lambda)
            test_metrics = vr.evaluate_epoch(model, test_loader, loss_func, metrics_func, reg_lambda)
        terminated = not np.isfinite(train_metrics["loss"])
        _log(logger, timer, epoch_index, oracle_calls, train_metrics["loss"], test_metrics, float("nan"), terminated)
        best_accuracy = vr._save_if_best(model, test_metrics["accuracy"], best_accuracy, target_dir)
        if terminated:
            break


def measurement_environment() -> dict:
    """
    概要: 計測環境（GPU，CPU，メモリ，ライブラリのバージョン，スレッド数，cuDNN の設定）を返す．
    引数: なし
    戻り値: info (dict)．
    """
    info = vr.environment_info()
    cpu_name = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            cpu_name = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    try:
        with open("/proc/meminfo") as f:
            mem_kib = int(next(line.split()[1] for line in f if line.startswith("MemTotal")))
    except (OSError, StopIteration):
        mem_kib = None
    info.update({
        "cpu": cpu_name,
        "logical_cpus": os.cpu_count(),
        "memory_gib": round(mem_kib / 1024 ** 2, 1) if mem_kib else None,
        "gpu_memory_gib": (round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
                           if torch.cuda.is_available() else None),
        "torch_num_threads": torch.get_num_threads(),
        "dataloader_num_workers": 0,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    })
    return info


def run_single_timed(task: dict, problem: dict) -> None:
    """
    概要: 1つの (手法, シード) について計測しながら学習し，`log.json` と `config.json` を保存する．
        すでに正常に終了した結果がある場合は何もしない．
    引数:
        task (dict)．`method`，`inner_start`，`seed`，`target_dir`，`run_config`，`experiment` を含む辞書．
        problem (dict)．`vr_image_training.run_variance_reduced` を参照．
    戻り値: なし
    """
    target_dir, run_config = task["target_dir"], task["run_config"]
    if vr.is_run_completed(target_dir, run_config["epochs"]):
        print(f"[skip] {target_dir}", flush=True)
        return
    os.makedirs(target_dir, exist_ok=True)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(task["seed"])

    logger = ResultLogger()
    logger.set_names(*METRIC_NAMES)
    if task["method"] == "SGD":
        run_timed_sgd(target_dir, problem, run_config, device, task["seed"], logger)
    else:
        run_timed_variance_reduced(task["method"], task["inner_start"], target_dir, problem, run_config,
                                   device, task["seed"], logger)
    logger.save(os.path.join(target_dir, "log.json"))

    config = {
        "experiment": task["experiment"],
        "method": task["method"],
        "inner_start": task["inner_start"],
        "seed": task["seed"],
        **run_config,
        "reg_lambda": problem["reg_lambda"],
        "evaluation_point": "final iterate" if task["method"] == "SGD" else "snapshot z_s",
        "sampling": "random_reshuffle",
        "wall_time_definition": "time_optimizer + time_svrg_full_grad",
        "terminated": bool(logger["terminated"][-1]),
        "recorded_epochs": len(logger["epoch"]) - 1,
        "environment": measurement_environment(),
        **problem.get("config_extra", {}),
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"[done] {target_dir}", flush=True)
