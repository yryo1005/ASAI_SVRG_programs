"""
E5（実験計画8章）：エポック内の平均点でも評価する SGD の学習ループ．

SGD の更新は既存の SGD（E3 の `timed_training.run_timed_sgd`，既存の `ex0024` の `run_sgd`）と同一である．
各エポックの終了時に，次の2点で学習データとテストデータを評価する．

| 評価点 | 定義 | ログの項目 |
|---|---|---|
| 最終点 | $ {\\mathbf w}^{K} $（既存の SGD と同じ） | `train_loss`，`train_accuracy`，`test_loss`，`test_accuracy`，`test_correct` |
| 平均点 | $ \\bar{\\mathbf w} = \\frac{1}{K}\\sum_{k=0}^{K-1}{\\mathbf w}^{k} $（更新前のパラメータの平均） | `avg_train_loss`，`avg_train_accuracy`，`avg_test_loss`，`avg_test_accuracy`，`avg_test_correct` |

平均点は，ASA-SVRG の `running_avg_params`（`optimizers/asai_svrg.py` の式(13)）と同じ添字と同じ逐次の式
$ \\bar{\\mathbf w}^{k+1} = \\frac{k}{k+1}\\bar{\\mathbf w}^{k} + \\frac{1}{k+1}{\\mathbf w}^{k} $ で計算する．
平均はエポックごとにリセットし，学習の軌跡には用いない．

学習の軌跡を変えないための規則（実験計画8.3節）：
- 最終点の評価は既存の SGD と同じく，シャッフルする学習用のローダーで行う．既存の実装では，この評価が
  シャッフルの乱数を消費して次のエポックの順序を決めるため，同じ順序で消費する必要がある．
- 平均点の評価は，パラメータを複製した別のモデルと，シャッフルしない学習データのローダーで行う．
  評価の前後で CPU と CUDA の乱数の状態を保存・復元する（`torch.random.fork_rng`）．
- $ s = 0 $ の平均点は未定義のため，初期パラメータ（最終点と同じ値）を記録する．
"""

import copy
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from cached_loader import CachedTensorLoader
from machine_learning_utils import ResultLogger, set_seed
from optimizers import SGD
import vr_image_training as vr

METRIC_NAMES = (
    "epoch",
    "oracle_calls",
    "elapsed_time",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
    "test_correct",
    "avg_train_loss",
    "avg_train_accuracy",
    "avg_test_loss",
    "avg_test_accuracy",
    "avg_test_correct",
    "terminated",
)


class EpochAverage:
    """
    概要: 1エポックの反復の平均点 $ \\bar{\\mathbf w} $ を，ASA-SVRG の `running_avg_params` と同じ式で逐次に計算する．
    """

    def __init__(self, params):
        """
        概要: 平均を0で初期化する．
        引数: params (iterable of torch.Tensor)．`model.parameters()`．形状と型の参照にのみ用いる．
        戻り値: なし
        """
        self.average = [torch.zeros_like(p) for p in params]
        self.count = 0

    def reset(self) -> None:
        """
        概要: エポックの開始時に平均を0に，点の数を0に戻す（ASA-SVRG の `begin_epoch` と同じ）．
        引数: なし
        戻り値: なし
        """
        self.average = [torch.zeros_like(a) for a in self.average]
        self.count = 0

    @torch.no_grad()
    def add(self, params) -> None:
        """
        概要: 更新前のパラメータ $ {\\mathbf w}^{k} $ を加える．
            $ \\bar{\\mathbf w}^{k+1} = \\frac{k}{k+1}\\bar{\\mathbf w}^{k} + \\frac{1}{k+1}{\\mathbf w}^{k} $．
        引数: params (iterable of torch.Tensor)．`model.parameters()`．読み出すだけで変更しない．
        戻り値: なし
        """
        k = self.count
        for a, p in zip(self.average, params):
            a.mul_(k / (k + 1)).add_(p, alpha=1.0 / (k + 1))
        self.count += 1

    def values(self) -> list:
        """
        概要: 現在の平均点のパラメータの複製を返す．
        引数: なし
        戻り値: params (list of torch.Tensor)．`model.parameters()` と同じ順序・形状．
        """
        return [a.clone() for a in self.average]


def sgd_train_epoch_with_average(model, dataloader, loss_func, reg_lambda, optimizer, average, desc) -> None:
    """
    概要: SGD の1エポック分の学習を行い，更新前のパラメータを `average` に加える．更新の演算は
        E3 の `timed_training.sgd_train_epoch` と同じである（平均への加算はパラメータと勾配を変えない）．
    引数:
        model (torch.nn.Module)，dataloader (CachedTensorLoader)，loss_func (callable)，reg_lambda (float)，
        optimizer (optimizers.SGD)，average (EpochAverage)，desc (str)．進捗バーの説明文．
    戻り値: なし
    """
    for inputs, teacher_signals in tqdm(dataloader, desc=desc, leave=False):
        optimizer.zero_grad()
        loss = loss_func(model(inputs), teacher_signals, model, reg_lambda)
        loss.backward()
        average.add(model.parameters())
        optimizer.step()


def make_eval_loader(train_loader: CachedTensorLoader, batch_size: int) -> CachedTensorLoader:
    """
    概要: 学習用のローダーと同じデータを，シャッフルせずに読み出す評価用のローダーを作る．
        学習用のローダーの乱数生成器は共有しない．
    引数: train_loader (CachedTensorLoader)．batch_size (int)．
    戻り値: eval_loader (CachedTensorLoader)．
    """
    return CachedTensorLoader(train_loader.inputs, train_loader.targets, batch_size, shuffle=False,
                              device=train_loader.device)


def evaluate_average_point(avg_model, params, train_eval_loader, test_loader, problem, device) -> tuple:
    """
    概要: 平均点のパラメータを別のモデルに複製し，学習データとテストデータで評価する．
        評価の前後で CPU と CUDA の乱数の状態を保存・復元し，学習の軌跡に影響させない．
    引数:
        avg_model (torch.nn.Module)．評価用のモデル（学習中のモデルとは別のインスタンス）．
        params (list of torch.Tensor)．平均点のパラメータ．
        train_eval_loader (CachedTensorLoader)．シャッフルしない学習データのローダー．
        test_loader (CachedTensorLoader)．テストデータのローダー．
        problem (dict)．`loss_func`，`metrics_func`，`reg_lambda` を含む．
        device (torch.device)．
    戻り値: (train_metrics, test_metrics) の組．どちらも {"loss", "accuracy", "correct"} の辞書．
    """
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        vr.set_model_params(avg_model, params)
        train_metrics = vr.evaluate_epoch(avg_model, train_eval_loader, problem["loss_func"],
                                          problem["metrics_func"], problem["reg_lambda"])
        test_metrics = vr.evaluate_epoch(avg_model, test_loader, problem["loss_func"],
                                         problem["metrics_func"], problem["reg_lambda"])
    return train_metrics, test_metrics


def _log(logger, epoch, oracle_calls, elapsed_time, final, average, terminated) -> None:
    """
    概要: 1つの評価点を記録する．
    引数:
        logger (ResultLogger)，epoch (int)，oracle_calls (int)，elapsed_time (float)．
        final (tuple)．最終点の (train_metrics, test_metrics)．
        average (tuple)．平均点の (train_metrics, test_metrics)．
        terminated (bool)．
    戻り値: なし
    """
    (train, test), (avg_train, avg_test) = final, average
    logger(epoch, oracle_calls, elapsed_time,
           train["loss"], train["accuracy"], test["loss"], test["accuracy"], test["correct"],
           avg_train["loss"], avg_train["accuracy"], avg_test["loss"], avg_test["accuracy"], avg_test["correct"],
           terminated)


def run_sgd_with_average(target_dir, problem, run_config, device, seed, logger) -> None:
    """
    概要: SGD で学習し，各エポックの終了時に最終点と平均点で評価して記録する．
    引数:
        target_dir (str)．結果の保存先．
        problem (dict)．`problems.build_problem` の戻り値．
        run_config (dict)．`learning_rate`，`batch_size`，`epochs` を含む．
        device (torch.device)．
        seed (int)．
        logger (ResultLogger)．`METRIC_NAMES` を登録したロガー．
    戻り値: なし
    """
    eta, batch_size, epochs = run_config["learning_rate"], run_config["batch_size"], run_config["epochs"]
    loss_func, metrics_func, reg_lambda = problem["loss_func"], problem["metrics_func"], problem["reg_lambda"]
    train_loader, test_loader = problem["load_dataloader"](seed=seed, batch_size=batch_size, device=device)
    train_eval_loader = make_eval_loader(train_loader, batch_size)
    N_train = len(train_loader.dataset)
    model = problem["build_model"](seed).to(device)
    avg_model = copy.deepcopy(model)  # 乱数を消費しない複製
    optimizer = SGD(model.parameters(), lr=eta)
    average = EpochAverage(model.parameters())
    oracle_calls = 0
    elapsed_time = 0.0

    final = (vr.evaluate_epoch(model, train_loader, loss_func, metrics_func, reg_lambda),
             vr.evaluate_epoch(model, test_loader, loss_func, metrics_func, reg_lambda))
    _log(logger, 0, oracle_calls, elapsed_time, final, final, False)
    best_accuracy = vr._save_if_best(model, final[1]["accuracy"], -1.0, target_dir)

    desc = f"SGD+avg bs={batch_size} seed={seed}"
    for epoch_index in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        start_time = time.time()
        average.reset()
        sgd_train_epoch_with_average(model, train_loader, loss_func, reg_lambda, optimizer, average,
                                     desc=f"{desc} ep{epoch_index}")
        elapsed_time += time.time() - start_time
        oracle_calls += N_train

        final = (vr.evaluate_epoch(model, train_loader, loss_func, metrics_func, reg_lambda),
                 vr.evaluate_epoch(model, test_loader, loss_func, metrics_func, reg_lambda))
        averaged = evaluate_average_point(avg_model, average.values(), train_eval_loader, test_loader,
                                          problem, device)
        terminated = not np.isfinite(final[0]["loss"])
        _log(logger, epoch_index, oracle_calls, elapsed_time, final, averaged, terminated)
        best_accuracy = vr._save_if_best(model, final[1]["accuracy"], best_accuracy, target_dir)
        if terminated:
            break


def run_single(task: dict, problem: dict) -> None:
    """
    概要: 1つの (データセット, バッチサイズ, シード) について学習し，`log.json` と `config.json` を保存する．
        すでに正常に終了した結果がある場合は何もしない．
    引数:
        task (dict)．`experiment`，`dataset`，`seed`，`target_dir`，`run_config` を含む．
        problem (dict)．`problems.build_problem` の戻り値．
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
    run_sgd_with_average(target_dir, problem, run_config, device, task["seed"], logger)
    logger.save(os.path.join(target_dir, "log.json"))

    config = {
        "experiment": task["experiment"],
        "method": "SGD",
        "dataset_key": task["dataset"],
        "seed": task["seed"],
        **run_config,
        "reg_lambda": problem["reg_lambda"],
        "evaluation_point": "final iterate w^K (train_*, test_*) and epoch average of pre-update iterates (avg_*)",
        "sampling": "random_reshuffle",
        "terminated": bool(logger["terminated"][-1]),
        "recorded_epochs": len(logger["epoch"]) - 1,
        "environment": vr.environment_info(),
        **problem.get("config_extra", {}),
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"[done] {target_dir}", flush=True)
