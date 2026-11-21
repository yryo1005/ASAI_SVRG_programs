"""
E2（スナップショットの選択方法と全勾配の推定方法の要因分離）の学習ループ．

`common/vr_image_training.py` の `iteration`，`train_epoch`，`_initial_snapshot_gradient`，`_end_outer_loop`，
`run_variance_reduced`，`run_single_experiment` をコピーし，次の点だけを変えた．
それ以外の関数（`evaluate_epoch`，`compute_full_gradient_and_metrics`，`compute_approx_error`，
`compute_gradient_statistics`，`set_model_params`，`is_run_completed` 等）は `common/vr_image_training.py` から import する．

1. 手法の分岐の一般化：全勾配を用いるかどうかを `method == "SVRG"` ではなく，集合
   `FULL_GRADIENT_METHODS = {"SVRG_final", "SVRG", "SVRG_average"}` への所属で判定する．
2. $ \\omega $ の記録（`実験計画.md` 1.3節）：内側ループの各ステップで，`optimizer.step` の直前の（更新前の）
   パラメータ $ {\\mathbf w}_{s-1}^{k} $ を $ {\\mathbf S}_{1} = \\sum_{k}{\\mathbf w}_{s-1}^{k} $（float64）と
   $ S_{2} = \\sum_{k}\\|{\\mathbf w}_{s-1}^{k}\\|^{2} $（float64）に加算する．外側ループの終了時に
   $ \\omega({\\mathbf z}) = S_{2} - 2\\langle {\\mathbf z}, {\\mathbf S}_{1}\\rangle + K\\|{\\mathbf z}\\|^{2} $ を計算し，
   `omega_snapshot`（$ {\\mathbf z} = {\\mathbf z}_{s} $）と `omega_average`（$ {\\mathbf z} = {\\mathbf S}_{1}/K $，$ \\omega $ の最小値）
   を記録する．加算は読み出しだけであり，学習の計算を変えない．
3. ログの項目：`common/vr_image_training.METRIC_NAMES` に `omega_snapshot`，`omega_average` を加える．

内側ループの開始点は `inner_start="snapshot"`（論文のAlgorithm 通り $ {\\mathbf w}_{s}^{0} \\leftarrow {\\mathbf z}_{s} $）のみとする．
"""

import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.join(os.path.dirname(_THIS_DIR), "common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from machine_learning_utils import ResultLogger, set_seed  # noqa: E402
from optimizers import ASAISVRG, NFGSVRG, NFGSVRGFinalPoint, SVRG, SVRGFinalPoint  # noqa: E402
from svrg_average import SVRGAverage  # noqa: E402
from vr_image_training import (METRIC_NAMES as E0_METRIC_NAMES, _save_if_best,  # noqa: E402
                               compute_approx_error, compute_full_gradient_and_metrics,
                               compute_gradient_statistics, environment_info, evaluate_epoch,
                               is_run_completed, set_model_params)

INNER_STARTS = ("snapshot",)

# 2×3 の要因配置の6手法．キーは結果のディレクトリの手法名である．
OPTIMIZER_CLASSES = {
    "SVRG_final": SVRGFinalPoint,
    "SVRG": SVRG,
    "SVRG_average": SVRGAverage,
    "NFG_SVRG_final": NFGSVRGFinalPoint,
    "NFG_SVRG": NFGSVRG,
    "ASAI_SVRG": ASAISVRG,
}
METHODS = tuple(OPTIMIZER_CLASSES)

# g_s として全勾配 ∇f(z_s) を用いる手法（残りの3手法は平均勾配を用いる）．
FULL_GRADIENT_METHODS = frozenset({"SVRG_final", "SVRG", "SVRG_average"})

# 本PCの `common/vr_image_training.METRIC_NAMES`（order_049）には既に omega の2項目が含まれるため，重複して加えない
# （統合時の修正，order_052）．どちらの場合も項目の並びは E0 の項目，omega_snapshot，omega_average の順となる．
METRIC_NAMES = tuple(E0_METRIC_NAMES) + tuple(
    name for name in ("omega_snapshot", "omega_average") if name not in E0_METRIC_NAMES)

__all__ = ["INNER_STARTS", "OPTIMIZER_CLASSES", "METHODS", "FULL_GRADIENT_METHODS", "METRIC_NAMES",
           "OmegaAccumulator", "iteration", "train_epoch", "run_variance_reduced", "run_single_experiment",
           "is_run_completed"]


class OmegaAccumulator:
    """
    概要: 内側ループの更新前のパラメータ $ {\\mathbf w}^{0},\\dots,{\\mathbf w}^{K-1} $ について
        $ {\\mathbf S}_{1} = \\sum_{k}{\\mathbf w}^{k} $ と $ S_{2} = \\sum_{k}\\|{\\mathbf w}^{k}\\|^{2} $ を float64 で逐次加算し，
        任意の $ {\\mathbf z} $ に対する $ \\omega({\\mathbf z}) = \\sum_{k}\\|{\\mathbf w}^{k} - {\\mathbf z}\\|^{2} $ を全パラメータを保持せずに計算する．
        元のパラメータのテンソルは読み出すだけで書き換えない（`.double()` は新しいテンソルを作る）．
    """

    def __init__(self):
        """
        概要: 空の状態で初期化する．
        引数: なし
        戻り値: なし
        """
        self.S1 = None
        self.S2 = None
        self.count = 0

    def reset(self) -> None:
        """
        概要: 外側ループの開始時に，加算の状態を空に戻す．
        引数: なし
        戻り値: なし
        """
        self.S1 = None
        self.S2 = None
        self.count = 0

    @torch.no_grad()
    def add(self, params) -> None:
        """
        概要: 1つの点 $ {\\mathbf w}^{k} $ を加算する．GPUとの同期を避けるため，$ S_{2} $ も0次元のテンソルで保持する．
        引数: params (iterable of torch.Tensor)．`model.parameters()` と同じ順序・形状のパラメータ．
        戻り値: なし
        """
        values = [p.detach().double() for p in params]
        if self.S1 is None:
            self.S1 = [torch.zeros_like(v) for v in values]
            self.S2 = torch.zeros((), dtype=torch.float64, device=values[0].device)
        for acc, v in zip(self.S1, values):
            acc.add_(v)
            self.S2.add_(torch.sum(v * v))
        self.count += 1

    @torch.no_grad()
    def omega(self, z_params) -> float:
        """
        概要: $ \\omega({\\mathbf z}) = S_{2} - 2\\langle {\\mathbf z}, {\\mathbf S}_{1}\\rangle + K\\|{\\mathbf z}\\|^{2} $ を float64 で計算する．
        引数: z_params (list of torch.Tensor)．$ {\\mathbf z} $（パラメータと同じ順序・形状．dtype は問わない）．
        戻り値: value (float)．加算した点がない場合は NaN．
        """
        if self.count == 0:
            return float("nan")
        inner = torch.zeros((), dtype=torch.float64, device=self.S2.device)
        norm_sq = torch.zeros((), dtype=torch.float64, device=self.S2.device)
        for z, s1 in zip(z_params, self.S1):
            z64 = z.detach().to(device=s1.device, dtype=torch.float64)
            inner += torch.sum(z64 * s1)
            norm_sq += torch.sum(z64 * z64)
        return float((self.S2 - 2.0 * inner + self.count * norm_sq).item())

    def mean_params(self) -> list:
        """
        概要: 加算した点の平均 $ {\\mathbf S}_{1}/K $（float64）を返す．$ \\omega $ を最小化する点である．
        引数: なし
        戻り値: params (list of torch.Tensor)．float64．
        """
        return [s1 / self.count for s1 in self.S1]

    def omega_pair(self, z_params) -> tuple:
        """
        概要: `omega_snapshot`（$ \\omega({\\mathbf z}_{s}) $）と `omega_average`（$ \\omega({\\mathbf S}_{1}/K) $）を返す．
        引数: z_params (list of torch.Tensor)．各手法が実際に選んだスナップショット $ {\\mathbf z}_{s} $．
        戻り値: (omega_snapshot, omega_average) の組 (float, float)．
        """
        if self.count == 0:
            return float("nan"), float("nan")
        return self.omega(z_params), self.omega(self.mean_params())


def iteration(model, inputs, teacher_signals, loss_func, reg_lambda, optimizer, snapshot_model,
              omega_acc=None) -> None:
    """
    概要: 1つのミニバッチで内側ループの1ステップを実行する（`vr_image_training.iteration` のコピー）．
        `omega_acc` を与えた場合は，`optimizer.step` の直前の（更新前の）パラメータを加算する．
    引数:
        model (torch.nn.Module)．現在のパラメータ $ {\\mathbf w}_{s}^{k} $ を保持するモデル．
        inputs (torch.Tensor)，形状 (B, ...)．入力．
        teacher_signals (torch.Tensor)，形状 (B,)．教師ラベル．
        loss_func (callable)．`loss_func(outputs, teacher_signals, model, reg_lambda)`．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
        optimizer (torch.optim.Optimizer)．SVRG系手法のOptimizer．
        snapshot_model (torch.nn.Module)．スナップショット $ {\\mathbf z}_{s} $ を保持するモデル．
        omega_acc (OmegaAccumulator) = None．$ \\omega $ の加算器．`None` の場合は記録しない．
    戻り値: なし
    """
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_func(outputs, teacher_signals, model, reg_lambda)
    loss.backward()

    snapshot_model.zero_grad()
    snapshot_outputs = snapshot_model(inputs)
    snapshot_loss = loss_func(snapshot_outputs, teacher_signals, snapshot_model, reg_lambda)
    snapshot_loss.backward()
    grad_at_snapshot = [p.grad.detach().clone() for p in snapshot_model.parameters()]
    if omega_acc is not None:
        # ASAISVRG の running_avg_params に加える点（更新前の w_s^k）と同じ点を加算する．
        omega_acc.add(model.parameters())
    optimizer.step(grad_at_snapshot)


def train_epoch(model, dataloader, loss_func, reg_lambda, optimizer, snapshot_model, desc, omega_acc=None) -> None:
    """
    概要: 1エポック分の内側ループを実行する（`vr_image_training.train_epoch` のコピー）．
    引数:
        model, loss_func, reg_lambda, optimizer, snapshot_model, omega_acc：`iteration` を参照．
        dataloader (CachedTensorLoader)．学習データのローダー．
        desc (str)．進捗バーの説明文．
    戻り値: なし
    """
    for inputs, teacher_signals in tqdm(dataloader, desc=desc, leave=False):
        iteration(model, inputs, teacher_signals, loss_func, reg_lambda, optimizer, snapshot_model, omega_acc)


def _initial_snapshot_gradient(method, optimizer, snapshot_model, train_loader, problem):
    """
    概要: 0エポック目のスナップショット勾配 $ {\\mathbf g}_{0} $ と全勾配を計算する．全勾配を用いる手法では
        $ {\\mathbf g}_{0} = \\nabla f({\\mathbf z}_{0}) $，平均勾配を用いる手法では $ {\\mathbf g}_{0} = {\\mathbf 0} $ である．
    引数:
        method (str)．手法名．
        optimizer (torch.optim.Optimizer)．
        snapshot_model (torch.nn.Module)．
        train_loader (CachedTensorLoader)．
        problem (dict)．`loss_func`，`metrics_func`，`reg_lambda` を含む辞書．
    戻り値:
        snapshot_grad (list of torch.Tensor)．$ {\\mathbf g}_{0} $．
        true_full_grad (list of torch.Tensor)．$ \\nabla f({\\mathbf z}_{0}) $．
        train_metrics (dict)．学習データの損失と分類精度．
    """
    true_full_grad, train_metrics = compute_full_gradient_and_metrics(
        snapshot_model, train_loader, problem["loss_func"], problem["metrics_func"], problem["reg_lambda"]
    )
    if method in FULL_GRADIENT_METHODS:
        return true_full_grad, true_full_grad, train_metrics
    return optimizer.get_snapshot_gradient(), true_full_grad, train_metrics


def _end_outer_loop(method, optimizer, snapshot_model, train_loader, problem):
    """
    概要: 外側ループの終了時に，新しいスナップショットにおける全勾配を計算する．全勾配を用いる手法では
        これを次の外側ループの $ {\\mathbf g}_{s} $ として設定する（近似誤差は既存の実装と同じく 0.0）．
        平均勾配を用いる手法では診断のためだけに計算する．
    引数:
        method, optimizer, snapshot_model, train_loader, problem：`_initial_snapshot_gradient` を参照．
    戻り値:
        train_metrics (dict)．学習データの損失と分類精度．
        approx_error (float)．$ \\|{\\mathbf e}_{s}\\|^{2} $．
        full_grad_norm_sq (float)．$ \\|\\nabla f({\\mathbf z}_{s})\\|^{2} $．
        approx_cos (float)．$ {\\mathbf g}_{s} $ と $ \\nabla f({\\mathbf z}_{s}) $ のコサイン類似度．
    """
    true_full_grad, train_metrics = compute_full_gradient_and_metrics(
        snapshot_model, train_loader, problem["loss_func"], problem["metrics_func"], problem["reg_lambda"]
    )
    if method in FULL_GRADIENT_METHODS:
        optimizer.set_snapshot_gradient(true_full_grad)
        full_norm_sq, approx_cos = compute_gradient_statistics(true_full_grad, true_full_grad)
        return train_metrics, 0.0, full_norm_sq, approx_cos

    snapshot_grad = optimizer.get_snapshot_gradient()
    approx_error = compute_approx_error(snapshot_grad, true_full_grad)
    full_norm_sq, approx_cos = compute_gradient_statistics(snapshot_grad, true_full_grad)
    return train_metrics, approx_error, full_norm_sq, approx_cos


def run_variance_reduced(method, inner_start, target_dir, problem, run_config, device, seed, logger,
                         record_omega: bool = True) -> None:
    """
    概要: SVRG系の6手法のいずれかで学習し，各外側ループの終了時の評価指標を `logger` に記録する
        （`vr_image_training.run_variance_reduced` のコピー）．評価はスナップショット $ {\\mathbf z}_{s} $ で行い，
        学習損失が非有限となった外側ループで打ち切る．
    引数:
        method (str)．`METHODS` のいずれか．
        inner_start (str)．"snapshot" のみ．
        target_dir (str)．結果の保存先ディレクトリ．
        problem (dict)．`load_dataloader(seed, batch_size, device)`，`build_model(seed)`，`loss_func`，
            `metrics_func`，`reg_lambda` を含む．
        run_config (dict)．`learning_rate`，`batch_size`，`epochs` を含む．
        device (torch.device)．
        seed (int)．乱数シード．
        logger (ResultLogger)．`METRIC_NAMES` を登録したロガー．
        record_omega (bool) = True．$ \\omega $ を記録するか．`False` の場合は `omega_*` を NaN とする
            （記録が学習に影響しないことの確認に用いる）．
    戻り値: なし
    """
    assert method in OPTIMIZER_CLASSES
    assert inner_start in INNER_STARTS
    eta, batch_size, epochs = run_config["learning_rate"], run_config["batch_size"], run_config["epochs"]
    loss_func, metrics_func, reg_lambda = problem["loss_func"], problem["metrics_func"], problem["reg_lambda"]
    uses_full_gradient = method in FULL_GRADIENT_METHODS

    train_loader, test_loader = problem["load_dataloader"](seed=seed, batch_size=batch_size, device=device)
    N_train = len(train_loader.dataset)
    K = len(train_loader)

    model = problem["build_model"](seed).to(device)
    # z_0 <- w_0．同じシードで初期化することで同一の初期値を持つ．
    snapshot_model = problem["build_model"](seed).to(device)

    optimizer = OPTIMIZER_CLASSES[method](model.parameters(), lr=eta, K=K)
    rng = np.random.default_rng(seed)
    omega_acc = OmegaAccumulator() if record_omega else None
    oracle_calls = 0

    snapshot_grad, true_full_grad_0, train_metrics_0 = _initial_snapshot_gradient(
        method, optimizer, snapshot_model, train_loader, problem
    )
    approx_error_0 = compute_approx_error(snapshot_grad, true_full_grad_0)
    full_norm_sq_0, approx_cos_0 = compute_gradient_statistics(snapshot_grad, true_full_grad_0)

    test_metrics = evaluate_epoch(snapshot_model, test_loader, loss_func, metrics_func, reg_lambda)
    # 0エポック目は内側ループがないため ω は NaN とする．
    logger(0, oracle_calls, 0.0, train_metrics_0["loss"], test_metrics["accuracy"], approx_error_0,
           test_metrics["correct"], full_norm_sq_0, approx_cos_0, False, float("nan"), float("nan"))
    best_accuracy = _save_if_best(snapshot_model, test_metrics["accuracy"], -1.0, target_dir)

    if uses_full_gradient:
        optimizer.set_snapshot_gradient(snapshot_grad)
        oracle_calls += N_train

    elapsed_time = 0.0
    desc_prefix = f"{method} {inner_start} bs={batch_size} seed={seed}"
    for epoch_index in tqdm(range(1, epochs + 1), desc=desc_prefix, leave=False):
        start_time = time.time()
        optimizer.begin_epoch(rng)
        if omega_acc is not None:
            omega_acc.reset()
        train_epoch(model, train_loader, loss_func, reg_lambda, optimizer, snapshot_model,
                    desc=f"{desc_prefix} ep{epoch_index}", omega_acc=omega_acc)
        oracle_calls += 2 * N_train

        optimizer.end_epoch()
        set_model_params(snapshot_model, optimizer.get_snapshot_params())
        # 論文のAlgorithm：次の内側ループを w_s^0 <- z_s から開始する．
        set_model_params(model, optimizer.get_snapshot_params())
        elapsed_time += time.time() - start_time

        if omega_acc is not None:
            omega_snapshot, omega_average = omega_acc.omega_pair(list(snapshot_model.parameters()))
        else:
            omega_snapshot, omega_average = float("nan"), float("nan")

        start_time = time.time()
        train_metrics, approx_error, full_norm_sq, approx_cos = _end_outer_loop(
            method, optimizer, snapshot_model, train_loader, problem
        )
        if uses_full_gradient:
            oracle_calls += N_train
            elapsed_time += time.time() - start_time

        start_time = time.time()
        test_metrics = evaluate_epoch(snapshot_model, test_loader, loss_func, metrics_func, reg_lambda)
        elapsed_time += time.time() - start_time

        terminated = not np.isfinite(train_metrics["loss"])
        logger(epoch_index, oracle_calls, elapsed_time, train_metrics["loss"], test_metrics["accuracy"],
               approx_error, test_metrics["correct"], full_norm_sq, approx_cos, terminated,
               omega_snapshot, omega_average)
        best_accuracy = _save_if_best(snapshot_model, test_metrics["accuracy"], best_accuracy, target_dir)
        if terminated:
            break


def run_single_experiment(task: dict, problem: dict) -> None:
    """
    概要: 1つの (手法, シード) の組について学習し，`log.json` と `config.json` を保存する
        （`vr_image_training.run_single_experiment` のコピー．E2 の学習ループを呼ぶ）．
        すでに正常に終了した結果がある場合は何もしない．
    引数:
        task (dict)．`method`，`inner_start`，`seed`，`target_dir`，`run_config`（`learning_rate`，
            `batch_size`，`epochs`），`experiment`，任意で `record_omega`（既定 True）を含む辞書．
        problem (dict)．`run_variance_reduced` を参照．追加で `config_extra` を含めてよい．
    戻り値: なし
    """
    target_dir = task["target_dir"]
    run_config = task["run_config"]
    if is_run_completed(target_dir, run_config["epochs"]):
        print(f"[skip] {target_dir}", flush=True)
        return
    os.makedirs(target_dir, exist_ok=True)
    record_omega = bool(task.get("record_omega", True))

    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(task["seed"])

    logger = ResultLogger()
    logger.set_names(*METRIC_NAMES)
    run_variance_reduced(task["method"], task["inner_start"], target_dir, problem, run_config,
                         device, task["seed"], logger, record_omega=record_omega)
    logger.save(os.path.join(target_dir, "log.json"))

    config = {
        "experiment": task["experiment"],
        "method": task["method"],
        "inner_start": task["inner_start"],
        "seed": task["seed"],
        **run_config,
        "reg_lambda": problem["reg_lambda"],
        "evaluation_point": "snapshot z_s",
        "sampling": "random_reshuffle",
        "snapshot_gradient": "full_gradient" if task["method"] in FULL_GRADIENT_METHODS else "averaged_gradient",
        "record_omega": record_omega,
        "terminated": bool(logger["terminated"][-1]),
        "recorded_epochs": len(logger["epoch"]) - 1,
        "environment": environment_info(),
        **problem.get("config_extra", {}),
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"[done] {target_dir}", flush=True)
