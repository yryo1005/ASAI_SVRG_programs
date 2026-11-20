"""
画像分類（CIFAR-10，ImageWoof）におけるSVRG系手法の学習ループ．

既存の実験（`ex0024_cifar10_alexnet_layernorm_longrun`，`ex0051_imagewoof_resnet18_bs128_longrun`）
の `run_variance_reduced` と同じ順序で同じ演算を行い，次の点だけを追加する．

1. 引数 `inner_start` による内側ループの開始点の切り替え（実験計画1.1節）．
   - `"last_iterate"`：既存の実装と同じ．次の内側ループを前の内側ループの最終点 $ {\\mathbf w}_{s-1}^{K} $
     から開始する．
   - `"snapshot"`：論文のAlgorithmと同じ．外側ループの終了時に `model` のパラメータを
     スナップショット $ {\\mathbf z}_{s} $ で上書きしてから，次の内側ループを開始する．
2. ログの項目の追加（実験計画1.3節）：`test_correct`，`full_grad_norm_sq`，`approx_cos`，
   `terminated`，`omega_snapshot`，`omega_average`．既存の項目（`epoch`，`oracle_calls`，`elapsed_time`，`train_loss`，
   `test_accuracy`，`approx_error`）は名前も計算方法も変えない．

3. E2（実験計画4章）のための手法の追加：`SVRG_average`（`SVRGAverage`），`SVRG_final`
   （`SVRGFinalPoint`），`NFG_SVRG_final`（`NFGSVRGFinalPoint`）．全勾配を用いる手法は
   `FULL_GRADIENT_METHODS` で判定する．$ \omega $ の記録は読み出しのみであり，学習の計算を変えない．

`"last_iterate"` を指定した場合，既存の実装と同じ値が得られることを
`programs/compare_with_existing_logs.py` で確認する．
"""

import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from machine_learning_utils import ResultLogger, set_seed
from optimizers import ASAISVRG, NFGSVRG, NFGSVRGFinalPoint, SVRG, SVRGAverage, SVRGFinalPoint

INNER_STARTS = ("last_iterate", "snapshot")

VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "SVRG": SVRG,
    "NFG_SVRG": NFGSVRG,
    "ASAI_SVRG": ASAISVRG,
    "SVRG_average": SVRGAverage,
    "SVRG_final": SVRGFinalPoint,
    "NFG_SVRG_final": NFGSVRGFinalPoint,
}

# スナップショット勾配として全勾配を用いる手法．外側ループごとに全勾配を計算して設定し，
# 勾配評価回数に初期の N と1エポックあたり 3N を数える．
FULL_GRADIENT_METHODS = ("SVRG", "SVRG_average", "SVRG_final")

METRIC_NAMES = (
    "epoch",
    "oracle_calls",
    "elapsed_time",
    "train_loss",
    "test_accuracy",
    "approx_error",
    "test_correct",
    "full_grad_norm_sq",
    "approx_cos",
    "terminated",
    "omega_snapshot",
    "omega_average",
)


class OmegaAccumulator:
    """
    概要: 内側ループの軌跡とスナップショットの距離
        $ \omega({\mathbf z}) = \sum_{k=0}^{K-1}\|{\mathbf w}^{k} - {\mathbf z}\|^{2} $（実験計画1.3節）を，
        全パラメータを保持せずに計算する．数値誤差を抑えるため，最初の点 $ {\mathbf w}^{0} $ を
        基準とした差 $ {\mathbf d}_{k} = {\mathbf w}^{k} - {\mathbf w}^{0} $ について
        $ {\mathbf S}_{1} = \sum_{k}{\mathbf d}_{k} $，$ S_{2} = \sum_{k}\|{\mathbf d}_{k}\|^{2} $ を
        float64で加算し，$ \omega({\mathbf z}) = S_{2} - 2\langle {\mathbf z} - {\mathbf w}^{0}, {\mathbf S}_{1}\rangle
        + K\|{\mathbf z} - {\mathbf w}^{0}\|^{2} $ とする．パラメータは読み出すだけで変更しない．
    """

    def __init__(self):
        """
        概要: 空の状態で初期化する．
        引数: なし
        戻り値: なし
        """
        self.reference = None
        self.sum_diff = None
        self.sum_sq = None
        self.count = 0

    @torch.no_grad()
    def add(self, params) -> None:
        """
        概要: 更新前のパラメータ $ {\mathbf w}^{k} $ を1点加える．
        引数: params (iterable of torch.Tensor)．`model.parameters()`．
        戻り値: なし
        """
        flat = torch.cat([p.detach().reshape(-1).double() for p in params])
        if self.reference is None:
            self.reference = flat
            self.sum_diff = torch.zeros_like(flat)
            self.sum_sq = torch.zeros((), dtype=torch.float64, device=flat.device)
        diff = flat - self.reference
        self.sum_diff += diff
        self.sum_sq += torch.dot(diff, diff)
        self.count += 1

    @torch.no_grad()
    def omega(self, z_params) -> tuple:
        """
        概要: 指定したスナップショットに対する $ \omega $ と，その最小値（平均点に対する値）を返す．
        引数: z_params (list of torch.Tensor)．スナップショット $ {\mathbf z} $ のパラメータ．
        戻り値:
            omega_snapshot (float)．$ \omega({\mathbf z}) $．
            omega_average (float)．$ \omega(\bar{\mathbf w}) = S_{2} - \|{\mathbf S}_{1}\|^{2}/K $．
        """
        if self.count == 0:
            return float("nan"), float("nan")
        z = torch.cat([p.detach().reshape(-1).double() for p in z_params]) - self.reference
        K = float(self.count)
        omega_snapshot = self.sum_sq - 2.0 * torch.dot(z, self.sum_diff) + K * torch.dot(z, z)
        omega_average = self.sum_sq - torch.dot(self.sum_diff, self.sum_diff) / K
        return float(omega_snapshot), float(omega_average)


def set_model_params(model: torch.nn.Module, param_values) -> None:
    """
    概要: モデルのパラメータを指定した値で上書きする．LayerNormを用いるモデルはバッファ
        （移動平均の統計量）を持たないため，パラメータのみを上書きすればよい．
    引数:
        model (torch.nn.Module)．上書き対象のモデル．
        param_values (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の値．
    戻り値: なし
    """
    with torch.no_grad():
        for p, v in zip(model.parameters(), param_values):
            p.copy_(v)


def iteration(model, inputs, teacher_signals, loss_func, reg_lambda, optimizer, snapshot_model, omega=None) -> None:
    """
    概要: 1つのミニバッチで内側ループの1ステップを実行する．同一ミニバッチについて `model`
        （現在のパラメータ $ {\\mathbf w}_{s}^{k} $）と `snapshot_model`（スナップショット
        $ {\\mathbf z}_{s} $）の勾配を計算し，`optimizer.step(grad_at_snapshot)` を呼ぶ．
    引数:
        model (torch.nn.Module)．現在のパラメータを保持するモデル．
        inputs (torch.Tensor)，形状 (B, ...)．入力．
        teacher_signals (torch.Tensor)，形状 (B,)．教師ラベル．
        loss_func (callable)．`loss_func(outputs, teacher_signals, model, reg_lambda)` で
            正則化項を含む損失（スカラーのテンソル）を返す関数．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
        optimizer (torch.optim.Optimizer)．SVRG系手法のOptimizer．
        snapshot_model (torch.nn.Module)．スナップショットを保持するモデル．
        omega (OmegaAccumulator) = None．指定した場合，更新前のパラメータを加える．
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
    if omega is not None:
        omega.add(model.parameters())
    optimizer.step(grad_at_snapshot)


def train_epoch(model, dataloader, loss_func, reg_lambda, optimizer, snapshot_model, desc, omega=None) -> None:
    """
    概要: 1エポック分の内側ループを実行する．
    引数:
        model, loss_func, reg_lambda, optimizer, snapshot_model：`iteration` を参照．
        dataloader (CachedTensorLoader)．学習データのローダー（ミニバッチは転送済み）．
        desc (str)．進捗バーの説明文．
        omega (OmegaAccumulator) = None．`iteration` を参照．
    戻り値: なし
    """
    for inputs, teacher_signals in tqdm(dataloader, desc=desc, leave=False):
        iteration(model, inputs, teacher_signals, loss_func, reg_lambda, optimizer, snapshot_model, omega)


def evaluate_epoch(model, dataloader, loss_func, metrics_func, reg_lambda) -> dict:
    """
    概要: データ全体に対する損失と分類精度を（学習せずに）計算する．分類精度は既存の実装と
        同じく，ミニバッチ内の平均値にミニバッチのデータ数を掛けて足し合わせ，最後に
        データ数で割る．これに加えて正解数（整数）を数える．
    引数:
        model (torch.nn.Module)．評価対象のモデル．
        dataloader (CachedTensorLoader)．評価するデータのローダー．
        loss_func, metrics_func (callable)．損失と評価指標を計算する関数．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: metrics (dict)．{"loss": float, "accuracy": float, "correct": int}．
    """
    total_metrics = {}
    total_count = 0
    total_correct = 0
    with torch.no_grad():
        for inputs, teacher_signals in dataloader:
            outputs = model(inputs)
            loss = loss_func(outputs, teacher_signals, model, reg_lambda)
            batch_metrics = metrics_func(outputs, teacher_signals)
            batch_metrics["loss"] = loss.item()
            total_correct += int((torch.argmax(outputs, dim=-1) == teacher_signals).sum().item())

            batch_size = inputs.shape[0]
            for key, value in batch_metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value * batch_size
            total_count += batch_size

    metrics = {key: value / total_count for key, value in total_metrics.items()}
    metrics["correct"] = total_correct
    return metrics


def compute_full_gradient_and_metrics(model, dataloader, loss_func, metrics_func, reg_lambda):
    """
    概要: 学習データ全体に対する勾配（全勾配）$ \\nabla f({\\mathbf z}) $ と，同じ走査で得られる
        損失・分類精度の平均を計算する．既存の実装と同じく，ミニバッチの平均勾配に
        ミニバッチのデータ数を掛けて足し合わせ，最後にデータ数で割る．
    引数:
        model (torch.nn.Module)．勾配を計算する点のパラメータを保持するモデル．
        dataloader (CachedTensorLoader)．学習データのローダー．
        loss_func, metrics_func (callable)．損失と評価指標を計算する関数．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値:
        grads (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の全勾配．
        metrics (dict)．{"loss": float, "accuracy": float}．
    """
    accumulated_grads = [torch.zeros_like(p) for p in model.parameters()]
    total_metrics = {}
    total_count = 0

    for inputs, teacher_signals in dataloader:
        batch_size = inputs.shape[0]
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, teacher_signals, model, reg_lambda)
        loss.backward()

        for acc, p in zip(accumulated_grads, model.parameters()):
            acc.add_(p.grad.detach(), alpha=batch_size)

        batch_metrics = metrics_func(outputs, teacher_signals)
        batch_metrics["loss"] = loss.item()
        for key, value in batch_metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value * batch_size
        total_count += batch_size

    grads = [acc / total_count for acc in accumulated_grads]
    metrics = {key: value / total_count for key, value in total_metrics.items()}
    return grads, metrics


def compute_approx_error(snapshot_gradient, true_full_gradient) -> float:
    """
    概要: 全勾配の近似誤差 $ \\|{\\mathbf e}_{s}\\|^{2} = \\|{\\mathbf g}_{s} - \\nabla f({\\mathbf z}_{s})\\|^{2} $
        を既存の実装と同じ方法（テンソルごとにfloat32で和をとり，Pythonのfloatで足し合わせる）
        で計算する．
    引数:
        snapshot_gradient (list of torch.Tensor)．スナップショット勾配 $ {\\mathbf g}_{s} $．
        true_full_gradient (list of torch.Tensor)．全勾配 $ \\nabla f({\\mathbf z}_{s}) $．
    戻り値: approx_error (float)．
    """
    return sum(
        torch.sum((g_s - g_true) ** 2).item()
        for g_s, g_true in zip(snapshot_gradient, true_full_gradient)
    )


def compute_gradient_statistics(snapshot_gradient, true_full_gradient) -> tuple:
    """
    概要: 全勾配のノルムの2乗 $ \\|\\nabla f({\\mathbf z}_{s})\\|^{2} $ と，$ {\\mathbf g}_{s} $ と
        $ \\nabla f({\\mathbf z}_{s}) $ のコサイン類似度をfloat64で計算する．
    引数:
        snapshot_gradient (list of torch.Tensor)．スナップショット勾配 $ {\\mathbf g}_{s} $．
        true_full_gradient (list of torch.Tensor)．全勾配 $ \\nabla f({\\mathbf z}_{s}) $．
    戻り値:
        full_grad_norm_sq (float)．$ \\|\\nabla f({\\mathbf z}_{s})\\|^{2} $．
        approx_cos (float)．コサイン類似度．どちらかのノルムが0の場合はNaN．
    """
    inner = 0.0
    g_norm_sq = 0.0
    full_norm_sq = 0.0
    for g_s, g_true in zip(snapshot_gradient, true_full_gradient):
        g_s64 = g_s.double()
        g_true64 = g_true.double()
        inner += torch.sum(g_s64 * g_true64).item()
        g_norm_sq += torch.sum(g_s64 ** 2).item()
        full_norm_sq += torch.sum(g_true64 ** 2).item()
    denom = float(np.sqrt(g_norm_sq) * np.sqrt(full_norm_sq))
    approx_cos = inner / denom if denom > 0.0 else float("nan")
    return full_norm_sq, approx_cos


def _save_if_best(model, test_accuracy, best_accuracy, target_dir) -> float:
    """
    概要: テスト精度が過去最高を更新した場合，モデルの重みを保存する．
    引数:
        model (torch.nn.Module)．保存対象のモデル．
        test_accuracy (float)．今回の外側ループのテスト精度．
        best_accuracy (float)．これまでの最高精度．
        target_dir (str)．保存先ディレクトリ．
    戻り値: best_accuracy (float)．更新後の最高精度．
    """
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))
        return test_accuracy
    return best_accuracy


def _initial_snapshot_gradient(method, optimizer, snapshot_model, train_loader, problem):
    """
    概要: 0エポック目（初期状態）のスナップショット勾配 $ {\\mathbf g}_{0} $ と全勾配を計算する．
        SVRGでは $ {\\mathbf g}_{0} = \\nabla f({\\mathbf z}_{0}) $，NFG-SVRGとASA-SVRGでは
        $ {\\mathbf g}_{0} = {\\mathbf 0} $ である．
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
    概要: 外側ループの終了時に，新しいスナップショットにおける全勾配を計算し，近似誤差の
        評価に必要な量を返す．SVRGではこの全勾配を次の外側ループの $ {\\mathbf g}_{s} $ として
        設定する（近似誤差は既存の実装と同じく0.0とする）．NFG-SVRGとASA-SVRGでは
        診断のためだけに全勾配を計算する．
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


def run_variance_reduced(method, inner_start, target_dir, problem, run_config, device, seed, logger) -> None:
    """
    概要: SVRG系手法（SVRG，NFG-SVRG，ASA-SVRG）で学習し，各外側ループの終了時の評価指標を
        `logger` に記録する．評価はスナップショット $ {\\mathbf z}_{s} $（`snapshot_model`）で行う．
        学習損失が非有限となった外側ループで打ち切り，それまでの記録を残す．
    引数:
        method (str)．`VARIANCE_REDUCED_OPTIMIZER_CLASSES` のキー（"SVRG"，"NFG_SVRG"，"ASAI_SVRG"，
            "SVRG_average"，"SVRG_final"，"NFG_SVRG_final"）．
        inner_start (str)．"last_iterate" または "snapshot"．
        target_dir (str)．結果の保存先ディレクトリ．
        problem (dict)．問題の定義．`load_dataloader(seed, batch_size, device)`，
            `build_model(seed)`，`loss_func`，`metrics_func`，`reg_lambda` を含む．
        run_config (dict)．`learning_rate`，`batch_size`，`epochs` を含む．任意で `record_omega`（既定 True）．
        device (torch.device)．
        seed (int)．乱数シード．
        logger (ResultLogger)．`METRIC_NAMES` を登録したロガー．
    戻り値: なし
    """
    assert method in VARIANCE_REDUCED_OPTIMIZER_CLASSES
    assert inner_start in INNER_STARTS
    eta, batch_size, epochs = run_config["learning_rate"], run_config["batch_size"], run_config["epochs"]
    loss_func, metrics_func, reg_lambda = problem["loss_func"], problem["metrics_func"], problem["reg_lambda"]

    train_loader, test_loader = problem["load_dataloader"](seed=seed, batch_size=batch_size, device=device)
    N_train = len(train_loader.dataset)
    K = len(train_loader)

    model = problem["build_model"](seed).to(device)
    # z_0 <- w_0．同じシードで初期化することで同一の初期値を持つ．
    snapshot_model = problem["build_model"](seed).to(device)

    optimizer = VARIANCE_REDUCED_OPTIMIZER_CLASSES[method](model.parameters(), lr=eta, K=K)
    rng = np.random.default_rng(seed)
    oracle_calls = 0

    snapshot_grad, true_full_grad_0, train_metrics_0 = _initial_snapshot_gradient(
        method, optimizer, snapshot_model, train_loader, problem
    )
    approx_error_0 = compute_approx_error(snapshot_grad, true_full_grad_0)
    full_norm_sq_0, approx_cos_0 = compute_gradient_statistics(snapshot_grad, true_full_grad_0)

    test_metrics = evaluate_epoch(snapshot_model, test_loader, loss_func, metrics_func, reg_lambda)
    logger(0, oracle_calls, 0.0, train_metrics_0["loss"], test_metrics["accuracy"], approx_error_0,
           test_metrics["correct"], full_norm_sq_0, approx_cos_0, False, float("nan"), float("nan"))
    best_accuracy = _save_if_best(snapshot_model, test_metrics["accuracy"], -1.0, target_dir)

    if method in FULL_GRADIENT_METHODS:
        optimizer.set_snapshot_gradient(snapshot_grad)
        oracle_calls += N_train

    elapsed_time = 0.0
    desc_prefix = f"{method} {inner_start} bs={batch_size} seed={seed}"
    for epoch_index in tqdm(range(1, epochs + 1), desc=desc_prefix, leave=False):
        start_time = time.time()
        optimizer.begin_epoch(rng)
        # 実時間の計測（E3）では run_config["record_omega"]=False として加算の時間を含めない．
        omega = OmegaAccumulator() if run_config.get("record_omega", True) else None
        train_epoch(model, train_loader, loss_func, reg_lambda, optimizer, snapshot_model,
                    desc=f"{desc_prefix} ep{epoch_index}", omega=omega)
        oracle_calls += 2 * N_train

        optimizer.end_epoch()
        set_model_params(snapshot_model, optimizer.get_snapshot_params())
        omega_snapshot, omega_average = (omega.omega(optimizer.get_snapshot_params()) if omega is not None
                                         else (float("nan"), float("nan")))
        if inner_start == "snapshot":
            # 論文のAlgorithm：次の内側ループを w_s^0 <- z_s から開始する．
            set_model_params(model, optimizer.get_snapshot_params())
        elapsed_time += time.time() - start_time

        start_time = time.time()
        train_metrics, approx_error, full_norm_sq, approx_cos = _end_outer_loop(
            method, optimizer, snapshot_model, train_loader, problem
        )
        if method in FULL_GRADIENT_METHODS:
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


def is_run_completed(target_dir: str, epochs: int) -> bool:
    """
    概要: 指定した条件の学習が正常に終了しているか（全エポックの記録がある，または打ち切りで
        終了している）を確認する．
    引数:
        target_dir (str)．結果の保存先ディレクトリ．
        epochs (int)．外側ループ数．
    戻り値: completed (bool)．
    """
    log_path = os.path.join(target_dir, "log.json")
    config_path = os.path.join(target_dir, "config.json")
    if not (os.path.exists(log_path) and os.path.exists(config_path)):
        return False
    try:
        logger = ResultLogger(log_path)
    except (json.JSONDecodeError, OSError):
        return False
    terminated = logger["terminated"]
    return len(logger["epoch"]) == epochs + 1 or (len(terminated) > 0 and bool(terminated[-1]))


def environment_info() -> dict:
    """
    概要: 実行環境（ライブラリのバージョン，GPU名）を辞書で返す．`config.json` に記録する．
    引数: なし
    戻り値: info (dict)．
    """
    import torchvision

    info = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import PIL

        info["pillow"] = PIL.__version__
    except ImportError:
        pass
    return info


def run_single_experiment(task: dict, problem: dict) -> None:
    """
    概要: 1つの (手法, 内側ループの開始点, シード) の組について学習し，`log.json` と
        `config.json` を保存する．すでに正常に終了した結果がある場合は何もしない．
    引数:
        task (dict)．`method`，`inner_start`，`seed`，`target_dir`，`run_config`（`learning_rate`，
            `batch_size`，`epochs`），`experiment` を含む辞書．
        problem (dict)．`run_variance_reduced` を参照．追加で `config_extra`（`config.json` に
            記録する問題固有の情報）を含めてよい．
    戻り値: なし
    """
    target_dir = task["target_dir"]
    run_config = task["run_config"]
    if is_run_completed(target_dir, run_config["epochs"]):
        print(f"[skip] {target_dir}", flush=True)
        return
    os.makedirs(target_dir, exist_ok=True)

    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(task["seed"])

    logger = ResultLogger()
    logger.set_names(*METRIC_NAMES)
    run_variance_reduced(task["method"], task["inner_start"], target_dir, problem, run_config,
                         device, task["seed"], logger)
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
        "terminated": bool(logger["terminated"][-1]),
        "recorded_epochs": len(logger["epoch"]) - 1,
        "environment": environment_info(),
        **problem.get("config_extra", {}),
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"[done] {target_dir}", flush=True)
