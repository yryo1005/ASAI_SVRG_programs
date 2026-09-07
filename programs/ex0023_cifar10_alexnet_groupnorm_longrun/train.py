"""
実験ex0023（GroupNorm・学習率0.001固定での長期エポック学習，誤差床の収束観察）の学習ループを
定義し，実験を実行するスクリプト．

`.orders/order_026.md` に対応する．ex0022（`report_025.md`）は，GroupNorm下でSGD・SVRG・
NFG SVRG・ASAI SVRGの4手法を比較し，学習率0.001では全条件で発散が生じず，ASAI SVRGが同一
オラクル呼び出し回数のもとでSVRGを一貫して上回ることを示した一方，学習率0.01ではバッチ
サイズ128でASAI SVRGが4/5 Seed発散するなど，学習率を上げることの恩恵とリスクが拮抗する
結果となった．

本実験は，`.orders/order_026.md` 2節の指示に基づき学習率0.01を対象から外し，最も安定した
挙動が確認されている学習率0.001に絞った上で，エポック数を大幅に延長し，近似誤差
$ \\|e_s\\|^2 $ および分類精度がプラトーに達するまで観察する．実験1（`report_021.md`）で
確認された「誤差床はtransientであり，十分なエポック数の下ではSVRGと同水準まで収束しうる」
という強凸設定での知見が，非凸・実践的なCNN設定でも成立するかを検証する．

## 既存結果を再利用しない理由（`.orders/order_026.md` 3節）

ex0022の学習率0.001の結果は，最終重み（`best_model.pth`）は保存されているが，各Optimizer
（NFG SVRG・ASAI SVRGの平均勾配・平均パラメータのrunning average，SVRGのスナップショット
状態等）の内部変数（`self.state`）はディスクに保存されていないため，これらの内部状態から
学習を再開することはできない．したがって本実験は全60条件（4手法×3バッチサイズ×5Seed）を
ゼロから再学習する．

## Optimizerクラスの選択とelapsed_timeの計上方法

`.orders/order_026.md` 4節の指示に基づき，比較手法はex0022と同一の実装（`SVRG`／`NFGSVRG`，
次のスナップショット点を内部ループのパラメータ列からランダムに選ぶ版）をそのまま用いる．
`elapsed_time`の計上方法（NFG SVRG・ASAI SVRGの診断専用フル勾配計算を除外する）も，
`.reports/report_025.md` 2.2節の修正をそのまま踏襲する．

## エポック数とプラトー判定

`.orders/order_026.md` 5節の初期値（総イテレーション数約4700の実験ex0022を約4倍に延長した
値，バッチサイズ512, 128, 32についてそれぞれ192, 48, 12エポック）を `EPOCHS_BY_BATCH_SIZE`
の初期値として用いる．`compute_trailing_relative_change` 関数は，学習完了後に近似誤差・
分類精度の軌跡がプラトーに達しているかを，末尾の複数エポックにわたる相対変化の最大値として
数値的に判定するために用いる（`.orders/order_026.md` 5節「近似誤差の変化率が十分小さく
なったことを数値的に確認する」に対応，`.ai/ai-dev-kit/machine_learning.md` の
`SpectralConv2d`バーンイン収束確認テストと同種の方法論）．
"""

import itertools
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

_PROGRAMS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from machine_learning_utils import ResultLogger, set_seed  # noqa: E402
from optimizers import ASAISVRG, NFGSVRG, SGD, SVRG  # noqa: E402

from data import load_dataloader  # noqa: E402
from model import (  # noqa: E402
    AlexNetCIFARNorm,
    load_model,
    loss_func,
    metrics_func,
    refresh_and_freeze_spectral_norm,
    set_model_params,
)

EXPERIMENT_NAME = "ex0023_cifar10_alexnet_groupnorm_longrun"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
BATCH_SIZES = [512, 128, 32]
LEARNING_RATE = 0.001  # `.orders/order_026.md` 4節の指示により固定（0.01は対象外）
NORM_TYPE = "groupnorm"
REG_LAMBDA = 5e-4  # 既存実験と同一

# バッチサイズごとのエポック数．`.orders/order_026.md` 5節の初期値（総イテレーション数
# 約4700のex0022を約4倍に延長，約18800）．プラトー判定の結果に応じて変更する場合がある
# （変更した値・判定根拠は `.reports/report_026.md` に明記する）．
EPOCHS_BY_BATCH_SIZE = {512: 192, 128: 48, 32: 12}

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "SVRG": SVRG,
    "NFG_SVRG": NFGSVRG,
    "ASAI_SVRG": ASAISVRG,
}


def hp_name(batch_size: int, epochs: int) -> str:
    """
    概要: ハイパーパラメータ条件名（ディレクトリ名）を構築する．
    引数:
        batch_size (int)．ミニバッチサイズ．
        epochs (int)．エポック数．
    戻り値: name (str)．
    """
    return f"lr{LEARNING_RATE}_bs{batch_size}_norm{NORM_TYPE}_lambda{REG_LAMBDA}_epochs{epochs}"


def compute_trailing_relative_change(values, window: int) -> float:
    """
    概要: 数値列の末尾 `window` 個について，連続する値の間の相対変化
        $ |v^{(t)}-v^{(t-1)}|/|v^{(t-1)}| $ の最大値を計算する．プラトー（変化が十分小さく
        なった状態）に達したかどうかを数値的に判定するために用いる
        （`.orders/order_026.md` 5節）．
    引数:
        values (Sequence[float])．時系列の数値列（例：エポックごとの近似誤差・分類精度）．
        window (int)．末尾から何個の値を対象にするか．`window + 1` 個以上の要素が必要．
    戻り値: max_relative_change (float)．末尾`window`区間における相対変化の最大値．
        非有限値（NaN・Inf）が含まれる場合は `float("inf")` を返す．
    """
    tail = np.asarray(values[-(window + 1):], dtype=np.float64)
    if not np.all(np.isfinite(tail)):
        return float("inf")
    diffs = np.abs(tail[1:] - tail[:-1])
    denom = np.abs(tail[:-1])
    denom = np.where(denom == 0.0, np.finfo(np.float64).eps, denom)
    return float(np.max(diffs / denom))


def iteration(model, inputs, teacher_signals, reg_lambda, optimizer, snapshot_model=None) -> dict:
    """
    概要: 1つのミニバッチのデータを学習する関数．`snapshot_model` が指定される場合（SVRG系
        手法），同一ミニバッチに対して `model`（現在のパラメータ $ w_s^k $）と
        `snapshot_model`（スナップショット $ z_s $）の双方でforward／backwardを実行し，
        2種類の勾配を用いて `optimizer.step(grad_at_snapshot)` を呼び出す．`snapshot_model`
        が `None` の場合（SGD），通常のforward／backward／`optimizer.step()` を実行する．
    引数:
        model (torch.nn.Module)．現在のパラメータを保持するモデル．
        inputs (torch.Tensor)，形状 (B, 3, 32, 32)．入力画像．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
        optimizer (torch.optim.Optimizer)．
        snapshot_model (torch.nn.Module) = None．SVRG系手法のスナップショットを保持するモデル．
    戻り値: metrics_to_value (dict)．{"loss": ..., "accuracy": ...}．
    """
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_func(outputs, teacher_signals, model, reg_lambda)
    loss.backward()

    if snapshot_model is not None:
        snapshot_model.zero_grad()
        snapshot_outputs = snapshot_model(inputs)
        snapshot_loss = loss_func(snapshot_outputs, teacher_signals, snapshot_model, reg_lambda)
        snapshot_loss.backward()
        grad_at_snapshot = [p.grad.detach().clone() for p in snapshot_model.parameters()]
        optimizer.step(grad_at_snapshot)
    else:
        optimizer.step()

    metrics = metrics_func(outputs, teacher_signals)
    metrics["loss"] = loss.item()
    return metrics


def train_epoch(model, dataloader, device, reg_lambda, optimizer, snapshot_model, desc) -> None:
    """
    概要: 1エポック分の学習（内部ループ）を実行する．
    引数:
        model, dataloader, device, reg_lambda, optimizer, snapshot_model：`iteration` 参照．
        desc (str)．tqdmの進捗バーの説明文．
    戻り値: なし
    """
    for inputs, teacher_signals in tqdm(dataloader, desc=desc, leave=False):
        inputs = inputs.to(device)
        teacher_signals = teacher_signals.to(device)
        iteration(model, inputs, teacher_signals, reg_lambda, optimizer, snapshot_model)


def evaluate_epoch(model, dataloader, device, reg_lambda) -> dict:
    """
    概要: 1つのデータローダーの全データを（学習せず）評価する関数．
    引数:
        model (torch.nn.Module)．
        dataloader (torch.utils.data.DataLoader)．
        device (torch.device)．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: metrics_to_value (dict)．1データあたりの評価・誤差の平均値の辞書．
    """
    total_metrics = {}
    total_count = 0
    with torch.no_grad():
        for inputs, teacher_signals in dataloader:
            inputs = inputs.to(device)
            teacher_signals = teacher_signals.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, teacher_signals, model, reg_lambda)
            batch_metrics = metrics_func(outputs, teacher_signals)
            batch_metrics["loss"] = loss.item()

            batch_size = inputs.shape[0]
            for key, value in batch_metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value * batch_size
            total_count += batch_size

    return {key: value / total_count for key, value in total_metrics.items()}


def compute_full_gradient_and_metrics(model, dataloader, device, reg_lambda):
    """
    概要: データローダー全体に対するフル勾配，および同じ1回の走査で得られる誤差・分類精度の
        平均値をまとめて計算する．
    引数:
        model (torch.nn.Module)．勾配・評価の対象となるパラメータを保持するモデル．
        dataloader (torch.utils.data.DataLoader)．データセット全体を走査するデータローダー．
        device (torch.device)．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値:
        grads (list of torch.Tensor)．`model.parameters()` と同じ順序・形状のフル勾配．
        metrics (dict)．{"loss": ..., "accuracy": ...} の1データあたりの平均値．
    """
    accumulated_grads = [torch.zeros_like(p) for p in model.parameters()]
    total_metrics = {}
    total_count = 0

    for inputs, teacher_signals in dataloader:
        inputs = inputs.to(device)
        teacher_signals = teacher_signals.to(device)
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
    概要: フル勾配の近似誤差 $ \\|e_s\\|^2 = \\|g_s - \\nabla f(z_s)\\|^2 $ を計算する．
    引数:
        snapshot_gradient (list of torch.Tensor)．スナップショット勾配 $ g_s $．
        true_full_gradient (list of torch.Tensor)．真のフル勾配 $ \\nabla f(z_s) $．
    戻り値: approx_error (float)．
    """
    return sum(
        torch.sum((g_s - g_true) ** 2).item()
        for g_s, g_true in zip(snapshot_gradient, true_full_gradient)
    )


def _save_if_best(model, test_accuracy, best_accuracy, target_dir):
    """
    概要: 検証用データに対する分類精度が過去最高を更新した場合，モデルの重みを保存する．
    引数:
        model (torch.nn.Module)．保存対象のモデル．
        test_accuracy (float)．今回のエポックの検証用データに対する分類精度．
        best_accuracy (float)．これまでの最高精度．
        target_dir (str)．保存先ディレクトリ．
    戻り値: best_accuracy (float)．更新後の最高精度．
    """
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))
        return test_accuracy
    return best_accuracy


def run_sgd(target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, seed, logger):
    """
    概要: SGDによる学習を実行し，各エポック終了時の評価指標を `logger` に記録する．
    引数:
        target_dir (str)．結果保存先ディレクトリ．
        load_dataloader_func (func)．
        eta (float)．学習率．
        batch_size (int)．ミニバッチサイズ．
        reg_lambda (float)．L2正則化係数．
        epochs (int)．
        device (torch.device)．
        seed (int)．乱数シード．
        logger (ResultLogger)．
    戻り値: なし
    """
    train_dataloader, test_dataloader = load_dataloader_func(seed=seed, batch_size=batch_size)
    N_train = len(train_dataloader.dataset)

    model = load_model(AlexNetCIFARNorm, seed=seed, norm_type=NORM_TYPE).to(device)
    optimizer = SGD(model.parameters(), lr=eta)

    oracle_calls = 0
    best_accuracy = -1.0

    train_metrics = evaluate_epoch(model, train_dataloader, device, reg_lambda)
    test_metrics = evaluate_epoch(model, test_dataloader, device, reg_lambda)
    logger(0, oracle_calls, 0.0, train_metrics["loss"], test_metrics["accuracy"], float("nan"))
    best_accuracy = _save_if_best(model, test_metrics["accuracy"], best_accuracy, target_dir)

    elapsed_time = 0.0
    desc = f"SGD bs={batch_size} eta={eta} seed={seed}"
    for epoch_index in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        start_time = time.time()
        train_epoch(
            model, train_dataloader, device, reg_lambda, optimizer, snapshot_model=None,
            desc=f"{desc} ep{epoch_index}",
        )
        oracle_calls += N_train
        elapsed_time += time.time() - start_time

        start_time = time.time()
        train_metrics = evaluate_epoch(model, train_dataloader, device, reg_lambda)
        test_metrics = evaluate_epoch(model, test_dataloader, device, reg_lambda)
        elapsed_time += time.time() - start_time

        logger(
            epoch_index, oracle_calls, elapsed_time,
            train_metrics["loss"], test_metrics["accuracy"], float("nan"),
        )
        best_accuracy = _save_if_best(model, test_metrics["accuracy"], best_accuracy, target_dir)


def run_variance_reduced(
    method, target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, seed, logger
):
    """
    概要: SVRG系手法（SVRG，NFG SVRG，ASAI SVRG）による学習を実行し，各エポック終了時の
        評価指標を `logger` に記録する．SVRGのみ，スナップショット勾配 $ g_s $ が真のフル
        勾配であり，これを次エポックの学習に用いるため，その計算時間は `elapsed_time` に
        計上する．NFG SVRG・ASAI SVRGは評価指標（近似誤差）算出のためだけに真のフル勾配を
        計算するため，この計算時間は `elapsed_time` から除外する（`.reports/report_025.md`
        2.2節の修正を踏襲）．
    引数:
        method (str)．"SVRG"，"NFG_SVRG"，"ASAI_SVRG" のいずれか．
        target_dir (str)．結果保存先ディレクトリ．
        load_dataloader_func (func)．
        eta (float)．学習率．
        batch_size (int)．ミニバッチサイズ．
        reg_lambda (float)．L2正則化係数．
        epochs (int)．外部ループ数 $ S $．
        device (torch.device)．
        seed (int)．乱数シード．
        logger (ResultLogger)．
    戻り値: なし
    """
    assert method in _VARIANCE_REDUCED_OPTIMIZER_CLASSES

    train_dataloader, test_dataloader = load_dataloader_func(seed=seed, batch_size=batch_size)
    N_train = len(train_dataloader.dataset)
    K = len(train_dataloader)  # 1エポックあたりのミニバッチ数

    model = load_model(AlexNetCIFARNorm, seed=seed, norm_type=NORM_TYPE).to(device)
    # z_0 <- w_0．snapshot_modelはmodelと同じseedで初期化することで同一の初期値を持つ．
    snapshot_model = load_model(AlexNetCIFARNorm, seed=seed, norm_type=NORM_TYPE).to(device)
    refresh_and_freeze_spectral_norm(snapshot_model)  # NORM_TYPE="groupnorm"のため何も行わない

    OptimizerClass = _VARIANCE_REDUCED_OPTIMIZER_CLASSES[method]
    optimizer = OptimizerClass(model.parameters(), lr=eta, K=K)
    rng = np.random.default_rng(seed)

    oracle_calls = 0

    if method == "SVRG":
        snapshot_grad, train_metrics_0 = compute_full_gradient_and_metrics(
            snapshot_model, train_dataloader, device, reg_lambda
        )
        true_full_grad_0 = snapshot_grad
    else:
        snapshot_grad = optimizer.get_snapshot_gradient()  # g_0 = 0
        true_full_grad_0, train_metrics_0 = compute_full_gradient_and_metrics(
            snapshot_model, train_dataloader, device, reg_lambda
        )
    approx_error_0 = compute_approx_error(snapshot_grad, true_full_grad_0)

    best_accuracy = -1.0
    test_metrics = evaluate_epoch(snapshot_model, test_dataloader, device, reg_lambda)
    logger(0, oracle_calls, 0.0, train_metrics_0["loss"], test_metrics["accuracy"], approx_error_0)
    best_accuracy = _save_if_best(snapshot_model, test_metrics["accuracy"], best_accuracy, target_dir)

    if method == "SVRG":
        optimizer.set_snapshot_gradient(snapshot_grad)
        oracle_calls += N_train

    elapsed_time = 0.0
    desc_prefix = f"{method} bs={batch_size} eta={eta} seed={seed}"
    for epoch_index in tqdm(range(1, epochs + 1), desc=desc_prefix, leave=False):
        start_time = time.time()
        optimizer.begin_epoch(rng)
        train_epoch(
            model, train_dataloader, device, reg_lambda, optimizer, snapshot_model,
            desc=f"{desc_prefix} ep{epoch_index}",
        )
        oracle_calls += 2 * N_train

        optimizer.end_epoch()
        set_model_params(snapshot_model, optimizer.get_snapshot_params(), source_model=model)
        refresh_and_freeze_spectral_norm(snapshot_model)
        elapsed_time += time.time() - start_time

        if method == "SVRG":
            start_time = time.time()
            snapshot_grad, train_metrics = compute_full_gradient_and_metrics(
                snapshot_model, train_dataloader, device, reg_lambda
            )
            oracle_calls += N_train
            optimizer.set_snapshot_gradient(snapshot_grad)
            approx_error = 0.0
            elapsed_time += time.time() - start_time
        else:
            # 評価指標算出のためだけの診断的なフル勾配計算であり，elapsed_timeには計上しない．
            snapshot_grad = optimizer.get_snapshot_gradient()
            true_full_grad, train_metrics = compute_full_gradient_and_metrics(
                snapshot_model, train_dataloader, device, reg_lambda
            )
            approx_error = compute_approx_error(snapshot_grad, true_full_grad)

        start_time = time.time()
        test_metrics = evaluate_epoch(snapshot_model, test_dataloader, device, reg_lambda)
        elapsed_time += time.time() - start_time

        logger(
            epoch_index, oracle_calls, elapsed_time,
            train_metrics["loss"], test_metrics["accuracy"], approx_error,
        )
        best_accuracy = _save_if_best(snapshot_model, test_metrics["accuracy"], best_accuracy, target_dir)

        if not np.isfinite(train_metrics["loss"]):
            # `.orders/order_026.md` 6節：長期学習で崩壊が生じた場合，これ以上学習を継続
            # しても意味がないため打ち切る．打ち切った時点までの記録はそのまま保存する．
            break


def is_run_completed(target_dir: str, epochs: int) -> bool:
    """
    概要: 指定した条件の学習が既に正常終了しているか確認する．崩壊により学習を打ち切った
        条件（`run_variance_reduced` 参照）も，記録されたログが存在すれば完了済みとして
        扱う（打ち切り後に再学習しても同じ崩壊を再現するだけであるため）．
    引数:
        target_dir (str)．結果を保存するディレクトリ．
        epochs (int)．期待されるエポック数．
    戻り値: completed (bool)．
    """
    log_path = os.path.join(target_dir, "log.json")
    if not os.path.exists(log_path):
        return False
    try:
        logger = ResultLogger(log_path)
        recorded_epochs = logger["epoch"]
        if len(recorded_epochs) == epochs + 1:
            return True
        # 崩壊による打ち切り：記録の最後のtrain_lossが非有限値であれば打ち切り完了とみなす．
        train_loss = logger["train_loss"]
        return len(train_loss) > 0 and not np.isfinite(train_loss[-1])
    except (json.JSONDecodeError, OSError):
        return False


def run_single_experiment(args):
    """
    概要: 1つの (手法, バッチサイズ, Seed) の組に対する学習を実行し，結果を保存する．
        すでに正常終了した結果（または崩壊による打ち切り）が存在する場合は学習をスキップする．
    引数: args (tuple)．(method, batch_size, seed) のタプル．
    戻り値: なし
    """
    method, batch_size, seed = args
    torch.set_num_threads(1)

    epochs = EPOCHS_BY_BATCH_SIZE[batch_size]
    name = hp_name(batch_size, epochs)
    target_dir = os.path.join(OUTPUT_ROOT, method, name, str(seed))
    if is_run_completed(target_dir, epochs):
        print(f"[skip] {method}/{name}/{seed} は既に完了しています．", flush=True)
        return

    os.makedirs(target_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    if method == "SGD":
        run_sgd(target_dir, load_dataloader, LEARNING_RATE, batch_size, REG_LAMBDA, epochs, device, seed, logger)
    else:
        run_variance_reduced(
            method, target_dir, load_dataloader, LEARNING_RATE, batch_size, REG_LAMBDA, epochs, device, seed, logger
        )

    logger.save(os.path.join(target_dir, "log.json"))

    train_dataloader, test_dataloader = load_dataloader(seed=seed, batch_size=batch_size)
    config = {
        "experiment": EXPERIMENT_NAME,
        "method": method,
        "seed": seed,
        "learning_rate": LEARNING_RATE,
        "batch_size": batch_size,
        "norm_type": NORM_TYPE,
        "reg_lambda": REG_LAMBDA,
        "epochs": epochs,
        "K": len(train_dataloader),
        "total_iterations": epochs * len(train_dataloader),
        "N_train": len(train_dataloader.dataset),
        "N_test": len(test_dataloader.dataset),
        "collapsed": len(logger["train_loss"]) < epochs + 1,
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{name}/{seed}", flush=True)


def main():
    """
    概要: 実験ex0023の全条件（4手法 × 3バッチサイズ × 5Seed = 60条件）をマルチプロセスで
        並列に学習する．学習率0.001固定，`.orders/order_026.md` 3節の指示により全条件を
        ゼロから再学習する（既存実験からのコピー再利用は行わない）．
    引数: なし
    戻り値: なし
    """
    print(f"バッチサイズ = {BATCH_SIZES}, 学習率 = {LEARNING_RATE}, 正規化層 = {NORM_TYPE}")
    print(f"バッチサイズごとのエポック数（初期値）: {EPOCHS_BY_BATCH_SIZE}")

    tasks = [
        (method, batch_size, seed)
        for method, batch_size, seed in itertools.product(METHODS, BATCH_SIZES, SEEDS)
    ]

    num_workers = min(8, len(tasks))
    print(f"並列プロセス数: {num_workers}（総タスク数: {len(tasks)}）")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks, chunksize=1)

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    main()
