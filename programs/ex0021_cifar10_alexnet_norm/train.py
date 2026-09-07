"""
実験ex0021（CIFAR-10における正規化層・バッチサイズの構造的変更による安定性の検証）の
学習ループを定義し，実験を実行するスクリプト．

`.orders/order_023.md` に対応する．実験2（`report_022.md`）は，CIFAR-10・AlexNet
（BatchNormalization・Dropout除去）という非凸・ミニバッチ設定において，NFG SVRG・
ASAI SVRGが学習率0.01ではほぼ全条件で，学習率0.001でもバッチサイズ8では発散するという
結果を得た．本実験は，学習率の再探索ではなく，(1) サンプル単位で完結する正規化層
（LayerNorm・GroupNorm）の追加，(2) バッチサイズの拡大（128, 256, 512）という構造的変更に
より，発散を緩和できるかを検証する．

## 比較手法

`.orders/order_023.md` 2.5節の指示に基づき，SGD・SVRGは実験2で既に安定性が確認済みのため
対象外とし，NFG SVRG（`NFGSVRGFinalPoint`）・ASAI SVRG（`ASAISVRG`）の2手法のみを比較する．

## エポック数の決定（バッチサイズ間で総イテレーション数を揃える）

バッチサイズを拡大すると1エポックあたりのイテレーション数 $ K=\\lceil N_{\\text{train}}/
\\text{batch size}\\rceil $ が減少するため，全条件でエポック数を12に固定すると総イテレーション
数（学習の進み具合の目安）がバッチサイズによって不公平になる．実験2の基準条件
（$ \\text{bs}=128 $，12エポック，$ K=391 $，総イテレーション数 $ 12\\times391=4692 $）を
基準とし，$ \\text{epoch} \\approx 4692/K $ を目安にエポック数を決定した．

| バッチサイズ | $ K $ | エポック数 | 総イテレーション数（epoch×K） |
| ---: | ---: | ---: | ---: |
| 128 | 391 | 12 | 4692 |
| 256 | 196 | 24 | 4704 |
| 512 | 98 | 48 | 4704 |

## スペクトル正規化の追加（`.orders/order_024.md`）

`.orders/order_024.md` の指示に基づき，正規化層の第4パターンとしてSpectralNorm
（`model.SpectralConv2d`）を追加した．SpectralNormは，正規化層とは異なり畳み込み層自体の
重みに制約を課すため，power iterationの補助バッファ（$ u, v $）を持つ．このバッファが
forwardのたびに更新される標準的な実装をそのまま用いると，スナップショット $ z_s $ に対する
勾配評価 $ \\nabla f_n(z_s) $ が呼び出しのたびに変化し，SVRG系手法の理論的前提（同一の
$ n $，同一の $ z_s $ に対する勾配は常に同じ値になること）を破る．そのため，`snapshot_model`
の重みを更新した直後に必ず `model.refresh_and_freeze_spectral_norm(snapshot_model)` を
呼び出し，バッファを凍結する（`run_variance_reduced` 参照）．この決定論性は
`tests/test_ex0021_cifar10_alexnet_norm.py::test_spectral_norm_snapshot_gradient_is_deterministic`
で検証している．
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
from optimizers import ASAISVRG, NFGSVRG  # noqa: E402

from data import load_dataloader  # noqa: E402
from model import (  # noqa: E402
    AlexNetCIFARNorm,
    load_model,
    loss_func,
    metrics_func,
    refresh_and_freeze_spectral_norm,
    set_model_params,
)

EXPERIMENT_NAME = "ex0021_cifar10_alexnet_norm"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["NFG_SVRG", "ASAI_SVRG"]
BATCH_SIZES = [128, 256, 512]
NORM_TYPES = ["none", "layernorm", "groupnorm", "spectralnorm"]
LEARNING_RATE = 0.001  # `.orders/order_023.md` 2節の指示により固定
REG_LAMBDA = 5e-4  # 実験2と同一

# バッチサイズごとのエポック数．total_iterations = epochs * K がおおむね一定になるよう決定
# （モジュールdocstring参照）．
EPOCHS_BY_BATCH_SIZE = {128: 12, 256: 24, 512: 48}

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "NFG_SVRG": NFGSVRG,
    "ASAI_SVRG": ASAISVRG,
}


def iteration(model, inputs, teacher_signals, reg_lambda, optimizer, snapshot_model) -> dict:
    """
    概要: 1つのミニバッチのデータを学習する関数．同一ミニバッチに対して `model`（現在の
        パラメータ $ w_s^k $）と `snapshot_model`（スナップショット $ z_s $）の双方で
        forward／backwardを実行し，2種類の勾配を用いて `optimizer.step(grad_at_snapshot)`
        を呼び出す．
    引数:
        model (torch.nn.Module)．現在のパラメータを保持するモデル．
        inputs (torch.Tensor)，形状 (B, 3, 32, 32)．入力画像．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
        optimizer (torch.optim.Optimizer)．
        snapshot_model (torch.nn.Module)．SVRG系手法のスナップショットを保持するモデル．
    戻り値: metrics_to_value (dict)．{"loss": ..., "accuracy": ...}．
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
    optimizer.step(grad_at_snapshot)

    metrics = metrics_func(outputs, teacher_signals)
    metrics["loss"] = loss.item()
    return metrics


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


def run_variance_reduced(
    method, target_dir, load_dataloader_func, norm_type, batch_size, reg_lambda,
    epochs, device, seed, logger,
):
    """
    概要: NFG SVRG・ASAI SVRGによる学習を実行し，各エポック終了時の評価指標を `logger` に
        記録する．
    引数:
        method (str)．"NFG_SVRG" または "ASAI_SVRG"．
        target_dir (str)．結果保存先ディレクトリ．
        load_dataloader_func (func)．
        norm_type (str)．正規化層の種類．
        batch_size (int)．ミニバッチサイズ．
        reg_lambda (float)．L2正則化係数．
        epochs (int)．外部ループ数 $ S $．
        device (torch.device)．
        seed (int)．乱数シード．
        logger (ResultLogger)．
    戻り値: なし
    """
    train_dataloader, test_dataloader = load_dataloader_func(seed=seed, batch_size=batch_size)
    N_train = len(train_dataloader.dataset)
    K = len(train_dataloader)  # 1エポックあたりのミニバッチ数

    model = load_model(AlexNetCIFARNorm, seed=seed, norm_type=norm_type).to(device)
    # z_0 <- w_0．snapshot_modelはmodelと同じseedで初期化することで同一の初期値を持つ．
    snapshot_model = load_model(AlexNetCIFARNorm, seed=seed, norm_type=norm_type).to(device)
    # norm_type="spectralnorm"の場合，z_0=w_0の重みに対しpower iterationバッファを確定させ
    # 凍結する．他のnorm_typeでは`SpectralConv2d`を含まないため何も行わない．
    refresh_and_freeze_spectral_norm(snapshot_model)

    OptimizerClass = _VARIANCE_REDUCED_OPTIMIZER_CLASSES[method]
    optimizer = OptimizerClass(model.parameters(), lr=LEARNING_RATE, K=K)
    rng = np.random.default_rng(seed)

    oracle_calls = 0

    snapshot_grad = optimizer.get_snapshot_gradient()  # g_0 = 0
    true_full_grad_0, train_metrics_0 = compute_full_gradient_and_metrics(
        snapshot_model, train_dataloader, device, reg_lambda
    )
    approx_error_0 = compute_approx_error(snapshot_grad, true_full_grad_0)

    best_accuracy = -1.0
    start_time = time.time()
    test_metrics = evaluate_epoch(snapshot_model, test_dataloader, device, reg_lambda)
    logger(
        0, oracle_calls, 0.0,
        train_metrics_0["loss"], test_metrics["accuracy"], approx_error_0,
    )
    best_accuracy = _save_if_best(snapshot_model, test_metrics["accuracy"], best_accuracy, target_dir)

    desc_prefix = f"{method} norm={norm_type} bs={batch_size} seed={seed}"
    for epoch_index in tqdm(range(1, epochs + 1), desc=desc_prefix, leave=False):
        optimizer.begin_epoch(rng)
        train_epoch(
            model, train_dataloader, device, reg_lambda, optimizer, snapshot_model,
            desc=f"{desc_prefix} ep{epoch_index}",
        )
        oracle_calls += 2 * N_train

        optimizer.end_epoch()
        set_model_params(snapshot_model, optimizer.get_snapshot_params(), source_model=model)
        # z_{s+1}の新しい重みに対しpower iterationバッファを更新・凍結する．
        refresh_and_freeze_spectral_norm(snapshot_model)

        snapshot_grad = optimizer.get_snapshot_gradient()
        true_full_grad, train_metrics = compute_full_gradient_and_metrics(
            snapshot_model, train_dataloader, device, reg_lambda
        )
        approx_error = compute_approx_error(snapshot_grad, true_full_grad)

        test_metrics = evaluate_epoch(snapshot_model, test_dataloader, device, reg_lambda)
        elapsed_time = time.time() - start_time

        logger(
            epoch_index, oracle_calls, elapsed_time,
            train_metrics["loss"], test_metrics["accuracy"], approx_error,
        )
        best_accuracy = _save_if_best(snapshot_model, test_metrics["accuracy"], best_accuracy, target_dir)


def is_run_completed(target_dir: str, epochs: int) -> bool:
    """
    概要: 指定した条件の学習が既に正常終了しているか確認する．
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
        return len(logger["epoch"]) == epochs + 1
    except (json.JSONDecodeError, OSError):
        return False


def run_single_experiment(args):
    """
    概要: 1つの (手法, バッチサイズ, 正規化層, Seed) の組に対する学習を実行し，結果を保存する．
        すでに正常終了した結果が存在する場合は学習をスキップする．
    引数: args (tuple)．(method, batch_size, norm_type, seed) のタプル．
    戻り値: なし
    """
    method, batch_size, norm_type, seed = args
    torch.set_num_threads(1)

    epochs = EPOCHS_BY_BATCH_SIZE[batch_size]
    hp_name = f"lr{LEARNING_RATE}_bs{batch_size}_norm{norm_type}_lambda{REG_LAMBDA}_epochs{epochs}"
    target_dir = os.path.join(OUTPUT_ROOT, method, hp_name, str(seed))
    if is_run_completed(target_dir, epochs):
        print(f"[skip] {method}/{hp_name}/{seed} は既に完了しています．", flush=True)
        return

    os.makedirs(target_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    run_variance_reduced(
        method, target_dir, load_dataloader, norm_type, batch_size, REG_LAMBDA,
        epochs, device, seed, logger,
    )

    logger.save(os.path.join(target_dir, "log.json"))

    train_dataloader, test_dataloader = load_dataloader(seed=seed, batch_size=batch_size)
    config = {
        "experiment": EXPERIMENT_NAME,
        "method": method,
        "seed": seed,
        "learning_rate": LEARNING_RATE,
        "batch_size": batch_size,
        "norm_type": norm_type,
        "reg_lambda": REG_LAMBDA,
        "epochs": epochs,
        "K": len(train_dataloader),
        "total_iterations": epochs * len(train_dataloader),
        "N_train": len(train_dataloader.dataset),
        "N_test": len(test_dataloader.dataset),
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{hp_name}/{seed}", flush=True)


def main():
    """
    概要: 実験ex0021の全条件（2手法 × 3バッチサイズ × 3正規化層 × 5Seed = 90条件）を
        マルチプロセスで並列に学習する．
    引数: なし
    戻り値: なし
    """
    print(
        f"バッチサイズ = {BATCH_SIZES}, 正規化層 = {NORM_TYPES}, "
        f"学習率 = {LEARNING_RATE}, lambda = {REG_LAMBDA}"
    )
    print(f"バッチサイズごとのエポック数: {EPOCHS_BY_BATCH_SIZE}")

    tasks = [
        (method, batch_size, norm_type, seed)
        for method, batch_size, norm_type, seed in itertools.product(
            METHODS, BATCH_SIZES, NORM_TYPES, SEEDS
        )
    ]

    num_workers = min(8, len(tasks))
    print(f"並列プロセス数: {num_workers}（総タスク数: {len(tasks)}）")

    # 実験2（`.reports/report_022.md` 4.1節）で判明した通り，タスクごとの実行時間が条件に
    # よって大きく異なりうるため，`chunksize=1` で動的に負荷分散する．
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks, chunksize=1)

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    main()
