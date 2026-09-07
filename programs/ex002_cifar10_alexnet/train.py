"""
実験2（CIFAR-10を用いた多値分類問題）の学習ループを定義し，実験を実行するスクリプト．

`.orders/order_022.md` の実験2に対応する．理論解析の仮定（平滑性・凸性）を満たさない非凸かつ
実践的な設定において，提案手法ASAI SVRGの実用的な有効性を検証する．実験0・実験1
（`report_020.md`／`report_021.md`）で得られた知見（ASAI SVRGの近似誤差がNFG SVRGより
一貫して小さいこと，内部ループ長 $ K $ と誤差床の恒久性の関係，学習率依存の不安定化）を
非凸・ミニバッチ学習の設定で検証する．

`@.ai/ai-dev-kit/machine_learning.md` が定める `loss_func`／`metrics_func`／`iteration`／
`epoch`／`train` の関数構成に従う．SVRG系手法（SVRG，NFG SVRG，ASAI SVRG）は現在のパラメータ
$ w_s^k $ とスナップショット $ z_s $ の双方における勾配を必要とするため，`iteration` 関数は
スナップショット専用モデル `snapshot_model` を任意引数として受け取れるよう拡張している
（`programs_old/ex002_cifar10_cnn/train.py` で確立した拡張と同一）．

## Optimizerクラスの再利用性

`programs/optimizers/` の各クラスは，`step()` が受け取る勾配を「1回のイテレーションで得られた
勾配」として扱うのみで，そのイテレーションが単一サンプルかミニバッチかを区別しない．
NFG SVRG・ASAI SVRGの平均勾配（running average）はイテレーション番号 $ k $ に基づく逐次更新
（$ \\bar{g}^{k+1} = \\frac{k}{k+1}\\bar{g}^k + \\frac{1}{k+1}\\nabla f_{n_s^k}(w_s^k) $）で
あり，$ \\nabla f_{n_s^k} $ がミニバッチ平均勾配であっても同じ更新式がそのまま成立する．
したがって，Optimizerクラス自体の変更は不要であり，内部ループ長 $ K $ を
$ \\lceil N_{\\text{train}}/\\text{batch size} \\rceil $ に設定し，各 `step()` 呼び出しに
ミニバッチ平均勾配を渡すことでミニバッチ学習に対応できる．
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
from model import AlexNetCIFAR, load_model, loss_func, metrics_func, set_model_params  # noqa: E402

EXPERIMENT_NAME = "ex002_cifar10_alexnet"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
BATCH_SIZES = [8, 32, 128]
LEARNING_RATES = [0.01, 0.001, 0.0001]
# 一般的なCNN学習で用いられるweight decayの範囲（1e-4〜1e-3程度）から，過学習の抑制と精度の
# バランスを踏まえ 5e-4 を採用する．全条件で共通に用いる．
REG_LAMBDA = 5e-4
# 実行前の1エポックあたりの実行時間概算（1プロセス，NVIDIA RTX PRO 6000）：
#   バッチサイズ128：SGD 24.3s, SVRG 32.7s, NFG_SVRG 36.3s, ASAI_SVRG 38.8s（計 132.1s）
#   バッチサイズ8　：SGD 95.0s, SVRG 197.8s, NFG_SVRG 202.8s, ASAI_SVRG 208.3s（計 703.9s）
# GPU使用率はいずれも13〜24%程度に留まり（Pythonループ・カーネル起動オーバーヘッドが支配的で
# GPU演算自体は律速していない），VRAM使用量も1プロセスあたり1GB未満と小さい．
# 4手法×3バッチサイズの合計は約1091秒/エポックであり，3学習率×5Seed分（15倍）で
# 約4.55時間/エポック（単一プロセス換算）を要する．全180条件×十分なエポック数を単一プロセスで
# 実行することは非現実的であるため，並列数を8（GPU使用率・VRAMに十分な余裕があることを踏まえた
# 値）に設定し，エポック数は限られた計算時間の中で収束傾向を確認できる範囲として12とした．
EPOCHS = 12

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "SVRG": SVRG,
    "NFG_SVRG": NFGSVRG,
    "ASAI_SVRG": ASAISVRG,
}


def iteration(model, inputs, teacher_signals, reg_lambda, optimizer=None, snapshot_model=None) -> dict:
    """
    概要: 1つのミニバッチのデータを学習/検証する関数．`optimizer` がSVRG系手法（
        `snapshot_model` が指定される場合）は，同一ミニバッチに対して `model`（現在の
        パラメータ $ w_s^k $）と `snapshot_model`（スナップショット $ z_s $）の双方で
        forward／backwardを実行し，2種類の勾配を用いて `optimizer.step(grad_at_snapshot)`
        を呼び出す．
    引数:
        model (torch.nn.Module)．現在のパラメータを保持するモデル．
        inputs (torch.Tensor)，形状 (B, 3, 32, 32)．入力画像．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
        optimizer (torch.optim.Optimizer) = None．
        snapshot_model (torch.nn.Module) = None．SVRG系手法のスナップショットを保持するモデル．
    戻り値: metrics_to_value (dict)．{"loss": ..., "accuracy": ...}．
    """
    if optimizer is None:
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_func(outputs, teacher_signals, model, reg_lambda)
    else:
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


def epoch(model, dataloader, device, reg_lambda, optimizer=None, snapshot_model=None) -> dict:
    """
    概要: 1つのデータローダーの全データを学習/検証する関数．
    引数:
        model (torch.nn.Module)．
        dataloader (torch.utils.data.DataLoader)．
        device (torch.device)．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
        optimizer (torch.optim.Optimizer) = None．
        snapshot_model (torch.nn.Module) = None．SVRG系手法のスナップショットを保持するモデル．
    戻り値: metrics_to_value (dict)．1データあたりの評価・誤差の平均値の辞書．
    """
    total_metrics = {}
    total_count = 0
    for inputs, teacher_signals in tqdm(dataloader, leave=False):
        inputs = inputs.to(device)
        teacher_signals = teacher_signals.to(device)

        batch_metrics = iteration(
            model, inputs, teacher_signals, reg_lambda, optimizer, snapshot_model
        )

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

    model = load_model(AlexNetCIFAR, seed=seed).to(device)
    optimizer = SGD(model.parameters(), lr=eta)

    oracle_calls = 0
    best_accuracy = -1.0
    start_time = time.time()

    with torch.no_grad():
        train_metrics = epoch(model, train_dataloader, device, reg_lambda, optimizer=None)
        test_metrics = epoch(model, test_dataloader, device, reg_lambda, optimizer=None)
    logger(
        0, oracle_calls, 0.0,
        train_metrics["loss"], test_metrics["accuracy"], float("nan"),
    )
    best_accuracy = _save_if_best(model, test_metrics["accuracy"], best_accuracy, target_dir)

    for epoch_index in tqdm(range(1, epochs + 1), desc=f"SGD bs={batch_size} eta={eta} seed={seed}", leave=False):
        train_metrics = epoch(model, train_dataloader, device, reg_lambda, optimizer=optimizer)
        oracle_calls += N_train

        with torch.no_grad():
            test_metrics = epoch(model, test_dataloader, device, reg_lambda, optimizer=None)
        elapsed_time = time.time() - start_time

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
        評価指標を `logger` に記録する．
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
    K = len(train_dataloader)  # 1エポックあたりのミニバッチ数 = ceil(N_train / batch_size)

    model = load_model(AlexNetCIFAR, seed=seed).to(device)
    # z_0 <- w_0．snapshot_modelはmodelと同じseedで初期化することで同一の初期値を持つ．
    snapshot_model = load_model(AlexNetCIFAR, seed=seed).to(device)

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
        snapshot_grad = optimizer.get_snapshot_gradient()
        true_full_grad_0, train_metrics_0 = compute_full_gradient_and_metrics(
            snapshot_model, train_dataloader, device, reg_lambda
        )

    approx_error_0 = compute_approx_error(snapshot_grad, true_full_grad_0)

    best_accuracy = -1.0
    start_time = time.time()
    with torch.no_grad():
        test_metrics = epoch(snapshot_model, test_dataloader, device, reg_lambda, optimizer=None)
    logger(
        0, oracle_calls, 0.0,
        train_metrics_0["loss"], test_metrics["accuracy"], approx_error_0,
    )
    best_accuracy = _save_if_best(snapshot_model, test_metrics["accuracy"], best_accuracy, target_dir)

    if method == "SVRG":
        optimizer.set_snapshot_gradient(snapshot_grad)
        oracle_calls += N_train

    desc = f"{method} bs={batch_size} eta={eta} seed={seed}"
    for epoch_index in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        optimizer.begin_epoch(rng)
        epoch(
            model, train_dataloader, device, reg_lambda,
            optimizer=optimizer, snapshot_model=snapshot_model,
        )
        oracle_calls += 2 * N_train

        optimizer.end_epoch()
        snapshot_params = optimizer.get_snapshot_params()
        set_model_params(snapshot_model, snapshot_params, source_model=model)

        if method == "SVRG":
            snapshot_grad, train_metrics = compute_full_gradient_and_metrics(
                snapshot_model, train_dataloader, device, reg_lambda
            )
            oracle_calls += N_train
            optimizer.set_snapshot_gradient(snapshot_grad)
            approx_error = 0.0
        else:
            snapshot_grad = optimizer.get_snapshot_gradient()
            true_full_grad, train_metrics = compute_full_gradient_and_metrics(
                snapshot_model, train_dataloader, device, reg_lambda
            )
            approx_error = compute_approx_error(snapshot_grad, true_full_grad)

        with torch.no_grad():
            test_metrics = epoch(snapshot_model, test_dataloader, device, reg_lambda, optimizer=None)
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
    概要: 1つの (手法, バッチサイズ, 学習率, Seed) の組に対する学習を実行し，結果を保存する．
        すでに正常終了した結果が存在する場合は学習をスキップする．
    引数: args (tuple)．(method, batch_size, eta, seed) のタプル．
    戻り値: なし
    """
    method, batch_size, eta, seed = args
    torch.set_num_threads(1)

    hp_name = f"lr{eta}_bs{batch_size}_lambda{REG_LAMBDA}_epochs{EPOCHS}"
    target_dir = os.path.join(OUTPUT_ROOT, method, hp_name, str(seed))
    if is_run_completed(target_dir, EPOCHS):
        print(f"[skip] {method}/{hp_name}/{seed} は既に完了しています．", flush=True)
        return

    os.makedirs(target_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    if method == "SGD":
        run_sgd(target_dir, load_dataloader, eta, batch_size, REG_LAMBDA, EPOCHS, device, seed, logger)
    else:
        run_variance_reduced(
            method, target_dir, load_dataloader, eta, batch_size, REG_LAMBDA, EPOCHS, device, seed, logger
        )

    logger.save(os.path.join(target_dir, "log.json"))

    train_dataloader, test_dataloader = load_dataloader(seed=seed, batch_size=batch_size)
    config = {
        "experiment": EXPERIMENT_NAME,
        "method": method,
        "seed": seed,
        "learning_rate": eta,
        "batch_size": batch_size,
        "reg_lambda": REG_LAMBDA,
        "epochs": EPOCHS,
        "K": len(train_dataloader),
        "N_train": len(train_dataloader.dataset),
        "N_test": len(test_dataloader.dataset),
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{hp_name}/{seed}", flush=True)


def main():
    """
    概要: 実験2の全条件（4手法 × 3バッチサイズ × 3学習率 × 5Seed = 180条件）をマルチ
        プロセスで並列に学習する．GPU使用率・VRAM使用量に余裕があること（モジュール
        docstringのEPOCHS算出根拠を参照）を踏まえ，並列プロセス数は8とする．
    引数: なし
    戻り値: なし
    """
    print(
        f"バッチサイズ = {BATCH_SIZES}, 学習率 = {LEARNING_RATES}, "
        f"lambda = {REG_LAMBDA}, epochs = {EPOCHS}"
    )

    tasks = [
        (method, batch_size, eta, seed)
        for method, batch_size, eta, seed in itertools.product(
            METHODS, BATCH_SIZES, LEARNING_RATES, SEEDS
        )
    ]

    num_workers = min(8, len(tasks))
    print(f"並列プロセス数: {num_workers}（総タスク数: {len(tasks)}）")

    # タスクごとの実行時間がバッチサイズにより大きく異なる（bs=8とbs=128で6〜8倍の差，
    # モジュールdocstring参照）ため，`Pool.map` の既定のチャンクサイズ（タスクを連続した
    # 大きな塊で各ワーカーに静的に割り当てる）では負荷分散が偏り，実行時間の長いタスクが
    # 集中したワーカーだけが後半まで動き続ける事態になりうる．`chunksize=1` を指定し，
    # ワーカーが1タスク完了するたびに次のタスクを動的に取得できるようにする．
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks, chunksize=1)

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    main()
