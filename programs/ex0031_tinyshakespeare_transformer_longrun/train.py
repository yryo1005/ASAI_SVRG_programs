"""
実験3 Stage C（長期学習によるASAI SVRGの効率性優位性の持続性検証）の学習ループを定義し，
実験を実行するスクリプト．

`.orders/order_029.md` に対応する．Stage B（`.reports/report_028.md`）は，バッチサイズ
512, 128, 32・学習率0.01, 0.001・12エポックのグリッドで，SGD・SVRG・NFG SVRG・ASAI SVRGの
4手法を比較し，オラクル呼び出し回数を揃えた場合にASAI SVRGが全6条件で一貫してSVRGを上回る
ことを示した．しかし，(a) いずれの条件でも精度がまだプラトーに達しておらず学習途中の値で
あったこと，(b) ex0023（`.reports/report_026.md`）で確認された「短いエポック数での安定判定
は崩壊直前で観測を止めていただけの可能性がある」という教訓，から，Stage Bの結果だけでは
効率性優位性の持続性・安定性の頑健性を十分に検証できていなかった．

本Stage Cは，`.orders/order_029.md` 3節の指示に基づき学習率を0.01に限定し，エポック数を
Stage Bの4倍（全バッチサイズ一律48エポック）に延長する．オーダーの指示（末尾「実験は
Epoch数を区別しやすいように ex0031 として実施してください」）に従い，本実験は
`ex0031_tinyshakespeare_transformer_longrun` として実装する．

## 既存結果を再利用しない理由

`.orders/order_029.md` 2節の指示に基づき，Stage Bの12エポック分の結果からの継続学習は
行わない．NFG SVRG・ASAI SVRGのOptimizer内部状態（running average・スナップショット等）は
ディスクに保存されていないため学習を再開できないという制約は，ex0023（`.reports/
report_026.md` 2.2節）と同様である．したがって本実験は全36条件（4手法×3バッチサイズ×
3Seed）をゼロから再学習する．

## 比較手法とelapsed_timeの計上方法

Stage A・Stage Bと同一の実装（`SGD`，`SVRG`／`NFGSVRG`，次のスナップショット点を内部ループの
パラメータ列からランダムに選ぶ，ASAI SVRG論文Algorithm 4の理論解析に整合する版）をそのまま
用いる．`elapsed_time`の計上方法（NFG SVRG・ASAI SVRGの診断専用フル勾配計算を除外する，
`.reports/report_025.md` 2.2節の修正）も踏襲する．

## 崩壊の検出とプラトー判定

`is_stuck_near_chance` 関数（Stage Aで導入）で，次文字予測精度がチャンスレベルに張り付く
「見えない崩壊」を検出する．`compute_trailing_relative_change` 関数（ex0023で導入，
`.reports/report_026.md` 2.4節）で，末尾エポックの相対変化からプラトー到達を数値的に判定
する．訓練損失が非有限値化した場合は学習を打ち切る（ex0023と同様）．
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

from data import build_vocabulary, load_dataloader, _download_raw_text  # noqa: E402
from model import (  # noqa: E402
    DecoderOnlyTransformer,
    load_model,
    loss_func,
    metrics_func,
    set_model_params,
)

EXPERIMENT_NAME = "ex0031_tinyshakespeare_transformer_longrun"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

SEEDS = [0, 1, 2]
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
BATCH_SIZES = [512, 128, 32]
LEARNING_RATE = 0.01  # `.orders/order_029.md` 3節の指示により固定（0.001は対象外）
REG_LAMBDA = 5e-4
EPOCHS = 48  # Stage B（12エポック）の4倍．全バッチサイズ一律

_VOCAB_SIZE = len(build_vocabulary(_download_raw_text())[1])

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
    return f"lr{LEARNING_RATE}_bs{batch_size}_lambda{REG_LAMBDA}_epochs{epochs}"


def is_stuck_near_chance(accuracies, vocab_size: int, window: int = 3, tolerance: float = 0.02) -> bool:
    """
    概要: 末尾 `window` エポックの次文字予測精度がすべてチャンスレベル
        （$ 1/\\text{vocab\\_size} $）付近に留まっているかを判定する．NaNにはならないが
        モデルが実質的に機能不全に陥る「見えない崩壊」の検出に用いる．
    引数:
        accuracies (Sequence[float])．エポックごとの次文字予測精度の列．
        vocab_size (int)．語彙サイズ．
        window (int) = 3．判定に用いる末尾のエポック数．
        tolerance (float) = 0.02．チャンスレベルからの許容乖離幅．
    戻り値: stuck (bool)．末尾`window`エポックの精度が全て
        `[1/vocab_size - tolerance, 1/vocab_size + tolerance]` に収まっていれば `True`．
        `window` 未満のエポック数しかない場合は `False`．
    """
    if len(accuracies) < window:
        return False
    chance = 1.0 / vocab_size
    tail = accuracies[-window:]
    return all(abs(a - chance) <= tolerance for a in tail)


def compute_trailing_relative_change(values, window: int) -> float:
    """
    概要: 数値列の末尾 `window` 個について，連続する値の間の相対変化
        $ |v^{(t)}-v^{(t-1)}|/|v^{(t-1)}| $ の最大値を計算する．プラトー（変化が十分小さく
        なった状態）に達したかどうかを数値的に判定するために用いる
        （`.orders/order_029.md` 4節3項，`.reports/report_026.md` 2.4節の方法論を踏襲）．
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
        model (torch.nn.Module)．
        inputs (torch.Tensor)，形状 (B, T)．入力トークン列．
        teacher_signals (torch.Tensor)，形状 (B, T)．教師トークン列．
        reg_lambda (float)．L2正則化係数．
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


def train_epoch(model, dataloader, device, reg_lambda, optimizer, snapshot_model=None, desc="") -> None:
    """
    概要: 1エポック分の学習（内部ループ）を実行する．
    引数: `iteration` 参照．desc (str)．tqdmの進捗バーの説明文．
    戻り値: なし
    """
    for inputs, teacher_signals in tqdm(dataloader, desc=desc, leave=False):
        inputs = inputs.to(device)
        teacher_signals = teacher_signals.to(device)
        iteration(model, inputs, teacher_signals, reg_lambda, optimizer, snapshot_model)


def evaluate_epoch(model, dataloader, device, reg_lambda) -> dict:
    """
    概要: 1つのデータローダーの全データを（学習せず）評価する関数．
    引数: model, dataloader, device, reg_lambda．
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
    概要: データローダー全体に対するフル勾配，および同じ1回の走査で得られる誤差・次文字
        予測精度の平均値をまとめて計算する．
    引数: model, dataloader, device, reg_lambda．
    戻り値:
        grads (list of torch.Tensor)．
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
    引数: snapshot_gradient, true_full_gradient (list of torch.Tensor)．
    戻り値: approx_error (float)．
    """
    return sum(
        torch.sum((g_s - g_true) ** 2).item()
        for g_s, g_true in zip(snapshot_gradient, true_full_gradient)
    )


def _save_if_best(model, test_accuracy, best_accuracy, target_dir):
    """
    概要: 検証用データに対する次文字予測精度が過去最高を更新した場合，モデルの重みを保存する．
    引数: model, test_accuracy, best_accuracy, target_dir．
    戻り値: best_accuracy (float)．
    """
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))
        return test_accuracy
    return best_accuracy


def run_sgd(target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, seed, logger):
    """
    概要: SGDによる学習を実行し，各エポック終了時の評価指標を `logger` に記録する．
    引数: target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, seed, logger．
    戻り値: なし
    """
    train_dataloader, test_dataloader = load_dataloader_func(seed=seed, batch_size=batch_size)
    N_train = len(train_dataloader.dataset)

    model = load_model(DecoderOnlyTransformer, seed=seed, vocab_size=_VOCAB_SIZE).to(device)
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

        if not np.isfinite(train_metrics["loss"]):
            break


def run_variance_reduced(
    method, target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, seed, logger
):
    """
    概要: SVRG系手法（SVRG，NFG SVRG，ASAI SVRG）による学習を実行し，各エポック終了時の
        評価指標を `logger` に記録する．SVRGのみ，スナップショット勾配 $ g_s $ が真のフル
        勾配であり，これを次エポックの学習に用いるため，その計算時間は `elapsed_time` に
        計上する．NFG SVRG・ASAI SVRGは評価指標（近似誤差）算出のためだけに真のフル勾配を
        計算するため，この計算時間は `elapsed_time` から除外する．
    引数: method, target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device,
        seed, logger．
    戻り値: なし
    """
    assert method in _VARIANCE_REDUCED_OPTIMIZER_CLASSES

    train_dataloader, test_dataloader = load_dataloader_func(seed=seed, batch_size=batch_size)
    N_train = len(train_dataloader.dataset)
    K = len(train_dataloader)

    model = load_model(DecoderOnlyTransformer, seed=seed, vocab_size=_VOCAB_SIZE).to(device)
    snapshot_model = load_model(DecoderOnlyTransformer, seed=seed, vocab_size=_VOCAB_SIZE).to(device)

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
            break


def is_run_completed(target_dir: str, epochs: int) -> bool:
    """
    概要: 指定した条件の学習が既に正常終了しているか確認する．崩壊により打ち切った条件も
        完了済みとして扱う．
    引数: target_dir (str)．epochs (int)．
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
        train_loss = logger["train_loss"]
        return len(train_loss) > 0 and not np.isfinite(train_loss[-1])
    except (json.JSONDecodeError, OSError):
        return False


def run_single_experiment(args):
    """
    概要: 1つの (手法, バッチサイズ, Seed) の組に対する学習を実行し，結果を保存する．
    引数: args (tuple)．(method, batch_size, seed) のタプル．
    戻り値: なし
    """
    method, batch_size, seed = args
    torch.set_num_threads(1)

    epochs = EPOCHS
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
        "reg_lambda": REG_LAMBDA,
        "epochs": epochs,
        "K": len(train_dataloader),
        "total_iterations": epochs * len(train_dataloader),
        "N_train": len(train_dataloader.dataset),
        "N_test": len(test_dataloader.dataset),
        "vocab_size": _VOCAB_SIZE,
        "collapsed": len(logger["train_loss"]) < epochs + 1,
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{name}/{seed}", flush=True)


def main():
    """
    概要: 実験3 Stage Cの全条件（4手法 × 3バッチサイズ × 3Seed = 36条件）をマルチプロセスで
        並列に学習する．学習率0.01固定，`.orders/order_029.md` 2節の指示により全条件を
        ゼロから再学習する（既存結果のコピー再利用は行わない）．
    引数: なし
    戻り値: なし
    """
    print(f"バッチサイズ = {BATCH_SIZES}, 学習率 = {LEARNING_RATE}, epochs = {EPOCHS}, vocab_size = {_VOCAB_SIZE}")

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
