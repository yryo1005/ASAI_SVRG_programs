"""
実験4（WikiText-2を用いた単語レベル言語モデリング，Decoder-only Transformer）Stage A
（安定性探索）の学習ループを定義し，実験を実行するスクリプト．

`.orders/order_030.md` に対応する．実験3（Tiny Shakespeare，`ex003`／`ex0031`系列）は，
オラクル呼び出し回数を揃えた場合にASAI SVRGがSVRGを一貫して上回ることを示した一方，長期学習
（`.reports/report_029.md`）の結果，全手法の最終精度がほぼ同水準に収束するという結果が判明
した．文字レベル・小規模コーパスというタスク難度の低さがこの原因である可能性を検証するため，
本実験4は単語レベル言語モデリング（WikiText-2，語彙サイズ約33,000）へ切り替える．

実験2・実験3と同様の教訓（正規化層なしのモデルでいきなり本比較を実施し発散した経験）を
踏まえ，本実験もStage A（安定性探索，NFG SVRG・ASAI SVRGの2手法のみ，短いエポック数）から
段階的に実施する．

## 比較手法（`.orders/order_030.md` 6節）

実験2以降で用いてきた `SVRG`／`NFGSVRG`（次のスナップショット点を内部ループのパラメータ
列からランダムに選ぶ版）と同じOptimizerクラス（`NFGSVRG`，`ASAISVRG`）をそのまま用いる．

## 崩壊の検出（`.orders/order_030.md` 7.1節）

- NaN発散：訓練損失の非有限値化を監視する．
- チャンスレベル張り付き：`is_stuck_near_chance` 関数で判定する．ただし，本実験の語彙サイズ
  （約33,000）は実験3（65）よりはるかに大きく，チャンスレベル（$ 1/|\\text{vocab}| $）自体が
  極めて小さい値（約0.00003）となるため，実験3の許容幅（0.02，チャンスレベルの約1.3倍）を
  そのまま流用すると，実際には学習が進んでいる（数%の精度に達している）条件まで許容範囲に
  含んでしまい，判定として機能しない．本実験ではチャンスレベルの約33倍に相当する0.001
  （0.1ポイント）を許容幅として用いる．なお，WikiText-2は単語頻度がZipf分布に従うため，
  最頻出単語（"the"，学習用テキストの約5.5%を占める）を常に出力するだけでもチャンスレベル
  よりはるかに高い精度（約5.5〜5.9%）が得られる．これは一様分布を仮定するチャンスレベルとは
  異なる「頻度バイアスへの縮退」という別の失敗モードであり，`is_stuck_near_chance`
  （一様チャンスレベル基準）では検出できない点に注意が必要である．このため，本実験では
  自動判定に加えて学習曲線を目視でも確認する．
- 近似誤差の急増：$ \\|e_s\\|^2 $ の推移を確認する．
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

from data import build_vocabulary, load_dataloader, _download_raw_text, SEQUENCE_LENGTH  # noqa: E402
from model import (  # noqa: E402
    DecoderOnlyTransformer,
    load_model,
    loss_func,
    metrics_func,
    set_model_params,
)

EXPERIMENT_NAME = "ex004_wikitext2_transformer"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

SEEDS = [0, 1, 2]
METHODS = ["NFG_SVRG", "ASAI_SVRG"]  # Stage A（order_030 7節）：SGD・SVRGは対象外
BATCH_SIZES = [512, 128, 32]
LEARNING_RATES = [0.01, 0.001]
REG_LAMBDA = 5e-4
EPOCHS = 12
NEAR_CHANCE_TOLERANCE = 0.001  # 本モジュールの docstring 参照（チャンスレベルの約33倍）

_VOCAB_SIZE = len(build_vocabulary(_download_raw_text("train"))[1])

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "NFG_SVRG": NFGSVRG,
    "ASAI_SVRG": ASAISVRG,
}


def hp_name(lr: float, batch_size: int, epochs: int) -> str:
    """
    概要: ハイパーパラメータ条件名（ディレクトリ名）を構築する．
    引数:
        lr (float)．学習率．
        batch_size (int)．ミニバッチサイズ．
        epochs (int)．エポック数．
    戻り値: name (str)．
    """
    return f"lr{lr}_bs{batch_size}_lambda{REG_LAMBDA}_epochs{epochs}"


def is_stuck_near_chance(accuracies, vocab_size: int, window: int = 3, tolerance: float = NEAR_CHANCE_TOLERANCE) -> bool:
    """
    概要: `.orders/order_030.md` 7.1節の指示に基づき，末尾 `window` エポックの次単語予測
        精度がすべてチャンスレベル（$ 1/\\text{vocab\\_size} $）付近に留まっているかを判定
        する．NaNにはならないがモデルが実質的に機能不全に陥る「見えない崩壊」の検出に用いる．
    引数:
        accuracies (Sequence[float])．エポックごとの次単語予測精度の列．
        vocab_size (int)．語彙サイズ．
        window (int) = 3．判定に用いる末尾のエポック数．
        tolerance (float) = 0.001．チャンスレベルからの許容乖離幅（本実験用に再検討した値，
            モジュールdocstring参照）．
    戻り値: stuck (bool)．末尾`window`エポックの精度が全て
        `[1/vocab_size - tolerance, 1/vocab_size + tolerance]` に収まっていれば `True`．
        `window` 未満のエポック数しかない場合は `False`．
    """
    if len(accuracies) < window:
        return False
    chance = 1.0 / vocab_size
    tail = accuracies[-window:]
    return all(abs(a - chance) <= tolerance for a in tail)


def iteration(model, inputs, teacher_signals, reg_lambda, optimizer, snapshot_model=None) -> dict:
    """
    概要: 1つのミニバッチのデータを学習する関数．`snapshot_model` が指定される場合（SVRG系
        手法），同一ミニバッチに対して `model`（現在のパラメータ $ w_s^k $）と
        `snapshot_model`（スナップショット $ z_s $）の双方でforward／backwardを実行し，
        2種類の勾配を用いて `optimizer.step(grad_at_snapshot)` を呼び出す．
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

    snapshot_model.zero_grad()
    snapshot_outputs = snapshot_model(inputs)
    snapshot_loss = loss_func(snapshot_outputs, teacher_signals, snapshot_model, reg_lambda)
    snapshot_loss.backward()
    grad_at_snapshot = [p.grad.detach().clone() for p in snapshot_model.parameters()]
    optimizer.step(grad_at_snapshot)

    metrics = metrics_func(outputs, teacher_signals)
    metrics["loss"] = loss.item()
    return metrics


def train_epoch(model, dataloader, device, reg_lambda, optimizer, snapshot_model, desc="") -> None:
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
    概要: データローダー全体に対するフル勾配，および同じ1回の走査で得られる誤差・次単語
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
    概要: 検証用データに対する次単語予測精度が過去最高を更新した場合，モデルの重みを保存する．
    引数: model, test_accuracy, best_accuracy, target_dir．
    戻り値: best_accuracy (float)．
    """
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))
        return test_accuracy
    return best_accuracy


def run_variance_reduced(
    method, target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, seed, logger
):
    """
    概要: SVRG系手法（NFG SVRG，ASAI SVRG）による学習を実行し，各エポック終了時の評価指標を
        `logger` に記録する．NFG SVRG・ASAI SVRGは評価指標（近似誤差）算出のためだけに真の
        フル勾配を計算するため，この計算時間は `elapsed_time` から除外する（`.reports/
        report_025.md` 2.2節の修正を踏襲）．
    引数:
        method (str)．"NFG_SVRG"，"ASAI_SVRG" のいずれか．
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
    K = len(train_dataloader)

    model = load_model(
        DecoderOnlyTransformer, seed=seed, vocab_size=_VOCAB_SIZE, max_seq_len=SEQUENCE_LENGTH
    ).to(device)
    snapshot_model = load_model(
        DecoderOnlyTransformer, seed=seed, vocab_size=_VOCAB_SIZE, max_seq_len=SEQUENCE_LENGTH
    ).to(device)

    OptimizerClass = _VARIANCE_REDUCED_OPTIMIZER_CLASSES[method]
    optimizer = OptimizerClass(model.parameters(), lr=eta, K=K)
    rng = np.random.default_rng(seed)

    oracle_calls = 0

    snapshot_grad = optimizer.get_snapshot_gradient()  # g_0 = 0
    true_full_grad_0, train_metrics_0 = compute_full_gradient_and_metrics(
        snapshot_model, train_dataloader, device, reg_lambda
    )
    approx_error_0 = compute_approx_error(snapshot_grad, true_full_grad_0)

    best_accuracy = -1.0
    test_metrics = evaluate_epoch(snapshot_model, test_dataloader, device, reg_lambda)
    logger(0, oracle_calls, 0.0, train_metrics_0["loss"], test_metrics["accuracy"], approx_error_0)
    best_accuracy = _save_if_best(snapshot_model, test_metrics["accuracy"], best_accuracy, target_dir)

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
    概要: 1つの (手法, バッチサイズ, 学習率, Seed) の組に対する学習を実行し，結果を保存する．
    引数: args (tuple)．(method, batch_size, eta, seed) のタプル．
    戻り値: なし
    """
    method, batch_size, eta, seed = args
    torch.set_num_threads(1)

    epochs = EPOCHS
    name = hp_name(eta, batch_size, epochs)
    target_dir = os.path.join(OUTPUT_ROOT, method, name, str(seed))
    if is_run_completed(target_dir, epochs):
        print(f"[skip] {method}/{name}/{seed} は既に完了しています．", flush=True)
        return

    os.makedirs(target_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    run_variance_reduced(
        method, target_dir, load_dataloader, eta, batch_size, REG_LAMBDA, epochs, device, seed, logger
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
        "epochs": epochs,
        "sequence_length": SEQUENCE_LENGTH,
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
    概要: 実験4 Stage Aの全条件（2手法 × 3バッチサイズ × 2学習率 × 3Seed = 36条件）を
        マルチプロセスで並列に学習する．
    引数: なし
    戻り値: なし
    """
    print(
        f"バッチサイズ = {BATCH_SIZES}, 学習率 = {LEARNING_RATES}, "
        f"lambda = {REG_LAMBDA}, epochs = {EPOCHS}, vocab_size = {_VOCAB_SIZE}, "
        f"sequence_length = {SEQUENCE_LENGTH}"
    )

    tasks = [
        (method, batch_size, eta, seed)
        for method, batch_size, eta, seed in itertools.product(
            METHODS, BATCH_SIZES, LEARNING_RATES, SEEDS
        )
    ]

    # 語彙サイズ（約33,000）に起因し，出力層のロジットテンソル（B×T×V）がバッチサイズ512の
    # 場合に約23.5GB（`torch.cuda.max_memory_allocated`実測，約27.8GB reserved）に達する．
    # 8並列（初期タスク順序ではバッチサイズ512の条件が6件連続して割り当てられる）を試したところ，
    # 実測を大幅に超えるVRAM要求（6×27.8GB≈167GB）によりGPU（WSL2のGPU仮想化層，dxg/vmbus）
    # が応答不能状態に陥る事象が発生した（詳細は`.reports/report_030.md`参照）．そのため，
    # 並列数1回の実測（root_prompt.md「並列化可能なプログラムは...」の指示）に基づき，
    # バッチサイズ512の条件が同時に2件実行されても安全な並列数（2×27.8GB≈55.6GB，
    # 総VRAM約102.6GBの半分程度）に制限する．
    num_workers = min(2, len(tasks))
    print(f"並列プロセス数: {num_workers}（総タスク数: {len(tasks)}）")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks, chunksize=1)

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    main()
