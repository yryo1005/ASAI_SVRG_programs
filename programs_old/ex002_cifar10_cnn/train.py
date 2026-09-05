"""
論文4.2節（CNNとCIFAR-10の多値分類問題）の数値実験を実行するスクリプト．

比較手法はSGD，SVRG，NFG SVRG，ASAI SVRG（提案手法）の4手法．各手法についてSeed値を5種類
（0〜4）で検証する．本実験は理論解析の仮定（平滑性・強凸性）を満たさない非凸な設定であるため，
- リプシッツ定数からの学習率の導出は行わず，学習率は適当な値を定める．
- 4.1節のオンライン学習（ミニバッチサイズ1）とは異なり，実際のユースケースを想定した
  適当なミニバッチサイズの下でミニバッチ学習を行う．
- 目的関数の真の最適値 f(w*) は非凸性のため求まらないため，目的関数の誤差 f(z_s) - f(w*) の
  代わりに，目的関数の値 f(z_s) 自体を記録する．

`@.ai/ai-dev-kit/root_prompt.md`／`@.ai/ai-dev-kit/machine_learning.md` が定める
loss_func / metrics_func / iteration / epoch / train の関数構成に従う．ただし，SVRG系手法
（SVRG，NFG SVRG，ASAI SVRG）は現在のパラメータ w_s^k とスナップショット z_s の双方における
勾配を必要とするため，`iteration` 関数はスナップショット専用モデル `snapshot_model`
（`model` と同一構造の別インスタンス）を任意引数として受け取れるよう拡張している．この点のみが
標準テンプレートからの変更点であり，それ以外（`loss_func`，`metrics_func`，`epoch` の
呼び出し方，`ResultLogger`，出力ディレクトリ規則等）は標準テンプレートに従う．
"""

import itertools
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_PROGRAMS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from machine_learning_utils import ResultLogger, set_seed  # noqa: E402
from optimizers.optimizers import ASAISVRG, NFGSVRG, SGD, SVRG  # noqa: E402

from data import load_dataloader  # noqa: E402
from model import CNNModel, load_model, set_model_params  # noqa: E402

EXPERIMENT_NAME = "ex002_cifar10_cnn"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs_old", EXPERIMENT_NAME)

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
BATCH_SIZE = 128
EPOCHS = 120
# 非凸設定のためリプシッツ定数は陽に求まらない．学習率は，全手法・全Seedで共通の値を用いる
# という制約（論文Algorithm 4の統一ワークフロー，手法間の相違はg_s/z_sの構成のみ）の下で，
# 事前の予備実験（lr=0.01, 0.005ではNFG SVRGが訓練の途中で発散した）に基づき，4手法すべてが
# 安定して学習できる値として lr=0.001 を選定した．詳細は .reports/report_004.md を参照．
LEARNING_RATE = 0.001

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "SVRG": SVRG,
    "NFG_SVRG": NFGSVRG,
    "ASAI_SVRG": ASAISVRG,
}


def loss_func(outputs: torch.Tensor, teacher_signals: torch.Tensor) -> torch.Tensor:
    """
    概要: モデルの出力と教師信号の誤差を計算する関数．論文(39)式の多値交差エントロピー損失．
    引数:
        outputs (torch.Tensor)，形状 (B, C)．モデルの出力（ロジット）．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号（クラスインデックス）．
    戻り値: loss (torch.Tensor)，形状 ()．1データあたりの誤差の平均値．
    """
    return nn.functional.cross_entropy(outputs, teacher_signals, reduction="mean")


def metrics_func(outputs: torch.Tensor, teacher_signals: torch.Tensor) -> dict:
    """
    概要: モデルの出力と教師信号の一致度（分類精度）を評価する関数．
    引数:
        outputs (torch.Tensor)，形状 (B, C)．モデルの出力（ロジット）．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号（クラスインデックス）．
    戻り値: metrics_to_value (dict)．{"accuracy": 1データあたりの正解率の平均値}．
    """
    with torch.no_grad():
        pred = outputs.argmax(dim=1)
        accuracy = (pred == teacher_signals).to(torch.float32).mean().item()
    return {"accuracy": accuracy}


def iteration(model, inputs, teacher_signals, optimizer=None, snapshot_model=None) -> dict:
    """
    概要: 1つのミニバッチのデータを学習/検証する関数．
        `optimizer` がSVRG系手法（`snapshot_model` が指定される場合）は，同一ミニバッチに
        対して `model`（現在のパラメータ w_s^k）と `snapshot_model`（スナップショット z_s）の
        双方でforward／backwardを実行し，2種類の勾配（`.grad`／`snapshot_model`から抽出した
        勾配列）を用いて `optimizer.step(grad_at_snapshot)` を呼び出す．これが標準の
        iteration関数から拡張している唯一の点である．
    引数:
        model (torch.nn.Module)．現在のパラメータを保持するモデル．
        inputs (torch.Tensor)，形状 (B, 3, 32, 32)．入力画像．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号．
        optimizer (torch.optim.Optimizer) = None．
        snapshot_model (torch.nn.Module) = None．SVRG系手法のスナップショットを保持するモデル．
    戻り値: metrics_to_value (dict)．{"loss": ..., "accuracy": ...}．
    """
    if optimizer is None:
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_func(outputs, teacher_signals)
    else:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, teacher_signals)
        loss.backward()

        if snapshot_model is not None:
            snapshot_model.zero_grad()
            snapshot_outputs = snapshot_model(inputs)
            snapshot_loss = loss_func(snapshot_outputs, teacher_signals)
            snapshot_loss.backward()
            grad_at_snapshot = [p.grad.detach().clone() for p in snapshot_model.parameters()]
            optimizer.step(grad_at_snapshot)
        else:
            optimizer.step()

    metrics = metrics_func(outputs, teacher_signals)
    metrics["loss"] = loss.item()
    return metrics


def epoch(model, dataloader, device, optimizer=None, snapshot_model=None) -> dict:
    """
    概要: 1つのデータローダーの全データを学習/検証する関数．
    引数:
        model (torch.nn.Module)．
        dataloader (torch.utils.data.DataLoader)．
        device (torch.device)．
        optimizer (torch.optim.Optimizer) = None．
        snapshot_model (torch.nn.Module) = None．SVRG系手法のスナップショットを保持するモデル．
    戻り値: metrics_to_value (dict)．1データあたりの評価・誤差の平均値の辞書．
    """
    total_metrics = {}
    total_count = 0
    for inputs, teacher_signals in tqdm(dataloader, leave=False):
        inputs = inputs.to(device)
        teacher_signals = teacher_signals.to(device)

        batch_metrics = iteration(model, inputs, teacher_signals, optimizer, snapshot_model)

        batch_size = inputs.shape[0]
        for key, value in batch_metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value * batch_size
        total_count += batch_size

    return {key: value / total_count for key, value in total_metrics.items()}


def compute_full_gradient_and_metrics(model, dataloader, device):
    """
    概要: データローダー全体に対するフル勾配，および同じ1回の走査で得られる
        誤差・分類精度の平均値をまとめて計算する．勾配計算と評価を別々に行うと
        データセット全体を2回走査することになるため，1回の走査で両方を得ることで
        計算コストを削減する．
    引数:
        model (torch.nn.Module)．勾配・評価の対象となるパラメータを保持するモデル．
        dataloader (torch.utils.data.DataLoader)．データセット全体を走査するデータローダー．
        device (torch.device)．
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
        loss = loss_func(outputs, teacher_signals)
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
    概要: フル勾配の近似誤差 ||e_s||^2 = ||g_s - ∇f(z_s)||^2 を計算する．
    引数:
        snapshot_gradient (list of torch.Tensor)．スナップショット勾配 g_s．
        true_full_gradient (list of torch.Tensor)．真のフル勾配 ∇f(z_s)．
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


def train_sgd(target_dir, ModelClass, load_dataloader_func, epochs, batch_size, device, seed, logger):
    """
    概要: SGDによる学習を実行し，各エポック終了時の評価指標を `logger` に記録する．
        検証用データに対する分類精度が過去最高を更新した場合，モデルの重みを保存する．
    引数:
        target_dir (str)．結果保存先ディレクトリ．
        ModelClass (torch.nn.Moduleのクラス)．
        load_dataloader_func (func)．
        epochs (int)．
        batch_size (int)．
        device (torch.device)．
        seed (int)．乱数シード．
        logger (ResultLogger)．
    戻り値: なし
    """
    train_dataloader, test_dataloader = load_dataloader_func(seed=seed, batch_size=batch_size)
    N_train = len(train_dataloader.dataset)

    model = load_model(ModelClass, seed=seed).to(device)
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    oracle_calls = 0
    best_accuracy = -1.0
    start_time = time.time()

    with torch.no_grad():
        train_metrics = epoch(model, train_dataloader, device, optimizer=None)
        test_metrics = epoch(model, test_dataloader, device, optimizer=None)
    logger(
        0, oracle_calls, 0.0,
        train_metrics["loss"], test_metrics["accuracy"], float("nan"),
    )
    best_accuracy = _save_if_best(model, test_metrics["accuracy"], best_accuracy, target_dir)

    for epoch_index in tqdm(range(1, epochs + 1), desc=f"SGD seed={seed}", leave=False):
        train_metrics = epoch(model, train_dataloader, device, optimizer=optimizer)
        oracle_calls += N_train

        with torch.no_grad():
            test_metrics = epoch(model, test_dataloader, device, optimizer=None)
        elapsed_time = time.time() - start_time

        logger(
            epoch_index, oracle_calls, elapsed_time,
            train_metrics["loss"], test_metrics["accuracy"], float("nan"),
        )
        best_accuracy = _save_if_best(model, test_metrics["accuracy"], best_accuracy, target_dir)


def train_variance_reduced(
    method, target_dir, ModelClass, load_dataloader_func, epochs, batch_size, device, seed, logger
):
    """
    概要: SVRG系手法（SVRG，NFG SVRG，ASAI SVRG）による学習を実行し，各エポック終了時の
        評価指標を `logger` に記録する．論文Algorithm 1〜3の外部ループ・内部ループ構造を，
        `epoch` 関数を内部ループ（1エポック分のミニバッチ列）として用いることで実装する．
    引数:
        method (str)．"SVRG"，"NFG_SVRG"，"ASAI_SVRG" のいずれか．
        target_dir (str)．結果保存先ディレクトリ．
        ModelClass (torch.nn.Moduleのクラス)．
        load_dataloader_func (func)．
        epochs (int)．外部ループ数 S．
        batch_size (int)．ミニバッチサイズ．
        device (torch.device)．
        seed (int)．乱数シード．
        logger (ResultLogger)．
    戻り値: なし
    """
    assert method in _VARIANCE_REDUCED_OPTIMIZER_CLASSES

    train_dataloader, test_dataloader = load_dataloader_func(seed=seed, batch_size=batch_size)
    N_train = len(train_dataloader.dataset)
    K = len(train_dataloader)  # 1エポックあたりのミニバッチ数

    model = load_model(ModelClass, seed=seed).to(device)
    # z_0 <- w_0．snapshot_modelはmodelと同じseedで初期化することで同一の初期値を持つ．
    snapshot_model = load_model(ModelClass, seed=seed).to(device)

    OptimizerClass = _VARIANCE_REDUCED_OPTIMIZER_CLASSES[method]
    optimizer = OptimizerClass(model.parameters(), lr=LEARNING_RATE, K=K)
    rng = np.random.default_rng(seed)

    oracle_calls = 0

    if method == "SVRG":
        # g_0: SVRGは真のフル勾配（論文Algorithm 1）．この計算コストは，epoch 0（学習開始前の
        # 初期状態）ではなく，epoch 1の学習に要するオラクル呼び出し回数として計上する．
        snapshot_grad, train_metrics_0 = compute_full_gradient_and_metrics(
            snapshot_model, train_dataloader, device
        )
        true_full_grad_0 = snapshot_grad
    else:
        # g_0 = 0（論文Algorithm 2, 3）．Optimizer初期化時に既に設定済み．
        snapshot_grad = optimizer.get_snapshot_gradient()
        true_full_grad_0, train_metrics_0 = compute_full_gradient_and_metrics(
            snapshot_model, train_dataloader, device
        )

    approx_error_0 = compute_approx_error(snapshot_grad, true_full_grad_0)

    best_accuracy = -1.0
    start_time = time.time()
    with torch.no_grad():
        test_metrics = epoch(snapshot_model, test_dataloader, device, optimizer=None)
    logger(
        0, oracle_calls, 0.0,
        train_metrics_0["loss"], test_metrics["accuracy"], approx_error_0,
    )
    best_accuracy = _save_if_best(snapshot_model, test_metrics["accuracy"], best_accuracy, target_dir)

    if method == "SVRG":
        optimizer.set_snapshot_gradient(snapshot_grad)
        oracle_calls += N_train

    for epoch_index in tqdm(range(1, epochs + 1), desc=f"{method} seed={seed}", leave=False):
        optimizer.begin_epoch(rng)
        epoch(model, train_dataloader, device, optimizer=optimizer, snapshot_model=snapshot_model)
        oracle_calls += 2 * N_train

        optimizer.end_epoch()
        snapshot_params = optimizer.get_snapshot_params()
        set_model_params(snapshot_model, snapshot_params, source_model=model)

        if method == "SVRG":
            snapshot_grad, train_metrics = compute_full_gradient_and_metrics(
                snapshot_model, train_dataloader, device
            )
            oracle_calls += N_train
            optimizer.set_snapshot_gradient(snapshot_grad)
            approx_error = 0.0
        else:
            snapshot_grad = optimizer.get_snapshot_gradient()
            true_full_grad, train_metrics = compute_full_gradient_and_metrics(
                snapshot_model, train_dataloader, device
            )
            approx_error = compute_approx_error(snapshot_grad, true_full_grad)

        with torch.no_grad():
            test_metrics = epoch(snapshot_model, test_dataloader, device, optimizer=None)
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
    概要: 1つの (手法, Seed) の組に対する学習を実行し，結果を保存する．
        すでに正常終了した結果が存在する場合は学習をスキップする．
    引数: args (tuple)．(method, seed) のタプル．
    戻り値: なし
    """
    method, seed = args

    hp_name = f"lr{LEARNING_RATE}_bs{BATCH_SIZE}_epochs{EPOCHS}"
    target_dir = os.path.join(OUTPUT_ROOT, method, hp_name, str(seed))
    if is_run_completed(target_dir, EPOCHS):
        print(f"[skip] {method}/{hp_name}/{seed} は既に完了しています．")
        return

    os.makedirs(target_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    if method == "SGD":
        train_sgd(target_dir, CNNModel, load_dataloader, EPOCHS, BATCH_SIZE, device, seed, logger)
    else:
        train_variance_reduced(
            method, target_dir, CNNModel, load_dataloader, EPOCHS, BATCH_SIZE, device, seed, logger
        )

    logger.save(os.path.join(target_dir, "log.json"))

    train_dataloader, test_dataloader = load_dataloader(seed=seed, batch_size=BATCH_SIZE)
    config = {
        "method": method,
        "seed": seed,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "N_train": len(train_dataloader.dataset),
        "N_test": len(test_dataloader.dataset),
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{hp_name}/{seed}")


def main():
    """
    概要: 実験4.2（CNNとCIFAR-10の多値分類問題）の全条件（4手法 x 5Seed）を
        マルチプロセスで並列に学習する．GPU（VRAM）の使用量を考慮し，並列プロセス数は
        4とする．
    引数: なし
    戻り値: なし
    """
    print(f"学習率 = {LEARNING_RATE}, ミニバッチサイズ = {BATCH_SIZE}, エポック数 = {EPOCHS}")

    tasks = [(method, seed) for method, seed in itertools.product(METHODS, SEEDS)]

    num_workers = min(4, len(tasks))
    print(f"並列プロセス数: {num_workers} (総タスク数: {len(tasks)})")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks)

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    main()
