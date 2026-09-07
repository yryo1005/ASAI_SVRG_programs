"""
実験ex0022（GroupNorm下での4手法比較，分散削減の効率性検証）の学習ループを定義し，
実験を実行するスクリプト．

`.orders/order_025.md` に対応する．実験2（`report_022.md`）・ex0021（`report_023.md`，
`report_024.md`）は，GroupNormの導入によりNFG SVRG・ASAI SVRGの発散が解消され，ASAI SVRGが
NFG SVRGに対して大きな優位性を示すことを確認したが，SGD・SVRGを含めた4手法比較（＝実験2
終了基準4「分散削減の利点」の検証）は未実施のまま残っていた．本実験は，GroupNorm固定のもと，
SGD・SVRG・NFG SVRG・ASAI SVRGの4手法をバッチサイズ512, 128, 32・学習率0.01, 0.001の
グリッドで比較する．

## 比較手法とOptimizerクラスの選択（`.orders/order_025.md` 4節との対応関係）

`.orders/order_025.md` 4節は比較手法として「SVRG（`SVRGFinalPoint`）」「NFG SVRG
（`NFGSVRGFinalPoint`）」と記載しているが，本実験が複製元とする実験2（`ex002_cifar10_alexnet`）・
ex0021（`ex0021_cifar10_alexnet_norm`）は，いずれも実際には `SVRGFinalPoint`／`NFGSVRGFinalPoint`
（NFG SVRG原論文Algorithm 1に忠実な，内部ループの最終パラメータを次のスナップショットとする版）
ではなく，`SVRG`／`NFGSVRG`（ASAI SVRG論文Algorithm 4の理論解析に合わせ，次のスナップショット点
$ z_{s+1} $ を内部ループのパラメータ列から一様ランダムに選ぶ版）を使用していた．これは
`report_022.md`・`report_023.md`・`report_024.md`の記述と実装の間の食い違いであり，本実験の
実装着手前に発見・報告した．ex0021の結果を3節の方針でコピーにより再利用する都合上，本実験でも
既存2実験と同じ `SVRG`／`NFGSVRG`（ランダム点採用版）に統一する．

## 実行時間（elapsed_time）の計上方法の修正

実験2・ex0021の実装は，NFG SVRG・ASAI SVRGについても評価指標（近似誤差）算出のためだけに
毎エポック計算する真のフル勾配（`compute_full_gradient_and_metrics`）の計算時間を，SVRGの
真のフル勾配計算時間（アルゴリズム自体に必要な計算）と区別せず `elapsed_time` に計上して
いた．これは `.orders/order_022.md` 7節「ASAI SVRG・NFG SVRGでは評価指標算出のためだけに
計算するフル勾配のコストを実行時間に含めない」という指示と一致しない．この食い違いも実装
着手前に発見・報告した実験2・ex0021のバグである．いずれの実験の報告（`report_022.md`〜
`report_024.md`）も `elapsed_time` を軸とした考察は行っておらず（オラクル呼び出し回数・
エポック数を評価軸として用いている），過去の結論への影響はない．本実験ではこの点を修正し，
`elapsed_time` の累積対象からNFG SVRG・ASAI SVRGの診断専用フル勾配計算を除外する．ただし
3節でコピーにより再利用する既存20条件の `elapsed_time` は，再学習をせずそのまま流用するため，
この修正は反映されない（該当条件のログにはこの限界がある旨を`.reports/report_025.md`に明記
する）．

## 既存結果の再利用（`.orders/order_025.md` 3節）

`outputs/ex0021_cifar10_alexnet_norm/` と `outputs/ex0022_cifar10_alexnet_groupnorm/` の
ハイパーパラメータ条件名（ディレクトリ名）は，学習率0.001・GroupNorm・バッチサイズ512, 128の
セルにおいて完全に一致するよう設計しており（`hp_name` 参照），該当4セル（NFG SVRG・ASAI SVRG
×バッチサイズ512, 128，各5Seed，計20条件）はディレクトリを直接コピーして再利用する
（`reuse_ex0021_results` 関数）．コピー前に，`tests/test_ex0022_cifar10_alexnet_groupnorm.py`
でモデルの初期値・データローダーの初期順序がex0021とex0022で完全に一致することを検証している．
"""

import itertools
import json
import multiprocessing
import os
import shutil
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

EXPERIMENT_NAME = "ex0022_cifar10_alexnet_groupnorm"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)
_EX0021_OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", "ex0021_cifar10_alexnet_norm")

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
BATCH_SIZES = [512, 128, 32]
LEARNING_RATES = [0.01, 0.001]
NORM_TYPE = "groupnorm"  # `.orders/order_025.md` 4節の指示により固定
REG_LAMBDA = 5e-4  # 実験2・ex0021と同一

# バッチサイズごとのエポック数．total_iterations = epochs * K がおおむね一定（約4700）になる
# よう決定（`.orders/order_025.md` 5節）．バッチサイズ512, 128はex0021と同一のため，
# 該当条件の結果を再利用できる．
EPOCHS_BY_BATCH_SIZE = {512: 48, 128: 12, 32: 3}

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "SVRG": SVRG,
    "NFG_SVRG": NFGSVRG,
    "ASAI_SVRG": ASAISVRG,
}

# ex0021からコピーにより再利用する条件（`.orders/order_025.md` 3節）．
# 学習率0.001・GroupNorm・バッチサイズ512, 128のNFG SVRG・ASAI SVRG，計4セル．
_REUSE_LEARNING_RATE = 0.001
_REUSE_CELLS = [
    (method, batch_size)
    for method in ["NFG_SVRG", "ASAI_SVRG"]
    for batch_size in [512, 128]
]


def hp_name(lr: float, batch_size: int, epochs: int) -> str:
    """
    概要: ハイパーパラメータ条件名（ディレクトリ名）を構築する．学習率0.001・GroupNorm・
        バッチサイズ512, 128の場合，`ex0021_cifar10_alexnet_norm` と完全に同一の文字列になる
        よう設計しており，これにより3節のコピーによる再利用が単純なディレクトリコピーで
        実現できる．
    引数:
        lr (float)．学習率．
        batch_size (int)．ミニバッチサイズ．
        epochs (int)．エポック数．
    戻り値: name (str)．
    """
    return f"lr{lr}_bs{batch_size}_norm{NORM_TYPE}_lambda{REG_LAMBDA}_epochs{epochs}"


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
        計算するため，この計算時間は `elapsed_time` から除外する（モジュールdocstring参照）．
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


def reuse_ex0021_results() -> None:
    """
    概要: `.orders/order_025.md` 3節に基づき，ex0021（学習率0.001・GroupNorm・バッチサイズ
        512, 128のNFG SVRG・ASAI SVRG，計4セル×5Seed=20条件）の既存結果を，ex0022の出力先
        へディレクトリごとコピーして再利用する．`hp_name` の設計により，該当条件では
        ex0021とex0022のハイパーパラメータ条件名の文字列が完全に一致するため，コピー時に
        ディレクトリ名のリネームは不要である．コピー元が存在しない，またはコピー先が既に
        完了済みの場合は何もしない（後者はスキップ，前者は`run_single_experiment`が新規に
        学習することで対応する）．
    引数: なし
    戻り値: なし
    """
    for method, batch_size in _REUSE_CELLS:
        epochs = EPOCHS_BY_BATCH_SIZE[batch_size]
        name = hp_name(_REUSE_LEARNING_RATE, batch_size, epochs)
        for seed in SEEDS:
            target_dir = os.path.join(OUTPUT_ROOT, method, name, str(seed))
            if is_run_completed(target_dir, epochs):
                continue

            source_dir = os.path.join(_EX0021_OUTPUT_ROOT, method, name, str(seed))
            if not is_run_completed(source_dir, epochs):
                print(f"[reuse-skip] コピー元が見つかりません: {source_dir}", flush=True)
                continue

            os.makedirs(target_dir, exist_ok=True)
            for filename in ["log.json", "config.json", "best_model.pth"]:
                source_path = os.path.join(source_dir, filename)
                if os.path.exists(source_path):
                    shutil.copy2(source_path, os.path.join(target_dir, filename))

            config_path = os.path.join(target_dir, "config.json")
            with open(config_path) as f:
                config = json.load(f)
            config["experiment"] = EXPERIMENT_NAME
            config["reused_from"] = os.path.relpath(source_dir, _PROJECT_ROOT)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)

            print(f"[reuse] {source_dir} -> {target_dir}", flush=True)


def run_single_experiment(args):
    """
    概要: 1つの (手法, バッチサイズ, 学習率, Seed) の組に対する学習を実行し，結果を保存する．
        すでに正常終了した結果が存在する場合（新規学習またはコピーによる再利用）は学習を
        スキップする．
    引数: args (tuple)．(method, batch_size, eta, seed) のタプル．
    戻り値: なし
    """
    method, batch_size, eta, seed = args
    torch.set_num_threads(1)

    epochs = EPOCHS_BY_BATCH_SIZE[batch_size]
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

    if method == "SGD":
        run_sgd(target_dir, load_dataloader, eta, batch_size, REG_LAMBDA, epochs, device, seed, logger)
    else:
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
        "norm_type": NORM_TYPE,
        "reg_lambda": REG_LAMBDA,
        "epochs": epochs,
        "K": len(train_dataloader),
        "total_iterations": epochs * len(train_dataloader),
        "N_train": len(train_dataloader.dataset),
        "N_test": len(test_dataloader.dataset),
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{name}/{seed}", flush=True)


def main():
    """
    概要: 実験ex0022の全条件（4手法 × 3バッチサイズ × 2学習率 × 5Seed = 120条件）のうち，
        ex0021から再利用可能な20条件をコピーし，残り100条件をマルチプロセスで並列に学習する．
    引数: なし
    戻り値: なし
    """
    print(
        f"バッチサイズ = {BATCH_SIZES}, 学習率 = {LEARNING_RATES}, "
        f"正規化層 = {NORM_TYPE}, lambda = {REG_LAMBDA}"
    )
    print(f"バッチサイズごとのエポック数: {EPOCHS_BY_BATCH_SIZE}")

    reuse_ex0021_results()

    tasks = [
        (method, batch_size, eta, seed)
        for method, batch_size, eta, seed in itertools.product(
            METHODS, BATCH_SIZES, LEARNING_RATES, SEEDS
        )
    ]

    num_workers = min(8, len(tasks))
    print(f"並列プロセス数: {num_workers}（総タスク数: {len(tasks)}）")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks, chunksize=1)

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    main()
