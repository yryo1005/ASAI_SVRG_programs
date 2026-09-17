"""
実験ex0024（LayerNorm・学習率0.001固定での長期エポック学習，GroupNormとの比較）の学習ループを
定義し，実験を実行するスクリプト．

`.orders/order_037.md` に対応する．ex0023（`report_026.md`）は，GroupNorm・学習率0.001固定・
長期エポック学習（バッチサイズ512, 128, 32についてそれぞれ192, 48, 12エポック）の下で，
SGD・SVRG・NFG SVRG・ASAI SVRGの4手法を比較し，ASAI SVRGが同一オラクル呼び出し回数のもとで
SVRGを一貫して上回る一方，NFG SVRGはエポック数を延長しても慢性的な振動から抜け出せない
ことを示した。本実験は，正規化層をGroupNormからLayerNormへ変更した場合に，同様の傾向が
維持されるかを検証する。

## モデル・データセット（`.orders/order_037.md` 2節）

`programs/ex0023_cifar10_alexnet_groupnorm_longrun/model.py` の `AlexNetCIFARNorm` は，
`.orders/order_023.md`・`.orders/order_024.md` により正規化層4パターン（なし，LayerNorm，
GroupNorm，SpectralNorm）を`norm_type`引数で選択できるよう既に実装済みであるため，
`model.py`はex0023から一切変更せず（バイト単位で完全に同一）そのまま複製している。
`data.py`もex0023と同一の前処理を用い，`EXPERIMENT_NAME`のみ本実験用に変更した。

## 正規化層の変更（`NORM_TYPE`）

ex0023の`NORM_TYPE = "groupnorm"`を`NORM_TYPE = "layernorm"`に変更した点のみが，ex0023との
実装上の差異である。`.orders/order_037.md`の指示（「ほかの実験条件はex0023と同様とします」）
に基づき，比較手法（SGD, SVRG, NFG SVRG, ASAI SVRG），バッチサイズ（512, 128, 32），学習率
（0.001固定），正則化係数（$\\lambda=5\\times10^{-4}$），Seed数（5），バッチサイズごとの
エポック数（`EPOCHS_BY_BATCH_SIZE`：512→192，128→48，32→12）は全てex0023と同一の値を用いる。

## Optimizerクラスの選択とelapsed_timeの計上方法

ex0023と同一の実装（`SVRG`／`NFGSVRG`，次のスナップショット点を内部ループのパラメータ列から
ランダムに選ぶ版）をそのまま用いる。`elapsed_time`の計上方法（NFG SVRG・ASAI SVRGの診断専用
フル勾配計算を除外する）も，`.reports/report_025.md` 2.2節の修正をそのまま踏襲する。

## 実行前の計算コスト見積もり（`.orders/order_037.md` 3節）

`@.ai/ai-dev-kit/root_prompt.md`・`machine_learning.md`の指示に基づき，グリッド中の最大
バッチサイズ（512）で1エポック全体を実際に1回実行し，GroupNorm（ex0023）との所要時間・VRAM
使用量の差を実測してから並列数を決定した。実測結果は`.reports/report_037.md`に記載する。

## SGDのみエポック数を倍にした追加学習（`.orders/order_038.md`）

チャットでの指示に基づき，全バッチサイズについてSGDのみを対象に，エポック数を
`EPOCHS_BY_BATCH_SIZE`の2倍（`SGD_DOUBLE_EPOCHS_BY_BATCH_SIZE`：512→384，128→96，32→24）に
延長した追加学習を行う。既存の60条件（4手法×3バッチサイズ×5Seed，`.orders/order_037.md`）は
変更せず保持し，新規にSGD×3バッチサイズ×5Seed＝15条件を追加する。`run_single_experiment`の
引数を`(method, batch_size, seed)`から`(method, batch_size, seed, epochs)`へ拡張し，
エポック数をタスクごとに明示的に指定できるようにした上で，タスク生成を`_build_main_tasks`
（既存の60条件）・`_build_sgd_double_epoch_tasks`（新規15条件）に分離した。`main`関数は
`run_grid`・`run_sgd_double`引数により両フェーズを独立に実行できる（`.orders/order_036.md`の
ex0051c拡張で確立した2フェーズ実行パターンを踏襲）。VRAM使用量は既存の4手法混在時（実測
約25.6GB，8プロセス合計）を下回ることが既知（SGDは他手法よりモデルインスタンスが1つ少なく
軽量）であるため，追加のVRAM実測は行わず8プロセス並列のまま実行した。計算コスト見積もりは，
既に完了しているex0024の実測`elapsed_time`（8プロセス並列実行下，資源競合を含む実測値）から
バッチサイズごとの1エポックあたり所要時間を算出し，2倍のエポック数に適用する方式を用いた。
実測結果は`.reports/report_037.md`に記載する。
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

EXPERIMENT_NAME = "ex0024_cifar10_alexnet_layernorm_longrun"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
BATCH_SIZES = [512, 128, 32]
LEARNING_RATE = 0.001  # ex0023と同一（`.orders/order_037.md`：他の実験条件はex0023と同様）
NORM_TYPE = "layernorm"  # ex0023からの変更点（GroupNorm -> LayerNorm）
REG_LAMBDA = 5e-4  # ex0023と同一

# バッチサイズごとのエポック数．ex0023（`.orders/order_026.md` 5節）と同一の値をそのまま用いる．
EPOCHS_BY_BATCH_SIZE = {512: 192, 128: 48, 32: 12}

# SGDのみエポック数を倍にした追加学習用（`.orders/order_038.md`）．
SGD_DOUBLE_EPOCHS_BY_BATCH_SIZE = {bs: 2 * ep for bs, ep in EPOCHS_BY_BATCH_SIZE.items()}

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
        なった状態）に達したかどうかを数値的に判定するために用いる．
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
    refresh_and_freeze_spectral_norm(snapshot_model)  # NORM_TYPE="layernorm"のため何も行わない

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
            # 長期学習で崩壊が生じた場合，これ以上学習を継続しても意味がないため打ち切る
            # （ex0023・`.orders/order_026.md` 6節と同一の方針）．打ち切った時点までの記録は
            # そのまま保存する．
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
    概要: 1つの (手法, バッチサイズ, Seed, エポック数) の組に対する学習を実行し，結果を保存
        する．すでに正常終了した結果（または崩壊による打ち切り）が存在する場合は学習を
        スキップする．`.orders/order_038.md`により，エポック数をタスクごとに明示的に指定
        できるよう引数を拡張した（同一バッチサイズでも，通常のグリッド（`EPOCHS_BY_BATCH_
        SIZE`）とSGDのみエポック数を倍にした追加学習（`SGD_DOUBLE_EPOCHS_BY_BATCH_SIZE`）
        とでエポック数が異なるため）．
    引数: args (tuple)．(method, batch_size, seed, epochs) のタプル．
    戻り値: なし
    """
    method, batch_size, seed, epochs = args
    torch.set_num_threads(1)

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


def _build_main_tasks() -> list:
    """
    概要: 実験ex0024の基本グリッド（4手法 × 3バッチサイズ × 5Seed = 60条件，
        `EPOCHS_BY_BATCH_SIZE`）のタスクリストを構築する．
    引数: なし
    戻り値: tasks (list of tuple)．(method, batch_size, seed, epochs) のリスト．
    """
    return [
        (method, batch_size, seed, EPOCHS_BY_BATCH_SIZE[batch_size])
        for method, batch_size, seed in itertools.product(METHODS, BATCH_SIZES, SEEDS)
    ]


def _build_sgd_double_epoch_tasks() -> list:
    """
    概要: SGDのみエポック数を倍にした追加学習（`.orders/order_038.md`）のタスクリストを
        構築する．3バッチサイズ × 5Seed = 15条件．
    引数: なし
    戻り値: tasks (list of tuple)．(method, batch_size, seed, epochs) のリスト．
    """
    return [
        ("SGD", batch_size, seed, SGD_DOUBLE_EPOCHS_BY_BATCH_SIZE[batch_size])
        for batch_size, seed in itertools.product(BATCH_SIZES, SEEDS)
    ]


def run_grid_phase() -> None:
    """
    概要: 実験ex0024の基本グリッド（60条件）を8プロセス並列で実行する．
    引数: なし
    戻り値: なし
    """
    tasks = _build_main_tasks()
    num_workers = min(8, len(tasks))
    print(f"基本グリッドタスク数: {len(tasks)}，並列プロセス数: {num_workers}")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks, chunksize=1)


def run_sgd_double_epoch_phase() -> None:
    """
    概要: SGDのみエポック数を倍にした追加学習（15条件，`.orders/order_038.md`）を8プロセス
        並列で実行する．
    引数: なし
    戻り値: なし
    """
    tasks = _build_sgd_double_epoch_tasks()
    num_workers = min(8, len(tasks))
    print(f"SGD倍エポックタスク数: {len(tasks)}，並列プロセス数: {num_workers}")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks, chunksize=1)


def main(run_grid: bool = True, run_sgd_double: bool = True) -> None:
    """
    概要: 実験ex0024の基本グリッド（4手法 × 3バッチサイズ × 5Seed = 60条件，
        `.orders/order_037.md`）と，SGDのみエポック数を倍にした追加学習（15条件，
        `.orders/order_038.md`）を実行する．`run_grid`・`run_sgd_double`引数により両
        フェーズを独立に実行できる（`.orders/order_036.md`のex0051c拡張で確立した2フェーズ
        実行パターンを踏襲）．基本グリッドは`is_run_completed`により完了済み条件を自動的に
        スキップする。
    引数:
        run_grid (bool) = True．基本グリッド（60条件）を実行するか。
        run_sgd_double (bool) = True．SGD倍エポック追加学習（15条件）を実行するか。
    戻り値: なし
    """
    print(f"バッチサイズ = {BATCH_SIZES}, 学習率 = {LEARNING_RATE}, 正規化層 = {NORM_TYPE}")
    print(f"バッチサイズごとのエポック数: {EPOCHS_BY_BATCH_SIZE}")
    print(f"SGD倍エポック（order_038）: {SGD_DOUBLE_EPOCHS_BY_BATCH_SIZE}")

    if run_grid:
        run_grid_phase()
    if run_sgd_double:
        run_sgd_double_epoch_phase()

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    # 基本グリッド（60条件）は完了済みのため，SGD倍エポック追加学習（order_038）のみ実行する．
    main(run_grid=False, run_sgd_double=True)
