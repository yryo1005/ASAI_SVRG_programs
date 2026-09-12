"""
実験5（Imagewoofを用いたResNet18画像分類，`ex005_imagewoof_resnet18`）Stage A（安定性探索）
・Stage B（4手法比較，長期学習）の学習ループを定義し，実験を実行するスクリプト．

`.orders/order_033.md`（Stage A）・`.orders/order_034.md`（Stage B）に対応する．CIFAR-10・
AlexNet（GroupNorm，ex0022/ex0023）での検証を踏まえ，より高解像度（$224\\times224$）・意味的
に難しい分類タスク（Imagewoof，ImageNet-1kから抽出した犬種10クラス）でSVRG系手法の挙動を
検証する．正規化層は，本論文で標準として報告するLayerNormに統一する（`model.py`のモジュール
docstring参照）．

実験2系列（ex002〜ex0023）・実験3〜4系列と同様の教訓（正規化層なしでいきなり本比較に進むと
広範な発散を招く）を踏まえ，Stage A（安定性探索，NFG SVRG・ASAI SVRGの2手法のみ，短い
エポック数）から段階的に実施した．Stage Aの結果（`report_033.md`）は，学習率0.001では
バッチサイズ128/64/32のいずれでも発散・崩壊が皆無であること，学習率0.01ではNFG SVRGが
全バッチサイズで崩壊しASAI SVRGもバッチサイズ64・32で崩壊することを示した．

## Stage B：4手法比較・長期学習（`.orders/order_034.md`）

Stage Aの結果・提案（10節）を踏まえ，学習率0.001に固定し，SGD・SVRGを比較対象に加えた
4手法比較を行う．バッチサイズは128・64のみを対象とし（32はStage Aで相対的に低い精度を
示したため補助的な条件として対象外），エポック数をバッチサイズ128で64，64で32へ大幅に延長
する（`STAGE_B_EPOCHS_BY_BATCH_SIZE`）．Seed数は3から5（0〜4）へ増やす．

Stage A（バッチサイズ128/64・学習率0.001・NFG SVRG／ASAI SVRG・Seed 0〜2）と条件が重複する
12条件についても，Optimizerの内部状態（running average等）が保存されておらず学習途中から
再開できないため（`.orders/order_034.md` 3節，ex0023（`report_026.md` 2.2節）と同様の理由），
**継続学習ではなくゼロから再学習する**．ただし，Stage Aのエポック数（12）とStage Bのエポック数
（64／32）が異なるため，`hp_name`が生成するディレクトリ名が自動的に異なり（`epochs12` vs
`epochs64`／`epochs32`），Stage Aの既存結果を誤って上書き・再利用することはない．

## 比較手法（`.orders/order_033.md` 6節，`.orders/order_034.md` 2節）

実験2系列以降で用いてきた `SGD`／`SVRG`／`NFGSVRG`（次のスナップショット点を内部ループの
パラメータ列からランダムに選ぶ版）をそのまま用いる．

## 崩壊の検出（`.orders/order_033.md` 8.1節）

- NaN発散：訓練損失の非有限値化を監視する．
- チャンスレベル張り付き：分類精度がチャンスレベル（$1/10=10\\%$）付近に固定されていないか
  `is_stuck_near_chance`関数で判定する．
- 近似誤差の急増：$ \\|e_s\\|^2 $ の推移を確認する．

## Stage A：実行前の計算コスト見積もり（`.orders/order_033.md` 7節）

学習実行前，ASAI SVRGの1イテレーション（2回のforward/backward）の実行時間をバッチサイズ
32・64・128でそれぞれ計測した（約152〜492ミリ秒／イテレーション，バッチサイズが大きいほど
1イテレーションは遅いが1エポックあたりの総時間はおおむね同程度（約35〜43秒）であった）．
VRAM使用量は，`.reports/report_030.md` 5.3節の教訓（VRAM実測はグリッド中の最大バッチサイズ
で行うこと）に従い，**最大バッチサイズ（128）で実測した**．結果は約3.4GB（`torch.cuda.
max_memory_allocated`），reservedベースで約6.1GBであり，8プロセス並列時の合計VRAM使用量
（$8\\times6.1\\text{GB}\\approx48.7\\text{GB}$）が総VRAM（約102.6GB）を十分下回ることを
確認した上で，`num_workers=8`を採用した．全36条件（12エポック）の総所要時間は，上記見積もり
に基づき約1時間程度と見積もられたが，実測では8プロセス並列時のGPU資源競合により約2.5時間を
要した（`report_033.md` 6節）．

## Stage B：実行前の計算コスト見積もり（`.orders/order_034.md` 4節）

4手法（SGD, SVRG, NFG SVRG, ASAI SVRG）それぞれについて，実データローダー（`load_dataloader`）
を用いた1イテレーションの実行時間をバッチサイズ128・64で計測した．バッチサイズ128では
SGD 517ms／NFG SVRG 411ms／SVRG 545ms／ASAI SVRG 573ms（1エポックあたり約29〜41秒，
$ K=71 $），バッチサイズ64ではSGD 251ms／NFG SVRG 280ms／SVRG 266ms／ASAI SVRG 276ms
（1エポックあたり約36〜40秒，$ K=142 $）であった．Stage Aの合成テンソルのみを用いた
簡易パイロットとは異なり，実データの読み込み・デコード・リサイズのコストが支配的であるため，
SVRG系手法（2回のforward/backward）とSGD（1回）の間で実行時間に大きな差が生じないことが
確認された．VRAM使用量は，最大バッチサイズ（128）で全4手法とも約5.9GB（reservedベース，
`torch.cuda.max_memory_reserved`）であり，Stage Aの実測値（約6.1GB）と同程度であることを
確認した．8プロセス並列時の合計VRAM使用量（$8\\times5.9\\text{GB}\\approx47.2\\text{GB}$）は
総VRAM（約102.6GB）を十分下回るため，Stage Aと同じ8プロセス並列を採用する．

上記の1エポックあたりの時間とエポック数（バッチサイズ128：64エポック，64：32エポック）から，
単一プロセスでの総所要時間（40条件の合計）は約19.7時間と見積もられる．8プロセス並列により
単純には約2.5時間まで短縮される計算だが，Stage Aで判明した並列実行時のGPU資源競合による
補正係数（実測は見積もりの約2.5倍，`report_033.md` 6節）を適用すると，実際の所要時間は
約6時間程度になると見込まれる．VRAM使用量には十分な余裕があるため，並列数の削減やエポック数の
削減は行わず，8プロセス並列のまま実行する．
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
    ResNet18LayerNorm,
    load_model,
    loss_func,
    metrics_func,
    set_model_params,
)

EXPERIMENT_NAME = "ex005_imagewoof_resnet18"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

REG_LAMBDA = 5e-4
NUM_CLASSES = 10
NEAR_CHANCE_TOLERANCE = 0.03  # チャンスレベル(0.1)の±0.03，本実験の10クラス設定向け

# --- Stage A（`.orders/order_033.md`）：安定性探索 ---
STAGE_A_SEEDS = [0, 1, 2]
STAGE_A_METHODS = ["NFG_SVRG", "ASAI_SVRG"]  # SGD・SVRGは対象外
STAGE_A_BATCH_SIZES = [128, 64, 32]
STAGE_A_LEARNING_RATES = [0.01, 0.001]
STAGE_A_EPOCHS = 12

# 後方互換のため，Stage Aの単体テスト（`tests/test_ex005_imagewoof_resnet18.py`）が参照する
# 従来の変数名も維持する．
SEEDS = STAGE_A_SEEDS
METHODS = STAGE_A_METHODS
BATCH_SIZES = STAGE_A_BATCH_SIZES
LEARNING_RATES = STAGE_A_LEARNING_RATES
EPOCHS = STAGE_A_EPOCHS

# --- Stage B（`.orders/order_034.md`）：4手法比較・長期学習 ---
STAGE_B_SEEDS = [0, 1, 2, 3, 4]
STAGE_B_METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
STAGE_B_BATCH_SIZES = [128, 64]  # order_034 2節：バッチサイズ32は対象外
STAGE_B_LEARNING_RATE = 0.001  # order_034 2節：学習率0.01は対象外
# order_034 2節：総イテレーション数（bs128:71*64=4544，bs64:142*32=4544）をおおむね揃える
STAGE_B_EPOCHS_BY_BATCH_SIZE = {128: 64, 64: 32}

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "SVRG": SVRG,
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


def is_stuck_near_chance(accuracies, num_classes: int, window: int = 3, tolerance: float = NEAR_CHANCE_TOLERANCE) -> bool:
    """
    概要: `.orders/order_033.md` 8.1節の指示に基づき，末尾 `window` エポックの分類精度が
        すべてチャンスレベル（$ 1/\\text{num\\_classes} $）付近に留まっているかを判定する．
        NaNにはならないがモデルが実質的に機能不全に陥る「見えない崩壊」の検出に用いる．
    引数:
        accuracies (Sequence[float])．エポックごとの分類精度の列．
        num_classes (int)．クラス数．
        window (int) = 3．判定に用いる末尾のエポック数．
        tolerance (float) = 0.03．チャンスレベルからの許容乖離幅．
    戻り値: stuck (bool)．末尾`window`エポックの精度が全て
        `[1/num_classes - tolerance, 1/num_classes + tolerance]` に収まっていれば `True`．
        `window` 未満のエポック数しかない場合は `False`．
    """
    if len(accuracies) < window:
        return False
    chance = 1.0 / num_classes
    tail = accuracies[-window:]
    return all(abs(a - chance) <= tolerance for a in tail)


def iteration(model, inputs, teacher_signals, reg_lambda, optimizer, snapshot_model=None) -> dict:
    """
    概要: 1つのミニバッチのデータを学習する関数．`snapshot_model` が指定される場合（SVRG系
        手法），同一ミニバッチに対して `model`（現在のパラメータ $ w_s^k $）と
        `snapshot_model`（スナップショット $ z_s $）の双方でforward／backwardを実行し，
        2種類の勾配を用いて `optimizer.step(grad_at_snapshot)` を呼び出す．`snapshot_model`
        が `None` の場合（SGD），通常のforward／backward／`optimizer.step()` を実行する．
    引数:
        model (torch.nn.Module)．
        inputs (torch.Tensor)，形状 (B, 3, 224, 224)．入力画像．
        teacher_signals (torch.Tensor)，形状 (B,)．教師ラベル．
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
    概要: データローダー全体に対するフル勾配，および同じ1回の走査で得られる誤差・分類精度の
        平均値をまとめて計算する．
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
    概要: 検証用データに対する分類精度が過去最高を更新した場合，モデルの重みを保存する．
    引数: model, test_accuracy, best_accuracy, target_dir．
    戻り値: best_accuracy (float)．
    """
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))
        return test_accuracy
    return best_accuracy


def run_sgd(target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, seed, logger):
    """
    概要: SGDによる学習を実行し，各エポック終了時の評価指標を `logger` に記録する
        （`.orders/order_034.md` Stage B）．
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

    model = load_model(ResNet18LayerNorm, seed=seed).to(device)
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
    K = len(train_dataloader)

    model = load_model(ResNet18LayerNorm, seed=seed).to(device)
    snapshot_model = load_model(ResNet18LayerNorm, seed=seed).to(device)

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
    概要: 1つの (手法, バッチサイズ, 学習率, エポック数, Seed) の組に対する学習を実行し，
        結果を保存する．エポック数を引数として明示的に受け取ることで，Stage A（12エポック
        固定）とStage B（バッチサイズごとに異なるエポック数）の双方に対応する．
    引数: args (tuple)．(method, batch_size, eta, epochs, seed) のタプル．
    戻り値: なし
    """
    method, batch_size, eta, epochs, seed = args
    torch.set_num_threads(1)

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
        "reg_lambda": REG_LAMBDA,
        "epochs": epochs,
        "K": len(train_dataloader),
        "total_iterations": epochs * len(train_dataloader),
        "N_train": len(train_dataloader.dataset),
        "N_test": len(test_dataloader.dataset),
        "num_classes": NUM_CLASSES,
        "collapsed": len(logger["train_loss"]) < epochs + 1,
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{name}/{seed}", flush=True)


def _build_stage_a_tasks() -> list:
    """
    概要: Stage A（`.orders/order_033.md`）の全条件（2手法×3バッチサイズ×2学習率×3Seed=
        36条件，12エポック）のタスクリストを構築する．全て学習済みのため，実行時は
        `is_run_completed` により全てスキップされる．
    引数: なし
    戻り値: tasks (list of tuple)．(method, batch_size, eta, epochs, seed) のリスト．
    """
    return [
        (method, batch_size, eta, STAGE_A_EPOCHS, seed)
        for method, batch_size, eta, seed in itertools.product(
            STAGE_A_METHODS, STAGE_A_BATCH_SIZES, STAGE_A_LEARNING_RATES, STAGE_A_SEEDS
        )
    ]


def _build_stage_b_tasks() -> list:
    """
    概要: Stage B（`.orders/order_034.md`）の全条件（4手法×2バッチサイズ×5Seed=40条件，
        学習率0.001固定，エポック数はバッチサイズごとに`STAGE_B_EPOCHS_BY_BATCH_SIZE`で
        指定）のタスクリストを構築する．
    引数: なし
    戻り値: tasks (list of tuple)．(method, batch_size, eta, epochs, seed) のリスト．
    """
    return [
        (method, batch_size, STAGE_B_LEARNING_RATE, STAGE_B_EPOCHS_BY_BATCH_SIZE[batch_size], seed)
        for method, batch_size, seed in itertools.product(
            STAGE_B_METHODS, STAGE_B_BATCH_SIZES, STAGE_B_SEEDS
        )
    ]


def main():
    """
    概要: 実験5のStage A（2手法×3バッチサイズ×2学習率×3Seed=36条件，12エポック，全て
        学習済みのためスキップされる）とStage B（4手法×2バッチサイズ×5Seed=40条件，
        学習率0.001固定，バッチサイズごとに64／32エポック）を合わせたタスクリストを
        マルチプロセスで並列に学習する．
    引数: なし
    戻り値: なし
    """
    stage_a_tasks = _build_stage_a_tasks()
    stage_b_tasks = _build_stage_b_tasks()
    tasks = stage_a_tasks + stage_b_tasks
    print(
        f"Stage Aタスク数: {len(stage_a_tasks)}（学習済みのためスキップ見込み），"
        f"Stage Bタスク数: {len(stage_b_tasks)}"
    )
    print(f"Stage B バッチサイズごとのエポック数: {STAGE_B_EPOCHS_BY_BATCH_SIZE}")

    # `.orders/order_034.md` 4節の指示に基づき，Stage Aと同一の最大バッチサイズ（128）・
    # 同一モデルであるため，Stage AのVRAM実測値（約6.1GB，reservedベース）を基本的に流用
    # しつつ，SGD・SVRGの学習ループ追加を踏まえ実行前に再確認した（`.reports/report_034.md`
    # 参照）．8プロセス並列時の合計VRAM使用量が総VRAMを十分下回ることを確認した上で，
    # Stage Aと同じ8プロセス並列を採用する．
    num_workers = min(8, len(tasks))
    print(f"並列プロセス数: {num_workers}（総タスク数: {len(tasks)}）")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks, chunksize=1)

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    main()
