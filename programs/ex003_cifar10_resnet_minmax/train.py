"""
NFG SVRG原論文（`references/No_Full_Grad_SVRG.pdf`）7節の実験（ResNet-18によるCIFAR-10
分類，min-max敵対的ロバスト性の定式化）を再現する数値実験スクリプト．

比較手法はSGD，SVRG，NFG（原論文Algorithm 1），ASAI SVRG（提案手法）の4手法．各手法について
Seed値を5種類（0〜4）で検証する．

`.orders/order_007.md` の指示に基づき，データセット・モデル・誤差関数・ハイパーパラメータを
原論文に可能な限り揃える．

- データセット：CIFAR-10（学習用50000枚，検証用10000枚）．
- モデル：ResNet-18（`model.py` の `ResNet18`．CIFAR-10向けに3x3畳み込みの初段を用いる標準的な
  適用方法）．
- 誤差関数：min-max敵対的ロバスト性の定式化
    min_w max_sigma (1/M) sum_i [ CE(w, x_i + sigma, y_i) ] + (lambda1/2)||w||^2 - (lambda2/2)||sigma||^2
  ここで sigma は画像1枚分の形状を持つ敵対的摂動（`model.py` の `MinMaxResNet18.sigma`）．
- ハイパーパラメータ：学習率 gamma = 0.01（w, sigma共通），lambda1 = lambda2 = 0.0005（原論文
  7節）．ミニバッチサイズは原論文が明記していないため，Ex001・Ex002と同様の128を用いる．
  原論文が用いる「ワーカー数M=5による分散環境の模擬」は，本実験では分散システムの詳細
  （通信・同期方式）に立ち入らない単一プロセスでのミニバッチ学習として簡略化する．

**min-maxの実装方法**：sigmaに対する勾配上昇（maximize）は，目的関数 L(w, sigma) に対する
sigmaの自然な勾配 dL/dsigma を反転させた F_sigma = -dL/dsigma を「勾配」として扱い，通常の
勾配降下（w ← w - eta*F_w, sigma ← sigma - eta*F_sigma = sigma + eta*dL/dsigma）を適用する
ことで実現する．原論文が変分不等式として定式化する F_i(z) = (grad_w f_i + lambda1*w;
-grad_sigma f_i + lambda2*sigma) は，この符号反転を演算子の定義に組み込んだものに一致する．
この方法により，`programs/optimizers/optimizers.py` の最適化手法クラス（SGD，SVRG系）を
一切変更せずに再利用できる（w とsigmaをモデルの結合パラメータ列として扱い，sigma成分の勾配の
符号のみをbackward()の直後に反転させる）．
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
from optimizers.optimizers import ASAISVRG, NFGSVRG, NFGSVRGFinalPoint, SGD, SVRGFinalPoint  # noqa: E402

from data import load_dataloader  # noqa: E402
from model import MinMaxResNet18, load_model, set_model_params  # noqa: E402

EXPERIMENT_NAME = "ex003_cifar10_resnet_minmax"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGD", "SVRG", "NFG", "ASAI_SVRG"]
BATCH_SIZE = 128
# ResNet-18は本リポジトリのEx002（3層CNN）より収束が大幅に速く（予備実験ではSGDが3エポックで
# 検証精度69%に到達），原論文が報告する「150エポック付近での安定化」を確認するには程遠いが，
# 実験時間の制約（1エポックあたりSVRG系手法で約80秒，4手法 x 5Seed）を考慮し30エポックとした．
EPOCHS = 30
# `.orders/order_008.md` の検証（ミニバッチサイズ1，NFG SVRGの2種のスナップショット構成の
# 比較，1Seed・8エポック）は，本ファイルの上記設定を一時的に変更して実行し，結果を
# `outputs/ex003_cifar10_resnet_minmax/{method}/lr0.01_bs1_lambda0.0005_epochs8/0/` に
# 保存した後，本設定に戻した（`.reports/report_008.md` 参照）．
# NFG SVRG原論文7節の設定に合わせる．
LEARNING_RATE = 0.01
LAMBDA1 = 0.0005  # w に対する正則化係数
LAMBDA2 = 0.0005  # sigma に対する正則化係数
# フル勾配（SVRGの真の勾配，NFG・ASAI SVRGの近似誤差診断）の計算に用いるミニバッチサイズ．
# フル勾配は全サンプルの勾配の平均であり，どのようなミニバッチ分割で計算しても数学的に
# 同一の結果になるため，内部ループの確率的勾配計算に用いるBATCH_SIZE（Algorithm 1に忠実に
# 保つ必要がある）とは独立に，計算効率のためこの値を用いる．
FULL_GRADIENT_BATCH_SIZE = 512

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    # 原論文と同一のスナップショット構成（内部ループの最終パラメータを採用）を用いる．
    "SVRG": SVRGFinalPoint,
    "NFG": NFGSVRGFinalPoint,
    # ASAI SVRG論文の理論解析上の都合による一様ランダム選択を用いるNFG SVRG．
    # `.orders/order_008.md` の検証（スナップショット構成方法の異なる2種類のNFG SVRGの比較）
    # のために追加．METHODSには含めず，個別に実行する．
    "NFG_SVRG": NFGSVRG,
    # ASAI SVRGは提案手法自身のスナップショット構成（平均パラメータ）をそのまま用いる．
    "ASAI_SVRG": ASAISVRG,
}


def loss_func(outputs: torch.Tensor, teacher_signals: torch.Tensor) -> torch.Tensor:
    """
    概要: モデルの出力と教師信号の誤差を計算する関数．多値交差エントロピー損失．
        min-max定式化における正則化項は含まない（報告用の解釈しやすい誤差として用いる）．
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


def compute_squared_norms(model: nn.Module):
    """
    概要: min-max定式化の正則化項に用いる ||w||^2（ResNet-18の全パラメータ）と
        ||sigma||^2 を計算する．
    引数: model (torch.nn.Module)．`MinMaxResNet18` のインスタンス．
    戻り値: (w_norm_sq, sigma_norm_sq) (torch.Tensor, torch.Tensor)．形状はいずれも ()．
    """
    w_norm_sq = sum(p.pow(2).sum() for p in model.resnet.parameters())
    sigma_norm_sq = model.sigma.pow(2).sum()
    return w_norm_sq, sigma_norm_sq


def backward_minmax_objective(model: nn.Module, inputs: torch.Tensor, teacher_signals: torch.Tensor):
    """
    概要: min-max定式化の目的関数
        L(w, sigma) = CE(w, x+sigma, y) + (lambda1/2)||w||^2 - (lambda2/2)||sigma||^2
        についてforward・backwardを実行し，sigma成分の勾配の符号を反転させることで，
        F(z) = (grad_w L; -grad_sigma L) を各パラメータの `.grad` に設定する．
        （sigmaに対する勾配上昇をvへの勾配降下として扱うための符号反転．モジュール冒頭の
        docstringを参照．）
    引数:
        model (torch.nn.Module)．`MinMaxResNet18` のインスタンス．
        inputs (torch.Tensor)，形状 (B, 3, 32, 32)．入力画像．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号．
    戻り値: outputs (torch.Tensor)，形状 (B, C)．モデルの出力（ロジット，勾配計算グラフなし）．
    """
    model.zero_grad(set_to_none=True)
    outputs = model(inputs)
    ce_loss = loss_func(outputs, teacher_signals)
    w_norm_sq, sigma_norm_sq = compute_squared_norms(model)
    regularized_loss = ce_loss + (LAMBDA1 / 2) * w_norm_sq - (LAMBDA2 / 2) * sigma_norm_sq
    regularized_loss.backward()
    model.sigma.grad.neg_()
    return outputs.detach()


def iteration(model, inputs, teacher_signals, optimizer=None, snapshot_model=None) -> dict:
    """
    概要: 1つのミニバッチのデータを学習/検証する関数．
        `optimizer` がSVRG系手法（`snapshot_model` が指定される場合）は，同一ミニバッチに
        対して `model`（現在のパラメータ）と `snapshot_model`（スナップショット）の双方で
        min-max目的関数のforward／backwardを実行し，2種類の勾配（F(z)）を用いて
        `optimizer.step(grad_at_snapshot)` を呼び出す．
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
        outputs = backward_minmax_objective(model, inputs, teacher_signals)
        loss = loss_func(outputs, teacher_signals)

        if snapshot_model is not None:
            backward_minmax_objective(snapshot_model, inputs, teacher_signals)
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
        ResNet-18はBatch Normalization層を含むため，`optimizer` が None の場合（検証用データの
        評価）は `model.eval()` により移動平均統計量を用いた評価モードに切り替え，`optimizer`
        が指定される場合（学習）は `model.train()`（および `snapshot_model` が指定される場合は
        それも）によりミニバッチ統計量を用いる学習モードに切り替える．
    引数:
        model (torch.nn.Module)．
        dataloader (torch.utils.data.DataLoader)．
        device (torch.device)．
        optimizer (torch.optim.Optimizer) = None．
        snapshot_model (torch.nn.Module) = None．SVRG系手法のスナップショットを保持するモデル．
    戻り値: metrics_to_value (dict)．1データあたりの評価・誤差の平均値の辞書．
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()
        if snapshot_model is not None:
            snapshot_model.train()

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
    概要: データローダー全体に対する min-max 目的関数のフル勾配 F(z)，および同じ1回の
        走査で得られる誤差・分類精度の平均値をまとめて計算する．勾配を計算するため，
        Batch Normalization層はミニバッチ統計量を用いる学習モード（`model.train()`）に
        設定する．
    引数:
        model (torch.nn.Module)．勾配・評価の対象となるパラメータを保持するモデル．
        dataloader (torch.utils.data.DataLoader)．データセット全体を走査するデータローダー．
        device (torch.device)．
    戻り値:
        grads (list of torch.Tensor)．`model.parameters()` と同じ順序・形状のフル勾配．
        metrics (dict)．{"loss": ..., "accuracy": ...} の1データあたりの平均値．
    """
    model.train()
    accumulated_grads = [torch.zeros_like(p) for p in model.parameters()]
    total_metrics = {}
    total_count = 0

    for inputs, teacher_signals in dataloader:
        inputs = inputs.to(device)
        teacher_signals = teacher_signals.to(device)
        batch_size = inputs.shape[0]

        outputs = backward_minmax_objective(model, inputs, teacher_signals)
        loss = loss_func(outputs, teacher_signals)

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
    概要: フル勾配の近似誤差 ||e_s||^2 = ||g_s - F(z_s)||^2 を計算する．
    引数:
        snapshot_gradient (list of torch.Tensor)．スナップショット勾配 g_s．
        true_full_gradient (list of torch.Tensor)．真のフル勾配 F(z_s)．
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
    logger.save(os.path.join(target_dir, "log.json"))  # 中断時に進捗を確認できるよう逐次保存

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
        logger.save(os.path.join(target_dir, "log.json"))  # 中断時に進捗を確認できるよう逐次保存


def train_variance_reduced(
    method, target_dir, ModelClass, load_dataloader_func, epochs, batch_size, device, seed, logger
):
    """
    概要: SVRG系手法（SVRG，NFG，ASAI SVRG）による学習を実行し，各エポック終了時の
        評価指標を `logger` に記録する．`epoch` 関数を内部ループ（1エポック分のミニバッチ列）
        として用いる．
    引数:
        method (str)．"SVRG"，"NFG"，"ASAI_SVRG" のいずれか．
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

    # フル勾配の計算専用に，計算効率のための大きいミニバッチサイズのDataLoaderを別途用意する
    # （内部ループの確率的勾配計算とは独立．モジュール冒頭のFULL_GRADIENT_BATCH_SIZEの
    # docstringを参照）．
    full_gradient_dataloader, _ = load_dataloader_func(seed=seed, batch_size=FULL_GRADIENT_BATCH_SIZE)

    model = load_model(ModelClass, seed=seed).to(device)
    # z_0 <- w_0．snapshot_modelはmodelと同じseedで初期化することで同一の初期値を持つ．
    snapshot_model = load_model(ModelClass, seed=seed).to(device)

    OptimizerClass = _VARIANCE_REDUCED_OPTIMIZER_CLASSES[method]
    optimizer = OptimizerClass(model.parameters(), lr=LEARNING_RATE, K=K)
    rng = np.random.default_rng(seed)

    oracle_calls = 0

    if method == "SVRG":
        # g_0: SVRGは真のフル勾配．この計算コストはepoch 1の学習コストとして計上する．
        snapshot_grad, train_metrics_0 = compute_full_gradient_and_metrics(
            snapshot_model, full_gradient_dataloader, device
        )
        true_full_grad_0 = snapshot_grad
    else:
        # g_0 = 0（NFG, ASAI SVRG）．Optimizer初期化時に既に設定済み．
        snapshot_grad = optimizer.get_snapshot_gradient()
        true_full_grad_0, train_metrics_0 = compute_full_gradient_and_metrics(
            snapshot_model, full_gradient_dataloader, device
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
    logger.save(os.path.join(target_dir, "log.json"))  # 中断時に進捗を確認できるよう逐次保存

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
                snapshot_model, full_gradient_dataloader, device
            )
            oracle_calls += N_train
            optimizer.set_snapshot_gradient(snapshot_grad)
            approx_error = 0.0
        else:
            snapshot_grad = optimizer.get_snapshot_gradient()
            true_full_grad, train_metrics = compute_full_gradient_and_metrics(
                snapshot_model, full_gradient_dataloader, device
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
        logger.save(os.path.join(target_dir, "log.json"))  # 中断時に進捗を確認できるよう逐次保存


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

    hp_name = f"lr{LEARNING_RATE}_bs{BATCH_SIZE}_lambda{LAMBDA1}_epochs{EPOCHS}"
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
        train_sgd(target_dir, MinMaxResNet18, load_dataloader, EPOCHS, BATCH_SIZE, device, seed, logger)
    else:
        train_variance_reduced(
            method, target_dir, MinMaxResNet18, load_dataloader, EPOCHS, BATCH_SIZE, device, seed, logger
        )

    logger.save(os.path.join(target_dir, "log.json"))

    config = {
        "method": method,
        "seed": seed,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "lambda1": LAMBDA1,
        "lambda2": LAMBDA2,
        "epochs": EPOCHS,
        "N_train": 50000,
        "N_test": 10000,
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{hp_name}/{seed}")


def main():
    """
    概要: Ex003（NFG SVRG原論文の実験再現）の全条件（4手法 x 5Seed）を
        マルチプロセスで並列に学習する．GPU（VRAM）の使用量を考慮し，並列プロセス数は4とする．
    引数: なし
    戻り値: なし
    """
    print(
        f"学習率 = {LEARNING_RATE}, ミニバッチサイズ = {BATCH_SIZE}, "
        f"lambda1 = {LAMBDA1}, lambda2 = {LAMBDA2}, エポック数 = {EPOCHS}"
    )

    tasks = [(method, seed) for method, seed in itertools.product(METHODS, SEEDS)]

    num_workers = min(4, len(tasks))
    print(f"並列プロセス数: {num_workers} (総タスク数: {len(tasks)})")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks)

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    main()
