"""
`.orders/order_010.md` の指示に基づく，σ（敵対的摂動）を外した純粋なCIFAR-10多値分類問題の
数値実験スクリプト（Ex005）．

`.reports/report_010.md` は，Ex003・Ex004（`programs/ex003_cifar10_resnet_minmax/`，
`programs/ex004_cifar10_resnet_minmax/`）で観測されたNFG SVRG・ASAI SVRGの不安定性（および
真のフル勾配を用いる古典的SVRGまでもが発散する現象）について，min-max（敵対的摂動）が
非凸・非凹の鞍点問題であり，同時勾配降下・上昇（simultaneous GDA）がこの種の問題で本質的に
不安定であることを主要因の候補として指摘した．本実験は，この仮説を検証するため，Ex004の
条件から**sigma（敵対的摂動）とmin-max構造のみを取り除き，他の条件（データセット，
ResNet-18モデル，学習率，ミニバッチサイズ，w に対するL2正則化係数，エポック数，M_WORKERSに
よる分散環境模擬，フル勾配計算時のBatch Normalization統計量の固定，`set_model_params`の
Batch Normalizationバッファ同期）はすべてEx004と同一に揃えた**，純粋な多値分類問題として
実装する．

- データセット：CIFAR-10（学習用50000枚，検証用10000枚）．
- モデル：ResNet-18（`model.py` の `ResNet18`．Ex003・Ex004と同一構造）．
- 誤差関数：L2正則化付き多値交差エントロピー損失
    min_w (1/M) sum_i [ CE(w, x_i, y_i) ] + (lambda1/2)||w||^2
  min-max構造（sigma，および正則化項 -(lambda2/2)||sigma||^2）は含まない．
- ハイパーパラメータ：学習率 gamma = 0.01，lambda1 = 0.0005（Ex004と同一）．ミニバッチサイズ
  128，エポック数30（Ex004と同一）．

`.orders/order_009.md` の指摘に基づくEx004の3点の修正のうち，min-max構造に依存しない次の2点は
本実験にもそのまま引き継ぐ．
1. **M_WORKERSワーカーによる分散環境の模倣**：グローバルミニバッチ（サイズ128）を，
   `M_WORKERS`（=5）個のローカルサブバッチに分割し，各サブバッチを独立にforwardすることで，
   各ワーカーが自身のローカルミニバッチに対してBatch Normalizationを適用する挙動を模倣する
   （`backward_objective_distributed`）．
2. **フル勾配計算時のBatch Normalization統計量の固定**：`compute_full_gradient_and_metrics`は
   `model.eval()`を用いることで，全データを走査する間の統計量の意図しない更新を防ぐ．

sigmaの正則化勾配のスケール調整（Ex004の修正3）は，sigma自体が存在しないため本実験には
該当しない．

さらに，`.orders/order_010.md` で報告された `set_model_params()` のBatch Normalizationバッファ
未同期バグの修正（`model.py` 参照）も適用済みであり，`snapshot_model` のBNバッファは各エポック
境界で `model` の現在値と同期される．
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
from model import ResNet18, load_model, set_model_params  # noqa: E402

EXPERIMENT_NAME = "ex005_cifar10_resnet_classification"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGD", "SVRG", "NFG", "ASAI_SVRG"]
BATCH_SIZE = 128
# Ex004と同一のエポック数．実験時間の制約（1エポックあたりSVRG系手法で約80秒，4手法 x 5Seed）を
# 考慮し，Ex003・Ex004と同じ30エポックとした．
EPOCHS = 30
# Ex004と同一の学習率・正則化係数．min-max構造を取り除いたため lambda2（sigmaの正則化係数）は
# 存在しない．
LEARNING_RATE = 0.01
LAMBDA1 = 0.0005  # w に対する正則化係数
# フル勾配（SVRGの真の勾配，NFG・ASAI SVRGの近似誤差診断）の計算に用いるミニバッチサイズ．
# フル勾配は全サンプルの勾配の平均であり，どのようなミニバッチ分割で計算しても数学的に
# 同一の結果になるため，内部ループの確率的勾配計算に用いるBATCH_SIZE（Algorithm 1に忠実に
# 保つ必要がある）とは独立に，計算効率のためこの値を用いる．フル勾配計算は
# `compute_full_gradient_and_metrics` により `model.eval()` で実行するため，ここでの
# ミニバッチ分割はBatch Normalizationの挙動に影響しない．
FULL_GRADIENT_BATCH_SIZE = 512
# Ex004と同一．グローバルミニバッチ（BATCH_SIZE）を M_WORKERS個のサブバッチに分割し，
# 各サブバッチを独立にforwardすることで，各ワーカーが自身のローカルミニバッチに対して
# Batch Normalizationを適用する挙動を模倣する．
M_WORKERS = 5

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    # 原論文と同一のスナップショット構成（内部ループの最終パラメータを採用）を用いる．
    "SVRG": SVRGFinalPoint,
    "NFG": NFGSVRGFinalPoint,
    # ASAI SVRG論文の理論解析上の都合による一様ランダム選択を用いるNFG SVRG．METHODSには
    # 含めず，Ex003・Ex004と同様に個別に実行する．
    "NFG_SVRG": NFGSVRG,
    # ASAI SVRGは提案手法自身のスナップショット構成（平均パラメータ）をそのまま用いる．
    "ASAI_SVRG": ASAISVRG,
}


def loss_func(outputs: torch.Tensor, teacher_signals: torch.Tensor) -> torch.Tensor:
    """
    概要: モデルの出力と教師信号の誤差を計算する関数．多値交差エントロピー損失．
        L2正則化項は含まない（報告用の解釈しやすい誤差として用いる）．
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


def compute_squared_norm(model: nn.Module) -> torch.Tensor:
    """
    概要: L2正則化項に用いる ||w||^2（ResNet-18の全パラメータ）を計算する．
    引数: model (torch.nn.Module)．`ResNet18` のインスタンス．
    戻り値: w_norm_sq (torch.Tensor)，形状 ()．
    """
    return sum(p.pow(2).sum() for p in model.parameters())


def backward_objective(model: nn.Module, inputs: torch.Tensor, teacher_signals: torch.Tensor):
    """
    概要: L2正則化付き多値分類の目的関数
        L(w) = CE(w, x, y) + (lambda1/2)||w||^2
        についてforward・backwardを実行し，勾配 grad_w L を各パラメータの `.grad` に設定する．
    引数:
        model (torch.nn.Module)．`ResNet18` のインスタンス．
        inputs (torch.Tensor)，形状 (B, 3, 32, 32)．入力画像．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号．
    戻り値: outputs (torch.Tensor)，形状 (B, C)．モデルの出力（ロジット，勾配計算グラフなし）．
    """
    model.zero_grad(set_to_none=True)
    outputs = model(inputs)
    ce_loss = loss_func(outputs, teacher_signals)
    w_norm_sq = compute_squared_norm(model)
    regularized_loss = ce_loss + (LAMBDA1 / 2) * w_norm_sq
    regularized_loss.backward()
    return outputs.detach()


def backward_objective_distributed(
    model: nn.Module, inputs: torch.Tensor, teacher_signals: torch.Tensor, num_workers: int = M_WORKERS
):
    """
    概要: グローバルミニバッチを `num_workers` 個のサブバッチ（ワーカー）に分割し，各サブバッチを
        独立にforwardすることで，分散環境のM=num_workersワーカーそれぞれが自身のローカル
        ミニバッチの統計量に基づいてBatch Normalizationを適用する挙動を模倣する
        （Ex004の`backward_minmax_objective_distributed`からmin-max構造を除いたもの）．
        データ項（交差エントロピー損失）の勾配は，各サブバッチの損失（reduction="mean"）に
        「サブバッチサイズ／グローバルバッチサイズ」の重みを掛けてbackward()することで，
        グローバルミニバッチ全体に対する平均勾配（1回のforwardで計算した場合と数学的に同一の
        値）を`.grad`に累積する．正則化項 (lambda1/2)||w||^2 の勾配は，サブバッチ数に依らず
        グローバルミニバッチ全体に対して1回だけ加える（`backward_objective`とスケールを
        一致させるため）．
    引数:
        model (torch.nn.Module)．`ResNet18` のインスタンス．
        inputs (torch.Tensor)，形状 (B, 3, 32, 32)．入力画像（グローバルミニバッチ）．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号．
        num_workers (int) = M_WORKERS．分割するサブバッチ（ワーカー）数．
    戻り値: outputs (torch.Tensor)，形状 (B, C)．各サブバッチのモデル出力を連結したもの
        （勾配計算グラフなし）．
    """
    model.zero_grad(set_to_none=True)

    total_batch_size = inputs.shape[0]
    input_chunks = torch.chunk(inputs, num_workers, dim=0)
    teacher_signal_chunks = torch.chunk(teacher_signals, num_workers, dim=0)

    all_outputs = []
    for sub_inputs, sub_teacher_signals in zip(input_chunks, teacher_signal_chunks):
        sub_batch_size = sub_inputs.shape[0]
        sub_outputs = model(sub_inputs)
        sub_ce_loss = loss_func(sub_outputs, sub_teacher_signals)
        weighted_sub_loss = sub_ce_loss * (sub_batch_size / total_batch_size)
        weighted_sub_loss.backward()
        all_outputs.append(sub_outputs.detach())

    w_norm_sq = compute_squared_norm(model)
    regularization = (LAMBDA1 / 2) * w_norm_sq
    regularization.backward()

    return torch.cat(all_outputs, dim=0)


def iteration(model, inputs, teacher_signals, optimizer=None, snapshot_model=None) -> dict:
    """
    概要: 1つのミニバッチのデータを学習/検証する関数．
        学習時（`optimizer` が指定される場合）は，M_WORKERS個のワーカーによる分散環境を
        模倣する `backward_objective_distributed` を用いる．
        `optimizer` がSVRG系手法（`snapshot_model` が指定される場合）は，同一ミニバッチに
        対して `model`（現在のパラメータ）と `snapshot_model`（スナップショット）の双方で
        目的関数のforward／backwardを実行し，2種類の勾配を用いて
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
        outputs = backward_objective_distributed(model, inputs, teacher_signals)
        loss = loss_func(outputs, teacher_signals)

        if snapshot_model is not None:
            backward_objective_distributed(snapshot_model, inputs, teacher_signals)
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
    概要: データローダー全体に対する目的関数のフル勾配 F(w)，および同じ1回の走査で得られる
        誤差・分類精度の平均値をまとめて計算する．全データを走査する間，Batch Normalization
        層が学習モード（`model.train()`）のままだと，移動平均統計量（running_mean，
        running_var）が走査の途中で更新され続け，走査の前半と後半で異なる関数を評価して
        しまい，スナップショット勾配としての数学的整合性が崩れる．そこで `model.eval()`
        （移動平均統計量を固定し，ミニバッチ統計量を用いないモード）に設定した上で
        forward・backwardを行う．`.eval()` はautogradの勾配計算グラフの構築を妨げないため，
        全パラメータに対する `.grad` は通常通り計算される．
    引数:
        model (torch.nn.Module)．勾配・評価の対象となるパラメータを保持するモデル．
        dataloader (torch.utils.data.DataLoader)．データセット全体を走査するデータローダー．
        device (torch.device)．
    戻り値:
        grads (list of torch.Tensor)．`model.parameters()` と同じ順序・形状のフル勾配．
        metrics (dict)．{"loss": ..., "accuracy": ...} の1データあたりの平均値．
    """
    model.eval()
    accumulated_grads = [torch.zeros_like(p) for p in model.parameters()]
    total_metrics = {}
    total_count = 0

    for inputs, teacher_signals in dataloader:
        inputs = inputs.to(device)
        teacher_signals = teacher_signals.to(device)
        batch_size = inputs.shape[0]

        outputs = backward_objective(model, inputs, teacher_signals)
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
        train_sgd(target_dir, ResNet18, load_dataloader, EPOCHS, BATCH_SIZE, device, seed, logger)
    else:
        train_variance_reduced(
            method, target_dir, ResNet18, load_dataloader, EPOCHS, BATCH_SIZE, device, seed, logger
        )

    logger.save(os.path.join(target_dir, "log.json"))

    config = {
        "method": method,
        "seed": seed,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "lambda1": LAMBDA1,
        "epochs": EPOCHS,
        "m_workers": M_WORKERS,
        "N_train": 50000,
        "N_test": 10000,
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{hp_name}/{seed}")


def main():
    """
    概要: Ex005（σを外した純粋なCIFAR-10分類問題）の全条件（4手法 x 5Seed）を
        マルチプロセスで並列に学習する．GPU（VRAM）の使用量を考慮し，並列プロセス数は4とする．
    引数: なし
    戻り値: なし
    """
    print(
        f"学習率 = {LEARNING_RATE}, ミニバッチサイズ = {BATCH_SIZE}, "
        f"lambda1 = {LAMBDA1}, エポック数 = {EPOCHS}"
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
