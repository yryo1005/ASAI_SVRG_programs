"""
E8（付録B.1，a9a の追加実験）の実験を実行するスクリプト．

E4（`ex008_e4_a9a/train.py`）をコピーし，`実験指示E8.md` 2節に従って次を加えた．
- ランダム点の手法 `SVRG_random`（`SVRG`），`NFG_SVRG_random`（`NFGSVRG`）．スナップショットの添字は
  `begin_epoch` で `rng.integers(0, K)` により1回選ぶ．`begin_epoch` には，添字の列の生成器
  `rng = default_rng(seed)` とは別の生成器 `snapshot_rng = default_rng([seed, 1])` を渡す（E1 の `ex006` と同じ）．
  最終点の手法と ASA-SVRG は `begin_epoch` で乱数を消費しないため，結果は E4 と変わらない．
- `--part`（A：BCE の `SVRG_random`・`NFG_SVRG_random`（`snapshot`），B：MSE の `ASAI_SVRG`（`last_iterate`），
  all：A と B）．`--loss`・`--methods`・`--inner-starts` のいずれかを明示した場合は，`--part` を使わない．
`SVRG`，`NFG_SVRG` は E4 と同じ最終点（`SVRGFinalPoint`，`NFGSVRGFinalPoint`）のままである．

以下は E4 の説明である．
条件は `実験計画.md` 6章と `実験指示E4.md` に基づく．E1（`ex006_e1_mushroom_rho_sampling/train.py`）を
コピーし，データとモデルを既存の a9a の実験と同一にした．次の仕組みは E1 のまま引き継いでいる．
`inner_start` の切り替え，乱数の消費位置（各外側ループの最初の `rng.permutation`），`--workers` による
並列実行，完了済みの試行のスキップ，ログの形式，SGD の実装．

| 項目 | `--loss mse`（`ex000_a9a_least_squares` の再実施） | `--loss bce`（`ex0001_a9a_logistic` の再実施） |
|---|---|---|
| データ | a9a，$ N = 32561 $，$ d = 123 $，9:1 に層化分割（$ N_{\\mathrm{train}} = 29304 $），標準化なし | 同左 |
| 目的関数 | $ \\frac{1}{N}\\sum_{n}(y_{n} - \\sigma({\\mathbf w}^{\\top}{\\mathbf x}_{n}))^{2} $，切片・正則化なし | BCE $ + \\frac{\\lambda}{2}(\\|{\\mathbf w}\\|^{2} + b^{2}) $ |
| 定数 | $ L = \\kappa_{\\max}\\max_{n}\\|{\\mathbf x}_{n}\\|^{2} = 2.1568199747494745 $ | $ L_{0} = \\frac{1}{4}\\max_{n}\\|\\tilde{\\mathbf x}_{n}\\|^{2} $，$ \\lambda = \\mu = 0.03 L_{0} = 0.1125 $，$ L = L_{0} + \\lambda = 3.8625 $ |
| 学習率 | $ \\eta = 0.99/(3L) $ | $ \\eta = 0.99/(3L) $（$ \\rho = 297.36 $） |
| ログの目的関数 | `objective_value` $ = f({\\mathbf z}_{s}) $ | `objective_gap` $ = f({\\mathbf z}_{s}) - f({\\mathbf w}^{*}) $ |

手法は SGD，SVRG（`SVRGFinalPoint`），NFG-SVRG（`NFGSVRGFinalPoint`），ASA-SVRG（`ASAISVRG`）．
内側ループは $ K = N_{\\mathrm{train}} $，ミニバッチサイズ1，random reshuffling．外側ループ数は100．シードは0〜4．
内側ループの開始点の既定は `"snapshot"`（$ {\\mathbf w}_{s}^{0} \\leftarrow {\\mathbf z}_{s} $，論文のAlgorithm通り）である．
`"last_iterate"` は旧実装の ASA-SVRG を再現できることの確認にのみ用いる．

実行例:
    python train.py --part all --workers 15
    python train.py --part A --seeds 0 --epochs 3 --output-root ../outputs/_verify/ex014_random
    python train.py --loss mse bce --methods SGD SVRG NFG_SVRG ASAI_SVRG --inner-starts snapshot --seeds 0 \\
        --epochs 3 --output-root ../outputs/_verify/ex014_e4
"""

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np  # noqa: E402
import scipy.optimize as scipy_opt  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAMS_DIR = os.path.dirname(_THIS_DIR)
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))
sys.path.insert(0, _THIS_DIR)

from machine_learning_utils import ResultLogger, set_seed  # noqa: E402
from optimizers import ASAISVRG, NFGSVRG, NFGSVRGFinalPoint, SGD, SVRG, SVRGFinalPoint  # noqa: E402

from data import load_dataloader  # noqa: E402
from model import (  # noqa: E402
    LOSSES, compute_accuracy_and_correct, compute_gradient, compute_loss, load_model, set_model_params,
)

EXPERIMENT_NAME = "ex014_e8_a9a_supplement"
OUTPUT_ROOT = os.path.join(_REVIEW_ROOT, "outputs", EXPERIMENT_NAME)

INNER_STARTS = ("last_iterate", "snapshot")
DEFAULT_INNER_STARTS = ["snapshot"]
SEEDS = [0, 1, 2, 3, 4]
ALL_METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG", "SVRG_random", "NFG_SVRG_random"]
# 全勾配を g_s に用いる手法（oracle_calls と full_grad_computations の数え方が同じ）．
FULL_GRADIENT_METHODS = ("SVRG", "SVRG_random")
# スナップショットを内側ループの点から一様ランダムに選ぶ手法（config.json に規則と乱数のシードを記録する）．
RANDOM_SNAPSHOT_METHODS = ("SVRG_random", "NFG_SVRG_random")
# 本実験の部（実験指示E8.md 0.1節）．各部は (損失, 手法, 内側ループの開始点) の組を持つ．
PARTS = {
    "A": {"losses": ["bce"], "methods": ["SVRG_random", "NFG_SVRG_random"], "inner_starts": ["snapshot"]},
    "B": {"losses": ["mse"], "methods": ["ASAI_SVRG"], "inner_starts": ["last_iterate"]},
}
PART_CHOICES = ("A", "B", "all")
SAMPLING = "random_reshuffle"
EPOCHS = 100
BATCH_SIZE = 1
LAMBDA_RATIO = 0.03
ETA_A_MARGIN = 0.99

# 実験指示E4.md 2.1節の表の値（既存の config.json の値）．データから計算した値と比較する．
EXPECTED_CONSTANTS = {
    "mse": {"L": 2.1568199747494745, "reg_lambda": 0.0},
    "bce": {"L": 3.8625, "reg_lambda": 0.1125},
}
CONSTANT_REL_TOL = 1e-12

# 目的関数の項目名は既存のログと同じにする（MSE は f の値，BCE は最適値との差）．
OBJECTIVE_KEYS = {"mse": "objective_value", "bce": "objective_gap"}


def metric_names(loss: str) -> tuple:
    """
    概要: `log.json` に記録する項目名を返す．先頭の8項目は既存のログ（`ex000`，`ex0001`）と同じ順序・名前である．
    引数: loss (str)．"mse" または "bce"．
    戻り値: names (tuple of str)．
    """
    return ("epoch", "oracle_calls", "full_grad_computations", "elapsed_time", OBJECTIVE_KEYS[loss],
            "grad_norm_sq", "accuracy", "approx_error", "test_correct", "full_grad_norm_sq", "approx_cos",
            "terminated")


_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "SVRG": SVRGFinalPoint,
    "NFG_SVRG": NFGSVRGFinalPoint,
    "ASAI_SVRG": ASAISVRG,
    "SVRG_random": SVRG,
    "NFG_SVRG_random": NFGSVRG,
}


def snapshot_rng_seed(seed: int) -> list:
    """
    概要: スナップショットの選択用の乱数生成器のシード `[seed, 1]` を返す（`ex006` と同じ）．添字の列の生成器
        （`default_rng(seed)`）とは独立の系列になる．
    引数: seed (int)．試行のシード．
    戻り値: seed_sequence (list of int)．長さ2．
    """
    return [seed, 1]


# ---------------------------------------------------------------------------
# 定数（L，λ）と最適値．既存の train.py の関数をコピーした．
# ---------------------------------------------------------------------------

def compute_kappa_max(z_half_width: float = 30.0, num_points: int = 400001) -> float:
    """
    概要: 1サンプル分の損失 $ l(z, y) = (y - \\sigma(z))^{2} $ の2階微分の絶対値の上界
        $ \\kappa_{\\max} = \\max_{z, y \\in \\{0, 1\\}} |l''(z, y)| $ を格子上の数値探索で求める
        （`programs/ex000_a9a_least_squares/train.py` と同一）．
    引数:
        z_half_width (float) = 30.0．探索区間 $ [-z_{\\mathrm{hw}}, z_{\\mathrm{hw}}] $ の半幅．
        num_points (int) = 400001．格子点の数．
    戻り値: kappa_max (float)．
    """
    z = np.linspace(-z_half_width, z_half_width, num_points)
    s = 1.0 / (1.0 + np.exp(-z))
    s1 = s * (1.0 - s)
    s2 = s1 * (1.0 - 2.0 * s)

    kappa_max = 0.0
    for y in (0.0, 1.0):
        second_derivative = 2.0 * s1 ** 2 + 2.0 * (s - y) * s2
        kappa_max = max(kappa_max, float(np.max(np.abs(second_derivative))))
    return kappa_max


def compute_smoothness_constants(A: np.ndarray) -> tuple[float, float]:
    """
    概要: MSE の平滑性定数 $ L_{\\mathrm{individual}} = \\kappa_{\\max}\\max_{n}\\|{\\mathbf x}_{n}\\|^{2} $ と
        $ L_{\\mathrm{average}} = \\kappa_{\\max}\\lambda_{\\max}({\\mathbf A}^{\\top}{\\mathbf A}/N) $ を計算する
        （`programs/ex000_a9a_least_squares/train.py` と同一）．学習率には $ L_{\\mathrm{individual}} $ を用いる．
    引数: A (numpy.ndarray)，形状 (N, d)．入力特徴量（0/1）．
    戻り値:
        L_individual (float)．
        L_average (float)．
    """
    N = A.shape[0]
    kappa_max = compute_kappa_max()
    row_norms_sq = (A ** 2).sum(axis=1)
    L_individual = kappa_max * float(row_norms_sq.max())
    eigvals = np.linalg.eigvalsh((A.T @ A) / N)
    L_average = kappa_max * float(eigvals.max())
    return L_individual, L_average


def compute_smoothness_constant(X: np.ndarray, reg_lambda: float) -> float:
    """
    概要: BCE の平滑性定数 $ L = \\frac{1}{4}\\max_{n}\\|\\tilde{\\mathbf x}_{n}\\|^{2} + \\lambda $ を計算する
        （$ \\tilde{\\mathbf x}_{n} $ は切片に対応する1を付加した特徴量．`programs/ex0001_a9a_logistic/train.py` と同一）．
    引数:
        X (numpy.ndarray)，形状 (N, d)．入力特徴量．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: L (float)．
    """
    X_augmented = np.hstack([X, np.ones((X.shape[0], 1))])
    row_norms_sq = (X_augmented ** 2).sum(axis=1)
    return 0.25 * float(row_norms_sq.max()) + reg_lambda


def compute_contraction_rate(eta: float, L: float, mu: float, K: int) -> float:
    """
    概要: 定理1の縮小率 $ \\rho = (K c_{2} + 2/\\mu)/(K c_{1}) $，$ c_{1} = 2\\eta(1-3\\eta L) $，
        $ c_{2} = 18\\eta^{2}L $ を計算する．
    引数:
        eta (float)．学習率．
        L (float)．平滑性定数．
        mu (float)．強凸性定数．
        K (int)．内側ループ長．
    戻り値: rho (float)．
    """
    c1 = 2.0 * eta * (1.0 - 3.0 * eta * L)
    c2 = 18.0 * eta ** 2 * L
    return (K * c2 + 2.0 / mu) / (K * c1)


def compute_optimal_solution(X_train: np.ndarray, y_train: np.ndarray, loss: str, reg_lambda: float):
    """
    概要: 目的関数の最適解と最適値を L-BFGS-B で求める（既存の2つの `compute_optimal_solution` と同一）．
        MSE は非凸のため，得られる値は参照値（`f_star_reference`）としてのみ用いる．
    引数:
        X_train (numpy.ndarray)，形状 (N, d)．学習用の特徴量．
        y_train (numpy.ndarray)，形状 (N,)．学習用の教師信号．
        loss (str)．"mse" または "bce"．
        reg_lambda (float)．L2正則化係数 $ \\lambda $（"bce" のみ）．
    戻り値:
        x_star (numpy.ndarray)，形状 (d,)（MSE）または (d+1,)（BCE，末尾が切片）．
        f_star (float)．
    """
    N, d = X_train.shape

    def mse_objective_and_grad(w):
        z = X_train @ w
        s = 1.0 / (1.0 + np.exp(-z))
        value = np.mean((y_train - s) ** 2)
        grad = (X_train.T @ (2.0 * (s - y_train) * s * (1.0 - s))) / N
        return value, grad

    def bce_objective_and_grad(wb):
        w, b = wb[:-1], wb[-1]
        p = 1.0 / (1.0 + np.exp(-(X_train @ w + b)))
        p_clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
        bce = np.mean(-(y_train * np.log(p_clipped) + (1.0 - y_train) * np.log(1.0 - p_clipped)))
        reg = 0.5 * reg_lambda * (w @ w + b * b)
        grad_w = (X_train.T @ (p - y_train)) / N + reg_lambda * w
        grad_b = np.mean(p - y_train) + reg_lambda * b
        return bce + reg, np.concatenate([grad_w, [grad_b]])

    if loss == "mse":
        fun, x0 = mse_objective_and_grad, np.zeros(d)
    else:
        fun, x0 = bce_objective_and_grad, np.zeros(d + 1)
    result = scipy_opt.minimize(fun, x0, jac=True, method="L-BFGS-B",
                                options={"maxiter": 20000, "ftol": 1e-18, "gtol": 1e-14})
    return result.x, float(result.fun)


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------

def evaluate_point(model, data, loss, reg_lambda, f_star):
    """
    概要: 与えられたパラメータにおける全勾配，そのノルムの2乗，目的関数（MSE は値，BCE は最適値との差），
        テスト精度，テストの正解数を計算する．この計算は `elapsed_time` と `oracle_calls` に含めない．
    引数:
        model (torch.nn.Module)．評価するパラメータを保持するモデル．
        data (tuple)．(X_train, y_train, X_test, y_test) の組（torch.Tensor）．
        loss (str)．"mse" または "bce"．
        reg_lambda (float)．L2正則化係数．
        f_star (float)．BCE の最適値（MSE では用いない）．
    戻り値:
        true_full_grad (list of torch.Tensor)．$ \\nabla f({\\mathbf z}_{s}) $．
        grad_norm_sq (float)．$ \\|\\nabla f({\\mathbf z}_{s})\\|^{2} $．
        objective (float)．MSE：$ f({\\mathbf z}_{s}) $，BCE：$ f({\\mathbf z}_{s}) - f({\\mathbf w}^{*}) $．
        accuracy (float)．テスト精度．
        test_correct (int)．テストデータの正解数．
    """
    X_train, y_train, X_test, y_test = data
    true_full_grad = compute_gradient(model, X_train, y_train, loss, reg_lambda)
    grad_norm_sq = sum(torch.sum(g ** 2).item() for g in true_full_grad)
    objective = compute_loss(model, X_train, y_train, loss, reg_lambda)
    if loss == "bce":
        objective = objective - f_star
    accuracy, test_correct = compute_accuracy_and_correct(model, X_test, y_test)
    return true_full_grad, grad_norm_sq, objective, accuracy, test_correct


def compute_approx_statistics(snapshot_grad: list, true_full_grad: list) -> tuple:
    """
    概要: 近似誤差 $ \\|{\\mathbf e}_{s}\\|^{2} = \\|{\\mathbf g}_{s} - \\nabla f({\\mathbf z}_{s})\\|^{2} $（既存と同じ計算）と，
        $ {\\mathbf g}_{s} $ と $ \\nabla f({\\mathbf z}_{s}) $ のコサイン類似度を計算する．
    引数:
        snapshot_grad (list of torch.Tensor)．$ {\\mathbf g}_{s} $．
        true_full_grad (list of torch.Tensor)．$ \\nabla f({\\mathbf z}_{s}) $．
    戻り値:
        approx_error (float)．
        approx_cos (float)．どちらかのノルムが0の場合はNaN．
    """
    approx_error = sum(torch.sum((g_s - g_true) ** 2).item() for g_s, g_true in zip(snapshot_grad, true_full_grad))
    inner = sum(torch.sum(g_s * g_true).item() for g_s, g_true in zip(snapshot_grad, true_full_grad))
    g_norm = np.sqrt(sum(torch.sum(g ** 2).item() for g in snapshot_grad))
    t_norm = np.sqrt(sum(torch.sum(g ** 2).item() for g in true_full_grad))
    approx_cos = float(inner / (g_norm * t_norm)) if g_norm * t_norm > 0.0 else float("nan")
    return approx_error, approx_cos


def get_epoch_indices(rng: np.random.Generator, N_train: int) -> np.ndarray:
    """
    概要: 1エポック分のデータの添字の列を random reshuffling で生成する．既存の実装と同じく，
        各外側ループの最初（`optimizer.begin_epoch(rng)` の直前）で `rng.permutation` を1回呼ぶ．
    引数:
        rng (numpy.random.Generator)．乱数生成器．
        N_train (int)．学習データ数（$ K $ と同じ）．
    戻り値: indices (numpy.ndarray)，形状 (N_train,)．int型．
    """
    return rng.permutation(N_train)


# ---------------------------------------------------------------------------
# 学習ループ
# ---------------------------------------------------------------------------

def run_inner_loop(model, snapshot_model, optimizer, X_train, y_train, loss, reg_lambda, indices) -> None:
    """
    概要: 1回の内側ループ（学習データを $ K = N_{\\mathrm{train}} $ 件，ミニバッチサイズ1）を実行する．
    引数:
        model (torch.nn.Module)．現在のパラメータ $ {\\mathbf w}_{s}^{k} $ を保持するモデル．
        snapshot_model (torch.nn.Module)．スナップショット $ {\\mathbf z}_{s} $ を保持するモデル．
        optimizer (torch.optim.Optimizer)．SVRG系の手法のOptimizer．
        X_train, y_train (torch.Tensor)．学習データ．
        loss (str)．"mse" または "bce"．
        reg_lambda (float)．L2正則化係数．
        indices (numpy.ndarray)，形状 (N_train,)．データを用いる順序．
    戻り値: なし
    """
    for index in indices:
        i = int(index)
        X_i, y_i = X_train[i:i + 1], y_train[i:i + 1]
        compute_gradient(model, X_i, y_i, loss, reg_lambda)
        grad_at_snapshot = compute_gradient(snapshot_model, X_i, y_i, loss, reg_lambda)
        optimizer.step(grad_at_snapshot)


def run_variance_reduced(method, inner_start, seed, data, loss, eta, reg_lambda, f_star, epochs, logger) -> None:
    """
    概要: SVRG系の手法（SVRG，NFG-SVRG，ASA-SVRG）で学習し，各外側ループの終了時の評価指標を
        `logger` に記録する．各外側ループで `rng.permutation` の後に `begin_epoch(snapshot_rng)` を呼ぶ．
        `snapshot_rng = default_rng([seed, 1])` は添字の列の `rng` とは別の生成器であり，ランダム点の手法だけが
        これを消費する（`ex006` と同じ）．目的関数の値が非有限になった外側ループで打ち切る．
    引数:
        method (str)．"SVRG"，"NFG_SVRG"，"ASAI_SVRG"，"SVRG_random"，"NFG_SVRG_random" のいずれか．
        inner_start (str)．"last_iterate" または "snapshot"．
        seed (int)．乱数シード．
        data (tuple)．(X_train, y_train, X_test, y_test) の組．
        loss (str)．"mse" または "bce"．
        eta (float)．学習率．
        reg_lambda (float)．L2正則化係数．
        f_star (float)．BCE の最適値．
        epochs (int)．外側ループ数．
        logger (ResultLogger)．`metric_names(loss)` を登録したロガー．
    戻り値: なし
    """
    X_train, y_train, _, _ = data
    N_train, input_dim = X_train.shape
    model = load_model(loss, seed=seed, input_dim=input_dim)
    snapshot_model = load_model(loss, seed=seed, input_dim=input_dim)
    optimizer = _VARIANCE_REDUCED_OPTIMIZER_CLASSES[method](model.parameters(), lr=eta, K=N_train)
    rng = np.random.default_rng(seed)
    snapshot_rng = np.random.default_rng(snapshot_rng_seed(seed))
    uses_full_gradient = method in FULL_GRADIENT_METHODS

    true_full_grad, grad_norm_sq, objective, accuracy, test_correct = evaluate_point(
        snapshot_model, data, loss, reg_lambda, f_star)
    if uses_full_gradient:
        start_time = time.perf_counter()
        snapshot_grad = compute_gradient(snapshot_model, X_train, y_train, loss, reg_lambda)
        elapsed_time = time.perf_counter() - start_time
        optimizer.set_snapshot_gradient(snapshot_grad)
    else:
        snapshot_grad = optimizer.get_snapshot_gradient()
        elapsed_time = 0.0
    approx_error, approx_cos = compute_approx_statistics(snapshot_grad, true_full_grad)
    logger(0, 0, 0.0, 0.0, objective, grad_norm_sq, accuracy, approx_error, test_correct, grad_norm_sq,
           approx_cos, False)
    oracle_calls = N_train if uses_full_gradient else 0

    for epoch in tqdm(range(1, epochs + 1), desc=f"{loss} {method} {inner_start} seed={seed}", leave=False):
        indices = get_epoch_indices(rng, N_train)
        start_time = time.perf_counter()
        optimizer.begin_epoch(snapshot_rng)
        run_inner_loop(model, snapshot_model, optimizer, X_train, y_train, loss, reg_lambda, indices)
        optimizer.end_epoch()
        set_model_params(snapshot_model, optimizer.get_snapshot_params(), source_model=model)
        if inner_start == "snapshot":
            # 論文のAlgorithm：次の内側ループを w_s^0 <- z_s から開始する．
            set_model_params(model, optimizer.get_snapshot_params(), source_model=model)
        if uses_full_gradient:
            snapshot_grad = compute_gradient(snapshot_model, X_train, y_train, loss, reg_lambda)
            optimizer.set_snapshot_gradient(snapshot_grad)
        else:
            snapshot_grad = optimizer.get_snapshot_gradient()
        elapsed_time += time.perf_counter() - start_time
        oracle_calls += 3 * N_train if uses_full_gradient else 2 * N_train

        true_full_grad, grad_norm_sq, objective, accuracy, test_correct = evaluate_point(
            snapshot_model, data, loss, reg_lambda, f_star)
        approx_error, approx_cos = compute_approx_statistics(snapshot_grad, true_full_grad)
        terminated = not np.isfinite(objective)
        logger(epoch, oracle_calls, oracle_calls / N_train, elapsed_time, objective, grad_norm_sq, accuracy,
               approx_error, test_correct, grad_norm_sq, approx_cos, terminated)
        if terminated:
            break


def run_sgd(seed, data, loss, eta, reg_lambda, f_star, epochs, logger) -> None:
    """
    概要: SGD（バッチサイズ1）で学習し，各エポック終了時の評価指標を `logger` に記録する．
        SVRG系の手法に固有の項目（`approx_error`，`approx_cos`）はNaNとする．目的関数の値が
        非有限になったエポックで打ち切る．
    引数:
        seed (int)．乱数シード．
        data (tuple)．(X_train, y_train, X_test, y_test) の組．
        loss (str)．"mse" または "bce"．
        eta (float)．学習率．
        reg_lambda (float)．L2正則化係数．
        f_star (float)．BCE の最適値．
        epochs (int)．エポック数．
        logger (ResultLogger)．`metric_names(loss)` を登録したロガー．
    戻り値: なし
    """
    X_train, y_train, _, _ = data
    N_train, input_dim = X_train.shape
    model = load_model(loss, seed=seed, input_dim=input_dim)
    optimizer = SGD(model.parameters(), lr=eta)
    rng = np.random.default_rng(seed)

    oracle_calls = 0
    elapsed_time = 0.0

    _, grad_norm_sq, objective, accuracy, test_correct = evaluate_point(model, data, loss, reg_lambda, f_star)
    logger(0, 0, 0.0, 0.0, objective, grad_norm_sq, accuracy, float("nan"), test_correct, grad_norm_sq,
           float("nan"), False)

    for epoch in tqdm(range(1, epochs + 1), desc=f"{loss} SGD seed={seed}", leave=False):
        indices = get_epoch_indices(rng, N_train)

        start_time = time.perf_counter()
        for index in indices:
            i = int(index)
            X_i, y_i = X_train[i:i + 1], y_train[i:i + 1]
            compute_gradient(model, X_i, y_i, loss, reg_lambda)
            optimizer.step()
        elapsed_time += time.perf_counter() - start_time
        oracle_calls += N_train

        _, grad_norm_sq, objective, accuracy, test_correct = evaluate_point(model, data, loss, reg_lambda, f_star)
        terminated = not np.isfinite(objective)
        logger(epoch, oracle_calls, oracle_calls / N_train, elapsed_time, objective, grad_norm_sq, accuracy,
               float("nan"), test_correct, grad_norm_sq, float("nan"), terminated)
        if terminated:
            break


# ---------------------------------------------------------------------------
# タスクの構築と実行
# ---------------------------------------------------------------------------

def is_run_completed(target_dir: str, epochs: int) -> bool:
    """
    概要: 指定した条件の学習が正常に終了しているかを確認する．
    引数:
        target_dir (str)．結果の保存先．
        epochs (int)．外側ループ数．
    戻り値: completed (bool)．
    """
    log_path = os.path.join(target_dir, "log.json")
    if not (os.path.exists(log_path) and os.path.exists(os.path.join(target_dir, "config.json"))):
        return False
    try:
        logger = ResultLogger(log_path)
    except (json.JSONDecodeError, OSError):
        return False
    terminated = logger["terminated"]
    return len(logger["epoch"]) == epochs + 1 or (len(terminated) > 0 and bool(terminated[-1]))


def build_config(task: dict, data: tuple, f_star: float, terminated: bool) -> dict:
    """
    概要: `config.json` に保存する辞書を作る．浮動小数点数は json の既定（17桁の repr）で保存する．
    引数:
        task (dict)．`build_tasks` が返すタスク．
        data (tuple)．(X_train, y_train, X_test, y_test) の組．
        f_star (float)．L-BFGS-B で求めた最適値．
        terminated (bool)．打ち切られたか．
    戻り値: config (dict)．
    """
    hp, loss, method = task["hyperparams"], task["loss"], task["method"]
    X_train, _, X_test, _ = data
    config = {
        "experiment": EXPERIMENT_NAME,
        "loss": loss,
        "method": method,
        "inner_start": task["inner_start"] if method != "SGD" else None,
        "sampling": SAMPLING,
        "seed": task["seed"],
        "eta": hp["eta"],
        "eta_rule": "0.99/(3L)",
        "L": hp["L"],
        "reg_lambda": hp["reg_lambda"],
        "mu": hp["mu"],
        "K": X_train.shape[0],
        "epochs": task["epochs"],
        "batch_size": BATCH_SIZE,
        "evaluation_point": "snapshot z_s" if method != "SGD" else "current point w_k",
        "N_train": X_train.shape[0],
        "N_test": X_test.shape[0],
        "d": X_train.shape[1],
        "terminated": terminated,
        "torch": torch.__version__,
    }
    if loss == "mse":
        config.update({"L_individual": hp["L_individual"], "L_average": hp["L_average"],
                       "f_star_reference": f_star})
    else:
        config.update({"L0": hp["L0"], "lambda_ratio": LAMBDA_RATIO, "rho": hp["rho"], "f_star": f_star})
    if "L_computed" in hp:
        config["L_computed"] = hp["L_computed"]
    if method in RANDOM_SNAPSHOT_METHODS:
        config["snapshot_rule"] = "uniform_random"
        config["snapshot_rng_seed"] = snapshot_rng_seed(task["seed"])
    return config


def run_single_experiment(task: dict) -> None:
    """
    概要: 1つの (損失, 手法, 内側ループの開始点, シード) の組について学習し，結果を保存する．
    引数: task (dict)．`build_tasks` が返すタスク．
    戻り値: なし
    """
    target_dir, hp, seed, loss = task["target_dir"], task["hyperparams"], task["seed"], task["loss"]
    if is_run_completed(target_dir, task["epochs"]):
        print(f"[skip] {target_dir}", flush=True)
        return
    os.makedirs(target_dir, exist_ok=True)
    torch.set_num_threads(1)

    set_seed(seed)
    train_dataloader, test_dataloader = load_dataloader(seed=seed, batch_size=BATCH_SIZE)
    X_train, y_train = train_dataloader.dataset.tensors
    X_test, y_test = test_dataloader.dataset.tensors
    data = (X_train, y_train, X_test, y_test)
    reg_lambda = hp["reg_lambda"]
    _, f_star = compute_optimal_solution(X_train.numpy(), y_train.numpy(), loss, reg_lambda)

    logger = ResultLogger()
    logger.set_names(*metric_names(loss))
    method = task["method"]
    if method == "SGD":
        run_sgd(seed, data, loss, hp["eta"], reg_lambda, f_star, task["epochs"], logger)
    else:
        run_variance_reduced(method, task["inner_start"], seed, data, loss, hp["eta"], reg_lambda, f_star,
                             task["epochs"], logger)
    logger.save(os.path.join(target_dir, "log.json"))

    config = build_config(task, data, f_star, bool(logger["terminated"][-1]))
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"[done] {target_dir}", flush=True)


def _check_constant(loss: str, name: str, computed: float) -> float:
    """
    概要: データから計算した定数を `EXPECTED_CONSTANTS` の値と比較する．相対誤差が `CONSTANT_REL_TOL`
        を超える場合は警告を表示し，表の値を返す（実験指示E4.md 2.1節）．
    引数:
        loss (str)．"mse" または "bce"．
        name (str)．"L" または "reg_lambda"．
        computed (float)．データから計算した値．
    戻り値: value (float)．用いる値．
    """
    expected = EXPECTED_CONSTANTS[loss][name]
    scale = max(abs(expected), np.finfo(np.float64).tiny)
    if expected == 0.0 and computed == 0.0:
        return computed
    if abs(computed - expected) / scale <= CONSTANT_REL_TOL:
        return computed
    print(f"[warning] {loss} の {name} の計算値 {computed!r} が表の値 {expected!r} と一致しないため，表の値を用いる．",
          flush=True)
    return expected


def build_hyperparameters(loss: str, epochs: int = EPOCHS) -> dict:
    """
    概要: 損失ごとの定数（$ L $，$ \\lambda $）と学習率 $ \\eta = 0.99/(3L) $ を，シード0の学習データから計算する
        （既存の `build_hyperparameter_list` と同じ）．
    引数:
        loss (str)．"mse" または "bce"．
        epochs (int) = EPOCHS．外側ループ数（ディレクトリ名に用いる）．
    戻り値: hyperparams (dict)．eta，L，reg_lambda，mu，hp_name と，MSE では L_individual・L_average，
        BCE では L0・rho を持つ．表の値を用いた場合は L_computed も持つ．
    """
    train_dataloader, _ = load_dataloader(seed=0, batch_size=BATCH_SIZE)
    X_train = train_dataloader.dataset.tensors[0].numpy()
    N_train = X_train.shape[0]
    if loss == "mse":
        L_individual, L_average = compute_smoothness_constants(X_train)
        L = _check_constant(loss, "L", L_individual)
        eta = ETA_A_MARGIN / (3.0 * L)
        hp = {"eta": eta, "L": L, "L_individual": L_individual, "L_average": L_average, "reg_lambda": 0.0,
              "mu": None, "hp_name": f"eta{eta:.6e}_K{N_train}_epochs{epochs}"}
        if L != L_individual:
            hp["L_computed"] = L_individual
        return hp
    L0 = compute_smoothness_constant(X_train, reg_lambda=0.0)
    reg_lambda = _check_constant(loss, "reg_lambda", LAMBDA_RATIO * L0)
    L_computed = L0 + reg_lambda
    L = _check_constant(loss, "L", L_computed)
    eta = ETA_A_MARGIN / (3.0 * L)
    hp = {"eta": eta, "L": L, "L0": L0, "reg_lambda": reg_lambda, "mu": reg_lambda,
          "rho": compute_contraction_rate(eta, L, reg_lambda, N_train),
          "hp_name": f"eta{eta:.6e}_lambda{reg_lambda:.6e}_K{N_train}_epochs{epochs}"}
    if L != L_computed:
        hp["L_computed"] = L_computed
    return hp


def build_tasks(losses, methods, inner_starts, seeds, epochs, output_root) -> list:
    """
    概要: 全ての (損失, 手法, 内側ループの開始点, シード) の組のタスクを構築する．SGDは開始点を持たないため，
        ディレクトリ名の該当部分を `inner_none` とする．保存先は
        `{output_root}/{loss}/{手法}/inner_{inner_start}_eta{η}[_lambda{λ}]_K{K}_epochs{epochs}/{seed}/`．
    引数:
        losses (list of str)．"mse"，"bce"．
        methods (list of str)．手法名．
        inner_starts (list of str)．内側ループの開始点（SGD以外に適用）．
        seeds (list of int)．シード．
        epochs (int)．外側ループ数．
        output_root (str)．結果の保存先のルート．
    戻り値: tasks (list of dict)．
    """
    tasks = []
    for loss in losses:
        hp = build_hyperparameters(loss, epochs)
        for method in methods:
            method_inner_starts = ["none"] if method == "SGD" else inner_starts
            for inner_start, seed in itertools.product(method_inner_starts, seeds):
                name = f"inner_{inner_start}_" + hp["hp_name"]
                tasks.append({
                    "loss": loss, "method": method, "inner_start": inner_start, "seed": seed, "epochs": epochs,
                    "hyperparams": hp, "target_dir": os.path.join(output_root, loss, method, name, str(seed)),
                })
    return tasks


def part_selections(part: str) -> list:
    """
    概要: 部（"A"，"B"，"all"）に対応する (損失, 手法, 内側ループの開始点) の選択の列を返す．
    引数: part (str)．"A"，"B"，"all" のいずれか．
    戻り値: selections (list of dict)．各要素は `losses`，`methods`，`inner_starts` を持つ．
    """
    names = ["A", "B"] if part == "all" else [part]
    return [PARTS[name] for name in names]


def selections_from_args(args: argparse.Namespace) -> list:
    """
    概要: コマンドライン引数からタスクの選択の列を作る．`--loss`・`--methods`・`--inner-starts` のいずれかを
        明示した場合は `--part` を使わず，明示しなかった項目は E4 の既定値（全ての損失・全ての手法・snapshot）とする．
    引数: args (argparse.Namespace)．`parse_args` の戻り値．
    戻り値: selections (list of dict)．
    """
    if args.loss is None and args.methods is None and args.inner_starts is None:
        return part_selections(args.part)
    return [{"losses": args.loss or list(LOSSES), "methods": args.methods or ALL_METHODS,
             "inner_starts": args.inner_starts or DEFAULT_INNER_STARTS}]


def build_selected_tasks(selections, seeds, epochs, output_root) -> list:
    """
    概要: 選択の列ごとに `build_tasks` でタスクを作り，連結する．
    引数:
        selections (list of dict)．`selections_from_args` の戻り値．
        seeds (list of int)．epochs (int)．output_root (str)．
    戻り値: tasks (list of dict)．
    """
    tasks = []
    for selection in selections:
        tasks += build_tasks(selection["losses"], selection["methods"], selection["inner_starts"], seeds, epochs,
                             output_root)
    return tasks


def parse_args(argv=None) -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．`--loss`，`--methods`，`--inner-starts` の既定値は None（未指定）とし，
        未指定なら `--part` でタスクを作る．
    引数: argv (list of str) = None．None なら `sys.argv[1:]`．
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", default="all", choices=PART_CHOICES)
    parser.add_argument("--loss", nargs="+", default=None, choices=LOSSES)
    parser.add_argument("--methods", nargs="+", default=None, choices=ALL_METHODS)
    parser.add_argument("--inner-starts", nargs="+", default=None, choices=INNER_STARTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=15)
    return parser.parse_args(argv)


def main() -> None:
    """
    概要: タスクを構築し，マルチプロセスで並列に学習する．長い試行（SVRG系）から先に投入する．
    引数: なし
    戻り値: なし
    """
    args = parse_args()
    tasks = build_selected_tasks(selections_from_args(args), args.seeds, args.epochs,
                                 os.path.abspath(args.output_root))
    for loss in dict.fromkeys(t["loss"] for t in tasks):
        hp = next(t["hyperparams"] for t in tasks if t["loss"] == loss)
        print(f"[{loss}] L = {hp['L']!r}, lambda = {hp['reg_lambda']!r}, eta = {hp['eta']!r}", flush=True)
    tasks.sort(key=lambda t: t["method"] == "SGD")
    num_workers = max(1, min(args.workers, len(tasks)))
    print(f"タスク数: {len(tasks)}，並列プロセス数: {num_workers}", flush=True)
    with Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks, chunksize=1)
    print("全ての学習が終了しました．", flush=True)


if __name__ == "__main__":
    main()
