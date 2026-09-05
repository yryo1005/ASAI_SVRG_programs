"""
実験1（Mushroomデータセットを用いた二値分類問題）の学習ループを定義し，実験を実行するスクリプト．

`.orders/order_021.md` の実験1に対応する．実験0（`report_020.md`）で観察された
(a) 学習率が理論上界 $ \\eta=1/(3L) $ に近い場合のNFG SVRG・ASAI SVRGの振動，
(b) ASAI SVRGの近似誤差 $ \\|e_s\\|^2 $ がNFG SVRGより一貫して小さい現象，
を，ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）のAssumption 1〜4（$ L $-平滑性，凸性，
$ \\mu $-強凸性，分散の一様有界性）を厳密に満たす設定で定量的に検証する．

## 目的関数

$$
f(w, b) = \\frac{1}{N}\\sum_{n=1}^N \\ell_{BCE}(y_n, \\hat{y}_n(w, b))
    + \\frac{\\lambda}{2}(\\|w\\|^2 + b^2), \\quad \\hat{y}_n(w, b) = \\sigma(w^\\top x_n + b)
$$

（`model.py` 参照．正則化項は $ w $ と切片 $ b $ の両方に課し，全パラメータについて
$ \\mu=\\lambda $ の強凸性を厳密に満たす．）

## 平滑性定数 $ L $ と正則化係数 $ \\lambda $

BCE損失の1サンプル分 $ \\ell_{BCE}(y_i, \\sigma(w^\\top x_i + b)) $ のヘッセ行列（$ w, b $ を
まとめた $ (d+1) $ 次元パラメータに関する）は $ \\sigma(z)(1-\\sigma(z)) \\cdot \\tilde{x}_i
\\tilde{x}_i^\\top \\preceq \\frac{1}{4}\\|\\tilde{x}_i\\|^2 \\cdot I $（$ \\tilde{x}_i =
[x_i; 1] $，切片項に対応する定数1を付加したベクトル）で上から抑えられることを用い，

$$
L = \\frac{1}{4}\\max_i \\|\\tilde{x}_i\\|^2 + \\lambda
$$

とする．正則化係数 $ \\lambda $ は，正則化なしの平滑性定数
$ L_0 = \\frac{1}{4}\\max_i\\|\\tilde{x}_i\\|^2 $ に対する比率で決定する．`.orders/
order_021.md` は例として比率 $ 10^{-2} $ を挙げているが，この比率では収縮係数 $ \\rho $
（式(30)）の最小値が0.79程度までしか下がらず（`build_hyperparameters` 内の事前調査），
目標とする $ \\rho \\approx 0.5 $ を達成できない．比率 $ 3 \\times 10^{-2} $ では $ \\rho $ の
最小値が約0.44となり，$ \\rho = 0.5 $ を満たす学習率が存在する．そのため本実験では
$ \\lambda = 3 \\times 10^{-2} \\times L_0 $ を採用する（条件数 $ \\mu/L \\approx 0.029 $）．

## 学習率

理論的な根拠を持つ次の3つの学習率を用いる．

1. $ \\eta_a = 0.99/(3L) $：ASAI SVRG論文 補題3の条件 $ 0 < \\eta < 1/(3L) $ の上界近傍．
   実験0（`report_020.md` 5.2節）で観察された学習率依存の不安定化が，強凸設定でも
   再現するかを確認する．
2. $ \\eta_b $：収縮係数 $ \\rho $（式(30)）が $ \\rho = 0.5 $ となるよう，$ (0, 1/(3L)) $ 上で
   数値的に逆算した学習率．$ \\rho(\\eta) $ は $ \\eta \\to 0 $ と $ \\eta \\to 1/(3L) $ の
   両極限で発散するU字形の関数であり，$ \\rho=0.5 $ を満たす解は最小値を挟んで2つ存在する．
   本実験では，より大きい方の解（最小値より右側，$ \\eta_a $ に近い側）を採用する．
3. $ \\eta_c = 1/(20L) $：実験0で安定した収束を示した学習率の規則をそのまま踏襲した値
   （$ \\rho(\\eta_c) \\approx 0.64 $）．

## 内部ループとスナップショット構成

内部ループはエポックごとのランダムリシャッフルにより学習用データを1巡するため，内部ループ長は
$ K = N_{\\text{train}} $ である．比較手法はSGD，SVRG（`SVRGFinalPoint`，最終点採用），
NFG SVRG（`NFGSVRGFinalPoint`，最終点採用），ASAI SVRG（`ASAISVRG`，平均パラメータ採用）の
4手法とする．

## 縦軸

強凸問題であり最適値 $ f(w^*) $ をL-BFGS-Bで高精度に推定できるため，目的関数は差分
$ f(z_s) - f(w^*) $（`objective_gap`）で評価する．
"""

import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import scipy.optimize as scipy_opt
import torch
from tqdm import tqdm

_PROGRAMS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from machine_learning_utils import ResultLogger, set_seed  # noqa: E402
from optimizers import ASAISVRG, NFGSVRGFinalPoint, SGD, SVRGFinalPoint  # noqa: E402

from data import load_dataloader  # noqa: E402
from model import (  # noqa: E402
    compute_accuracy,
    compute_gradient,
    compute_loss,
    load_model,
    set_model_params,
)

EXPERIMENT_NAME = "ex001_mushroom_logistic"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
EPOCHS = 100
BATCH_SIZE = 1

# 正則化なしの平滑性定数 L_0 に対する正則化係数 lambda の比率．
LAMBDA_RATIO = 0.03
# 収縮係数 rho の目標値（学習率 eta_b の逆算に用いる）．
TARGET_RHO = 0.5
# eta_a（上界近傍）に用いる 1/(3L) に対する係数（ちょうど 1/(3L) では c_1=0 となり数値的に
# 不安定なため，わずかに小さい値を用いる）．
ETA_A_MARGIN = 0.99

METRIC_NAMES = (
    "epoch",
    "oracle_calls",
    "full_grad_computations",
    "elapsed_time",
    "objective_gap",
    "grad_norm_sq",
    "accuracy",
    "approx_error",
)

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "SVRG": SVRGFinalPoint,
    "NFG_SVRG": NFGSVRGFinalPoint,
    "ASAI_SVRG": ASAISVRG,
}


def compute_smoothness_constant(X: np.ndarray, reg_lambda: float) -> float:
    """
    概要: L2正則化付き二値交差エントロピー損失の平滑性定数 $ L $ を計算する．BCE損失の
        1サンプル分のヘッセ行列（切片項を含む $ (d+1) $ 次元パラメータに関する）が
        $ \\frac{1}{4}\\|\\tilde{x}_i\\|^2 \\cdot I $（$ \\tilde{x}_i $ は切片項に対応する
        定数1を付加した特徴量ベクトル）で上から抑えられることを用いる．
    引数:
        X (numpy.ndarray)，形状 (N, d)．標準化済みの入力特徴量．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: L (float)．平滑性定数．
    """
    X_augmented = np.hstack([X, np.ones((X.shape[0], 1))])
    row_norms_sq = (X_augmented ** 2).sum(axis=1)
    return 0.25 * float(row_norms_sq.max()) + reg_lambda


def compute_contraction_rate(eta: float, L: float, mu: float, K: int) -> float:
    """
    概要: ASAI SVRG論文 定理1（式(30)）が定める収縮係数 $ \\rho $ を計算する．
        $ c_1 = 2\\eta(1-3\\eta L) $，$ c_2 = 18\\eta^2 L $（式(28)）を用い，
        $ \\rho = (K c_2 + 2/\\mu) / (K c_1) $ とする．
    引数:
        eta (float)．学習率 $ \\eta $（$ 0 < \\eta < 1/(3L) $ を満たす必要がある）．
        L (float)．平滑性定数．
        mu (float)．強凸性定数 $ \\mu $（本実験では $ \\mu = \\lambda $）．
        K (int)．内部ループ長．
    戻り値: rho (float)．収縮係数．
    """
    c1 = 2.0 * eta * (1.0 - 3.0 * eta * L)
    c2 = 18.0 * eta ** 2 * L
    return (K * c2 + 2.0 / mu) / (K * c1)


def solve_eta_for_target_rho(L: float, mu: float, K: int, target_rho: float) -> float:
    """
    概要: 収縮係数 $ \\rho(\\eta) $（`compute_contraction_rate`）が目標値 `target_rho` と
        一致する学習率 $ \\eta \\in (0, 1/(3L)) $ を数値的に求める．$ \\rho(\\eta) $ は
        $ \\eta \\to 0 $，$ \\eta \\to 1/(3L) $ の両極限で発散するU字形の関数であり，
        最小値を挟んで2つの解を持つ．本関数は，まず $ \\rho $ を最小化する $ \\eta $ を
        `scipy.optimize.minimize_scalar` で求め，その右側（より大きい学習率側）の区間で
        二分法（`scipy.optimize.brentq`）により解を求める．
    引数:
        L (float)．平滑性定数．
        mu (float)．強凸性定数．
        K (int)．内部ループ長．
        target_rho (float)．目標とする収縮係数．
    戻り値: eta (float)．$ \\rho(\\eta) = $ `target_rho` を満たす学習率（大きい方の解）．
    """
    eta_max = 1.0 / (3.0 * L)

    def rho(eta):
        return compute_contraction_rate(eta, L, mu, K)

    result = scipy_opt.minimize_scalar(
        rho, bounds=(eta_max * 1e-5, eta_max * (1.0 - 1e-5)), method="bounded"
    )
    if result.fun >= target_rho:
        raise ValueError(
            f"rhoの最小値（{result.fun:.4f}）が目標値（{target_rho}）を下回らないため，"
            "target_rhoを満たす学習率が存在しません．lambda_ratioを大きくしてください．"
        )

    return scipy_opt.brentq(
        lambda eta: rho(eta) - target_rho, result.x, eta_max * (1.0 - 1e-6)
    )


def compute_optimal_solution(X_train: np.ndarray, y_train: np.ndarray, reg_lambda: float):
    """
    概要: 学習用データにおける目的関数 $ f(w, b) $ の最適解 $ (w^*, b^*) $ および最適値
        $ f(w^*, b^*) $ を，scipyのL-BFGS-Bにより求める．目的関数は $ \\mu $-強凸であるため，
        大域的最適解が一意に定まる．
    引数:
        X_train (numpy.ndarray)，形状 (N, d)．標準化済みの学習用特徴量．
        y_train (numpy.ndarray)，形状 (N,)．学習用の教師信号．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値:
        x_star (numpy.ndarray)，形状 (d+1,)．最適解（末尾が切片 $ b^* $）．
        f_star (float)．最適値 $ f(w^*, b^*) $．
    """
    N, d = X_train.shape

    def objective_and_grad(wb):
        w, b = wb[:-1], wb[-1]
        z = X_train @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        p_clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
        bce = np.mean(-(y_train * np.log(p_clipped) + (1.0 - y_train) * np.log(1.0 - p_clipped)))
        reg = 0.5 * reg_lambda * (w @ w + b * b)
        grad_w = (X_train.T @ (p - y_train)) / N + reg_lambda * w
        grad_b = np.mean(p - y_train) + reg_lambda * b
        return bce + reg, np.concatenate([grad_w, [grad_b]])

    result = scipy_opt.minimize(
        objective_and_grad,
        np.zeros(d + 1),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 20000, "ftol": 1e-18, "gtol": 1e-14},
    )
    return result.x, float(result.fun)


def draw_sample(X: torch.Tensor, y: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    概要: 指定したインデックスの単一サンプルを，バッチサイズ1のミニバッチとして取り出す．
    引数:
        X (torch.Tensor)，形状 (N, d)．特徴量．
        y (torch.Tensor)，形状 (N,)．教師信号．
        index (int)．抽出するサンプルのインデックス．
    戻り値: (X_i, y_i)．形状 (1, d) と (1,) のテンソルの組．
    """
    return X[index:index + 1], y[index:index + 1]


def evaluate_point(model, X_train, y_train, X_test, y_test, reg_lambda, f_star):
    """
    概要: 与えられたモデルのパラメータにおける評価指標（真のフル勾配とそのノルムの2乗，
        目的関数の差分 $ f(z_s)-f(w^*) $，検証用データに対する分類精度）をまとめて計算する．
        本関数の計算コストは評価指標の算出のためだけに要するものであり，学習の実行時間
        （`elapsed_time`）およびオラクル呼び出し回数には計上しない．
    引数:
        model (torch.nn.Module)．評価対象のパラメータを保持するモデル．
        X_train, y_train, X_test, y_test (torch.Tensor)．学習用・検証用データ．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
        f_star (float)．最適値 $ f(w^*) $．
    戻り値:
        true_full_grad (list of torch.Tensor)．真のフル勾配 $ \\nabla f(z_s) $．
        grad_norm_sq (float)．$ \\|\\nabla f(z_s)\\|^2 $．
        objective_gap (float)．$ f(z_s) - f(w^*) $．
        accuracy (float)．検証用データに対する分類精度．
    """
    true_full_grad = compute_gradient(model, X_train, y_train, reg_lambda)
    grad_norm_sq = sum(torch.sum(g ** 2).item() for g in true_full_grad)
    objective_gap = compute_loss(model, X_train, y_train, reg_lambda) - f_star
    accuracy = compute_accuracy(model, X_test, y_test)
    return true_full_grad, grad_norm_sq, objective_gap, accuracy


def compute_approx_error(snapshot_grad: list, true_full_grad: list) -> float:
    """
    概要: フル勾配の近似誤差 $ \\|e_s\\|^2 = \\|g_s - \\nabla f(z_s)\\|^2 $ を計算する．
    引数:
        snapshot_grad (list of torch.Tensor)．スナップショット勾配 $ g_s $．
        true_full_grad (list of torch.Tensor)．真のフル勾配 $ \\nabla f(z_s) $．
    戻り値: approx_error (float)．
    """
    return sum(
        torch.sum((g_s - g_true) ** 2).item()
        for g_s, g_true in zip(snapshot_grad, true_full_grad)
    )


def run_sgd(seed, X_train, y_train, X_test, y_test, eta, reg_lambda, f_star, epochs, logger):
    """
    概要: SGD（オンライン学習，バッチサイズ1）による学習を実行し，各エポック終了時の評価指標を
        `logger` に記録する．データはエポックごとにランダムリシャッフルし1巡する．
    引数:
        seed (int)．乱数シード．
        X_train, y_train, X_test, y_test (torch.Tensor)．学習用・検証用データ．
        eta (float)．学習率 $ \\eta $．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
        f_star (float)．最適値 $ f(w^*) $．
        epochs (int)．エポック数．
        logger (ResultLogger)．評価指標を記録するロガー．
    戻り値: なし
    """
    N_train = X_train.shape[0]
    input_dim = X_train.shape[1]
    model = load_model(seed=seed, input_dim=input_dim)
    optimizer = SGD(model.parameters(), lr=eta)
    rng = np.random.default_rng(seed)

    oracle_calls = 0
    elapsed_time = 0.0

    _, grad_norm_sq, objective_gap, accuracy = evaluate_point(
        model, X_train, y_train, X_test, y_test, reg_lambda, f_star
    )
    logger(0, 0, 0.0, 0.0, objective_gap, grad_norm_sq, accuracy, float("nan"))

    for epoch in tqdm(range(1, epochs + 1), desc=f"SGD seed={seed}", leave=False):
        permutation = rng.permutation(N_train)

        start_time = time.perf_counter()
        for index in permutation:
            X_i, y_i = draw_sample(X_train, y_train, int(index))
            compute_gradient(model, X_i, y_i, reg_lambda)
            optimizer.step()
        elapsed_time += time.perf_counter() - start_time
        oracle_calls += N_train

        _, grad_norm_sq, objective_gap, accuracy = evaluate_point(
            model, X_train, y_train, X_test, y_test, reg_lambda, f_star
        )
        logger(
            epoch, oracle_calls, oracle_calls / N_train, elapsed_time,
            objective_gap, grad_norm_sq, accuracy, float("nan"),
        )


def run_variance_reduced(
    method, seed, X_train, y_train, X_test, y_test, eta, reg_lambda, f_star, epochs, logger
):
    """
    概要: SVRG系手法（SVRG，NFG SVRG，ASAI SVRG）による学習を実行し，各エポック終了時の評価
        指標を `logger` に記録する．外部ループ・内部ループ構造は，`programs/optimizers/` の
        対応するOptimizerクラスの `begin_epoch`／`step`／`end_epoch` を用いて実行する．
        内部ループはエポックごとのランダムリシャッフルにより学習用データを1巡するため，
        内部ループ長は $ K = N_{\\text{train}} $ である．
    引数:
        method (str)．"SVRG"，"NFG_SVRG"，"ASAI_SVRG" のいずれか．
        seed (int)．乱数シード．
        X_train, y_train, X_test, y_test (torch.Tensor)．学習用・検証用データ．
        eta (float)．学習率 $ \\eta $．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
        f_star (float)．最適値 $ f(w^*) $．
        epochs (int)．外部ループ数 $ S $．
        logger (ResultLogger)．評価指標を記録するロガー．
    戻り値: なし
    """
    N_train = X_train.shape[0]
    input_dim = X_train.shape[1]
    model = load_model(seed=seed, input_dim=input_dim)
    # z_0 <- w_0．snapshot_modelはmodelと同じseedで初期化することで同一の初期値を持つ．
    snapshot_model = load_model(seed=seed, input_dim=input_dim)

    OptimizerClass = _VARIANCE_REDUCED_OPTIMIZER_CLASSES[method]
    optimizer = OptimizerClass(model.parameters(), lr=eta, K=N_train)
    rng = np.random.default_rng(seed)

    oracle_calls = 0
    elapsed_time = 0.0

    true_full_grad, grad_norm_sq, objective_gap, accuracy = evaluate_point(
        snapshot_model, X_train, y_train, X_test, y_test, reg_lambda, f_star
    )
    if method == "SVRG":
        start_time = time.perf_counter()
        snapshot_grad = compute_gradient(snapshot_model, X_train, y_train, reg_lambda)
        initial_full_grad_time = time.perf_counter() - start_time
        optimizer.set_snapshot_gradient(snapshot_grad)
    else:
        snapshot_grad = optimizer.get_snapshot_gradient()
        initial_full_grad_time = 0.0

    logger(
        0, 0, 0.0, 0.0, objective_gap, grad_norm_sq, accuracy,
        compute_approx_error(snapshot_grad, true_full_grad),
    )

    elapsed_time += initial_full_grad_time
    if method == "SVRG":
        oracle_calls += N_train

    for epoch in tqdm(range(1, epochs + 1), desc=f"{method} seed={seed}", leave=False):
        permutation = rng.permutation(N_train)

        start_time = time.perf_counter()
        optimizer.begin_epoch(rng)
        for index in permutation:
            X_i, y_i = draw_sample(X_train, y_train, int(index))
            compute_gradient(model, X_i, y_i, reg_lambda)
            grad_at_snapshot = compute_gradient(snapshot_model, X_i, y_i, reg_lambda)
            optimizer.step(grad_at_snapshot)

        optimizer.end_epoch()
        set_model_params(snapshot_model, optimizer.get_snapshot_params(), source_model=model)

        if method == "SVRG":
            snapshot_grad = compute_gradient(snapshot_model, X_train, y_train, reg_lambda)
            optimizer.set_snapshot_gradient(snapshot_grad)
        else:
            snapshot_grad = optimizer.get_snapshot_gradient()
        elapsed_time += time.perf_counter() - start_time

        oracle_calls += 2 * N_train
        if method == "SVRG":
            oracle_calls += N_train

        true_full_grad, grad_norm_sq, objective_gap, accuracy = evaluate_point(
            snapshot_model, X_train, y_train, X_test, y_test, reg_lambda, f_star
        )
        logger(
            epoch, oracle_calls, oracle_calls / N_train, elapsed_time,
            objective_gap, grad_norm_sq, accuracy,
            compute_approx_error(snapshot_grad, true_full_grad),
        )


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
    概要: 1つの (手法, ハイパーパラメータ, Seed) の組に対する学習を実行し，結果を保存する．
        すでに正常終了した結果が存在する場合は学習をスキップする．
    引数: args (tuple)．(method, seed, hyperparams) のタプル．
    戻り値: なし
    """
    method, seed, hyperparams = args
    torch.set_num_threads(1)

    eta = hyperparams["eta"]
    reg_lambda = hyperparams["reg_lambda"]
    hp_name = hyperparams["hp_name"]
    epochs = EPOCHS

    target_dir = os.path.join(OUTPUT_ROOT, method, hp_name, str(seed))
    if is_run_completed(target_dir, epochs):
        print(f"[skip] {method}/{hp_name}/{seed} は既に完了しています．", flush=True)
        return

    os.makedirs(target_dir, exist_ok=True)

    set_seed(seed)
    train_dataloader, test_dataloader = load_dataloader(seed=seed, batch_size=BATCH_SIZE)
    X_train, y_train = train_dataloader.dataset.tensors
    X_test, y_test = test_dataloader.dataset.tensors

    _, f_star = compute_optimal_solution(X_train.numpy(), y_train.numpy(), reg_lambda)

    logger = ResultLogger()
    logger.set_names(*METRIC_NAMES)

    if method == "SGD":
        run_sgd(seed, X_train, y_train, X_test, y_test, eta, reg_lambda, f_star, epochs, logger)
    else:
        run_variance_reduced(
            method, seed, X_train, y_train, X_test, y_test, eta, reg_lambda, f_star, epochs, logger
        )

    logger.save(os.path.join(target_dir, "log.json"))

    config = {
        "experiment": EXPERIMENT_NAME,
        "method": method,
        "seed": seed,
        "eta": eta,
        "eta_rule": hyperparams["eta_rule"],
        "eta_description": hyperparams["eta_description"],
        "reg_lambda": reg_lambda,
        "lambda_ratio": LAMBDA_RATIO,
        "K": X_train.shape[0],
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "sampling": "random_reshuffle",
        "L": hyperparams["L"],
        "mu": reg_lambda,
        "condition_number": reg_lambda / hyperparams["L"],
        "rho": hyperparams["rho"],
        "N_train": X_train.shape[0],
        "N_test": X_test.shape[0],
        "d": X_train.shape[1],
        "f_star": f_star,
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{hp_name}/{seed}", flush=True)


def build_hyperparameter_list() -> list:
    """
    概要: 全手法・全Seedで共通して用いる学習率の候補を構築する．平滑性定数 $ L $，
        正則化係数 $ \\lambda $（`LAMBDA_RATIO` と $ L_0 $ から決定），および3つの学習率
        （モジュールdocstring参照）を計算する．
    引数: なし
    戻り値: hyperparameter_list (list of dict)．eta，eta_rule，eta_description，reg_lambda，
        hp_name，L，rho を含む辞書のリスト．
    """
    train_dataloader, _ = load_dataloader(seed=0, batch_size=BATCH_SIZE)
    X_train, _ = train_dataloader.dataset.tensors
    X_train_np = X_train.numpy()
    N_train = X_train_np.shape[0]

    L0 = compute_smoothness_constant(X_train_np, reg_lambda=0.0)
    reg_lambda = LAMBDA_RATIO * L0
    L = L0 + reg_lambda
    mu = reg_lambda

    eta_max = 1.0 / (3.0 * L)
    eta_a = ETA_A_MARGIN * eta_max
    eta_b = solve_eta_for_target_rho(L, mu, N_train, TARGET_RHO)
    eta_c = 1.0 / (20.0 * L)

    rules = [
        ("eta_a_near_upper_bound", eta_a, "0.99/(3L)．補題3の上界近傍"),
        ("eta_b_rho=0.5", eta_b, f"rho(eta)={TARGET_RHO}を満たす学習率（数値的に逆算，大きい方の解）"),
        ("eta_c=1/(20L)", eta_c, "実験0で安定した収束を示した学習率規則の踏襲"),
    ]

    return [
        {
            "eta": eta,
            "eta_rule": rule,
            "eta_description": description,
            "reg_lambda": reg_lambda,
            "hp_name": f"eta{eta:.6e}_lambda{reg_lambda:.6e}_K{N_train}_epochs{EPOCHS}",
            "L": L,
            "rho": compute_contraction_rate(eta, L, mu, N_train),
        }
        for rule, eta, description in rules
    ]


def main():
    """
    概要: 実験1の全条件（4手法 × 3学習率 × 5Seed = 60条件）をマルチプロセスで並列に学習する．
    引数: なし
    戻り値: なし
    """
    hyperparameter_list = build_hyperparameter_list()
    print(
        f"L = {hyperparameter_list[0]['L']:.6f}, "
        f"lambda = {hyperparameter_list[0]['reg_lambda']:.6f}, "
        f"kappa = mu/L = {hyperparameter_list[0]['reg_lambda']/hyperparameter_list[0]['L']:.6f}, "
        f"epochs = {EPOCHS}"
    )
    for hyperparams in hyperparameter_list:
        print(
            f"  eta = {hyperparams['eta']:.6e}  rho = {hyperparams['rho']:.4f}  "
            f"({hyperparams['eta_rule']})"
        )

    tasks = [
        (method, seed, hyperparams)
        for method, hyperparams, seed in itertools.product(METHODS, hyperparameter_list, SEEDS)
    ]

    # `.reports/report_020.md` 5.6節で判明した通り，本マシンは32物理コア／64論理スレッド
    # （2-way SMT）である．論理スレッド数（`os.cpu_count()`）まで並列化すると物理コア数を
    # 超過し，SMT・メモリ帯域の競合により実行時間の比較（手法間・バッチ間）が不安定になる．
    # そのため物理コア数相当（論理スレッド数の半分）に並列数を制限し，全条件を同一の
    # 並列度・競合条件下で実行することで，実行時間の比較可能性を確保する．
    physical_cores = max(1, (os.cpu_count() or 2) // 2)
    num_workers = min(len(tasks), physical_cores)
    print(f"並列プロセス数: {num_workers}（総タスク数: {len(tasks)}，物理コア数相当に制限）")

    with Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks)

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    main()
