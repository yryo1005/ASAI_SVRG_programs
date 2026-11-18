"""
E1（Mushroomにおける $ \\rho $ の異なる学習率での収束挙動とサンプリング方式の影響）の
実験を実行するスクリプト．

条件は実験計画（`実験計画.md` 3章）および `実験指示E1.md` に基づく．

| 項目 | 設定 |
|---|---|
| データ | Mushroom（$ N = 8124 $，22特徴量の順序符号化，9:1に分割，学習データの統計量で標準化） |
| 目的関数 | $ f({\\mathbf w}, b) = \\frac{1}{N}\\sum_{n}\\ell_{\\mathrm{BCE}} + \\frac{\\lambda}{2}(\\|{\\mathbf w}\\|^{2} + b^{2}) $ |
| 定数 | $ L = \\frac{1}{4}\\max_{n}\\|\\tilde{\\mathbf x}_{n}\\|^{2} + \\lambda $，$ \\lambda = \\mu = 0.03 L_{0} $，$ K = N_{\\mathrm{train}} $，ミニバッチサイズ1 |
| 学習率 | $ \\rho \\in \\{0.5, 0.64, 0.8, 1.0, 2.0, 10, 298.4\\} $ に対応する $ \\eta $（大きい側の解） |
| エポック数 | 100 |
| 手法 | SGD，SVRG（最終点，`SVRGFinalPoint`），NFG-SVRG（最終点，`NFGSVRGFinalPoint`），ASA-SVRG（`ASAISVRG`） |
| サンプリング | `random_reshuffle`（全7 $ \\rho $），`uniform_with_replacement`（$ \\rho \\in \\{0.5, 1.0, 298.4\\} $ のみ） |
| `inner_start` | 既定は `"snapshot"`（論文のAlgorithm通り）．`"last_iterate"` は既存実装との一致確認にのみ使用する |
| シード | 0〜4 |

$ \\rho = 298.4 $ は $ \\eta = 0.99/(3L) $（論文の図1）に対応するため，この条件だけは二分法ではなく
直接この式で $ \\eta $ を計算する（既存の結果と同一の値にするため）．他の条件は，E0と同じ
`solve_eta_for_target_rho`（大きい側の解）で $ \\eta $ を求める．

`uniform_with_replacement` では，1エポック分の添字を `rng.integers(0, N_train, size=N_train)` で
まとめて生成する．生成するタイミングは `random_reshuffle` の `rng.permutation` と同じ位置とし，
乱数の消費の順序を揃える（`get_epoch_indices` 関数）．

学習ループの構造はE0（`ex001_e0_mushroom_logistic/train.py`）を踏襲する．SGDは
`programs/ex001_mushroom_logistic/train.py` の `run_sgd` を移植し，同じ2つのサンプリング方式と
打切り条件（目的関数誤差が非有限になった外側ループで打切り）を追加した．

実行例:
    python train.py
    python train.py --rhos 0.5 1.0 298.4 --samplings random_reshuffle --methods SGD --seeds 0
    python train.py --inner-starts last_iterate --rhos 298.4 --samplings random_reshuffle \\
        --methods SVRG NFG_SVRG ASAI_SVRG --seeds 0 --output-root ../../outputs/_verify/ex006
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
from optimizers import ASAISVRG, NFGSVRGFinalPoint, SGD, SVRGFinalPoint  # noqa: E402

from data import load_dataloader  # noqa: E402
from model import compute_accuracy, compute_gradient, compute_loss, load_model, set_model_params  # noqa: E402

EXPERIMENT_NAME = "ex006_e1_mushroom_rho_sampling"
OUTPUT_ROOT = os.path.join(_REVIEW_ROOT, "outputs", EXPERIMENT_NAME)

INNER_STARTS = ("last_iterate", "snapshot")
DEFAULT_INNER_STARTS = ["snapshot"]
SEEDS = [0, 1, 2, 3, 4]
ALL_METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
ALL_SAMPLINGS = ("random_reshuffle", "uniform_with_replacement")
RHO_CONDITIONS = [0.5, 0.64, 0.8, 1.0, 2.0, 10.0, 298.4]
UNIFORM_WITH_REPLACEMENT_RHOS = [0.5, 1.0, 298.4]
EPOCHS = 100
BATCH_SIZE = 1
LAMBDA_RATIO = 0.03
ETA_A_MARGIN = 0.99
RHO_298_4 = 298.4

METRIC_NAMES = (
    "epoch",
    "oracle_calls",
    "full_grad_computations",
    "elapsed_time",
    "objective_gap",
    "grad_norm_sq",
    "accuracy",
    "approx_error",
    "test_correct",
    "full_grad_norm_sq",
    "approx_cos",
    "terminated",
)

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "SVRG": SVRGFinalPoint,
    "NFG_SVRG": NFGSVRGFinalPoint,
    "ASAI_SVRG": ASAISVRG,
}


def compute_smoothness_constant(X: np.ndarray, reg_lambda: float) -> float:
    """
    概要: 正則化付きBCE損失の平滑性定数 $ L = \\frac{1}{4}\\max_{n}\\|\\tilde{\\mathbf x}_{n}\\|^{2} + \\lambda $ を計算する．
    引数:
        X (numpy.ndarray)，形状 (N, d)．標準化済みの特徴量．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: L (float)．
    """
    X_augmented = np.hstack([X, np.ones((X.shape[0], 1))])
    return 0.25 * float((X_augmented ** 2).sum(axis=1).max()) + reg_lambda


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


def solve_eta_for_target_rho(L: float, mu: float, K: int, target_rho: float) -> float:
    """
    概要: $ \\rho(\\eta) $ = `target_rho` となる学習率のうち大きい方の解を，E0と同じ方法
        （最小点を `minimize_scalar` で求め，その右側で `brentq`）で求める．
    引数:
        L (float)．平滑性定数．
        mu (float)．強凸性定数．
        K (int)．内側ループ長．
        target_rho (float)．目標の縮小率．
    戻り値: eta (float)．
    """
    eta_max = 1.0 / (3.0 * L)

    def rho(eta):
        return compute_contraction_rate(eta, L, mu, K)

    result = scipy_opt.minimize_scalar(rho, bounds=(eta_max * 1e-5, eta_max * (1.0 - 1e-5)), method="bounded")
    if result.fun >= target_rho:
        raise ValueError(f"rhoの最小値（{result.fun:.4f}）が目標値（{target_rho}）以上である．")
    return scipy_opt.brentq(lambda eta: rho(eta) - target_rho, result.x, eta_max * (1.0 - 1e-6))


RHO_0_64 = 0.64


def solve_eta_for_rho_condition(L: float, mu: float, K: int, target_rho: float) -> float:
    """
    概要: `target_rho` に対応する学習率 $ \\eta $ を求める．$ \\rho = 298.4 $ と $ \\rho = 0.64 $ は，
        既存の結果（E0のeta_a・eta_cの各規則）と同一の値にするため，それぞれ $ \\eta = 0.99/(3L) $，
        $ \\eta = 1/(20L) $ を直接用いる（実験計画.md 3.3節の表の注記）．他の条件は二分法で解く．
    引数:
        L (float)．平滑性定数．
        mu (float)．強凸性定数．
        K (int)．内側ループ長．
        target_rho (float)．目標の縮小率．
    戻り値: eta (float)．
    """
    if target_rho == RHO_298_4:
        return ETA_A_MARGIN / (3.0 * L)
    if target_rho == RHO_0_64:
        return 1.0 / (20.0 * L)
    return solve_eta_for_target_rho(L, mu, K, target_rho)


def compute_optimal_solution(X_train: np.ndarray, y_train: np.ndarray, reg_lambda: float):
    """
    概要: 最適解と最適値 $ f({\\mathbf w}^{*}) $ をL-BFGS-Bで求める（E0と同一）．
    引数:
        X_train (numpy.ndarray)，形状 (N, d)．標準化済みの特徴量．
        y_train (numpy.ndarray)，形状 (N,)．教師信号．
        reg_lambda (float)．L2正則化係数．
    戻り値:
        x_star (numpy.ndarray)，形状 (d+1,)．最適解（末尾が切片）．
        f_star (float)．最適値．
    """
    N, d = X_train.shape

    def objective_and_grad(wb):
        w, b = wb[:-1], wb[-1]
        p = 1.0 / (1.0 + np.exp(-(X_train @ w + b)))
        p_clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
        bce = np.mean(-(y_train * np.log(p_clipped) + (1.0 - y_train) * np.log(1.0 - p_clipped)))
        reg = 0.5 * reg_lambda * (w @ w + b * b)
        grad_w = (X_train.T @ (p - y_train)) / N + reg_lambda * w
        grad_b = np.mean(p - y_train) + reg_lambda * b
        return bce + reg, np.concatenate([grad_w, [grad_b]])

    result = scipy_opt.minimize(objective_and_grad, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                                options={"maxiter": 20000, "ftol": 1e-18, "gtol": 1e-14})
    return result.x, float(result.fun)


def evaluate_point(model, X_train, y_train, X_test, y_test, reg_lambda, f_star):
    """
    概要: 与えられたパラメータにおける全勾配，そのノルムの2乗，目的関数誤差，テスト精度，
        正解数を計算する．
    引数:
        model (torch.nn.Module)．評価するパラメータを保持するモデル．
        X_train, y_train, X_test, y_test (torch.Tensor)．学習・テストデータ．
        reg_lambda (float)．L2正則化係数．
        f_star (float)．最適値．
    戻り値:
        true_full_grad (list of torch.Tensor)．$ \\nabla f({\\mathbf z}_{s}) $．
        grad_norm_sq (float)．$ \\|\\nabla f({\\mathbf z}_{s})\\|^{2} $．
        objective_gap (float)．$ f({\\mathbf z}_{s}) - f({\\mathbf w}^{*}) $．
        accuracy (float)．テスト精度．
        test_correct (int)．テストデータの正解数．
    """
    true_full_grad = compute_gradient(model, X_train, y_train, reg_lambda)
    grad_norm_sq = sum(torch.sum(g ** 2).item() for g in true_full_grad)
    objective_gap = compute_loss(model, X_train, y_train, reg_lambda) - f_star
    accuracy = compute_accuracy(model, X_test, y_test)
    with torch.no_grad():
        pred = (torch.sigmoid(model(X_test)) >= 0.5).to(y_test.dtype)
        test_correct = int((pred == y_test).sum().item())
    return true_full_grad, grad_norm_sq, objective_gap, accuracy, test_correct


def compute_approx_statistics(snapshot_grad: list, true_full_grad: list) -> tuple:
    """
    概要: 近似誤差 $ \\|{\\mathbf e}_{s}\\|^{2} $（E0と同じ計算）と，$ {\\mathbf g}_{s} $ と
        $ \\nabla f({\\mathbf z}_{s}) $ のコサイン類似度を計算する．
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


def get_epoch_indices(rng: np.random.Generator, N_train: int, sampling: str) -> np.ndarray:
    """
    概要: 1エポック分のデータの添字列を，指定したサンプリング方式で生成する．乱数の消費を
        `random_reshuffle` と `uniform_with_replacement` とで同じ位置（外側ループの最初）に揃える．
    引数:
        rng (numpy.random.Generator)．乱数生成器．
        N_train (int)．学習データ数（$ K $ と同じ）．
        sampling (str)．"random_reshuffle" または "uniform_with_replacement"．
    戻り値: indices (numpy.ndarray)，形状 (N_train,)．int型．
    """
    if sampling == "random_reshuffle":
        return rng.permutation(N_train)
    if sampling == "uniform_with_replacement":
        return rng.integers(0, N_train, size=N_train)
    raise ValueError(f"未知のサンプリング方式: {sampling}")


def run_inner_loop(model, snapshot_model, optimizer, X_train, y_train, reg_lambda, indices) -> None:
    """
    概要: 1回の内側ループ（学習データを $ K = N_{\\mathrm{train}} $ 件，ミニバッチサイズ1）を実行する．
    引数:
        model (torch.nn.Module)．現在のパラメータ $ {\\mathbf w}_{s}^{k} $ を保持するモデル．
        snapshot_model (torch.nn.Module)．スナップショット $ {\\mathbf z}_{s} $ を保持するモデル．
        optimizer (torch.optim.Optimizer)．SVRG系手法のOptimizer．
        X_train, y_train (torch.Tensor)．学習データ．
        reg_lambda (float)．L2正則化係数．
        indices (numpy.ndarray)，形状 (N_train,)．データを用いる順序（重複を含み得る）．
    戻り値: なし
    """
    for index in indices:
        i = int(index)
        X_i, y_i = X_train[i:i + 1], y_train[i:i + 1]
        compute_gradient(model, X_i, y_i, reg_lambda)
        grad_at_snapshot = compute_gradient(snapshot_model, X_i, y_i, reg_lambda)
        optimizer.step(grad_at_snapshot)


def run_variance_reduced(method, inner_start, sampling, seed, data, eta, reg_lambda, f_star, epochs,
                         logger) -> None:
    """
    概要: SVRG系手法（SVRG，NFG-SVRG，ASA-SVRG）で学習し，各外側ループの終了時の評価指標を
        `logger` に記録する．E0と同じ順序で乱数を消費する（各エポックで `get_epoch_indices` の後に
        `begin_epoch(rng)`）．目的関数誤差が非有限になった外側ループで打ち切る．
    引数:
        method (str)．"SVRG"，"NFG_SVRG"，"ASAI_SVRG" のいずれか．
        inner_start (str)．"last_iterate" または "snapshot"．
        sampling (str)．"random_reshuffle" または "uniform_with_replacement"．
        seed (int)．乱数シード．
        data (tuple)．(X_train, y_train, X_test, y_test) の組．
        eta (float)．学習率．
        reg_lambda (float)．L2正則化係数．
        f_star (float)．最適値．
        epochs (int)．外側ループ数．
        logger (ResultLogger)．`METRIC_NAMES` を登録したロガー．
    戻り値: なし
    """
    X_train, y_train, X_test, y_test = data
    N_train, input_dim = X_train.shape
    model = load_model(seed=seed, input_dim=input_dim)
    snapshot_model = load_model(seed=seed, input_dim=input_dim)
    optimizer = _VARIANCE_REDUCED_OPTIMIZER_CLASSES[method](model.parameters(), lr=eta, K=N_train)
    rng = np.random.default_rng(seed)

    true_full_grad, grad_norm_sq, objective_gap, accuracy, test_correct = evaluate_point(
        snapshot_model, X_train, y_train, X_test, y_test, reg_lambda, f_star)
    if method == "SVRG":
        start_time = time.perf_counter()
        snapshot_grad = compute_gradient(snapshot_model, X_train, y_train, reg_lambda)
        elapsed_time = time.perf_counter() - start_time
        optimizer.set_snapshot_gradient(snapshot_grad)
    else:
        snapshot_grad = optimizer.get_snapshot_gradient()
        elapsed_time = 0.0
    approx_error, approx_cos = compute_approx_statistics(snapshot_grad, true_full_grad)
    logger(0, 0, 0.0, 0.0, objective_gap, grad_norm_sq, accuracy, approx_error, test_correct, grad_norm_sq,
           approx_cos, False)
    oracle_calls = N_train if method == "SVRG" else 0

    for epoch in tqdm(range(1, epochs + 1), desc=f"{method} {inner_start} {sampling} seed={seed}", leave=False):
        indices = get_epoch_indices(rng, N_train, sampling)
        start_time = time.perf_counter()
        optimizer.begin_epoch(rng)
        run_inner_loop(model, snapshot_model, optimizer, X_train, y_train, reg_lambda, indices)
        optimizer.end_epoch()
        set_model_params(snapshot_model, optimizer.get_snapshot_params(), source_model=model)
        if inner_start == "snapshot":
            # 論文のAlgorithm：次の内側ループを w_s^0 <- z_s から開始する．
            set_model_params(model, optimizer.get_snapshot_params(), source_model=model)
        if method == "SVRG":
            snapshot_grad = compute_gradient(snapshot_model, X_train, y_train, reg_lambda)
            optimizer.set_snapshot_gradient(snapshot_grad)
        else:
            snapshot_grad = optimizer.get_snapshot_gradient()
        elapsed_time += time.perf_counter() - start_time
        oracle_calls += 3 * N_train if method == "SVRG" else 2 * N_train

        true_full_grad, grad_norm_sq, objective_gap, accuracy, test_correct = evaluate_point(
            snapshot_model, X_train, y_train, X_test, y_test, reg_lambda, f_star)
        approx_error, approx_cos = compute_approx_statistics(snapshot_grad, true_full_grad)
        terminated = not np.isfinite(objective_gap)
        logger(epoch, oracle_calls, oracle_calls / N_train, elapsed_time, objective_gap, grad_norm_sq, accuracy,
               approx_error, test_correct, grad_norm_sq, approx_cos, terminated)
        if terminated:
            break


def run_sgd(sampling, seed, data, eta, reg_lambda, f_star, epochs, logger) -> None:
    """
    概要: SGD（オンライン学習，バッチサイズ1）で学習し，各エポック終了時の評価指標を `logger` に
        記録する．SVRG系固有の項目（`approx_error`，`approx_cos`）はNaNとする．目的関数誤差が
        非有限になった外側ループで打ち切る．
    引数:
        sampling (str)．"random_reshuffle" または "uniform_with_replacement"．
        seed (int)．乱数シード．
        data (tuple)．(X_train, y_train, X_test, y_test) の組．
        eta (float)．学習率．
        reg_lambda (float)．L2正則化係数．
        f_star (float)．最適値．
        epochs (int)．エポック数．
        logger (ResultLogger)．`METRIC_NAMES` を登録したロガー．
    戻り値: なし
    """
    X_train, y_train, X_test, y_test = data
    N_train, input_dim = X_train.shape
    model = load_model(seed=seed, input_dim=input_dim)
    optimizer = SGD(model.parameters(), lr=eta)
    rng = np.random.default_rng(seed)

    oracle_calls = 0
    elapsed_time = 0.0

    _, grad_norm_sq, objective_gap, accuracy, test_correct = evaluate_point(
        model, X_train, y_train, X_test, y_test, reg_lambda, f_star)
    logger(0, 0, 0.0, 0.0, objective_gap, grad_norm_sq, accuracy, float("nan"), test_correct, grad_norm_sq,
           float("nan"), False)

    for epoch in tqdm(range(1, epochs + 1), desc=f"SGD {sampling} seed={seed}", leave=False):
        indices = get_epoch_indices(rng, N_train, sampling)

        start_time = time.perf_counter()
        for index in indices:
            i = int(index)
            X_i, y_i = X_train[i:i + 1], y_train[i:i + 1]
            compute_gradient(model, X_i, y_i, reg_lambda)
            optimizer.step()
        elapsed_time += time.perf_counter() - start_time
        oracle_calls += N_train

        _, grad_norm_sq, objective_gap, accuracy, test_correct = evaluate_point(
            model, X_train, y_train, X_test, y_test, reg_lambda, f_star)
        terminated = not np.isfinite(objective_gap)
        logger(epoch, oracle_calls, oracle_calls / N_train, elapsed_time, objective_gap, grad_norm_sq, accuracy,
               float("nan"), test_correct, grad_norm_sq, float("nan"), terminated)
        if terminated:
            break


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


def run_single_experiment(task: dict) -> None:
    """
    概要: 1つの (手法, $ \\rho $, サンプリング方式, 内側ループの開始点, シード) の組について
        学習し，結果を保存する．
    引数: task (dict)．`build_tasks` が返すタスク．
    戻り値: なし
    """
    target_dir, hyperparams, seed = task["target_dir"], task["hyperparams"], task["seed"]
    if is_run_completed(target_dir, task["epochs"]):
        print(f"[skip] {target_dir}", flush=True)
        return
    os.makedirs(target_dir, exist_ok=True)
    torch.set_num_threads(1)

    set_seed(seed)
    train_dataloader, test_dataloader = load_dataloader(seed=seed, batch_size=BATCH_SIZE)
    X_train, y_train = train_dataloader.dataset.tensors
    X_test, y_test = test_dataloader.dataset.tensors
    reg_lambda = hyperparams["reg_lambda"]
    _, f_star = compute_optimal_solution(X_train.numpy(), y_train.numpy(), reg_lambda)

    logger = ResultLogger()
    logger.set_names(*METRIC_NAMES)
    method, sampling = task["method"], task["sampling"]
    if method == "SGD":
        run_sgd(sampling, seed, (X_train, y_train, X_test, y_test), hyperparams["eta"], reg_lambda, f_star,
               task["epochs"], logger)
    else:
        run_variance_reduced(method, task["inner_start"], sampling, seed, (X_train, y_train, X_test, y_test),
                             hyperparams["eta"], reg_lambda, f_star, task["epochs"], logger)
    logger.save(os.path.join(target_dir, "log.json"))

    config = {
        "experiment": EXPERIMENT_NAME,
        "method": method,
        "inner_start": task["inner_start"] if method != "SGD" else None,
        "sampling": sampling,
        "seed": seed,
        "eta": hyperparams["eta"],
        "rho": hyperparams["rho"],
        "target_rho": hyperparams["target_rho"],
        "reg_lambda": reg_lambda,
        "lambda_ratio": LAMBDA_RATIO,
        "K": X_train.shape[0],
        "epochs": task["epochs"],
        "batch_size": BATCH_SIZE,
        "evaluation_point": "snapshot z_s" if method != "SGD" else "current point w_k",
        "L": hyperparams["L"],
        "mu": reg_lambda,
        "N_train": X_train.shape[0],
        "N_test": X_test.shape[0],
        "d": X_train.shape[1],
        "f_star": f_star,
        "terminated": bool(logger["terminated"][-1]),
        "torch": torch.__version__,
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"[done] {target_dir}", flush=True)


def build_hyperparameters(rhos: list) -> dict:
    """
    概要: 指定した $ \\rho $ の各値について，対応する学習率 $ \\eta $ と定数（$ L $，$ \\lambda $）を
        計算する．$ L $，$ \\lambda $ はE0と同一の方法で求める（$ \\lambda = \\mu = 0.03 L_{0} $）．
    引数: rhos (list of float)．目標の縮小率 $ \\rho $ の値．
    戻り値: hyperparameters (dict)．キーが `target_rho`，値が eta，rho（$ \\eta $ から計算し直した値），
        target_rho，reg_lambda，L，hp_name（$ \\rho $ を除く部分）を含む辞書．
    """
    train_dataloader, _ = load_dataloader(seed=0, batch_size=BATCH_SIZE)
    X_train = train_dataloader.dataset.tensors[0].numpy()
    N_train = X_train.shape[0]
    L0 = compute_smoothness_constant(X_train, reg_lambda=0.0)
    reg_lambda = LAMBDA_RATIO * L0
    L = L0 + reg_lambda
    hyperparameters = {}
    for target_rho in rhos:
        eta = solve_eta_for_rho_condition(L, reg_lambda, N_train, target_rho)
        hyperparameters[target_rho] = {
            "eta": eta,
            "rho": compute_contraction_rate(eta, L, reg_lambda, N_train),
            "target_rho": target_rho,
            "reg_lambda": reg_lambda,
            "hp_name": f"rho{target_rho}_eta{eta:.6e}_lambda{reg_lambda:.6e}_K{N_train}_epochs{{epochs}}",
            "L": L,
        }
    return hyperparameters


def build_tasks(methods, samplings, rhos, inner_starts, seeds, epochs, output_root) -> list:
    """
    概要: 全ての (手法, サンプリング方式, $ \\rho $, 内側ループの開始点, シード) の組のタスクを
        構築する．`uniform_with_replacement` は `UNIFORM_WITH_REPLACEMENT_RHOS` に含まれる
        $ \\rho $ のみに限定する．SGDは `inner_start` を持たないため，ディレクトリ名の該当部分を
        `"none"` とする．
    引数:
        methods (list of str)．手法名（"SGD" を含む）．
        samplings (list of str)．サンプリング方式．
        rhos (list of float)．目標の縮小率 $ \\rho $ の値．
        inner_starts (list of str)．内側ループの開始点（SGD以外に適用）．
        seeds (list of int)．シード．
        epochs (int)．外側ループ数．
        output_root (str)．結果の保存先のルート．
    戻り値: tasks (list of dict)．
    """
    hyperparameters = build_hyperparameters(rhos)
    tasks = []
    for method, sampling in itertools.product(methods, samplings):
        allowed_rhos = rhos if sampling == "random_reshuffle" else \
            [r for r in rhos if r in UNIFORM_WITH_REPLACEMENT_RHOS]
        method_inner_starts = ["none"] if method == "SGD" else inner_starts
        for target_rho, inner_start, seed in itertools.product(allowed_rhos, method_inner_starts, seeds):
            hyperparams = hyperparameters[target_rho]
            name = f"inner_{inner_start}_{sampling}_" + hyperparams["hp_name"].format(epochs=epochs)
            tasks.append({
                "method": method, "sampling": sampling, "inner_start": inner_start, "seed": seed,
                "epochs": epochs, "hyperparams": hyperparams,
                "target_dir": os.path.join(output_root, method, name, str(seed)),
            })
    return tasks


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--samplings", nargs="+", default=list(ALL_SAMPLINGS), choices=ALL_SAMPLINGS)
    parser.add_argument("--rhos", nargs="+", type=float, default=RHO_CONDITIONS)
    parser.add_argument("--inner-starts", nargs="+", default=DEFAULT_INNER_STARTS, choices=INNER_STARTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    """
    概要: タスクを構築し，マルチプロセスで並列に学習する．
    引数: なし
    戻り値: なし
    """
    args = parse_args()
    tasks = build_tasks(args.methods, args.samplings, args.rhos, args.inner_starts, args.seeds, args.epochs,
                        os.path.abspath(args.output_root))
    if tasks:
        h = tasks[0]["hyperparams"]
        print(f"L = {h['L']:.6f}, lambda = {h['reg_lambda']:.6f}", flush=True)
    num_workers = max(1, min(args.workers, len(tasks)))
    print(f"タスク数: {len(tasks)}，並列プロセス数: {num_workers}", flush=True)
    with Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks, chunksize=1)
    print("全ての学習が終了しました．", flush=True)


if __name__ == "__main__":
    main()
