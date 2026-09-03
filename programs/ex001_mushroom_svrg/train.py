"""
論文4.1節（マッシュルームデータセットの二値分類問題）の数値実験を実行するスクリプト．

比較手法はSGD，SVRG，NFG SVRG，ASAI SVRG（提案手法）の4手法．各手法についてSeed値を5種類
（0〜4）で検証し，論文で指定される評価指標
    - 勾配計算回数（オラクル呼び出し回数，#grad/N），エポック数，経過時間（wall-clock time）
    - 目的関数の誤差 f(z_s) - f(w*)，分類精度（Accuracy），フル勾配の近似誤差 ||e_s||^2
を記録する．

SVRG，NFG SVRG，ASAI SVRGは外部ループ・内部ループから成る特有の反復構造（論文Algorithm 1-4）を
持ち，ミニバッチのDataLoaderを毎エポック順に消費する通常の教師あり学習の学習ループとは根本的に
構造が異なる．そのため，本スクリプトでは `@.ai/ai-dev-kit/machine_learning.md` が定めるiteration/
epoch/train関数によるテンプレートは用いず，Algorithm 1-4に忠実な専用の学習ループを実装する．
`set_seed`，`ResultLogger`，出力ディレクトリ規則等，テンプレートと両立する部分はそのまま踏襲する．

`.orders/order_003.md` の指示に基づき，勾配はPyTorchの標準的な自動微分（`loss.backward()`）に
より取得する．SVRG系手法は現在のパラメータ w_s^k とスナップショット z_s の双方における勾配を
必要とするため，`model`（現在のパラメータ，最適化手法クラスが直接更新する）と
`snapshot_model`（スナップショット z_s を保持する，`model` と同一構造の別インスタンス）の
2つのモデルインスタンスを用意し，それぞれに対して forward／backward を実行することで
両方の勾配を得る．
"""

import copy
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

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
from optimizers.optimizers import ASAISVRG, NFGSVRG, SGD, SVRG  # noqa: E402

from data import load_dataloader  # noqa: E402
from model import (  # noqa: E402
    compute_accuracy,
    compute_gradient,
    compute_loss,
    load_model,
    set_model_params,
)

EXPERIMENT_NAME = "ex001_mushroom_svrg"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
REG_LAMBDA = 1e-2
EPOCHS = 60
BATCH_SIZE = 1


def compute_smoothness_constant(X: np.ndarray, reg_lambda: float) -> float:
    """
    概要: L2正則化付き二値交差エントロピー損失の平滑性定数 L（Assumption 1）を計算する．
        ロジスティック損失のヘッセ行列は (1/N) X^T D X （D_ii ≤ 1/4）で上から抑えられるため，
        L = (1/4) λ_max(X^T X / N) + λ として評価する．
    引数:
        X (numpy.ndarray)，形状 (N, d)．標準化済みの入力特徴量．
        reg_lambda (float)．L2正則化係数 λ．
    戻り値: L (float)．平滑性定数．
    """
    N = X.shape[0]
    XtX_over_N = (X.T @ X) / N
    eigvals = np.linalg.eigvalsh(XtX_over_N)
    return 0.25 * eigvals.max() + reg_lambda


def compute_optimal_solution(X_train: np.ndarray, y_train: np.ndarray, reg_lambda: float):
    """
    概要: 学習用データにおける目的関数 f(w) の最適解 w* および最適値 f(w*) を，
        scipyのL-BFGS-Bにより高精度に求める．
    引数:
        X_train (numpy.ndarray)，形状 (N, d)．標準化済みの学習用特徴量．
        y_train (numpy.ndarray)，形状 (N,)．学習用の教師信号．
        reg_lambda (float)．L2正則化係数 λ．
    戻り値:
        params_star ((torch.Tensor, torch.Tensor))．最適な (weight, bias)．
        f_star (float)．最適値 f(w*)．
    """
    d = X_train.shape[1]
    N = X_train.shape[0]

    def objective_and_grad(theta):
        w = theta[:d]
        b = theta[d]
        z = X_train @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-15
        bce = -np.mean(y_train * np.log(p + eps) + (1 - y_train) * np.log(1 - p + eps))
        loss = bce + 0.5 * reg_lambda * np.sum(w ** 2)

        error = p - y_train
        grad_w = (error @ X_train) / N + reg_lambda * w
        grad_b = error.mean()
        grad = np.concatenate([grad_w, [grad_b]])
        return loss, grad

    theta0 = np.zeros(d + 1)
    result = scipy_opt.minimize(
        objective_and_grad,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 20000, "ftol": 1e-18, "gtol": 1e-14},
    )

    weight_star = torch.tensor(result.x[:d], dtype=torch.float64).reshape(1, -1)
    bias_star = torch.tensor([result.x[d]], dtype=torch.float64)
    return (weight_star, bias_star), float(result.fun)


def draw_batch(X: torch.Tensor, y: torch.Tensor, index: int):
    """
    概要: 指定したインデックスの単一サンプルをバッチとして取り出す．
    引数:
        X (torch.Tensor)，形状 (N, d)．
        y (torch.Tensor)，形状 (N,)．
        index (int)．抽出するサンプルのインデックス．
    戻り値: (x_i, y_i)．形状 (1, d) と (1,) のテンソルの組．
    """
    return X[index:index + 1], y[index:index + 1]


def evaluate(model, X_train, y_train, X_test, y_test, reg_lambda, f_star):
    """
    概要: 与えられたモデルのパラメータにおける目的関数の誤差・分類精度を評価する．
    引数:
        model (torch.nn.Module)．評価対象のパラメータを保持するモデル．
        X_train, y_train, X_test, y_test (torch.Tensor)．学習用・検証用データ．
        reg_lambda (float)．L2正則化係数 λ．
        f_star (float)．最適値 f(w*)．
    戻り値: (objective_gap, accuracy) (float, float)．
    """
    objective_gap = compute_loss(model, X_train, y_train, reg_lambda) - f_star
    accuracy = compute_accuracy(model, X_test, y_test)
    return objective_gap, accuracy


def run_sgd(seed, X_train, y_train, X_test, y_test, eta, reg_lambda, epochs, f_star, logger):
    """
    概要: SGD（(4)式）による学習を実行し，各エポック終了時の評価指標を `logger` に記録する．
        勾配はPyTorchの自動微分（`loss.backward()`）により取得する．
    引数:
        seed (int)．乱数シード．
        X_train, y_train, X_test, y_test (torch.Tensor)．学習用・検証用データ．
        eta (float)．学習率 η．
        reg_lambda (float)．L2正則化係数 λ．
        epochs (int)．エポック数．
        f_star (float)．最適値 f(w*)．
        logger (ResultLogger)．評価指標を記録するロガー．
    戻り値: なし
    """
    N_train = X_train.shape[0]
    model = load_model(seed=seed)
    optimizer = SGD(model.parameters(), lr=eta)
    rng = np.random.default_rng(seed)

    oracle_calls = 0
    start_time = time.time()

    objective_gap, accuracy = evaluate(model, X_train, y_train, X_test, y_test, reg_lambda, f_star)
    logger(0, oracle_calls, 0.0, objective_gap, accuracy, float("nan"))

    for epoch in tqdm(range(1, epochs + 1), desc=f"SGD seed={seed}", leave=False):
        for _ in range(N_train):
            idx = int(rng.integers(0, N_train))
            x_i, y_i = draw_batch(X_train, y_train, idx)
            compute_gradient(model, x_i, y_i, reg_lambda)  # model.parameters()の.gradを更新
            optimizer.step()
            oracle_calls += 1

        elapsed_time = time.time() - start_time
        objective_gap, accuracy = evaluate(
            model, X_train, y_train, X_test, y_test, reg_lambda, f_star
        )
        logger(epoch, oracle_calls, elapsed_time, objective_gap, accuracy, float("nan"))


_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "SVRG": SVRG,
    "NFG_SVRG": NFGSVRG,
    "ASAI_SVRG": ASAISVRG,
}


def run_variance_reduced(
    method, seed, X_train, y_train, X_test, y_test, eta, reg_lambda, K, epochs, f_star, logger
):
    """
    概要: SVRG系手法（SVRG，NFG SVRG，ASAI SVRG）による学習を実行し，各エポック終了時の
        評価指標を `logger` に記録する．論文のAlgorithm 1〜3に対応する外部ループ・内部ループ
        構造を，`programs/optimizers/optimizers.py` の対応するOptimizerクラス
        （`SVRG`，`NFGSVRG`，`ASAISVRG`）の `begin_epoch`/`step`/`end_epoch` を用いて実行する．
        3手法とも外部ループの構成（フル勾配の計算，最終スナップショットの取得・評価）は共通で
        あるため，本関数はOptimizerクラスの多相的なインターフェースを介して1つの関数で
        3手法をまとめて扱う（Optimizerクラス自体はそれぞれ独立に陽実装されている）．
        勾配はPyTorchの自動微分（`loss.backward()`）により取得する．内部ループでは，現在の
        パラメータ w_s^k を保持する `model` とスナップショット z_s を保持する
        `snapshot_model` の2つのモデルインスタンスに対してそれぞれforward／backwardを実行し，
        同一サンプルに対する2種類の勾配を得る．
    引数:
        method (str)．"SVRG"，"NFG_SVRG"，"ASAI_SVRG" のいずれか．
        seed (int)．乱数シード．
        X_train, y_train, X_test, y_test (torch.Tensor)．学習用・検証用データ．
        eta (float)．学習率 η．
        reg_lambda (float)．L2正則化係数 λ．
        K (int)．内部ループ長．
        epochs (int)．外部ループ数 S．
        f_star (float)．最適値 f(w*)．
        logger (ResultLogger)．評価指標を記録するロガー．
    戻り値: なし
    """
    assert method in _VARIANCE_REDUCED_OPTIMIZER_CLASSES

    N_train = X_train.shape[0]
    model = load_model(seed=seed)
    # z_0 <- w_0．snapshot_modelはmodelと同じseedで初期化することで同一の初期値を持つ．
    snapshot_model = load_model(seed=seed)

    OptimizerClass = _VARIANCE_REDUCED_OPTIMIZER_CLASSES[method]
    optimizer = OptimizerClass(model.parameters(), lr=eta, K=K)
    rng = np.random.default_rng(seed)

    oracle_calls = 0

    if method == "SVRG":
        # g_0: SVRGは真のフル勾配（論文Algorithm 1）．この計算コストは，epoch 0（学習開始前の
        # 初期状態）ではなく，epoch 1の学習に要するオラクル呼び出し回数として計上する．
        snapshot_grad = compute_gradient(snapshot_model, X_train, y_train, reg_lambda)
    else:
        # g_0 = 0（論文Algorithm 2, 3）．Optimizer初期化時に既に設定済み．
        snapshot_grad = optimizer.get_snapshot_gradient()

    start_time = time.time()

    true_full_grad_0 = compute_gradient(snapshot_model, X_train, y_train, reg_lambda)
    approx_error_0 = sum(
        torch.sum((g_s - g_true) ** 2).item()
        for g_s, g_true in zip(snapshot_grad, true_full_grad_0)
    )
    objective_gap, accuracy = evaluate(
        snapshot_model, X_train, y_train, X_test, y_test, reg_lambda, f_star
    )
    logger(0, oracle_calls, 0.0, objective_gap, accuracy, approx_error_0)

    if method == "SVRG":
        optimizer.set_snapshot_gradient(snapshot_grad)
        oracle_calls += N_train

    for epoch in tqdm(range(1, epochs + 1), desc=f"{method} seed={seed}", leave=False):
        optimizer.begin_epoch(rng)

        for _ in range(K):
            idx = int(rng.integers(0, N_train))
            x_i, y_i = draw_batch(X_train, y_train, idx)

            compute_gradient(model, x_i, y_i, reg_lambda)  # model.parameters()の.gradを更新
            grad_at_snapshot = compute_gradient(snapshot_model, x_i, y_i, reg_lambda)
            oracle_calls += 2

            optimizer.step(grad_at_snapshot)

        optimizer.end_epoch()
        snapshot_params = optimizer.get_snapshot_params()
        set_model_params(snapshot_model, snapshot_params)

        if method == "SVRG":
            snapshot_grad = compute_gradient(snapshot_model, X_train, y_train, reg_lambda)
            oracle_calls += N_train
            optimizer.set_snapshot_gradient(snapshot_grad)
        else:
            snapshot_grad = optimizer.get_snapshot_gradient()

        elapsed_time = time.time() - start_time

        true_full_grad = compute_gradient(snapshot_model, X_train, y_train, reg_lambda)
        if method == "SVRG":
            approx_error = 0.0
        else:
            approx_error = sum(
                torch.sum((g_s - g_true) ** 2).item()
                for g_s, g_true in zip(snapshot_grad, true_full_grad)
            )

        objective_gap, accuracy = evaluate(
            snapshot_model, X_train, y_train, X_test, y_test, reg_lambda, f_star
        )
        logger(epoch, oracle_calls, elapsed_time, objective_gap, accuracy, approx_error)


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
    引数: args (tuple)．(method, seed, hyperparams) のタプル．hyperparamsはη, λ, K, epochsを含む辞書．
    戻り値: なし
    """
    method, seed, hyperparams = args
    torch.set_num_threads(1)

    eta = hyperparams["eta"]
    reg_lambda = hyperparams["reg_lambda"]
    K = hyperparams["K"]
    epochs = hyperparams["epochs"]
    hp_name = hyperparams["hp_name"]

    target_dir = os.path.join(OUTPUT_ROOT, method, hp_name, str(seed))
    if is_run_completed(target_dir, epochs):
        print(f"[skip] {method}/{hp_name}/{seed} は既に完了しています．")
        return

    os.makedirs(target_dir, exist_ok=True)

    set_seed(seed)
    train_dataloader, test_dataloader = load_dataloader(seed=seed, batch_size=BATCH_SIZE)
    X_train, y_train = train_dataloader.dataset.tensors
    X_test, y_test = test_dataloader.dataset.tensors

    params_star, f_star = compute_optimal_solution(
        X_train.numpy(), y_train.numpy(), reg_lambda
    )

    logger = ResultLogger()
    logger.set_names(
        "epoch", "oracle_calls", "elapsed_time", "objective_gap", "accuracy", "approx_error"
    )

    if method == "SGD":
        run_sgd(seed, X_train, y_train, X_test, y_test, eta, reg_lambda, epochs, f_star, logger)
    else:
        run_variance_reduced(
            method, seed, X_train, y_train, X_test, y_test, eta, reg_lambda, K, epochs,
            f_star, logger,
        )

    logger.save(os.path.join(target_dir, "log.json"))

    weight_star, bias_star = params_star
    config = {
        "method": method,
        "seed": seed,
        "eta": eta,
        "reg_lambda": reg_lambda,
        "K": K,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "N_train": X_train.shape[0],
        "N_test": X_test.shape[0],
        "f_star": f_star,
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{hp_name}/{seed}")


def build_hyperparameters():
    """
    概要: 全手法・全Seedで共通して用いるハイパーパラメータを構築する．
        論文Algorithm 4が示す通り，SVRG系手法（SVRG，NFG SVRG，ASAI SVRG）の相違点は
        スナップショットg_sの計算方法とz_sの更新規則のみであり，学習率η・内部ループ長K・
        外部ループ数（エポック数）は全手法で共通の値を用いる．
        論文はこれらの数値そのものを指定していないため，データから決まる平滑性定数 L と
        正則化係数 λ から，理論保証の条件 0 < η < 1/(3L) を満たすように η = 0.5 / (3L) を定める．
    引数: なし
    戻り値: hyperparams (dict)．eta, reg_lambda, K, epochs, hp_name を含む辞書．
    """
    train_dataloader, _ = load_dataloader(seed=0, batch_size=BATCH_SIZE)
    X_train, _ = train_dataloader.dataset.tensors
    N_train = X_train.shape[0]

    L = compute_smoothness_constant(X_train.numpy(), REG_LAMBDA)
    eta = 0.5 / (3.0 * L)
    K = N_train

    hp_name = f"eta{eta:.6f}_lambda{REG_LAMBDA}_K{K}_epochs{EPOCHS}"

    return {
        "eta": eta,
        "reg_lambda": REG_LAMBDA,
        "K": K,
        "epochs": EPOCHS,
        "hp_name": hp_name,
        "L": L,
    }


def main():
    """
    概要: 実験4.1（マッシュルームデータセットの二値分類問題）の全条件（4手法 x 5Seed）を
        マルチプロセスで並列に学習する．
    引数: なし
    戻り値: なし
    """
    hyperparams = build_hyperparameters()
    print(f"学習率 eta = {hyperparams['eta']:.6f}, L = {hyperparams['L']:.6f}, "
          f"lambda = {hyperparams['reg_lambda']}, K = {hyperparams['K']}, epochs = {hyperparams['epochs']}")

    tasks = [
        (method, seed, hyperparams)
        for method, seed in itertools.product(METHODS, SEEDS)
    ]

    num_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"並列プロセス数: {num_workers} (総タスク数: {len(tasks)})")

    with Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks)

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    main()
