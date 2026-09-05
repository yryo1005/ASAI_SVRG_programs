"""
`.orders/order_011.md` の指示に基づく，`references/No_Full_Grad_SVRG.pdf` 付録A.1
（LEAST SQUARES REGRESSION，式(8)）の数値実験を実行するスクリプト（Ex006）．

式(8)の非線形最小二乗損失 f(x) = (1/n) Σ_i (y_i - h_i)^2，h_i = 1/(1+exp(-A_i・x)) を，
LIBSVM Dataのa9aデータセット（`data.py`）上で最小化する．原論文の付録A.1は，同じ問題に対し
SO NFG-SVRG／RR NFG-SVRG／SVRG（理論ステップ幅）と，それぞれのチューニング済みステップ幅版
（tuned）を比較しているが，`.orders/order_011.md` の指示によりtuned版は実装せず，本リポジトリの
従来の実験（Ex001等）と同様にSGD，SVRG，NFG（SVRG），ASAI SVRGの4手法を比較する．

原論文Algorithm 1（No Full Grad SVRG）の記述（次エポックのスナップショットとして内部ループの
最終パラメータ ω_{s+1} = x_s^n をそのまま採用する）に忠実な実装である
`programs/optimizers/optimizers.py` の `SVRGFinalPoint`／`NFGSVRGFinalPoint` を用いる
（Ex003〜Ex005で確立した，本論文の再現実験における標準的な選択）．学習率は，原論文Theorem 1
（非凸設定，Algorithm 1）が与える理論的な上界 γ ≤ 1/(20Ln) の半分の値を，SVRG・NFGに加え
SGD・ASAI SVRGにも共通して用いる（詳細は `build_hyperparameters` を参照）．

評価指標として，原論文Figure 3・4と同じ「真のフル勾配のノルムの2乗 ||∇f(z_s)||^2」
（`grad_norm_sq`）を主軸に記録する．そのほか，目的関数の値 f(z_s)（`objective_value`），
分類精度（`accuracy`，原論文には無いが学習の進行を確認する補助指標），NFG・ASAI SVRGの
フル勾配の近似誤差 ||e_s||^2（`approx_error`）も記録する．
"""

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
from optimizers.optimizers import ASAISVRG, NFGSVRGFinalPoint, SGD, SVRGFinalPoint  # noqa: E402

from data import load_dataloader  # noqa: E402
from model import (  # noqa: E402
    compute_accuracy,
    compute_gradient,
    compute_loss,
    load_model,
    set_model_params,
)

EXPERIMENT_NAME = "ex006_a9a_least_squares"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs_old", EXPERIMENT_NAME)

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGD", "SVRG", "NFG", "ASAI_SVRG"]
EPOCHS = 150
BATCH_SIZE = 1

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    # 原論文Algorithm 1（No Full Grad SVRG）と同一のスナップショット構成（内部ループの
    # 最終パラメータを採用）を用いる．Ex003〜Ex005と同じ選択．
    "SVRG": SVRGFinalPoint,
    "NFG": NFGSVRGFinalPoint,
    # ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）自身のスナップショット構成
    # （内部ループのパラメータ列の平均）をそのまま用いる．
    "ASAI_SVRG": ASAISVRG,
}


def _kappa_max_squared_sigmoid_loss(z_half_width: float = 30.0, num_points: int = 400001) -> float:
    """
    概要: 非線形最小二乗損失の1サンプル分 l(z, y) = (y - σ(z))^2（σ: シグモイド関数）について，
        zに関する2階微分 l''(z, y) の絶対値の上界 κ_max = max_{z, y∈{0,1}} |l''(z, y)| を，
        十分細かい格子上の数値探索により求める．z が絶対値の大きい領域ではσ(z)は0または1に
        飽和し l''(z, y) → 0 となるため，有限区間 [-z_half_width, z_half_width] の格子探索で
        十分な精度が得られる．
        l''(z, y) = 2σ'(z)^2 + 2(σ(z) - y)σ''(z)（σ'(z) = σ(z)(1-σ(z))，
        σ''(z) = σ'(z)(1-2σ(z))）である．
    引数:
        z_half_width (float) = 30.0．探索区間 [-z_half_width, z_half_width] の半幅．
        num_points (int) = 400001．格子点数．
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


def compute_smoothness_constant(A: np.ndarray) -> float:
    """
    概要: 非線形最小二乗損失 f(x) = (1/n)Σ(y_i - σ(A_i・x))^2（Assumption 1）の平滑性定数 L を
        計算する．1サンプル分の損失のヘッセ行列は κ_max・a_i a_i^T で上から抑えられる
        （`_kappa_max_squared_sigmoid_loss` 参照）ため，
        L = κ_max・λ_max(A^T A / N) として評価する．
    引数: A (numpy.ndarray)，形状 (N, d)．入力特徴量（0/1のOne-Hotベクトル，無加工）．
    戻り値: L (float)．平滑性定数．
    """
    N = A.shape[0]
    kappa_max = _kappa_max_squared_sigmoid_loss()
    AtA_over_N = (A.T @ A) / N
    eigvals = np.linalg.eigvalsh(AtA_over_N)
    return kappa_max * eigvals.max()


def compute_optimal_solution(A_train: np.ndarray, y_train: np.ndarray):
    """
    概要: 学習用データにおける目的関数 f(x)（式(8)）の（局所）最適解 x* および最適値 f(x*) を，
        scipyのL-BFGS-Bにより求める．f(x)は非凸であるため，得られる解は大域的最適解である
        保証はなく，あくまで目的関数の値の基準点（reference）として用いる．
    引数:
        A_train (numpy.ndarray)，形状 (N, d)．学習用特徴量．
        y_train (numpy.ndarray)，形状 (N,)．学習用の教師信号．
    戻り値:
        x_star (torch.Tensor)，形状 (1, d)．（局所）最適解．
        f_star (float)．最適値 f(x*)．
    """
    d = A_train.shape[1]
    N = A_train.shape[0]

    def objective_and_grad(w):
        z = A_train @ w
        s = 1.0 / (1.0 + np.exp(-z))
        loss = np.mean((y_train - s) ** 2)
        grad = (A_train.T @ (2.0 * (s - y_train) * s * (1.0 - s))) / N
        return loss, grad

    w0 = np.zeros(d)
    result = scipy_opt.minimize(
        objective_and_grad,
        w0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 20000, "ftol": 1e-18, "gtol": 1e-14},
    )

    x_star = torch.tensor(result.x, dtype=torch.float64).reshape(1, -1)
    return x_star, float(result.fun)


def draw_batch(A: torch.Tensor, y: torch.Tensor, index: int):
    """
    概要: 指定したインデックスの単一サンプルをバッチとして取り出す．
    引数:
        A (torch.Tensor)，形状 (N, d)．
        y (torch.Tensor)，形状 (N,)．
        index (int)．抽出するサンプルのインデックス．
    戻り値: (a_i, y_i)．形状 (1, d) と (1,) のテンソルの組．
    """
    return A[index:index + 1], y[index:index + 1]


def evaluate_point(model, A_train, y_train, A_test, y_test):
    """
    概要: 与えられたモデルのパラメータにおける評価指標（真のフル勾配とそのノルムの2乗，
        目的関数の値，分類精度）をまとめて計算する．真のフル勾配は，学習用データ全体に対する
        `compute_gradient` の1回の呼び出しで計算する．
    引数:
        model (torch.nn.Module)．評価対象のパラメータを保持するモデル．
        A_train, y_train, A_test, y_test (torch.Tensor)．学習用・検証用データ．
    戻り値:
        true_full_grad (list of torch.Tensor)．真のフル勾配 ∇f(x)．
        grad_norm_sq (float)．||∇f(x)||^2．
        objective_value (float)．目的関数の値 f(x)．
        accuracy (float)．検証用データに対する分類精度．
    """
    true_full_grad = compute_gradient(model, A_train, y_train)
    grad_norm_sq = sum(torch.sum(g ** 2).item() for g in true_full_grad)
    objective_value = compute_loss(model, A_train, y_train)
    accuracy = compute_accuracy(model, A_test, y_test)
    return true_full_grad, grad_norm_sq, objective_value, accuracy


def run_sgd(seed, A_train, y_train, A_test, y_test, eta, epochs, logger):
    """
    概要: SGDによる学習を実行し，各エポック終了時の評価指標を `logger` に記録する．
        勾配はPyTorchの自動微分（`loss.backward()`）により取得する．
    引数:
        seed (int)．乱数シード．
        A_train, y_train, A_test, y_test (torch.Tensor)．学習用・検証用データ．
        eta (float)．学習率 η．
        epochs (int)．エポック数．
        logger (ResultLogger)．評価指標を記録するロガー．
    戻り値: なし
    """
    N_train = A_train.shape[0]
    model = load_model(seed=seed)
    optimizer = SGD(model.parameters(), lr=eta)
    rng = np.random.default_rng(seed)

    oracle_calls = 0
    start_time = time.time()

    _, grad_norm_sq, objective_value, accuracy = evaluate_point(model, A_train, y_train, A_test, y_test)
    logger(0, oracle_calls, 0.0, objective_value, grad_norm_sq, accuracy, float("nan"))

    for epoch in tqdm(range(1, epochs + 1), desc=f"SGD seed={seed}", leave=False):
        for _ in range(N_train):
            idx = int(rng.integers(0, N_train))
            a_i, y_i = draw_batch(A_train, y_train, idx)
            compute_gradient(model, a_i, y_i)  # model.parameters()の.gradを更新
            optimizer.step()
            oracle_calls += 1

        elapsed_time = time.time() - start_time
        _, grad_norm_sq, objective_value, accuracy = evaluate_point(
            model, A_train, y_train, A_test, y_test
        )
        logger(epoch, oracle_calls, elapsed_time, objective_value, grad_norm_sq, accuracy, float("nan"))


def run_variance_reduced(
    method, seed, A_train, y_train, A_test, y_test, eta, K, epochs, logger
):
    """
    概要: SVRG系手法（SVRG，NFG，ASAI SVRG）による学習を実行し，各エポック終了時の評価指標を
        `logger` に記録する．外部ループ・内部ループ構造は，
        `programs/optimizers/optimizers.py` の対応するOptimizerクラスの
        `begin_epoch`/`step`/`end_epoch` を用いて実行する．
    引数:
        method (str)．"SVRG"，"NFG"，"ASAI_SVRG" のいずれか．
        seed (int)．乱数シード．
        A_train, y_train, A_test, y_test (torch.Tensor)．学習用・検証用データ．
        eta (float)．学習率 η．
        K (int)．内部ループ長．
        epochs (int)．外部ループ数 S．
        logger (ResultLogger)．評価指標を記録するロガー．
    戻り値: なし
    """
    assert method in _VARIANCE_REDUCED_OPTIMIZER_CLASSES

    N_train = A_train.shape[0]
    model = load_model(seed=seed)
    # z_0 <- x_0．snapshot_modelはmodelと同じseedで初期化することで同一の初期値を持つ．
    snapshot_model = load_model(seed=seed)

    OptimizerClass = _VARIANCE_REDUCED_OPTIMIZER_CLASSES[method]
    optimizer = OptimizerClass(model.parameters(), lr=eta, K=K)
    rng = np.random.default_rng(seed)

    oracle_calls = 0

    if method == "SVRG":
        # g_0: SVRGは真のフル勾配．この計算コストはepoch 0（学習開始前の初期状態）ではなく，
        # epoch 1の学習に要するオラクル呼び出し回数として計上する．
        snapshot_grad = compute_gradient(snapshot_model, A_train, y_train)
    else:
        # g_0 = 0．Optimizer初期化時に既に設定済み．
        snapshot_grad = optimizer.get_snapshot_gradient()

    start_time = time.time()

    true_full_grad_0, grad_norm_sq_0, objective_value_0, accuracy_0 = evaluate_point(
        snapshot_model, A_train, y_train, A_test, y_test
    )
    approx_error_0 = sum(
        torch.sum((g_s - g_true) ** 2).item()
        for g_s, g_true in zip(snapshot_grad, true_full_grad_0)
    )
    logger(0, oracle_calls, 0.0, objective_value_0, grad_norm_sq_0, accuracy_0, approx_error_0)

    if method == "SVRG":
        optimizer.set_snapshot_gradient(snapshot_grad)
        oracle_calls += N_train

    for epoch in tqdm(range(1, epochs + 1), desc=f"{method} seed={seed}", leave=False):
        optimizer.begin_epoch(rng)

        for _ in range(K):
            idx = int(rng.integers(0, N_train))
            a_i, y_i = draw_batch(A_train, y_train, idx)

            compute_gradient(model, a_i, y_i)  # model.parameters()の.gradを更新
            grad_at_snapshot = compute_gradient(snapshot_model, a_i, y_i)
            oracle_calls += 2

            optimizer.step(grad_at_snapshot)

        optimizer.end_epoch()
        snapshot_params = optimizer.get_snapshot_params()
        set_model_params(snapshot_model, snapshot_params, source_model=model)

        if method == "SVRG":
            snapshot_grad = compute_gradient(snapshot_model, A_train, y_train)
            oracle_calls += N_train
            optimizer.set_snapshot_gradient(snapshot_grad)
        else:
            snapshot_grad = optimizer.get_snapshot_gradient()

        elapsed_time = time.time() - start_time

        true_full_grad, grad_norm_sq, objective_value, accuracy = evaluate_point(
            snapshot_model, A_train, y_train, A_test, y_test
        )
        if method == "SVRG":
            approx_error = 0.0
        else:
            approx_error = sum(
                torch.sum((g_s - g_true) ** 2).item()
                for g_s, g_true in zip(snapshot_grad, true_full_grad)
            )

        logger(epoch, oracle_calls, elapsed_time, objective_value, grad_norm_sq, accuracy, approx_error)


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
    引数: args (tuple)．(method, seed, hyperparams) のタプル．hyperparamsはeta, K, epochsを含む辞書．
    戻り値: なし
    """
    method, seed, hyperparams = args
    torch.set_num_threads(1)

    eta = hyperparams["eta"]
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
    A_train, y_train = train_dataloader.dataset.tensors
    A_test, y_test = test_dataloader.dataset.tensors

    x_star, f_star = compute_optimal_solution(A_train.numpy(), y_train.numpy())

    logger = ResultLogger()
    logger.set_names(
        "epoch", "oracle_calls", "elapsed_time", "objective_value", "grad_norm_sq", "accuracy", "approx_error"
    )

    if method == "SGD":
        run_sgd(seed, A_train, y_train, A_test, y_test, eta, epochs, logger)
    else:
        run_variance_reduced(method, seed, A_train, y_train, A_test, y_test, eta, K, epochs, logger)

    logger.save(os.path.join(target_dir, "log.json"))

    config = {
        "method": method,
        "seed": seed,
        "eta": eta,
        "K": K,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "N_train": A_train.shape[0],
        "N_test": A_test.shape[0],
        "f_star_reference": f_star,
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{hp_name}/{seed}")


def build_hyperparameters():
    """
    概要: 全手法・全Seedで共通して用いるハイパーパラメータを構築する．
        `references/No_Full_Grad_SVRG.pdf` Theorem 1（非凸設定，Algorithm 1 = NFG SVRG）は，
        学習率が γ ≤ 1/(20Ln) を満たせば，Algorithm 1が O(nL/ε^2) 回のオラクル呼び出しで
        ε精度に到達することを保証する（n: サンプル数，L: 平滑性定数）．本実験では，古典的SVRG
        （原論文の比較対象）にも同一の学習率を用いる．論文はこの理論的な学習率の具体的な数値を
        与えていないため，データから計算した平滑性定数 L を用いて，上界の半分
        η = 0.5 / (20Ln) を実際の学習率として採用する．学習率の理論的な根拠を持たないSGD・
        ASAI SVRGについても，`.orders/order_011.md` の指示（「記載のないパラメータは妥当に
        定める」）に基づき，同一の η を用いる．内部ループ長 K は，原論文のAlgorithm 1・
        Theorem 1における n（サンプル数）にそのまま対応させ，K = N_train とする．
    引数: なし
    戻り値: hyperparams (dict)．eta, K, epochs, hp_name を含む辞書．
    """
    train_dataloader, _ = load_dataloader(seed=0, batch_size=BATCH_SIZE)
    A_train, _ = train_dataloader.dataset.tensors
    N_train = A_train.shape[0]

    L = compute_smoothness_constant(A_train.numpy())
    eta = 0.5 / (20.0 * L * N_train)
    K = N_train

    hp_name = f"eta{eta:.10e}_K{K}_epochs{EPOCHS}"

    return {
        "eta": eta,
        "K": K,
        "epochs": EPOCHS,
        "hp_name": hp_name,
        "L": L,
    }


def main():
    """
    概要: 実験A.1（LEAST SQUARES REGRESSION，a9aデータセット）の全条件（4手法 x 5Seed）を
        マルチプロセスで並列に学習する．
    引数: なし
    戻り値: なし
    """
    hyperparams = build_hyperparameters()
    print(f"学習率 eta = {hyperparams['eta']:.6e}, L = {hyperparams['L']:.6f}, "
          f"K = {hyperparams['K']}, epochs = {hyperparams['epochs']}")

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
