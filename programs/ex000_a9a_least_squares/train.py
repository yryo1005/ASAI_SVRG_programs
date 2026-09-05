"""
実験0（a9aデータセットを用いた二値分類問題）の学習ループを定義し，実験を実行するスクリプト．

`.orders/order_020.md` の実験0に対応する．NFG SVRG原論文（Medyakov, Molodtsov, Chezhegov,
Rebrikov, Beznosikov, "Variance Reduction Methods Do Not Need to Compute Full Gradients:
Improved Efficiency through Shuffling", 2025, `references/No_Full_Grad_SVRG.pdf`）付録A.1
（LEAST SQUARES REGRESSION，式(8)）の非凸実験設定を再現し，本リポジトリの実装（SVRG，
NFG SVRG）が先行研究の挙動を正しく再現できることを確認する検証実験である．

実験終了の基準は，原論文Figure 3（横軸「フル勾配の計算回数」に対する縦軸
$ \\|\\nabla f(x^k)\\|^2 $）と定性的に同じ傾向，すなわち
(a) $ \\|\\nabla f\\|^2 $ が対数軸上でほぼ直線的に多桁にわたり減少すること，
(b) 理論的なステップ幅の下でNFG SVRGとSVRGの収束曲線がほぼ重なること，
が再現できることである．

## 比較手法

`.orders/order_020.md` はSGD，SVRG，NFG SVRGの3手法比較を指示している（ASAI SVRGは原論文に
存在しない手法であるため必須ではないとしている）．一方で，提案手法ASAI SVRGが論文全体を通じて
一貫して比較対象に含まれることが望ましいとのユーザーからの追加指示に基づき，本実験でも
ASAI SVRGを含めた4手法を比較する．SVRG・NFG SVRGは，原論文Algorithm 1の記述（次エポックの
スナップショットとして内部ループの最終パラメータ ω_{s+1} = x_s^n を採用する）に忠実な
`programs/optimizers/` の `SVRGFinalPoint`／`NFGSVRGFinalPoint` を用いる．ASAI SVRGは，
ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）Algorithm 3に対応する `ASAISVRG`（内部ループの
パラメータ列の平均をスナップショットとする，提案手法自体のアルゴリズム）を用いる．学習率は
SVRG・NFG SVRGと共通のものを用いる（`.orders/order_011.md` の指示に基づくEx006と同様の扱い）．

## データの抽出方法

原論文の中核はシャッフリングの発見的手法（shuffling heuristic）であり，NFG SVRGのフル勾配の
近似（原論文の式(5)・式(6)）は，1エポックの内部ループで各データを丁度1回ずつ用いることに
よって成立する．そのため，本実験では4手法すべてでランダムリシャッフル（Random Reshuffle，RR，
原論文Figure 3の「RR NFG-SVRG」に対応）を用い，各エポックで学習用データの順列を引き直して
1巡する．したがって内部ループ長は $ K = N_{\\text{train}} $ である．

## 学習率

`.orders/order_020.md` の指示に従い，学習率はデータから推定した平滑性定数 $ L $ に基づいて
解析的に決定し，チューニング（グリッド探索等）は行わない．理論的な根拠を持つ3つの値を用いる
（詳細は `build_hyperparameter_list` を参照）．
"""

import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

# 本実験は60条件をマルチプロセスで並列実行するため，各プロセス内でBLAS（NumPy・SciPy）が
# さらにスレッドを起動するとスレッド数がCPUコア数を大きく超過し，かえって実行が遅くなる．
# NumPy・PyTorchのimportより前に環境変数を設定し，各プロセスを1スレッドに制限する．
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

EXPERIMENT_NAME = "ex000_a9a_least_squares"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
EPOCHS = 100
BATCH_SIZE = 1

# 記録する評価指標の名前．
METRIC_NAMES = (
    "epoch",
    "oracle_calls",
    "full_grad_computations",
    "elapsed_time",
    "objective_value",
    "grad_norm_sq",
    "accuracy",
    "approx_error",
)

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    # 原論文Algorithm 1（No Full Grad SVRG）および同論文が比較対象とする古典的SVRGと同一の
    # スナップショット構成（内部ループの最終パラメータを採用）を用いる．
    "SVRG": SVRGFinalPoint,
    "NFG_SVRG": NFGSVRGFinalPoint,
    # ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）自身のスナップショット構成
    # （内部ループのパラメータ列の平均）をそのまま用いる．
    "ASAI_SVRG": ASAISVRG,
}


def compute_kappa_max(z_half_width: float = 30.0, num_points: int = 400001) -> float:
    """
    概要: 非線形最小二乗損失の1サンプル分 l(z, y) = (y - σ(z))^2（σ: シグモイド関数）について，
        z に関する2階微分 l''(z, y) の絶対値の上界
        κ_max = max_{z, y∈{0,1}} |l''(z, y)| を，十分細かい格子上の数値探索により求める．
        l''(z, y) = 2σ'(z)^2 + 2(σ(z) - y)σ''(z)（σ'(z) = σ(z)(1-σ(z))，
        σ''(z) = σ'(z)(1-2σ(z))）である．z の絶対値が大きい領域では σ(z) は0または1に飽和し
        l''(z, y) → 0 となるため，有限区間 [-z_half_width, z_half_width] の格子探索で十分な
        精度が得られる．
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


def compute_smoothness_constants(A: np.ndarray) -> tuple[float, float]:
    """
    概要: 非線形最小二乗損失の平滑性定数を計算する．1サンプル分の損失
        l_i(x) = (y_i - σ(A_i・x))^2 のヘッセ行列は l''(z_i, y_i)・A_i A_i^T であり，
        その固有値は κ_max・||A_i||^2 で上から抑えられる．
        原論文およびASAI SVRG論文のAssumption 1は「各 f_i が L-平滑であること」を要求するため，
        本実験ではサンプルごとの平滑性定数の最大値
        L_individual = κ_max・max_i ||A_i||^2 を学習率の決定に用いる．
        比較のため，経験損失 f の平均としての平滑性定数
        L_average = κ_max・λ_max(A^T A / N) も併せて返す．
    引数: A (numpy.ndarray)，形状 (N, d)．入力特徴量（0/1のOne-Hotベクトル，無加工）．
    戻り値:
        L_individual (float)．Assumption 1を満たす平滑性定数．
        L_average (float)．経験損失 f の平滑性定数．
    """
    N = A.shape[0]
    kappa_max = compute_kappa_max()

    row_norms_sq = (A ** 2).sum(axis=1)
    L_individual = kappa_max * float(row_norms_sq.max())

    eigvals = np.linalg.eigvalsh((A.T @ A) / N)
    L_average = kappa_max * float(eigvals.max())

    return L_individual, L_average


def compute_optimal_solution(A_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, float]:
    """
    概要: 学習用データにおける目的関数 f(x)（式(8)）の（局所）最適解 x* および最適値 f(x*) を，
        scipyのL-BFGS-Bにより求める．f(x) は非凸であるため，得られる解は大域的最適解である
        保証はなく，あくまで目的関数の値の基準点（reference）として用いる．
    引数:
        A_train (numpy.ndarray)，形状 (N, d)．学習用の特徴量．
        y_train (numpy.ndarray)，形状 (N,)．学習用の教師信号．
    戻り値:
        x_star (numpy.ndarray)，形状 (d,)．（局所）最適解．
        f_star (float)．最適値 f(x*)．
    """
    N, d = A_train.shape

    def objective_and_grad(w):
        z = A_train @ w
        s = 1.0 / (1.0 + np.exp(-z))
        loss = np.mean((y_train - s) ** 2)
        grad = (A_train.T @ (2.0 * (s - y_train) * s * (1.0 - s))) / N
        return loss, grad

    result = scipy_opt.minimize(
        objective_and_grad,
        np.zeros(d),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 20000, "ftol": 1e-18, "gtol": 1e-14},
    )
    return result.x, float(result.fun)


def draw_sample(A: torch.Tensor, y: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    概要: 指定したインデックスの単一サンプルを，バッチサイズ1のミニバッチとして取り出す．
    引数:
        A (torch.Tensor)，形状 (N, d)．特徴量．
        y (torch.Tensor)，形状 (N,)．教師信号．
        index (int)．抽出するサンプルのインデックス．
    戻り値: (A_i, y_i)．形状 (1, d) と (1,) のテンソルの組．
    """
    return A[index:index + 1], y[index:index + 1]


def evaluate_point(model, A_train, y_train, A_test, y_test) -> tuple[list, float, float, float]:
    """
    概要: 与えられたモデルのパラメータにおける評価指標（真のフル勾配とそのノルムの2乗，
        目的関数の値，検証用データに対する分類精度）をまとめて計算する．
        真のフル勾配は，学習用データ全体に対する `compute_gradient` の1回の呼び出しで計算する．
        本関数の計算コストは評価指標の算出のためだけに要するものであり，学習の実行時間
        （`elapsed_time`）およびオラクル呼び出し回数には計上しない．
    引数:
        model (torch.nn.Module)．評価対象のパラメータを保持するモデル．
        A_train (torch.Tensor)，形状 (N_train, d)．学習用の特徴量．
        y_train (torch.Tensor)，形状 (N_train,)．学習用の教師信号．
        A_test (torch.Tensor)，形状 (N_test, d)．検証用の特徴量．
        y_test (torch.Tensor)，形状 (N_test,)．検証用の教師信号．
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


def compute_approx_error(snapshot_grad: list, true_full_grad: list) -> float:
    """
    概要: フル勾配の近似誤差 ||e_s||^2 = ||g_s - ∇f(z_s)||^2 を計算する．
    引数:
        snapshot_grad (list of torch.Tensor)．スナップショット勾配 g_s．
        true_full_grad (list of torch.Tensor)．真のフル勾配 ∇f(z_s)．
    戻り値: approx_error (float)．
    """
    return sum(
        torch.sum((g_s - g_true) ** 2).item()
        for g_s, g_true in zip(snapshot_grad, true_full_grad)
    )


def run_sgd(seed, A_train, y_train, A_test, y_test, eta, epochs, logger):
    """
    概要: SGD（オンライン学習，バッチサイズ1）による学習を実行し，各エポック終了時の評価指標を
        `logger` に記録する．データはエポックごとにランダムリシャッフルし1巡する．
    引数:
        seed (int)．乱数シード．
        A_train (torch.Tensor)，形状 (N_train, d)．学習用の特徴量．
        y_train (torch.Tensor)，形状 (N_train,)．学習用の教師信号．
        A_test (torch.Tensor)，形状 (N_test, d)．検証用の特徴量．
        y_test (torch.Tensor)，形状 (N_test,)．検証用の教師信号．
        eta (float)．学習率 η．
        epochs (int)．エポック数．
        logger (ResultLogger)．評価指標を記録するロガー．
    戻り値: なし
    """
    N_train = A_train.shape[0]
    model = load_model(seed=seed, input_dim=A_train.shape[1])
    optimizer = SGD(model.parameters(), lr=eta)
    rng = np.random.default_rng(seed)

    oracle_calls = 0
    elapsed_time = 0.0

    _, grad_norm_sq, objective_value, accuracy = evaluate_point(
        model, A_train, y_train, A_test, y_test
    )
    logger(0, 0, 0.0, 0.0, objective_value, grad_norm_sq, accuracy, float("nan"))

    for epoch in tqdm(range(1, epochs + 1), desc=f"SGD seed={seed}", leave=False):
        permutation = rng.permutation(N_train)

        start_time = time.perf_counter()
        for index in permutation:
            A_i, y_i = draw_sample(A_train, y_train, int(index))
            compute_gradient(model, A_i, y_i)  # model.parameters()の.gradを更新
            optimizer.step()
        elapsed_time += time.perf_counter() - start_time
        oracle_calls += N_train

        _, grad_norm_sq, objective_value, accuracy = evaluate_point(
            model, A_train, y_train, A_test, y_test
        )
        logger(
            epoch, oracle_calls, oracle_calls / N_train, elapsed_time,
            objective_value, grad_norm_sq, accuracy, float("nan"),
        )


def run_variance_reduced(method, seed, A_train, y_train, A_test, y_test, eta, epochs, logger):
    """
    概要: SVRG系手法（SVRG，NFG SVRG）による学習を実行し，各エポック終了時の評価指標を
        `logger` に記録する．外部ループ・内部ループ構造は，`programs/optimizers/` の対応する
        Optimizerクラスの `begin_epoch`／`step`／`end_epoch` を用いて実行する．
        内部ループはエポックごとのランダムリシャッフルにより学習用データを1巡するため，
        内部ループ長は K = N_train である．
    引数:
        method (str)．"SVRG" または "NFG_SVRG"．
        seed (int)．乱数シード．
        A_train (torch.Tensor)，形状 (N_train, d)．学習用の特徴量．
        y_train (torch.Tensor)，形状 (N_train,)．学習用の教師信号．
        A_test (torch.Tensor)，形状 (N_test, d)．検証用の特徴量．
        y_test (torch.Tensor)，形状 (N_test,)．検証用の教師信号．
        eta (float)．学習率 η．
        epochs (int)．外部ループ数 S．
        logger (ResultLogger)．評価指標を記録するロガー．
    戻り値: なし
    """
    N_train = A_train.shape[0]
    model = load_model(seed=seed, input_dim=A_train.shape[1])
    # z_0 <- w_0．snapshot_modelはmodelと同じseedで初期化することで同一の初期値を持つ．
    snapshot_model = load_model(seed=seed, input_dim=A_train.shape[1])

    OptimizerClass = _VARIANCE_REDUCED_OPTIMIZER_CLASSES[method]
    optimizer = OptimizerClass(model.parameters(), lr=eta, K=N_train)
    rng = np.random.default_rng(seed)

    oracle_calls = 0
    elapsed_time = 0.0

    true_full_grad, grad_norm_sq, objective_value, accuracy = evaluate_point(
        snapshot_model, A_train, y_train, A_test, y_test
    )
    if method == "SVRG":
        # g_0: SVRGは真のフル勾配．この計算コストはepoch 0（学習開始前の初期状態）ではなく，
        # epoch 1の学習に要するオラクル呼び出し回数として計上する．
        start_time = time.perf_counter()
        snapshot_grad = compute_gradient(snapshot_model, A_train, y_train)
        initial_full_grad_time = time.perf_counter() - start_time
        optimizer.set_snapshot_gradient(snapshot_grad)
    else:
        # g_0 = 0．Optimizer初期化時に既に設定済み．
        snapshot_grad = optimizer.get_snapshot_gradient()
        initial_full_grad_time = 0.0

    logger(
        0, 0, 0.0, 0.0, objective_value, grad_norm_sq, accuracy,
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
            A_i, y_i = draw_sample(A_train, y_train, int(index))
            compute_gradient(model, A_i, y_i)  # model.parameters()の.gradを更新
            grad_at_snapshot = compute_gradient(snapshot_model, A_i, y_i)
            optimizer.step(grad_at_snapshot)

        optimizer.end_epoch()
        set_model_params(snapshot_model, optimizer.get_snapshot_params(), source_model=model)

        if method == "SVRG":
            snapshot_grad = compute_gradient(snapshot_model, A_train, y_train)
            optimizer.set_snapshot_gradient(snapshot_grad)
        else:
            snapshot_grad = optimizer.get_snapshot_gradient()
        elapsed_time += time.perf_counter() - start_time

        oracle_calls += 2 * N_train
        if method == "SVRG":
            oracle_calls += N_train

        true_full_grad, grad_norm_sq, objective_value, accuracy = evaluate_point(
            snapshot_model, A_train, y_train, A_test, y_test
        )
        logger(
            epoch, oracle_calls, oracle_calls / N_train, elapsed_time,
            objective_value, grad_norm_sq, accuracy,
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
    引数: args (tuple)．(method, seed, hyperparams) のタプル．hyperparams は eta，hp_name
        などを含む辞書．
    戻り値: なし
    """
    method, seed, hyperparams = args
    torch.set_num_threads(1)

    eta = hyperparams["eta"]
    hp_name = hyperparams["hp_name"]
    epochs = EPOCHS

    target_dir = os.path.join(OUTPUT_ROOT, method, hp_name, str(seed))
    if is_run_completed(target_dir, epochs):
        print(f"[skip] {method}/{hp_name}/{seed} は既に完了しています．", flush=True)
        return

    os.makedirs(target_dir, exist_ok=True)

    set_seed(seed)
    train_dataloader, test_dataloader = load_dataloader(seed=seed, batch_size=BATCH_SIZE)
    A_train, y_train = train_dataloader.dataset.tensors
    A_test, y_test = test_dataloader.dataset.tensors

    _, f_star = compute_optimal_solution(A_train.numpy(), y_train.numpy())

    logger = ResultLogger()
    logger.set_names(*METRIC_NAMES)

    if method == "SGD":
        run_sgd(seed, A_train, y_train, A_test, y_test, eta, epochs, logger)
    else:
        run_variance_reduced(
            method, seed, A_train, y_train, A_test, y_test, eta, epochs, logger
        )

    logger.save(os.path.join(target_dir, "log.json"))

    config = {
        "experiment": EXPERIMENT_NAME,
        "method": method,
        "seed": seed,
        "eta": eta,
        "eta_rule": hyperparams["eta_rule"],
        "eta_description": hyperparams["eta_description"],
        "K": A_train.shape[0],
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "sampling": "random_reshuffle",
        "L_individual": hyperparams["L_individual"],
        "L_average": hyperparams["L_average"],
        "N_train": A_train.shape[0],
        "N_test": A_test.shape[0],
        "d": A_train.shape[1],
        "f_star_reference": f_star,
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{hp_name}/{seed}", flush=True)


def build_hyperparameter_list() -> list:
    """
    概要: 全手法・全Seedで共通して用いる学習率の候補を構築する．`.orders/order_020.md` の指示に
        従い，学習率はデータから推定した平滑性定数 $ L $ に基づいて解析的に決定し，
        チューニング（グリッド探索等）は行わない．理論的な根拠を持つ次の3つの値を用いる．

        1. η = 1/(3L)．ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）の式(28)が定める
           係数 c_1 = 2η(1 - 3ηL) が正となるための上界であり，SVRG系手法の解析が成立する
           最大の学習率に対応する．
        2. η = 1/(20L)．NFG SVRG原論文Theorem 1が与える上界 γ ≤ 1/(20Ln) の，データ数 n に
           よる縮小を除いた値である．原論文の定数 1/20 を保ちながら，1エポックあたりの
           実質的な移動量が n に依存して消失しないようにした，実用的な理論ステップ幅である．
        3. η = 1/(20Ln)．NFG SVRG原論文Theorem 1（非凸設定）の上界そのものである．

        いずれの値も上界 1/(3L) 以下であり，`.orders/order_020.md` の条件を満たす．
        平滑性定数は，原論文およびASAI SVRG論文のAssumption 1（各 f_i が L-平滑）に対応する
        L_individual を用いる．
    引数: なし
    戻り値: hyperparameter_list (list of dict)．eta，eta_rule，eta_description，hp_name，
        L_individual，L_average を含む辞書のリスト．
    """
    train_dataloader, _ = load_dataloader(seed=0, batch_size=BATCH_SIZE)
    A_train, _ = train_dataloader.dataset.tensors
    N_train = A_train.shape[0]

    L_individual, L_average = compute_smoothness_constants(A_train.numpy())

    rules = [
        (
            "1/(3L)",
            1.0 / (3.0 * L_individual),
            "ASAI SVRG論文 式(28) の c_1 = 2η(1 - 3ηL) > 0 を満たす上界",
        ),
        (
            "1/(20L)",
            1.0 / (20.0 * L_individual),
            "NFG SVRG原論文 Theorem 1 の上界 1/(20Ln) から データ数 n による縮小を除いた値",
        ),
        (
            "1/(20Ln)",
            1.0 / (20.0 * L_individual * N_train),
            "NFG SVRG原論文 Theorem 1（非凸設定）が与える上界そのもの",
        ),
    ]

    return [
        {
            "eta": eta,
            "eta_rule": rule,
            "eta_description": description,
            "hp_name": f"eta{eta:.6e}_K{N_train}_epochs{EPOCHS}",
            "L_individual": L_individual,
            "L_average": L_average,
        }
        for rule, eta, description in rules
    ]


def main():
    """
    概要: 実験0の全条件（4手法 × 3学習率 × 5Seed = 60条件）をマルチプロセスで並列に学習する．
    引数: なし
    戻り値: なし
    """
    hyperparameter_list = build_hyperparameter_list()
    print(
        f"L_individual = {hyperparameter_list[0]['L_individual']:.6f}, "
        f"L_average = {hyperparameter_list[0]['L_average']:.6f}, epochs = {EPOCHS}"
    )
    for hyperparams in hyperparameter_list:
        print(f"  eta = {hyperparams['eta']:.6e}  ({hyperparams['eta_rule']})")

    tasks = [
        (method, seed, hyperparams)
        for method, hyperparams, seed in itertools.product(METHODS, hyperparameter_list, SEEDS)
    ]

    num_workers = min(len(tasks), os.cpu_count() or 1)
    print(f"並列プロセス数: {num_workers}（総タスク数: {len(tasks)}）")

    with Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks)

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    main()
