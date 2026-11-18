"""
E0（内側ループの開始点の影響確認）のMushroom・ロジスティック回帰の実験を実行するスクリプト．

条件は論文4.1節（既存の実験 `ex001_mushroom_logistic`）と同一である．

| 項目 | 設定 |
|---|---|
| データ | Mushroom（$ N = 8124 $，22特徴量の順序符号化，9:1に分割，学習データの統計量で標準化） |
| 目的関数 | $ f({\\mathbf w}, b) = \\frac{1}{N}\\sum_{n}\\ell_{\\mathrm{BCE}} + \\frac{\\lambda}{2}(\\|{\\mathbf w}\\|^{2} + b^{2}) $ |
| 定数 | $ L = \\frac{1}{4}\\max_{n}\\|\\tilde{\\mathbf x}_{n}\\|^{2} + \\lambda $，$ \\lambda = \\mu = 0.03 L_{0} $，$ K = N_{\\mathrm{train}} $，ミニバッチサイズ1 |
| 学習率 | $ \\eta = 0.99/(3L) $（$ \\rho = 298.4 $，論文の図1） |
| エポック数 | 100 |
| 手法 | ASA-SVRG（`ASAISVRG`）．SVRGとNFG-SVRGは最終点をスナップショットとする（`SVRGFinalPoint`，`NFGSVRGFinalPoint`）ため，開始点の要因の影響を受けない |
| 要因 | `inner_start` ∈ {"last_iterate", "snapshot"} |
| シード | 0〜4 |

学習ループは既存の実装（`ex001_mushroom_logistic/train.py` の `run_variance_reduced`）と同じ
順序で同じ演算を行い，`inner_start="snapshot"` の場合のみ外側ループの終了時に `model` の
パラメータをスナップショット $ {\\mathbf z}_{s} $ で上書きする．ログには既存の項目に加えて
`test_correct`，`full_grad_norm_sq`，`approx_cos`，`terminated` を記録する．

実行例:
    python train.py
    python train.py --inner-starts last_iterate --seeds 0 --epochs 3 --output-root ../../outputs/_verify/ex001
    python train.py --eta-rules eta_a eta_b eta_c --methods SVRG NFG_SVRG ASAI_SVRG   # 追加の条件
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
from optimizers import ASAISVRG, NFGSVRGFinalPoint, SVRGFinalPoint  # noqa: E402

from data import load_dataloader  # noqa: E402
from model import compute_accuracy, compute_gradient, compute_loss, load_model, set_model_params  # noqa: E402

EXPERIMENT_NAME = "ex001_e0_mushroom_logistic"
OUTPUT_ROOT = os.path.join(_REVIEW_ROOT, "outputs", EXPERIMENT_NAME)

INNER_STARTS = ("last_iterate", "snapshot")
SEEDS = [0, 1, 2, 3, 4]
ALL_METHODS = ["SVRG", "NFG_SVRG", "ASAI_SVRG"]
DEFAULT_METHODS = ["ASAI_SVRG"]
DEFAULT_ETA_RULES = ["eta_a"]
EPOCHS = 100
BATCH_SIZE = 1
LAMBDA_RATIO = 0.03
TARGET_RHO = 0.5
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
    概要: $ \\rho(\\eta) $ = `target_rho` となる学習率のうち大きい方の解を，既存の実装と同じ方法
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


def compute_optimal_solution(X_train: np.ndarray, y_train: np.ndarray, reg_lambda: float):
    """
    概要: 最適解と最適値 $ f({\\mathbf w}^{*}) $ をL-BFGS-Bで求める（既存の実装と同一）．
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
    概要: スナップショットにおける全勾配，そのノルムの2乗，目的関数誤差，テスト精度，正解数を計算する．
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
    概要: 近似誤差 $ \\|{\\mathbf e}_{s}\\|^{2} $（既存の実装と同じ計算）と，$ {\\mathbf g}_{s} $ と
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


def run_inner_loop(model, snapshot_model, optimizer, X_train, y_train, reg_lambda, permutation) -> None:
    """
    概要: 1回の内側ループ（学習データを1巡，ミニバッチサイズ1）を実行する．
    引数:
        model (torch.nn.Module)．現在のパラメータ $ {\\mathbf w}_{s}^{k} $ を保持するモデル．
        snapshot_model (torch.nn.Module)．スナップショット $ {\\mathbf z}_{s} $ を保持するモデル．
        optimizer (torch.optim.Optimizer)．SVRG系手法のOptimizer．
        X_train, y_train (torch.Tensor)．学習データ．
        reg_lambda (float)．L2正則化係数．
        permutation (numpy.ndarray)，形状 (N,)．データを用いる順序．
    戻り値: なし
    """
    for index in permutation:
        i = int(index)
        X_i, y_i = X_train[i:i + 1], y_train[i:i + 1]
        compute_gradient(model, X_i, y_i, reg_lambda)
        grad_at_snapshot = compute_gradient(snapshot_model, X_i, y_i, reg_lambda)
        optimizer.step(grad_at_snapshot)


def run_variance_reduced(method, inner_start, seed, data, eta, reg_lambda, f_star, epochs, logger) -> None:
    """
    概要: SVRG系手法で学習し，各外側ループの終了時の評価指標を `logger` に記録する．既存の実装と
        同じ順序で乱数を消費する（各エポックで `rng.permutation` の後に `begin_epoch(rng)`）．
    引数:
        method (str)．"SVRG"，"NFG_SVRG"，"ASAI_SVRG" のいずれか．
        inner_start (str)．"last_iterate" または "snapshot"．
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

    for epoch in tqdm(range(1, epochs + 1), desc=f"{method} {inner_start} seed={seed}", leave=False):
        permutation = rng.permutation(N_train)
        start_time = time.perf_counter()
        optimizer.begin_epoch(rng)
        run_inner_loop(model, snapshot_model, optimizer, X_train, y_train, reg_lambda, permutation)
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
    概要: 1つの (手法, 学習率, 内側ループの開始点, シード) の組について学習し，結果を保存する．
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
    run_variance_reduced(task["method"], task["inner_start"], seed, (X_train, y_train, X_test, y_test),
                         hyperparams["eta"], reg_lambda, f_star, task["epochs"], logger)
    logger.save(os.path.join(target_dir, "log.json"))

    config = {
        "experiment": EXPERIMENT_NAME,
        "method": task["method"],
        "inner_start": task["inner_start"],
        "seed": seed,
        "eta": hyperparams["eta"],
        "eta_rule": hyperparams["eta_rule"],
        "reg_lambda": reg_lambda,
        "lambda_ratio": LAMBDA_RATIO,
        "K": X_train.shape[0],
        "epochs": task["epochs"],
        "batch_size": BATCH_SIZE,
        "sampling": "random_reshuffle",
        "evaluation_point": "snapshot z_s",
        "L": hyperparams["L"],
        "mu": reg_lambda,
        "rho": hyperparams["rho"],
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


def build_hyperparameters() -> dict:
    """
    概要: 既存の実験と同じ3つの学習率（$ \\eta_a = 0.99/(3L) $，$ \\rho = 0.5 $ となる $ \\eta_b $，
        $ \\eta_c = 1/(20L) $）と定数を計算する．
    引数: なし
    戻り値: hyperparameters (dict)．キーが "eta_a"，"eta_b"，"eta_c"，値が eta，eta_rule，reg_lambda，
        hp_name（`inner_start` を除く部分），L，rho を含む辞書．
    """
    train_dataloader, _ = load_dataloader(seed=0, batch_size=BATCH_SIZE)
    X_train = train_dataloader.dataset.tensors[0].numpy()
    N_train = X_train.shape[0]
    L0 = compute_smoothness_constant(X_train, reg_lambda=0.0)
    reg_lambda = LAMBDA_RATIO * L0
    L = L0 + reg_lambda
    etas = {
        "eta_a": ETA_A_MARGIN / (3.0 * L),
        "eta_b": solve_eta_for_target_rho(L, reg_lambda, N_train, TARGET_RHO),
        "eta_c": 1.0 / (20.0 * L),
    }
    return {
        rule: {
            "eta": eta,
            "eta_rule": rule,
            "reg_lambda": reg_lambda,
            "hp_name": f"eta{eta:.6e}_lambda{reg_lambda:.6e}_K{N_train}_epochs{{epochs}}",
            "L": L,
            "rho": compute_contraction_rate(eta, L, reg_lambda, N_train),
        }
        for rule, eta in etas.items()
    }


def build_tasks(methods, eta_rules, inner_starts, seeds, epochs, output_root) -> list:
    """
    概要: 全ての (手法, 学習率, 内側ループの開始点, シード) の組のタスクを構築する．
    引数:
        methods (list of str)．手法名．
        eta_rules (list of str)．学習率の規則（"eta_a"，"eta_b"，"eta_c"）．
        inner_starts (list of str)．内側ループの開始点．
        seeds (list of int)．シード．
        epochs (int)．外側ループ数．
        output_root (str)．結果の保存先のルート．
    戻り値: tasks (list of dict)．
    """
    hyperparameters = build_hyperparameters()
    tasks = []
    for method, rule, inner_start, seed in itertools.product(methods, eta_rules, inner_starts, seeds):
        hyperparams = hyperparameters[rule]
        name = f"inner_{inner_start}_" + hyperparams["hp_name"].format(epochs=epochs)
        tasks.append({
            "method": method, "inner_start": inner_start, "seed": seed, "epochs": epochs,
            "hyperparams": hyperparams, "target_dir": os.path.join(output_root, method, name, str(seed)),
        })
    return tasks


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=ALL_METHODS)
    parser.add_argument("--eta-rules", nargs="+", default=DEFAULT_ETA_RULES, choices=["eta_a", "eta_b", "eta_c"])
    parser.add_argument("--inner-starts", nargs="+", default=list(INNER_STARTS), choices=INNER_STARTS)
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
    tasks = build_tasks(args.methods, args.eta_rules, args.inner_starts, args.seeds, args.epochs,
                        os.path.abspath(args.output_root))
    for task in tasks[:1]:
        h = task["hyperparams"]
        print(f"L = {h['L']:.6f}, lambda = {h['reg_lambda']:.6f}", flush=True)
    num_workers = max(1, min(args.workers, len(tasks)))
    print(f"タスク数: {len(tasks)}，並列プロセス数: {num_workers}", flush=True)
    with Pool(processes=num_workers) as pool:
        pool.map(run_single_experiment, tasks, chunksize=1)
    print("全ての学習が終了しました．", flush=True)


if __name__ == "__main__":
    main()
