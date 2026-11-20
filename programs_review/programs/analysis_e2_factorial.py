"""
E2（スナップショットの選択方法と全勾配の推定方法の要因分離，CIFAR-10）の結果を集計し，図と表を作成するモジュール．

`実験計画.md` 4.4節・4.5節と `実験指示E2.md` 2.4節に従う．

評価量：
- 学習損失 `train_loss`，テスト精度 `test_accuracy`，近似誤差 $ \\|{\\mathbf e}_{s}\\|^{2} $（`approx_error`），
  相対近似誤差 $ \\|{\\mathbf e}_{s}\\|^{2}/\\|\\nabla f({\\mathbf z}_{s})\\|^{2} $（後処理），`approx_cos`．
- $ \\omega({\\mathbf z}_{s})/K $（スナップショットと内側ループの軌跡の平均二乗距離）と比 $ \\omega({\\mathbf z}_{s})/\\omega_{\\min} $
  （`omega_snapshot / omega_average`．平均点の手法では常に1）．
- 精度の変化量：(1) `実験計画.md` 4.4節の定義．学習後半（エポック25〜48）の $ |a_{s} - a_{s-1}| $ の平均．
  (2) report_044 と同じ定義（`analysis_e0_with_sgd.roughness`，予算 $ B = 90 $，$ B/2 < \\#\\mathrm{grad}/N \\le B $ の隣り合う評価点の差の標準偏差）．

最終精度：(a) $ \\#\\mathrm{grad}/N \\le 90 $ の最後の評価点，(b) 48エポック目（打ち切りの場合は打ち切り直前の評価点）．
学習後半の中央値（近似誤差，相対近似誤差，$ \\omega $）はエポック25〜48の評価点の中央値をシードごとに求め，シード間で平均 ± 標準誤差を示す．

出力：
- `outputs/ex007_e2_cifar10_snapshot_gradient_factorial/figures/*.png`
- `outputs/e2_factorial_summary.md`

使い方（`programs_review/programs` で実行する）:
    python analysis_e2_factorial.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis_e0 import DEFAULT_LOG_FLOOR, OUTPUTS_DIR, _fmt, final_value  # noqa: E402
from analysis_e0_with_sgd import roughness  # noqa: E402

EXPERIMENT = "ex007_e2_cifar10_snapshot_gradient_factorial"
EXPERIMENT_DIR = os.path.join(OUTPUTS_DIR, EXPERIMENT)
FIGURE_DIR = os.path.join(EXPERIMENT_DIR, "figures")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "e2_factorial_summary.md")

BUDGET = 90
EPOCHS = 48
LATTER_HALF_FIRST_EPOCH = 25
RATIO_TOLERANCE = 1e-6

# 要因配置．手法名 → (スナップショットの選択方法, g_s の推定方法)．
FACTORS = {
    "SVRG_final": ("final", "full"),
    "SVRG": ("random", "full"),
    "SVRG_average": ("average", "full"),
    "NFG_SVRG_final": ("final", "averaged"),
    "NFG_SVRG": ("random", "averaged"),
    "ASAI_SVRG": ("average", "averaged"),
}
METHODS = list(FACTORS)
FULL_GRADIENT_METHODS = [m for m, (_, g) in FACTORS.items() if g == "full"]
AVERAGED_GRADIENT_METHODS = [m for m, (_, g) in FACTORS.items() if g == "averaged"]
SNAPSHOT_RULES = ("final", "random", "average")
METHOD_LABELS = {
    "SVRG_final": "SVRG-final", "SVRG": "SVRG-random", "SVRG_average": "SVRG-average",
    "NFG_SVRG_final": "NFG-final", "NFG_SVRG": "NFG-random", "ASAI_SVRG": "ASA-SVRG",
}
SNAPSHOT_COLORS = {"final": "#1f77b4", "random": "#2ca02c", "average": "#d62728"}
GRADIENT_STYLES = {"full": "-", "averaged": "--"}

# 対応のある差（指示書2.4節）．(ラベル, 手法A, 手法B) で A − B を計算する．
APPROX_ERROR_EFFECTS = [("近似誤差の効果（最終点）", "NFG_SVRG_final", "SVRG_final"),
                        ("近似誤差の効果（ランダム点）", "NFG_SVRG", "SVRG"),
                        ("近似誤差の効果（平均点）", "ASAI_SVRG", "SVRG_average")]
SNAPSHOT_EFFECTS = [("スナップショットの効果（全勾配，平均点 − ランダム点）", "SVRG_average", "SVRG"),
                    ("スナップショットの効果（全勾配，最終点 − ランダム点）", "SVRG_final", "SVRG"),
                    ("スナップショットの効果（平均勾配，平均点 − ランダム点）", "ASAI_SVRG", "NFG_SVRG")]


# ---------------------------------------------------------------------------
# 読み込みと派生量
# ---------------------------------------------------------------------------

def _derived_series(log: dict, method: str, n_train: float, K: int) -> dict:
    """
    概要: ログから後処理の量（相対近似誤差，$ \\omega/K $，$ \\omega $ の比）を計算する．
    引数:
        log (dict)．`log.json` の内容．
        method (str)．手法名．
        n_train (float)．学習データ数．
        K (int)．内側ループの長さ．
    戻り値: series (dict)．キーが指標名，値が numpy.ndarray（評価点の数）．
    """
    err = np.asarray(log["approx_error"], dtype=np.float64)
    full = np.asarray(log["full_grad_norm_sq"], dtype=np.float64)
    omega_s = np.asarray(log.get("omega_snapshot", [np.nan] * len(err)), dtype=np.float64)
    omega_a = np.asarray(log.get("omega_average", [np.nan] * len(err)), dtype=np.float64)
    epoch = np.asarray(log["epoch"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(full > 0, err / full, np.nan)
        # 平均勾配の手法の1エポック目は内側ループが動かないため ω = 0 であり，記録される値は展開式の桁落ちによる
        # 丸め誤差（負の値を含む）である．このエポックの ω/K と比は定義しない（NaN とする）．
        stationary = (epoch == 1) if FACTORS[method][1] == "averaged" else np.zeros_like(epoch, dtype=bool)
        valid = ~stationary & (omega_a > 0)
        ratio = np.where(valid, omega_s / omega_a, np.nan)
        omega_per_k = np.where(valid, omega_s / K, np.nan)
    if FACTORS[method][1] == "full":
        rel = np.zeros_like(err)
    return {"rel_approx_error": rel, "omega_per_K": omega_per_k, "omega_ratio": ratio}


def load_runs(experiment_dir: str = EXPERIMENT_DIR) -> list:
    """
    概要: 実験ディレクトリ以下の全ての試行を読み込む．
    引数: experiment_dir (str)．`outputs/ex007_...`．
    戻り値: runs (list of dict)．各要素は `method`，`seed`，`config`，`log`（派生量を追加した辞書），
        `x`（$ \\#\\mathrm{grad}/N $），`epoch`（エポック数）を持つ．
    """
    runs = []
    for method in METHODS:
        method_dir = os.path.join(experiment_dir, method)
        if not os.path.isdir(method_dir):
            continue
        for condition in sorted(os.listdir(method_dir)):
            condition_dir = os.path.join(method_dir, condition)
            for seed in sorted(os.listdir(condition_dir)):
                run_dir = os.path.join(condition_dir, seed)
                log_path, config_path = os.path.join(run_dir, "log.json"), os.path.join(run_dir, "config.json")
                if not (seed.isdigit() and os.path.exists(log_path) and os.path.exists(config_path)):
                    continue
                with open(log_path) as f:
                    log = json.load(f)
                with open(config_path) as f:
                    config = json.load(f)
                n_train = n_train_from_log(method, log)
                K = int(np.ceil(n_train / config["batch_size"]))
                log.update(_derived_series(log, method, n_train, K))
                runs.append({"method": method, "seed": int(seed), "condition": condition, "config": config,
                             "log": log, "K": K, "n_train": n_train,
                             "x": np.asarray(log["oracle_calls"], dtype=np.float64) / n_train,
                             "epoch": np.asarray(log["epoch"], dtype=np.float64)})
    return runs


def n_train_from_log(method: str, log: dict) -> float:
    """
    概要: 学習データ数 $ N $ を `oracle_calls` から逆算する．1エポック目の終了時の値は，全勾配の手法で $ 4N $
        （初期の全勾配 $ N $ + $ 3N $），平均勾配の手法で $ 2N $ である．
    引数: method (str)．手法名．log (dict)．`log.json` の内容．
    戻り値: n_train (float)．
    """
    per_first_epoch = 4 if FACTORS[method][1] == "full" else 2
    return float(log["oracle_calls"][1]) / per_first_epoch


def select(runs: list, method: str) -> list:
    """
    概要: 手法で試行を絞り込み，シード順に並べる．
    引数: runs (list of dict)，method (str)．
    戻り値: selected (list of dict)．
    """
    return sorted([r for r in runs if r["method"] == method], key=lambda r: r["seed"])


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------

def mean_and_se(runs: list, key: str, axis: str, log_scale: bool, budget: float = np.inf):
    """
    概要: 全シードの記録がある評価点について，指標の平均と標準誤差を計算する．`log_scale` の場合は
        `analysis_e0.mean_and_se` と同じく常用対数の平均（幾何平均）とその標準誤差とする（下限 `DEFAULT_LOG_FLOOR`）．
        NaN の評価点（0エポック目の ω など）は除く．
    引数:
        runs (list of dict)．同じ手法の試行．
        key (str)．指標名．
        axis (str)．"x"（$ \\#\\mathrm{grad}/N $）または "epoch"．
        log_scale (bool)．対数で平均するか．
        budget (float) = inf．横軸の上限．
    戻り値: (x, mean, se) の組．いずれも numpy.ndarray．
    """
    if not runs:
        return np.array([]), np.array([]), np.array([])
    length = min(len(r["log"][key]) for r in runs)
    x = runs[0][axis][:length]
    values = np.array([np.asarray(r["log"][key][:length], dtype=np.float64) for r in runs])
    if log_scale:
        values = np.log10(np.maximum(values, DEFAULT_LOG_FLOOR))
    mask = (x <= budget + 1e-9) & np.all(np.isfinite(values), axis=0)
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(mean)
    return x[mask], mean[mask], se[mask]


def _plot_metric(ax, runs, methods, key, axis, log_scale, ylabel):
    """
    概要: 1つの指標について，手法ごとの平均 ± 標準誤差を描く．同じスナップショットの手法は同じ色，
        全勾配の手法は実線，平均勾配の手法は破線とする．
    引数:
        ax (matplotlib.axes.Axes)．描画先．
        runs (list of dict)．全試行．
        methods (list of str)．描く手法．
        key (str)．指標名．
        axis (str)．"x" または "epoch"．
        log_scale (bool)．縦軸を対数にするか．
        ylabel (str)．縦軸のラベル．
    戻り値: なし
    """
    for method in methods:
        selected = select(runs, method)
        if not selected or key not in selected[0]["log"]:
            continue
        x, mean, se = mean_and_se(selected, key, axis, log_scale)
        if log_scale:
            y, lower, upper = 10 ** mean, 10 ** (mean - se), 10 ** (mean + se)
        else:
            y, lower, upper = mean, mean - se, mean + se
        snapshot_rule, gradient = FACTORS[method]
        ax.plot(x, y, GRADIENT_STYLES[gradient], color=SNAPSHOT_COLORS[snapshot_rule], lw=1.4,
                label=f"{METHOD_LABELS[method]} (n={len(selected)})")
        ax.fill_between(x, lower, upper, color=SNAPSHOT_COLORS[snapshot_rule], alpha=0.15, lw=0)
    if log_scale:
        ax.set_yscale("log")
    if axis == "x":
        ax.axvline(BUDGET, color="gray", ls=":", lw=1.0)
        ax.set_xlabel(r"$\#\mathrm{grad}/N$")
    else:
        ax.set_xlabel("Epoch (outer loop $s$)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


METRIC_PANELS = {
    "loss": ("train_loss", False, "Training loss"),
    "accuracy": ("test_accuracy", False, "Test accuracy"),
    "approx_error": ("approx_error", True, r"$\|\mathbf{e}_s\|^2$"),
    "rel_approx_error": ("rel_approx_error", True, r"$\|\mathbf{e}_s\|^2/\|\nabla f(\mathbf{z}_s)\|^2$"),
    "approx_cos": ("approx_cos", False, r"$\cos(\mathbf{g}_s, \nabla f(\mathbf{z}_s))$"),
    "omega_per_K": ("omega_per_K", True, r"$\omega(\mathbf{z}_s)/K$"),
    "omega_ratio": ("omega_ratio", True, r"$\omega(\mathbf{z}_s)/\omega_{\min}$"),
}


def _save(fig, name: str) -> str:
    """
    概要: 図を `FIGURE_DIR` に保存して閉じる．
    引数: fig (matplotlib.figure.Figure)，name (str)．ファイル名（拡張子なし）．
    戻り値: path (str)．
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_six_methods(runs: list) -> list:
    """
    概要: 図1：6手法の学習損失・テスト精度・$ \\omega/K $・$ \\omega $ の比，平均勾配の3手法の近似誤差・相対近似誤差・`approx_cos` を，
        横軸 $ \\#\\mathrm{grad}/N $ とエポック数の2種類で描く．
    引数: runs (list of dict)．
    戻り値: paths (list of str)．
    """
    paths = []
    for name, (key, log_scale, ylabel) in METRIC_PANELS.items():
        methods = AVERAGED_GRADIENT_METHODS if name in ("approx_error", "rel_approx_error", "approx_cos") else METHODS
        for axis in ("x", "epoch"):
            fig, ax = plt.subplots(figsize=(5.5, 5))
            _plot_metric(ax, runs, methods, key, axis, log_scale, ylabel)
            ax.set_title("CIFAR-10 (AlexNet, LayerNorm, bs 128), E2\nsolid: full gradient, dashed: averaged gradient",
                         fontsize=8.5)
            ax.legend(fontsize=7, loc="best")
            paths.append(_save(fig, f"e2_{name}_{'grad' if axis == 'x' else 'epoch'}"))
    return paths


def plot_omega_and_error(runs: list) -> str:
    """
    概要: 図2：平均勾配の3手法について，$ \\omega({\\mathbf z}_{s})/K $ と近似誤差の推移（横軸はエポック数）を左右に並べる．
    引数: runs (list of dict)．
    戻り値: path (str)．
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, name in zip(axes, ("omega_per_K", "approx_error")):
        key, log_scale, ylabel = METRIC_PANELS[name]
        _plot_metric(ax, runs, AVERAGED_GRADIENT_METHODS, key, "epoch", log_scale, ylabel)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(r"Averaged-gradient methods: $\omega(\mathbf{z}_s)/K$ and approximation error", fontsize=10)
    return _save(fig, "e2_omega_vs_approx_error_epoch")


def plot_pairs_by_snapshot(runs: list) -> list:
    """
    概要: 図3：スナップショットの選択方法ごとに，全勾配と平均勾配の2手法だけを重ねた図（横軸はエポック数）．
        同じエポックにおける差が近似誤差の有無だけによる差となる．各図は学習損失とテスト精度の2枚の軸を持つ．
    引数: runs (list of dict)．
    戻り値: paths (list of str)．
    """
    paths = []
    for rule in SNAPSHOT_RULES:
        methods = [m for m, (r, _) in FACTORS.items() if r == rule]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
        for ax, name in zip(axes, ("loss", "accuracy")):
            key, log_scale, ylabel = METRIC_PANELS[name]
            _plot_metric(ax, runs, methods, key, "epoch", log_scale, ylabel)
            ax.legend(fontsize=7.5, loc="best")
        fig.suptitle(f"Snapshot: {rule} point (solid: full gradient, dashed: averaged gradient)", fontsize=10)
        paths.append(_save(fig, f"e2_pair_{rule}_epoch"))
    return paths


# ---------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------

def final_at_epoch(run: dict, key: str, epoch: int = EPOCHS) -> float:
    """
    概要: 指定したエポック以下の最後の有限な評価点の値を返す（打ち切りの場合は打ち切り直前）．
    引数: run (dict)，key (str)，epoch (int) = 48．
    戻り値: value (float)．
    """
    values = np.asarray(run["log"][key], dtype=np.float64)
    loss = np.asarray(run["log"]["train_loss"], dtype=np.float64)[:len(values)]
    valid = (run["epoch"][:len(values)] <= epoch) & np.isfinite(loss)
    return float(values[np.where(valid)[0][-1]])


def final_at_budget(run: dict, key: str) -> float:
    """
    概要: $ \\#\\mathrm{grad}/N \\le 90 $ の最後の評価点の値を返す（`analysis_e0.final_value`）．
    引数: run (dict)，key (str)．
    戻り値: value (float)．
    """
    return final_value(run, key, BUDGET)


def epoch_at_budget(run: dict) -> int:
    """
    概要: $ \\#\\mathrm{grad}/N \\le 90 $ の最後の評価点のエポック数を返す．
    引数: run (dict)．
    戻り値: epoch (int)．
    """
    return int(final_value(run, "epoch", BUDGET))


def accuracy_change_plan(run: dict) -> float:
    """
    概要: 精度の変化量（`実験計画.md` 4.4節の定義）：エポック25〜48の $ |a_{s} - a_{s-1}| $ の平均．
    引数: run (dict)．
    戻り値: value (float)．該当する評価点がない場合は NaN．
    """
    acc = np.asarray(run["log"]["test_accuracy"], dtype=np.float64)
    epoch = run["epoch"][:len(acc)]
    diffs = np.abs(np.diff(acc))  # diffs[i] = |a_{i+1} - a_i|
    mask = (epoch[1:] >= LATTER_HALF_FIRST_EPOCH) & (epoch[1:] <= EPOCHS) & np.isfinite(diffs)
    return float(diffs[mask].mean()) if mask.any() else float("nan")


def accuracy_change_report044(run: dict) -> float:
    """
    概要: 精度の変化量（report_044 の定義）：`analysis_e0_with_sgd.roughness`（予算90）．
    引数: run (dict)．
    戻り値: value (float)．
    """
    return roughness(run, "test_accuracy", BUDGET)


def latter_half_median(run: dict, key: str) -> float:
    """
    概要: 学習後半（エポック25〜48）の有限な評価点の中央値を返す．
    引数: run (dict)，key (str)．
    戻り値: value (float)．
    """
    values = np.asarray(run["log"][key], dtype=np.float64)
    epoch = run["epoch"][:len(values)]
    mask = (epoch >= LATTER_HALF_FIRST_EPOCH) & (epoch <= EPOCHS) & np.isfinite(values)
    return float(np.median(values[mask])) if mask.any() else float("nan")


def max_ratio_deviation(run: dict) -> float:
    """
    概要: $ |\\omega({\\mathbf z}_{s})/\\omega_{\\min} - 1| $ の全エポックでの最大値を返す（比が定義されるエポックのみ）．
    引数: run (dict)．
    戻り値: value (float)．
    """
    ratio = np.asarray(run["log"]["omega_ratio"], dtype=np.float64)
    ratio = ratio[np.isfinite(ratio)]
    return float(np.max(np.abs(ratio - 1.0))) if ratio.size else float("nan")


def _sci(values) -> str:
    """
    概要: 値の列を指数表記の「平均 ± 標準誤差」にする．
    引数: values (list of float)．
    戻り値: text (str)．
    """
    return _fmt(values, sci=True)


def summary_table(runs: list) -> str:
    """
    概要: 6手法の最終精度 (a)(b)，学習損失 (a)(b)，精度の変化量2種類，学習後半の近似誤差・相対近似誤差・$ \\omega $ の比の
        中央値，打ち切りの数の表を作る．
    引数: runs (list of dict)．
    戻り値: text (str)．Markdown．
    """
    lines = ["| 手法 | スナップショット | $ {\\mathbf g}_{s} $ | 試行数 | 打ち切り | (a) の評価点のエポック | "
             "最終精度 (a) $ \\#\\mathrm{grad}/N \\le 90 $ | 最終精度 (b) 48エポック | 学習損失 (a) | 学習損失 (b) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        sel = select(runs, method)
        if not sel:
            continue
        rule, grad = FACTORS[method]
        n_term = sum(bool(r["log"]["terminated"][-1]) for r in sel)
        epochs_a = sorted({epoch_at_budget(r) for r in sel})
        lines.append(f"| {METHOD_LABELS[method]} | {rule} | {grad} | {len(sel)} | {n_term} | "
                     f"{', '.join(map(str, epochs_a))} | "
                     f"{_fmt([final_at_budget(r, 'test_accuracy') for r in sel])} | "
                     f"{_fmt([final_at_epoch(r, 'test_accuracy') for r in sel])} | "
                     f"{_fmt([final_at_budget(r, 'train_loss') for r in sel])} | "
                     f"{_fmt([final_at_epoch(r, 'train_loss') for r in sel])} |")
    lines += ["", "| 手法 | 精度の変化量（計画4.4節：エポック25〜48の $ |a_{s}-a_{s-1}| $ の平均） | "
              "精度の変化量（report_044：$ 45 < \\#\\mathrm{grad}/N \\le 90 $ の差の標準偏差） | "
              "近似誤差の中央値（後半） | 相対近似誤差の中央値（後半） | `approx_cos` の中央値（後半） | "
              "$ \\omega({\\mathbf z}_{s})/K $ の中央値（後半） | $ \\omega({\\mathbf z}_{s})/\\omega_{\\min} $ の中央値（後半） |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        sel = select(runs, method)
        if not sel:
            continue
        is_full = FACTORS[method][1] == "full"
        err = "0（全勾配）" if is_full else _sci([latter_half_median(r, "approx_error") for r in sel])
        rel = "0（全勾配）" if is_full else _sci([latter_half_median(r, "rel_approx_error") for r in sel])
        cos = "1（全勾配）" if is_full else _fmt([latter_half_median(r, "approx_cos") for r in sel])
        lines.append(f"| {METHOD_LABELS[method]} | {_fmt([accuracy_change_plan(r) for r in sel])} | "
                     f"{_fmt([accuracy_change_report044(r) for r in sel])} | {err} | {rel} | {cos} | "
                     f"{_sci([latter_half_median(r, 'omega_per_K') for r in sel])} | "
                     f"{_fmt([latter_half_median(r, 'omega_ratio') for r in sel], digits=6)} |")
    return "\n".join(lines) + "\n"


def ratio_check(runs: list) -> str:
    """
    概要: 平均点の手法（`SVRG_average`，`ASAI_SVRG`）で $ \\omega({\\mathbf z}_{s})/\\omega_{\\min} $ が1から $ 10^{-6} $ 以上ずれた試行を報告する．
    引数: runs (list of dict)．
    戻り値: text (str)．Markdown の箇条書き．
    """
    lines = []
    for method in ("SVRG_average", "ASAI_SVRG"):
        sel = select(runs, method)
        if not sel:
            continue
        devs = {r["seed"]: max_ratio_deviation(r) for r in sel}
        worst = max(devs.values())
        flagged = [s for s, d in devs.items() if d >= RATIO_TOLERANCE]
        status = (f"1から {RATIO_TOLERANCE:.0e} 以上ずれたシード：{flagged}（数値誤差）" if flagged
                  else f"全シードで1からのずれは {RATIO_TOLERANCE:.0e} 未満")
        lines.append(f"- {METHOD_LABELS[method]}：$ |\\omega({{\\mathbf z}}_{{s}})/\\omega_{{\\min}} - 1| $ の最大値 "
                     f"{worst:.3e}（シードごと：{', '.join(f'{s}: {d:.2e}' for s, d in devs.items())}）．{status}．")
    return "\n".join(lines) + "\n"


def paired_table(runs: list, pairs: list) -> str:
    """
    概要: 同じシードの2手法の差（A − B）を，シードごとと平均 ± 標準誤差で示す表を作る．比べる値は最終精度 (a)(b) と学習損失 (a)(b)．
    引数: runs (list of dict)．pairs (list of tuple)．(ラベル, 手法A, 手法B)．
    戻り値: text (str)．Markdown．
    """
    metrics = [("最終精度 (a)", lambda r: final_at_budget(r, "test_accuracy")),
               ("最終精度 (b)", lambda r: final_at_epoch(r, "test_accuracy")),
               ("学習損失 (a)", lambda r: final_at_budget(r, "train_loss")),
               ("学習損失 (b)", lambda r: final_at_epoch(r, "train_loss"))]
    lines = ["| 比較 | A − B | 指標 | シードごとの差 | 平均 ± 標準誤差 |", "|---|---|---|---|---:|"]
    for label, a, b in pairs:
        run_a = {r["seed"]: r for r in select(runs, a)}
        run_b = {r["seed"]: r for r in select(runs, b)}
        seeds = sorted(set(run_a) & set(run_b))
        if not seeds:
            continue
        for name, func in metrics:
            diffs = [func(run_a[s]) - func(run_b[s]) for s in seeds]
            per_seed = ", ".join(f"{s}: {d:+.4f}" for s, d in zip(seeds, diffs))
            lines.append(f"| {label} | {METHOD_LABELS[a]} − {METHOD_LABELS[b]} | {name} | {per_seed} | {_fmt(diffs)} |")
    return "\n".join(lines) + "\n"


def omega_order_table(runs: list) -> str:
    """
    概要: 平均勾配の3手法について，学習後半の $ \\omega({\\mathbf z}_{s})/K $ の中央値と近似誤差の中央値の大小の順序をシードごとに比べる．
    引数: runs (list of dict)．
    戻り値: text (str)．Markdown．
    """
    methods = AVERAGED_GRADIENT_METHODS
    by_method = {m: {r["seed"]: r for r in select(runs, m)} for m in methods}
    seeds = sorted(set.intersection(*(set(v) for v in by_method.values()))) if all(by_method.values()) else []
    lines = ["| シード | $ \\omega/K $ の小さい順 | 近似誤差の小さい順 | 相対近似誤差の小さい順 | 順序の一致（近似誤差） | 順序の一致（相対近似誤差） |",
             "|---:|---|---|---|---|---|"]
    n_match, n_match_rel = 0, 0
    for s in seeds:
        omega = {m: latter_half_median(by_method[m][s], "omega_per_K") for m in methods}
        err = {m: latter_half_median(by_method[m][s], "approx_error") for m in methods}
        rel = {m: latter_half_median(by_method[m][s], "rel_approx_error") for m in methods}
        order = [sorted(methods, key=d.get) for d in (omega, err, rel)]
        match, match_rel = order[0] == order[1], order[0] == order[2]
        n_match += match
        n_match_rel += match_rel
        names = [" < ".join(METHOD_LABELS[m] for m in o) for o in order]
        lines.append(f"| {s} | {names[0]} | {names[1]} | {names[2]} | {'○' if match else '×'} | {'○' if match_rel else '×'} |")
    means = {m: np.mean([latter_half_median(r, "omega_per_K") for r in by_method[m].values()]) for m in methods}
    errs = {m: np.mean([latter_half_median(r, "approx_error") for r in by_method[m].values()]) for m in methods}
    lines.append("")
    lines.append(f"一致したシード数：近似誤差 {n_match}/{len(seeds)}，相対近似誤差 {n_match_rel}/{len(seeds)}．"
                 f"シード平均の順序：$ \\omega/K $ は {' < '.join(METHOD_LABELS[m] for m in sorted(methods, key=means.get))}，"
                 f"近似誤差は {' < '.join(METHOD_LABELS[m] for m in sorted(methods, key=errs.get))}．")
    lines.append("")
    lines.append("全エポック（2〜48）をまとめた，シードごとの $ \\log_{10}(\\omega/K) $ と $ \\log_{10}\\|{\\mathbf e}_{s}\\|^{2} $ の相関係数（手法ごと）：")
    lines.append("")
    for m in methods:
        corrs = []
        for r in by_method[m].values():
            om = np.asarray(r["log"]["omega_per_K"], dtype=np.float64)
            er = np.asarray(r["log"]["approx_error"], dtype=np.float64)[:len(om)]
            mask = np.isfinite(om) & np.isfinite(er) & (om > 0) & (er > 0)
            if mask.sum() > 2:
                corrs.append(float(np.corrcoef(np.log10(om[mask]), np.log10(er[mask]))[0, 1]))
        lines.append(f"- {METHOD_LABELS[m]}：{_fmt(corrs, digits=3)}")
    return "\n".join(lines) + "\n"


def write_summary(experiment_dir: str = EXPERIMENT_DIR, output: str = SUMMARY_PATH) -> str:
    """
    概要: 図と表を作成し，`outputs/e2_factorial_summary.md` に保存する．
    引数: experiment_dir (str)．結果のディレクトリ．output (str)．表の保存先．
    戻り値: path (str)．保存した Markdown のパス．
    """
    runs = load_runs(experiment_dir)
    if not runs:
        raise FileNotFoundError(f"結果がありません: {experiment_dir}")
    paths = plot_six_methods(runs) + [plot_omega_and_error(runs)] + plot_pairs_by_snapshot(runs)
    sections = [
        "# E2：スナップショットの選択方法と全勾配の推定方法の要因分離（CIFAR-10，集計結果）\n",
        "CIFAR-10・AlexNet（LayerNorm），$ \\eta = 10^{-3} $，バッチサイズ128（$ K = 391 $），48エポック，"
        "$ \\lambda = 5\\times10^{-4} $，`inner_start=\"snapshot\"`．値はシード間の平均 ± 標準誤差である．\n",
        "- 最終値 (a)：$ \\#\\mathrm{grad}/N \\le 90 $ の最後の評価点（予算をそろえた比較）．全勾配の手法は約30エポック，"
        "平均勾配の手法は45エポックの評価点に当たる．",
        "- 最終値 (b)：48エポック目（同じエポック数の比較．全勾配の手法は $ \\#\\mathrm{grad}/N \\approx 145 $，平均勾配の手法は96）．",
        "- 打ち切りの場合は打ち切り直前の評価点を用いる．学習後半の中央値はエポック25〜48の評価点の中央値をシードごとに求めた値である．",
        "- 平均勾配の手法の1エポック目は $ {\\mathbf g}_{0} = {\\mathbf 0} $，$ {\\mathbf w}_{0}^{0} = {\\mathbf z}_{0} $ より補正勾配が常に $ {\\mathbf 0} $ であり，"
        "内側ループが動かない（$ \\omega = 0 $）．このエポックの $ \\omega $ の比は定義しない．\n",
        "## 6手法の集計\n",
        summary_table(runs),
        "\n平均点の手法の $ \\omega({\\mathbf z}_{s})/\\omega_{\\min} $ の確認：\n",
        ratio_check(runs),
        "\n## 対応のある差：近似誤差の効果（同じスナップショット，平均勾配 − 全勾配）\n",
        paired_table(runs, APPROX_ERROR_EFFECTS),
        "\n## 対応のある差：スナップショットの効果（同じ $ {\\mathbf g}_{s} $ の推定方法）\n",
        paired_table(runs, SNAPSHOT_EFFECTS),
        "\n## 平均勾配の3手法における $ \\omega $ と近似誤差の順序（学習後半の中央値）\n",
        omega_order_table(runs),
        "\n図：\n",
        "\n".join(f"- `{os.path.relpath(p, OUTPUTS_DIR)}`" for p in paths) + "\n",
    ]
    with open(output, "w") as f:
        f.write("\n".join(sections))
    return output


if __name__ == "__main__":
    print(write_summary())
