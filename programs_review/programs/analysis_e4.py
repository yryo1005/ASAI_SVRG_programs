"""
E4（付録B.1，a9a の MSE・BCE の再実施）の結果を集計し，図と表を作成するモジュール．

`実験計画.md` 6.5節と `実験指示E4.md` 2.5節の出力を作成する．
- 表（`outputs/e4_summary.md`）：各実験・手法について，$ \\#\\mathrm{grad}/N \\le 31 $ の最後の評価点と100外側ループ
  終了時の値を5シードの平均 ± 標準誤差で示す．項目は主な評価量（MSE：$ \\|\\nabla f({\\mathbf z}_{s})\\|^{2} $，
  BCE：目的関数誤差），近似誤差，テスト精度，打ち切りの数．BCE では E1 と同じ定義の実測の縮小率 $ \\hat{r} $ も示す．
  旧実装の ASA-SVRG（既存のログ，`last_iterate`）の同じ値を参考の行として示す．
- 図（`outputs/ex008_e4_a9a/figures/`）：横軸 $ \\#\\mathrm{grad}/N \\le 31 $，5シードの平均 ± 標準誤差．
  MSE：(A) $ \\|\\nabla f({\\mathbf z}_{s})\\|^{2} $，(B) テスト精度，(C) 近似誤差．BCE：(A) 目的関数誤差，(B) テスト精度，
  (C) 近似誤差．各図について，旧実装の ASA-SVRG を破線で重ねた確認用の図（`*_with_old.png`）も作る．
  全区間（100外側ループ）の図（`*_full.png`）も参考として作る．

対数軸の量は，E1（`analysis_e1.py`）と同じく，下限で切り上げた値の常用対数を平均し（幾何平均），
その標準誤差の幅を描く．下限は E1 と同じく，目的関数誤差が $ 10^{-16} $，近似誤差が $ 10^{-300} $ である．
$ \\|\\nabla f({\\mathbf z}_{s})\\|^{2} $ は $ 10^{-16} $ より小さい値まで有意に減少し得るため（2乗のノルムの丸め誤差の水準は
$ 10^{-30} $ 程度），近似誤差と同じ $ 10^{-300} $ とする．

使い方:
    python analysis_e4.py --existing mse=../../outputs/ex000_a9a_least_squares bce=../../outputs/ex0001_a9a_logistic
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_PROGRAMS_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
OUTPUTS_DIR = os.path.join(REVIEW_ROOT, "outputs")
EXPERIMENT = "ex008_e4_a9a"
EXPERIMENT_DIR = os.path.join(OUTPUTS_DIR, EXPERIMENT)

LOSSES = ("mse", "bce")
LOSS_LABELS = {"mse": "a9a, MSE (nonconvex)", "bce": "a9a, BCE + L2 (strongly convex)"}
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
METHOD_LABELS = {"SGD": "SGD", "SVRG": "SVRG", "NFG_SVRG": "NFG-SVRG", "ASAI_SVRG": "ASA-SVRG"}
METHOD_COLORS = {"SGD": "tab:gray", "SVRG": "tab:blue", "NFG_SVRG": "tab:green", "ASAI_SVRG": "tab:red"}
OLD_LABEL = "ASA-SVRG (old impl., last iterate)"
APPROX_METHODS = ["NFG_SVRG", "ASAI_SVRG"]
BUDGET_SHORT = 31
LOSS_FLOOR = 1e-16
SMALL_FLOOR = 1e-300
HAT_R_THRESHOLD = 1e-12

# 各実験の (A) の量．
PRIMARY_KEYS = {"mse": "grad_norm_sq", "bce": "objective_gap"}
PANELS = {
    "A": {"mse": ("grad_norm_sq", True, SMALL_FLOOR, r"$\|\nabla f(\mathbf{z}_s)\|^2$"),
          "bce": ("objective_gap", True, LOSS_FLOOR, r"$f(\mathbf{z}_s)-f(\mathbf{w}^*)$")},
    "B": {"mse": ("accuracy", False, None, "test accuracy"), "bce": ("accuracy", False, None, "test accuracy")},
    "C": {"mse": ("approx_error", True, SMALL_FLOOR, r"$\|\mathbf{e}_s\|^2$"),
          "bce": ("approx_error", True, SMALL_FLOOR, r"$\|\mathbf{e}_s\|^2$")},
    # 補足（MSE のみ）：非凸の MSE では，シグモイド関数の飽和により f が大きいまま勾配が消える点がある
    # （例：全てを負例と予測する点 f = 正例の割合 ≈ 0.2408）．(A) の解釈のため f(z_s) の値も描く．
    "D": {"mse": ("objective_value", False, None, r"$f(\mathbf{z}_s)$")},
}
PANEL_NAMES = {"A": "primary", "B": "accuracy", "C": "approx_error", "D": "objective_value"}


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------

def _read_run(run_dir: str) -> dict:
    """
    概要: 1つの試行の `log.json` と `config.json` を読み込む．
    引数: run_dir (str)．試行のディレクトリ．
    戻り値: run (dict)．`config`，`log`，`x`（$ \\#\\mathrm{grad}/N $ の numpy.ndarray）を持つ．どちらかが無い場合は None．
    """
    log_path, config_path = os.path.join(run_dir, "log.json"), os.path.join(run_dir, "config.json")
    if not (os.path.exists(log_path) and os.path.exists(config_path)):
        return None
    with open(log_path) as f:
        log = json.load(f)
    with open(config_path) as f:
        config = json.load(f)
    x = np.asarray(log["oracle_calls"], dtype=np.float64) / float(config["N_train"])
    return {"config": config, "log": log, "x": x}


def load_runs(experiment_dir: str = EXPERIMENT_DIR) -> list:
    """
    概要: E4 の全ての試行を読み込む．保存先は `{experiment_dir}/{loss}/{手法}/{条件}/{seed}/`．
    引数: experiment_dir (str) = EXPERIMENT_DIR．
    戻り値: runs (list of dict)．各要素は `loss`，`method`，`condition`，`seed`，`config`，`log`，`x` を持つ．
    """
    runs = []
    for loss in LOSSES:
        for method in METHODS:
            method_dir = os.path.join(experiment_dir, loss, method)
            if not os.path.isdir(method_dir):
                continue
            for condition in sorted(os.listdir(method_dir)):
                condition_dir = os.path.join(method_dir, condition)
                for seed in sorted(os.listdir(condition_dir)) if os.path.isdir(condition_dir) else []:
                    run = _read_run(os.path.join(condition_dir, seed))
                    if run is not None:
                        run.update({"loss": loss, "method": method, "condition": condition, "seed": int(seed)})
                        runs.append(run)
    return runs


def strip_inner_prefix(condition: str) -> str:
    """
    概要: 新しい条件名から `inner_{none,snapshot,last_iterate}_` を除き，既存の条件名にする．
    引数: condition (str)．
    戻り値: old_condition (str)．
    """
    for start in ("none", "snapshot", "last_iterate"):
        prefix = f"inner_{start}_"
        if condition.startswith(prefix):
            return condition[len(prefix):]
    return condition


def load_existing_asa(existing_root: str, runs: list, loss: str) -> list:
    """
    概要: 旧実装の ASA-SVRG（既存のログ）のうち，新しい試行と同じ学習率の条件の試行を読み込む．新しい条件名から
        `inner_{inner_start}_` を除いたものが既存の条件名である．
    引数:
        existing_root (str)．既存の実験ディレクトリ（`ex000_a9a_least_squares` または `ex0001_a9a_logistic`）．
        runs (list of dict)．`load_runs` の戻り値．
        loss (str)．"mse" または "bce"．
    戻り値: old_runs (list of dict)．`load_runs` と同じ形式（`method` は "ASAI_SVRG_old"）．
    """
    conditions = {strip_inner_prefix(r["condition"]) for r in runs if r["loss"] == loss and r["method"] == "ASAI_SVRG"}
    old_runs = []
    for condition in sorted(conditions):
        condition_dir = os.path.join(existing_root, "ASAI_SVRG", condition)
        if not os.path.isdir(condition_dir):
            continue
        for seed in sorted(os.listdir(condition_dir)):
            run = _read_run(os.path.join(condition_dir, seed))
            if run is not None:
                run.update({"loss": loss, "method": "ASAI_SVRG_old", "condition": condition, "seed": int(seed)})
                old_runs.append(run)
    return old_runs


def select_runs(runs: list, loss: str, method: str) -> list:
    """
    概要: 損失と手法で試行を絞り込み，シード順に並べる．`inner_last_iterate` の条件は除く．
    引数: runs (list of dict)．loss (str)．method (str)．
    戻り値: selected (list of dict)．
    """
    selected = [r for r in runs if r["loss"] == loss and r["method"] == method
                and not r["condition"].startswith("inner_last_iterate")]
    return sorted(selected, key=lambda r: r["seed"])


# ---------------------------------------------------------------------------
# 統計量
# ---------------------------------------------------------------------------

def mean_and_se(runs: list, key: str, budget, log_scale: bool, floor):
    """
    概要: 全シードの記録が揃っている評価点について，指標の平均と標準誤差を計算する．`log_scale` の場合は
        `floor` で切り上げた値の常用対数の平均（幾何平均）とその標準誤差を返す（E1 と同じ）．
    引数:
        runs (list of dict)．同一条件の試行．
        key (str)．指標名．
        budget (float または None)．横軸の上限．None の場合は全区間．
        log_scale (bool)．対数で平均するか．
        floor (float または None)．対数をとる前の下限．
    戻り値:
        x (numpy.ndarray)，形状 (T,)．
        mean (numpy.ndarray)，形状 (T,)．
        se (numpy.ndarray)，形状 (T,)．
    """
    if not runs or key not in runs[0]["log"]:
        return np.array([]), np.array([]), np.array([])
    length = min(len(r["log"][key]) for r in runs)
    x = runs[0]["x"][:length]
    values = np.array([np.asarray(r["log"][key][:length], dtype=np.float64) for r in runs])
    if log_scale:
        values = np.log10(np.maximum(values, floor))
    mask = x <= budget + 1e-9 if budget is not None else np.ones_like(x, dtype=bool)
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(mean)
    return x[mask], mean[mask], se[mask]


def objective_key(run: dict) -> str:
    """
    概要: 試行のログの目的関数の項目名（MSE：`objective_value`，BCE：`objective_gap`）を返す．
    引数: run (dict)．
    戻り値: key (str)．
    """
    return "objective_value" if "objective_value" in run["log"] else "objective_gap"


def final_value(run: dict, key: str, budget=None) -> float:
    """
    概要: 予算以下の最後の評価点の値を返す（打ち切られた試行は打ち切り直前の有限の評価点）．
    引数: run (dict)．key (str)．budget (float または None) = None．
    戻り値: value (float)．
    """
    values = np.asarray(run["log"][key], dtype=np.float64)
    objective = np.asarray(run["log"][objective_key(run)], dtype=np.float64)
    x = run["x"][:len(values)]
    mask = np.isfinite(objective[:len(values)])
    if budget is not None:
        mask = mask & (x <= budget + 1e-9)
    idx = np.where(mask)[0]
    return float(values[idx[-1]]) if idx.size else float("nan")


def final_x(run: dict, budget=None) -> float:
    """
    概要: `final_value` が用いる評価点の $ \\#\\mathrm{grad}/N $ を返す．
    引数: run (dict)．budget (float または None) = None．
    戻り値: x (float)．
    """
    objective = np.asarray(run["log"][objective_key(run)], dtype=np.float64)
    mask = np.isfinite(objective)
    if budget is not None:
        mask = mask & (run["x"][:len(objective)] <= budget + 1e-9)
    idx = np.where(mask)[0]
    return float(run["x"][idx[-1]]) if idx.size else float("nan")


def compute_hat_r(run: dict) -> dict:
    """
    概要: 実測の縮小率 $ \\hat{r} = (\\Delta_{s_{\\mathrm{end}}}/\\Delta_{1})^{1/(s_{\\mathrm{end}}-1)} $ を計算する（E1 と同じ定義）．
        $ \\Delta_{s} $ は目的関数誤差，$ s_{\\mathrm{end}} $ は $ \\Delta_{s} > 10^{-12} $ が $ s = 1 $ から連続する最後の
        外側ループである．$ s = 0 $ は除外する．
    引数: run (dict)．試行（BCE）．
    戻り値: result (dict)．`s_end`，`hat_r`（float または None），`fast`（$ s_{\\mathrm{end}} < 3 $ か）．
    """
    delta = np.asarray(run["log"]["objective_gap"], dtype=np.float64)
    s_end = 0
    for s in range(1, len(delta)):
        if np.isfinite(delta[s]) and delta[s] > HAT_R_THRESHOLD:
            s_end = s
        else:
            break
    if s_end < 3:
        return {"s_end": s_end, "hat_r": None, "fast": True}
    hat_r = float((delta[s_end] / delta[1]) ** (1.0 / (s_end - 1)))
    return {"s_end": s_end, "hat_r": hat_r, "fast": False}


def accuracy_seed_spread(runs: list, budget=BUDGET_SHORT) -> float:
    """
    概要: テスト精度の試行間のばらつきとして，$ \\#\\mathrm{grad}/N \\le $ `budget` の各評価点（0を除く）における
        5シードの標本標準偏差を，評価点について平均した値を返す．
    引数: runs (list of dict)．同一条件の試行．budget (float) = BUDGET_SHORT．
    戻り値: spread (float)．
    """
    if len(runs) < 2:
        return float("nan")
    length = min(len(r["log"]["accuracy"]) for r in runs)
    x = runs[0]["x"][:length]
    values = np.array([np.asarray(r["log"]["accuracy"][:length], dtype=np.float64) for r in runs])
    mask = (x > 0) & (x <= budget + 1e-9)
    return float(values[:, mask].std(axis=0, ddof=1).mean())


def _fmt(values: list, digits: int = 4, sci: bool = False) -> str:
    """
    概要: 値の列を「平均 ± 標準誤差」の文字列にする．
    引数: values (list of float)．digits (int) = 4．sci (bool) = False．
    戻り値: text (str)．
    """
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "—"
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
    if sci:
        return f"{mean:.3e} ± {se:.1e}"
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def _fmt_geo(values: list, floor: float) -> str:
    """
    概要: 対数軸の量について，幾何平均（`floor` で切り上げた値の常用対数の平均）を文字列にする．
    引数: values (list of float)．floor (float)．
    戻り値: text (str)．
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "—"
    return f"{10 ** np.log10(np.maximum(arr, floor)).mean():.2e}"


# ---------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------

def _row_label(method: str) -> str:
    """
    概要: 表の行の手法名を返す．
    引数: method (str)．
    戻り値: label (str)．
    """
    return "ASA-SVRG（旧実装，`last_iterate`，参考）" if method == "ASAI_SVRG_old" else METHOD_LABELS[method]


def summary_table(runs: list, old_runs: dict, loss: str) -> str:
    """
    概要: 1つの実験（損失）について，主な評価量・近似誤差・テスト精度を $ \\#\\mathrm{grad}/N \\le 31 $ の最後の評価点と
        100外側ループ終了時で示す表（Markdown）を作る．
    引数:
        runs (list of dict)．`load_runs` の戻り値．
        old_runs (dict)．損失から旧実装の ASA-SVRG の試行への対応．
        loss (str)．"mse" または "bce"．
    戻り値: text (str)．
    """
    primary = PRIMARY_KEYS[loss]
    primary_label = "$ \\Vert\\nabla f({\\mathbf z}_{s})\\Vert^{2} $" if loss == "mse" else "目的関数誤差"
    header = (f"| 手法 | シード数 | 評価点の $ \\#\\mathrm{{grad}}/N $ | {primary_label}（≤31） | {primary_label}（100外側ループ） | "
              "近似誤差（≤31） | 近似誤差（100外側ループ） | テスト精度（≤31） | テスト精度（100外側ループ） | "
              "テスト精度の試行間の標準偏差（≤31の平均） | 打ち切り数 |")
    if loss == "mse":
        header += " $ f({\\mathbf z}_{s}) $（≤31） | $ f({\\mathbf z}_{s}) $（100外側ループ） |"
    if loss == "bce":
        header += " 実測の縮小率 $ \\hat{r} $ |"
    lines = [header, "|" + "---|" * (header.count(" | ") + 1)]
    groups = [(m, select_runs(runs, loss, m)) for m in METHODS] + [("ASAI_SVRG_old", old_runs.get(loss, []))]
    for method, selected in groups:
        if not selected:
            continue
        x31 = sorted({final_x(r, BUDGET_SHORT) for r in selected})
        xfull = sorted({final_x(r) for r in selected})
        cells = [_row_label(method), str(len(selected)), f"{', '.join(f'{v:g}' for v in x31)} / "
                 f"{', '.join(f'{v:g}' for v in xfull)}"]
        for key in (primary, "approx_error", "accuracy"):
            for budget in (BUDGET_SHORT, None):
                values = [final_value(r, key, budget) for r in selected]
                if method == "SGD" and key == "approx_error":
                    cells.append("（適用外）")
                elif key == "accuracy":
                    cells.append(_fmt(values))
                else:
                    cells.append(_fmt(values, sci=True))
        cells.append(f"{accuracy_seed_spread(selected):.4f}")
        cells.append(str(sum(bool(r["log"].get("terminated", [False])[-1]) for r in selected)))
        if loss == "mse":
            for budget in (BUDGET_SHORT, None):
                cells.append(_fmt([final_value(r, "objective_value", budget) for r in selected], digits=6))
        if loss == "bce":
            if method == "SGD":
                cells.append("（適用外）")
            else:
                infos = [compute_hat_r(r) for r in selected]
                text = _fmt([i["hat_r"] for i in infos if i["hat_r"] is not None], digits=4)
                n_fast = sum(i["fast"] for i in infos)
                cells.append(text + (f"（{n_fast}シードは1〜2外側ループで $ 10^{{-12}} $ に到達）" if n_fast else ""))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def seed_objective_table(runs: list, old_runs: dict) -> str:
    """
    概要: MSE について，シードごとの100外側ループ終了時の $ f({\\mathbf z}_{s}) $ とテスト精度，および参照値
        `f_star_reference` を示す表を作る．シグモイド関数の飽和により勾配が消える点に留まった試行を見分けるために用いる．
    引数: runs (list of dict)．old_runs (dict)．
    戻り値: text (str)．
    """
    groups = [(m, select_runs(runs, "mse", m)) for m in METHODS] + [("ASAI_SVRG_old", old_runs.get("mse", []))]
    seeds = sorted({r["seed"] for _, sel in groups for r in sel})
    lines = ["| 手法 | " + " | ".join(f"シード{s}" for s in seeds) + " |", "|---|" + "---:|" * len(seeds)]
    for method, selected in groups:
        if not selected:
            continue
        by_seed = {r["seed"]: r for r in selected}
        cells = [f"{final_value(by_seed[s], 'objective_value'):.6f}（{final_value(by_seed[s], 'accuracy'):.4f}）"
                 if s in by_seed else "—" for s in seeds]
        lines.append(f"| {_row_label(method)} | " + " | ".join(cells) + " |")
    refs = {r["seed"]: r["config"].get("f_star_reference") for r in select_runs(runs, "mse", "SVRG")}
    lines.append("| 参照値 `f_star_reference` | " + " | ".join(f"{refs[s]:.6f}" if refs.get(s) else "—" for s in seeds)
                 + " |")
    return "\n".join(lines) + "\n"


def early_approx_table(runs: list, old_runs: dict, loss: str, checkpoints=(2, 4, 8, 16, 30)) -> str:
    """
    概要: 学習の前半の近似誤差の幾何平均（5シード）を，$ \\#\\mathrm{grad}/N $ の数点で示す表を作る
        （実験計画6.6節の論点2のため）．NFG-SVRG と ASA-SVRG の比（NFG/ASA）も示す．
    引数:
        runs (list of dict)．old_runs (dict)．loss (str)．checkpoints (tuple of float)．評価点の $ \\#\\mathrm{grad}/N $．
    戻り値: text (str)．
    """
    lines = ["| 手法 | " + " | ".join(f"$ \\#\\mathrm{{grad}}/N = {c} $" for c in checkpoints) + " |",
             "|---|" + "---:|" * len(checkpoints)]
    geo = {}
    for method, selected in [(m, select_runs(runs, loss, m)) for m in APPROX_METHODS] + \
            [("ASAI_SVRG_old", old_runs.get(loss, []))]:
        if not selected:
            continue
        row = []
        for c in checkpoints:
            values = []
            for r in selected:
                idx = np.where(np.isclose(r["x"], c))[0]
                if idx.size and idx[0] < len(r["log"]["approx_error"]):
                    values.append(r["log"]["approx_error"][idx[0]])
            row.append(_fmt_geo(values, SMALL_FLOOR))
            geo[(method, c)] = 10 ** np.log10(np.maximum(np.asarray(values, dtype=np.float64), SMALL_FLOOR)).mean() \
                if values else float("nan")
        lines.append(f"| {_row_label(method)} | " + " | ".join(row) + " |")
    ratio = [geo.get(("NFG_SVRG", c), np.nan) / geo.get(("ASAI_SVRG", c), np.nan) for c in checkpoints]
    lines.append("| 比 NFG-SVRG / ASA-SVRG | " + " | ".join(f"{v:.1e}" for v in ratio) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------

def _draw(ax, runs, key, budget, log_scale, floor, color, label, style="-"):
    """
    概要: 1つの手法の平均 ± 標準誤差の曲線を描く．
    引数:
        ax (matplotlib.axes.Axes)．runs (list of dict)．key (str)．budget (float または None)．log_scale (bool)．
        floor (float または None)．color (str)．label (str)．style (str) = "-"．
    戻り値: drawn (bool)．描いたか．
    """
    x, mean, se = mean_and_se(runs, key, budget, log_scale, floor)
    if x.size == 0:
        return False
    if log_scale:
        y, lower, upper = 10 ** mean, 10 ** (mean - se), 10 ** (mean + se)
    else:
        y, lower, upper = mean, mean - se, mean + se
    ax.plot(x, y, style, color=color, lw=1.4, label=f"{label} (n={len(runs)})", marker="o", ms=2)
    ax.fill_between(x, lower, upper, color=color, alpha=0.15, lw=0)
    return True


def plot_panel(experiment_dir: str, runs: list, old_runs: dict, loss: str, panel: str, budget, with_old: bool) -> str:
    """
    概要: 1つの実験の1つの図（(A)，(B)，(C)）を作成して保存する．
    引数:
        experiment_dir (str)．保存先の実験ディレクトリ．
        runs (list of dict)．old_runs (dict)．loss (str)．panel (str)．"A"，"B"，"C"．
        budget (float または None)．横軸の上限．with_old (bool)．旧実装の ASA-SVRG を破線で重ねるか．
    戻り値: path (str)．保存した図のパス（描く試行が無い場合は None）．
    """
    key, log_scale, floor, ylabel = PANELS[panel][loss]
    methods = APPROX_METHODS if panel == "C" else METHODS
    fig, ax = plt.subplots(figsize=(5, 5))
    drawn = False
    for method in methods:
        drawn |= _draw(ax, select_runs(runs, loss, method), key, budget, log_scale, floor, METHOD_COLORS[method],
                       METHOD_LABELS[method])
    if with_old and old_runs.get(loss):
        drawn |= _draw(ax, old_runs[loss], key, budget, log_scale, floor, "black", OLD_LABEL, style="--")
    if not drawn:
        plt.close(fig)
        return None
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\#\mathrm{grad}/N$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"({panel}) {LOSS_LABELS[loss]}", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    save_dir = os.path.join(experiment_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)
    suffix = ("budget31" if budget is not None else "full") + ("_with_old" if with_old else "")
    path = os.path.join(save_dir, f"e4_{loss}_{panel}_{PANEL_NAMES[panel]}_{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# まとめ
# ---------------------------------------------------------------------------

def write_summary(existing: dict = None, experiment_dir: str = EXPERIMENT_DIR) -> dict:
    """
    概要: 全ての図と `outputs/e4_summary.md` を作成する．
    引数:
        existing (dict) = None．損失から既存の実験ディレクトリ（旧実装の ASA-SVRG を含む）への対応．
        experiment_dir (str) = EXPERIMENT_DIR．
    戻り値: result (dict)．`summary_path`（str）と `figures`（損失から図のパスのリストへの辞書）．
    """
    existing = existing or {}
    runs = load_runs(experiment_dir)
    old_runs = {loss: load_existing_asa(path, runs, loss) for loss, path in existing.items()}
    figures = {loss: [] for loss in LOSSES}
    for loss in LOSSES:
        for panel in [k for k in PANELS if loss in PANELS[k]]:
            for budget, with_old in ((BUDGET_SHORT, False), (BUDGET_SHORT, True), (None, False), (None, True)):
                if with_old and not old_runs.get(loss):
                    continue
                p = plot_panel(experiment_dir, runs, old_runs, loss, panel, budget, with_old)
                if p:
                    figures[loss].append(p)

    sections = ["# E4：付録B.1（a9a，MSE・BCE）の再実施（集計結果）\n",
                "値は5シードの平均 ± 標準誤差である．「≤31」は $ \\#\\mathrm{grad}/N \\le 31 $ の最後の評価点，"
                "「100外側ループ」は100外側ループ終了時（打ち切りの場合は打ち切り直前）の評価点の値である．"
                "評価点の $ \\#\\mathrm{grad}/N $ の列は「≤31 の評価点 / 最終の評価点」を示す．"
                "SVRG系の手法は内側ループの開始点を `inner_start=\"snapshot\"`（$ {\\mathbf w}_{s}^{0} \\leftarrow {\\mathbf z}_{s} $）とした．"
                "旧実装の ASA-SVRG は既存のログ（`last_iterate`）であり，参考として示す．"
                "テスト精度の試行間の標準偏差は，$ 0 < \\#\\mathrm{grad}/N \\le 31 $ の各評価点における5シードの標本標準偏差の平均である．\n"]
    for loss in LOSSES:
        sections.append(f"## {LOSS_LABELS[loss]}\n")
        sections.append(summary_table(runs, old_runs, loss))
        if loss == "mse":
            sections.append("\nシードごとの100外側ループ終了時の $ f({\\mathbf z}_{s}) $（括弧内はテスト精度）．"
                            "参照値 `f_star_reference` は L-BFGS-B で求めた値である：\n")
            sections.append(seed_objective_table(runs, old_runs))
        sections.append("\n学習の前半の近似誤差（5シードの幾何平均）：\n")
        sections.append(early_approx_table(runs, old_runs, loss))
        if existing.get(loss):
            sections.append(f"\n旧実装のログ：`{existing[loss]}`\n")
    sections.append("\n## 図\n")
    for loss in LOSSES:
        sections.append("\n".join(f"- `{os.path.relpath(p, OUTPUTS_DIR)}`" for p in figures[loss]))
    path = os.path.join(OUTPUTS_DIR, "e4_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(sections) + "\n")
    return {"summary_path": path, "figures": figures}


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--existing", nargs="*", default=[], help="損失=既存の実験ディレクトリ（旧実装の ASA-SVRG）")
    parser.add_argument("--experiment-dir", default=EXPERIMENT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = write_summary(dict(item.split("=", 1) for item in args.existing), args.experiment_dir)
    print(result["summary_path"])
