"""
E1（Mushroomにおける $ \\rho $ の異なる学習率での収束挙動とサンプリング方式の影響）の結果を
集計し，図と表を作成するモジュール．

実験計画3.5節・3.6節，および `実験指示E1.md` 2.5節の出力を作成する．
- 各 `sampling` × 手法について，横軸 $ \\#\\mathrm{grad}/N $，縦軸 目的関数誤差（対数）とし，
  $ \\rho $ の異なる曲線を重ねた図（横軸 $ \\le 31 $ と全区間）．
- 近似誤差（NFG-SVRG・ASA-SVRG）についても同様の図．
- 2つの `sampling` を手法ごとに重ねた図（$ \\rho \\in \\{0.5, 1.0, 298.4\\} $）．
- 表（`outputs/e1_summary.md`）：各 (sampling, $ \\rho $, 手法) について，$ \\eta $，$ \\rho $，
  最終の目的関数誤差，実測の縮小率 $ \\hat{r} $，最終のテスト精度，誤差床，打ち切りの数を
  平均 ± 標準誤差（5シード）で示す．

実験計画 3.4節（SVRG・NFG-SVRG のランダム点での再実施）の出力も作成する．
- 手法 `SVRG_random`，`NFG_SVRG_random` を上の図と表に加える（最終点の表はそのまま残す）．
- random reshuffling の $ \\hat{r} $ と，独立一様サンプリングの誤差床の，ランダム点と最終点の比較表．
- ランダム点と最終点の対応のある差（同じシード，同じ添字の列）．
- ASA-SVRG とランダム点の NFG-SVRG の近似誤差の比．
- ランダム点の手法の各シードの曲線を薄く重ねた確認用の図．

使い方:
    python analysis_e1.py            # 図と表を outputs/ 以下に保存する
    python analysis_e1.py --this-pc-root ../outputs/_verify/ex006_final_this_pc
        # 本PCで再実行した最終点の試行と既存のログ（別のPCで実行）の差を
        # outputs/ex006_e1_mushroom_rho_sampling/consistency_final_this_pc.md に保存する
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from itertools import product as itertools_product  # noqa: E402

_PROGRAMS_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
OUTPUTS_DIR = os.path.join(REVIEW_ROOT, "outputs")
EXPERIMENT = "ex006_e1_mushroom_rho_sampling"
EXPERIMENT_DIR = os.path.join(OUTPUTS_DIR, EXPERIMENT)

METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG", "SVRG_random", "NFG_SVRG_random"]
METHOD_LABELS = {"SGD": "SGD", "SVRG": "SVRG", "NFG_SVRG": "NFG-SVRG", "ASAI_SVRG": "ASA-SVRG",
                 "SVRG_random": "SVRG (random)", "NFG_SVRG_random": "NFG-SVRG (random)"}
APPROX_METHODS = ["NFG_SVRG", "ASAI_SVRG", "NFG_SVRG_random"]
# ランダム点の手法と，対応する最終点の手法．
RANDOM_POINT_PAIRS = {"SVRG_random": "SVRG", "NFG_SVRG_random": "NFG_SVRG"}
# 全勾配を用いる（近似誤差が0の）手法．
FULL_GRADIENT_METHODS = ("SVRG", "SVRG_random")
SAMPLINGS = ("random_reshuffle", "uniform_with_replacement")
SAMPLING_LABELS = {"random_reshuffle": "random reshuffling", "uniform_with_replacement": "uniform w/ replacement"}
SAMPLING_STYLES = {"random_reshuffle": "-", "uniform_with_replacement": "--"}
RHO_CONDITIONS = [0.5, 0.64, 0.8, 1.0, 2.0, 10.0, 298.4]
OVERLAY_RHOS = [0.5, 1.0, 298.4]
BUDGET_SHORT = 31
LOSS_FLOOR = 1e-16
APPROX_FLOOR = 1e-300
HAT_R_THRESHOLD = 1e-12
TAIL_WINDOW = 20

_RHO_CMAP = plt.get_cmap("viridis")
RHO_COLORS = {rho: _RHO_CMAP(i / max(len(RHO_CONDITIONS) - 1, 1)) for i, rho in enumerate(RHO_CONDITIONS)}


def load_runs(experiment_dir: str = EXPERIMENT_DIR) -> list:
    """
    概要: 実験ディレクトリ以下の全ての試行（`log.json` と `config.json` の組）を読み込む．
    引数: experiment_dir (str) = EXPERIMENT_DIR．
    戻り値: runs (list of dict)．各要素は `method`，`sampling`，`target_rho`，`inner_start`，`seed`，
        `config`，`log`，`x`（$ \\#\\mathrm{grad}/N $ の numpy.ndarray）を持つ．
    """
    runs = []
    if not os.path.isdir(experiment_dir):
        return runs
    for method in sorted(os.listdir(experiment_dir)):
        method_dir = os.path.join(experiment_dir, method)
        if not os.path.isdir(method_dir) or method not in METHODS:
            continue
        for condition in sorted(os.listdir(method_dir)):
            condition_dir = os.path.join(method_dir, condition)
            if not os.path.isdir(condition_dir):
                continue
            for seed in sorted(os.listdir(condition_dir)):
                run_dir = os.path.join(condition_dir, seed)
                log_path, config_path = os.path.join(run_dir, "log.json"), os.path.join(run_dir, "config.json")
                if not (os.path.exists(log_path) and os.path.exists(config_path)):
                    continue
                with open(log_path) as f:
                    log = json.load(f)
                with open(config_path) as f:
                    config = json.load(f)
                n_train = float(config["N_train"])
                runs.append({
                    "method": method, "sampling": config["sampling"], "target_rho": config["target_rho"],
                    "inner_start": config["inner_start"], "seed": int(seed), "condition": condition,
                    "config": config, "log": log,
                    "x": np.asarray(log["oracle_calls"], dtype=np.float64) / n_train,
                })
    return runs


def select_runs(runs: list, method: str, sampling: str, target_rho: float) -> list:
    """
    概要: 手法・サンプリング方式・$ \\rho $ で試行を絞り込み，シード順に並べる．
    引数:
        runs (list of dict)．`load_runs` の戻り値．
        method (str)．手法名．
        sampling (str)．サンプリング方式．
        target_rho (float)．目標の縮小率．
    戻り値: selected (list of dict)．
    """
    selected = [r for r in runs if r["method"] == method and r["sampling"] == sampling
               and abs(r["target_rho"] - target_rho) < 1e-9]
    return sorted(selected, key=lambda r: r["seed"])


def mean_and_se(runs: list, key: str, budget, log_scale: bool = False, floor: float = LOSS_FLOOR):
    """
    概要: 全シードの記録が揃っている評価点（最も早く打ち切られた試行まで）について，指標の平均と
        標準誤差を計算する．`log_scale` の場合は常用対数の平均（幾何平均）とその標準誤差を返す．
    引数:
        runs (list of dict)．同一条件の試行．
        key (str)．指標名．
        budget (float または None)．横軸の上限．`None` の場合は全区間を使う．
        log_scale (bool) = False．対数で平均するか．
        floor (float)．対数をとる前に値を切り上げる下限．
    戻り値:
        x (numpy.ndarray)，形状 (T,)．$ \\#\\mathrm{grad}/N $．
        mean (numpy.ndarray)，形状 (T,)．
        se (numpy.ndarray)，形状 (T,)．
    """
    if not runs:
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


def final_value(run: dict, key: str, budget=None) -> float:
    """
    概要: 予算以下の最後の評価点の値を返す（打ち切られた試行は打ち切り直前の有限の評価点）．
    引数: run (dict)．key (str)．budget (float または None) = None．
    戻り値: value (float)．
    """
    values = np.asarray(run["log"][key], dtype=np.float64)
    loss = np.asarray(run["log"]["objective_gap"], dtype=np.float64)
    x = run["x"][:len(values)]
    mask = np.isfinite(loss[:len(values)])
    if budget is not None:
        mask = mask & (x <= budget + 1e-9)
    idx = np.where(mask)[0]
    return float(values[idx[-1]]) if idx.size else float("nan")


def compute_hat_r(run: dict) -> dict:
    """
    概要: 実測の縮小率 $ \\hat{r} = (\\Delta_{s_{\\mathrm{end}}}/\\Delta_1)^{1/(s_{\\mathrm{end}}-1)} $ を計算する．
        $ \\Delta_s $ は目的関数誤差，$ s_{\\mathrm{end}} $ は $ \\Delta_s > 10^{-12} $ が $ s=1 $ から連続する
        最後の外側ループである．$ s=0 $ は除外する．SGDには適用しない．
    引数: run (dict)．試行．
    戻り値: result (dict)．`s_end`，`hat_r`（floatまたはNone），`fast`（$ s_{\\mathrm{end}} < 3 $ か）を持つ．
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


def tail_median(run: dict, key: str, window: int = TAIL_WINDOW) -> float:
    """
    概要: 最後の `window` 外側ループにおける指標の中央値（誤差床）を計算する．
    引数: run (dict)．key (str)．window (int) = TAIL_WINDOW．
    戻り値: value (float)．有限の値が無い場合はNaN．
    """
    values = np.asarray(run["log"][key], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    tail = values[-window:]
    return float(np.median(tail))


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


def rhos_for_sampling(sampling: str) -> list:
    """
    概要: 指定したサンプリング方式で実施された $ \\rho $ の一覧を返す．
    引数: sampling (str)．
    戻り値: rhos (list of float)．
    """
    return RHO_CONDITIONS if sampling == "random_reshuffle" else OVERLAY_RHOS


def plot_rho_overlay(experiment_dir: str, runs: list, sampling: str, method: str, key: str, budget,
                     log_scale: bool, ylabel: str, floor: float, suffix: str) -> str:
    """
    概要: 1つの (sampling, 手法) について，$ \\rho $ の異なる曲線を重ねた図を作成して保存する．
    引数:
        experiment_dir (str)．保存先の実験ディレクトリ．
        runs (list of dict)．`load_runs` の戻り値．
        sampling (str)．サンプリング方式．
        method (str)．手法名．
        key (str)．指標名．
        budget (float または None)．横軸の上限．
        log_scale (bool)．縦軸を対数にするか．
        ylabel (str)．縦軸のラベル．
        floor (float)．対数で平均する際の下限．
        suffix (str)．ファイル名の末尾（"budget31" または "full"）．
    戻り値: path (str)．保存した図のパス（描く試行が無い場合は None）．
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    drawn = False
    for rho in rhos_for_sampling(sampling):
        selected = select_runs(runs, method, sampling, rho)
        if not selected or key not in selected[0]["log"]:
            continue
        x, mean, se = mean_and_se(selected, key, budget, log_scale, floor)
        if x.size == 0:
            continue
        drawn = True
        if log_scale:
            y, lower, upper = 10 ** mean, 10 ** (mean - se), 10 ** (mean + se)
        else:
            y, lower, upper = mean, mean - se, mean + se
        ax.plot(x, y, "-", color=RHO_COLORS[rho], lw=1.4, label=rf"$\rho={rho:g}$ (n={len(selected)})")
        ax.fill_between(x, lower, upper, color=RHO_COLORS[rho], alpha=0.15, lw=0)
    if not drawn:
        plt.close(fig)
        return None
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\#\mathrm{grad}/N$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{METHOD_LABELS[method]}, {SAMPLING_LABELS[sampling]}", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    save_dir = os.path.join(experiment_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)
    name = "loss" if key == "objective_gap" else "approx_error"
    path = os.path.join(save_dir, f"e1_{name}_{sampling}_{method}_{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sampling_overlay(experiment_dir: str, runs: list, method: str, key: str, log_scale: bool, ylabel: str,
                          floor: float) -> str:
    """
    概要: 1つの手法について，2つの `sampling` を重ねた図（$ \\rho \\in \\{0.5, 1.0, 298.4\\} $）を作成する．
    引数:
        experiment_dir (str)．保存先の実験ディレクトリ．
        runs (list of dict)．`load_runs` の戻り値．
        method (str)．手法名．
        key (str)．指標名．
        log_scale (bool)．縦軸を対数にするか．
        ylabel (str)．縦軸のラベル．
        floor (float)．対数で平均する際の下限．
    戻り値: path (str)．保存した図のパス（描く試行が無い場合は None）．
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    drawn = False
    for rho in OVERLAY_RHOS:
        for sampling in SAMPLINGS:
            selected = select_runs(runs, method, sampling, rho)
            if not selected or key not in selected[0]["log"]:
                continue
            x, mean, se = mean_and_se(selected, key, None, log_scale, floor)
            if x.size == 0:
                continue
            drawn = True
            if log_scale:
                y, lower, upper = 10 ** mean, 10 ** (mean - se), 10 ** (mean + se)
            else:
                y, lower, upper = mean, mean - se, mean + se
            ax.plot(x, y, SAMPLING_STYLES[sampling], color=RHO_COLORS[rho], lw=1.4,
                    label=rf"$\rho={rho:g}$, {SAMPLING_LABELS[sampling]}")
            ax.fill_between(x, lower, upper, color=RHO_COLORS[rho], alpha=0.12, lw=0)
    if not drawn:
        plt.close(fig)
        return None
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\#\mathrm{grad}/N$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{METHOD_LABELS[method]}", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    save_dir = os.path.join(experiment_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)
    name = "loss" if key == "objective_gap" else "approx_error"
    path = os.path.join(save_dir, f"e1_sampling_overlay_{method}_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def summary_table(runs: list) -> str:
    """
    概要: 各 (sampling, $ \\rho $, 手法) について，$ \\eta $，$ \\rho $，最終の目的関数誤差，
        実測の縮小率 $ \\hat{r} $，最終のテスト精度，誤差床，打ち切りの数を平均 ± 標準誤差
        （5シード）で示す表（Markdown）を作る．
    引数: runs (list of dict)．`load_runs` の戻り値．
    戻り値: text (str)．
    """
    lines = ["| sampling | ρ（目標） | 手法 | η | ρ（実測） | 最終の目的関数誤差 | 実測の縮小率 r̂ | "
             "最終のテスト精度 | 誤差床（目的関数誤差） | 誤差床（近似誤差） | 打ち切り数 |",
             "|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for sampling in SAMPLINGS:
        for rho in rhos_for_sampling(sampling):
            for method in METHODS:
                selected = select_runs(runs, method, sampling, rho)
                if not selected:
                    continue
                eta = [r["config"]["eta"] for r in selected]
                actual_rho = [r["config"]["rho"] for r in selected]
                final_gap = [final_value(r, "objective_gap") for r in selected]
                final_acc = [final_value(r, "accuracy") for r in selected]
                gap_floor = [tail_median(r, "objective_gap") for r in selected]
                n_terminated = sum(bool(r["log"].get("terminated", [False])[-1]) for r in selected)
                if method == "SGD":
                    hat_r_text = "（適用外）"
                    approx_floor_text = "（適用外）"
                else:
                    hat_r_infos = [compute_hat_r(r) for r in selected]
                    n_fast = sum(1 for info in hat_r_infos if info["fast"])
                    hat_r_values = [info["hat_r"] for info in hat_r_infos if info["hat_r"] is not None]
                    hat_r_text = _fmt(hat_r_values, digits=4)
                    if n_fast:
                        hat_r_text += f"（{n_fast}シードは1〜2エポックで$10^{{-12}}$に到達）"
                    approx_floor_text = _fmt([tail_median(r, "approx_error") for r in selected], sci=True)
                lines.append(
                    f"| {SAMPLING_LABELS[sampling]} | {rho:g} | {METHOD_LABELS[method]} | {_fmt(eta, digits=6)} | "
                    f"{_fmt(actual_rho, digits=4)} | {_fmt(final_gap, sci=True)} | {hat_r_text} | "
                    f"{_fmt(final_acc)} | {_fmt(gap_floor, sci=True)} | {approx_floor_text} | {n_terminated} |")
    return "\n".join(lines) + "\n"


def _paired(runs: list, method_a: str, method_b: str, sampling: str, rho: float) -> list:
    """
    概要: 2つの手法の同じシードの試行の組を返す．
    引数:
        runs (list of dict)．`load_runs` の戻り値．
        method_a, method_b (str)．手法名．
        sampling (str)．サンプリング方式．
        rho (float)．目標の縮小率．
    戻り値: pairs (list of tuple)．(run_a, run_b) の組のシード順の列．
    """
    by_seed_b = {r["seed"]: r for r in select_runs(runs, method_b, sampling, rho)}
    return [(r, by_seed_b[r["seed"]]) for r in select_runs(runs, method_a, sampling, rho) if r["seed"] in by_seed_b]


def _log10_ratio(a: float, b: float, floor: float = LOSS_FLOOR) -> float:
    """
    概要: $ \\log_{10}(\\max(a, f)/\\max(b, f)) $ を計算する（$ f $ は下限）．
    引数: a, b (float)．floor (float) = LOSS_FLOOR．
    戻り値: value (float)．どちらかが非有限の場合はNaN．
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(np.log10(max(a, floor)) - np.log10(max(b, floor)))


def _hat_r_text(selected: list) -> str:
    """
    概要: 試行の列の $ \\hat{r} $ を「平均 ± 標準誤差」の文字列にする（1〜2エポックで到達した数を注記）．
    引数: selected (list of dict)．
    戻り値: text (str)．
    """
    infos = [compute_hat_r(r) for r in selected]
    text = _fmt([i["hat_r"] for i in infos if i["hat_r"] is not None], digits=4)
    n_fast = sum(1 for i in infos if i["fast"])
    return text + (f"（{n_fast}シードは1〜2エポックで到達）" if n_fast else "")


def hat_r_comparison_table(runs: list) -> str:
    """
    概要: random reshuffling の各 $ \\rho $ について，SVRG・NFG-SVRG のランダム点と最終点の
        $ \\hat{r} $，$ s_{\\mathrm{end}} $，$ \\hat{r} < \\rho $ を満たすシード数，ASA-SVRG の $ \\hat{r} $ を示す表を作る．
    引数: runs (list of dict)．
    戻り値: text (str)．Markdown の表．
    """
    lines = ["| ρ（目標） | SVRG（ランダム点） | SVRG（最終点） | NFG-SVRG（ランダム点） | NFG-SVRG（最終点） | "
             "ASA-SVRG | s_end（ランダム点 SVRG／NFG） | r̂ < ρ のシード数（ランダム点 SVRG／NFG） |",
             "|---:|---|---|---|---|---|---|---|"]
    for rho in RHO_CONDITIONS:
        cells, s_end_cells, below_cells = [], [], []
        for method in ("SVRG_random", "SVRG", "NFG_SVRG_random", "NFG_SVRG", "ASAI_SVRG"):
            cells.append(_hat_r_text(select_runs(runs, method, "random_reshuffle", rho)))
        for method in ("SVRG_random", "NFG_SVRG_random"):
            infos = [compute_hat_r(r) for r in select_runs(runs, method, "random_reshuffle", rho)]
            s_end_cells.append(_fmt([i["s_end"] for i in infos], digits=1))
            below = sum(1 for i in infos if i["hat_r"] is not None and i["hat_r"] < rho)
            below_cells.append(f"{below}/{len(infos)}")
        lines.append(f"| {rho:g} | " + " | ".join(cells) + f" | {' ／ '.join(s_end_cells)} | "
                     f"{' ／ '.join(below_cells)} |")
    return "\n".join(lines) + "\n"


def floor_comparison_table(runs: list) -> str:
    """
    概要: 独立一様サンプリングの各 $ \\rho $ について，誤差床（最後の20外側ループの中央値）の目的関数誤差と
        近似誤差を，ランダム点・最終点・ASA-SVRG で比べる表を作る．
    引数: runs (list of dict)．
    戻り値: text (str)．Markdown の表．
    """
    methods = ("SVRG_random", "SVRG", "NFG_SVRG_random", "NFG_SVRG", "ASAI_SVRG")
    lines = ["| ρ（目標） | 指標 | SVRG（ランダム点） | SVRG（最終点） | NFG-SVRG（ランダム点） | NFG-SVRG（最終点） | "
             "ASA-SVRG |",
             "|---:|---|" + "---|" * len(methods)]
    for rho in OVERLAY_RHOS:
        for key, label in (("objective_gap", "目的関数誤差"), ("approx_error", "近似誤差")):
            cells = []
            for method in methods:
                if key == "approx_error" and method in FULL_GRADIENT_METHODS:
                    cells.append("（0）")
                    continue
                cells.append(_fmt([tail_median(r, key) for r in select_runs(runs, method, "uniform_with_replacement",
                                                                               rho)], sci=True))
            lines.append(f"| {rho:g} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def paired_difference_table(runs: list) -> str:
    """
    概要: ランダム点と最終点の対応のある差（同じシード，同じ添字の列）を示す表を作る．
        random reshuffling では $ \\hat{r} $ の差，独立一様サンプリングでは誤差床の常用対数の比を示す．
        両者で最終の目的関数誤差の常用対数の比も示す．
    引数: runs (list of dict)．
    戻り値: text (str)．Markdown の表．
    """
    lines = ["| 手法 | sampling | ρ（目標） | r̂（ランダム点 − 最終点） | "
             "log10 誤差床の比（目的関数誤差） | log10 誤差床の比（近似誤差） | log10 最終の目的関数誤差の比 | n |",
             "|---|---|---:|---|---|---|---|---:|"]
    for random_method, final_method in RANDOM_POINT_PAIRS.items():
        for sampling in SAMPLINGS:
            for rho in rhos_for_sampling(sampling):
                pairs = _paired(runs, random_method, final_method, sampling, rho)
                if not pairs:
                    continue
                d_hat_r = []
                for a, b in pairs:
                    ra, rb = compute_hat_r(a)["hat_r"], compute_hat_r(b)["hat_r"]
                    if ra is not None and rb is not None:
                        d_hat_r.append(ra - rb)
                gap_floor = [_log10_ratio(tail_median(a, "objective_gap"), tail_median(b, "objective_gap"))
                             for a, b in pairs]
                final_gap = [_log10_ratio(final_value(a, "objective_gap"), final_value(b, "objective_gap"))
                             for a, b in pairs]
                is_uniform = sampling == "uniform_with_replacement"
                hat_r_text = "—" if is_uniform else _fmt(d_hat_r, digits=4)
                floor_text = _fmt(gap_floor, digits=3) if is_uniform else "—"
                if random_method in FULL_GRADIENT_METHODS:
                    approx_floor = "（0）"
                elif is_uniform:
                    approx_floor = _fmt([_log10_ratio(tail_median(a, "approx_error"), tail_median(b, "approx_error"),
                                                      APPROX_FLOOR) for a, b in pairs], digits=3)
                else:
                    approx_floor = "—"
                lines.append(f"| {METHOD_LABELS[final_method]} | {SAMPLING_LABELS[sampling]} | {rho:g} | "
                             f"{hat_r_text} | {floor_text} | {approx_floor} | {_fmt(final_gap, digits=3)} | "
                             f"{len(pairs)} |")
    return "\n".join(lines) + "\n"


def approx_error_ratio_table(runs: list, budget: float = BUDGET_SHORT) -> str:
    """
    概要: ASA-SVRG とランダム点の NFG-SVRG の近似誤差の比（同じシード）を示す表を作る．
        $ \\#\\mathrm{grad}/N \\le $ `budget` の各外側ループ（$ s \\ge 1 $）の $ \\log_{10} $ の比の中央値，
        その比が負（ASA-SVRG が小さい）の外側ループの割合，誤差床の比を示す．
    引数: runs (list of dict)．budget (float) = BUDGET_SHORT．
    戻り値: text (str)．Markdown の表．
    """
    lines = [f"| sampling | ρ（目標） | log10(ASA/NFG(random)) の中央値（#grad/N ≤ {budget}） | "
             "ASA が小さい外側ループの割合 | log10 誤差床の比（ASA/NFG(random)） | n |",
             "|---|---:|---|---|---|---:|"]
    for sampling in SAMPLINGS:
        for rho in rhos_for_sampling(sampling):
            pairs = _paired(runs, "ASAI_SVRG", "NFG_SVRG_random", sampling, rho)
            if not pairs:
                continue
            medians, fractions, floors = [], [], []
            for a, b in pairs:
                length = min(len(a["log"]["approx_error"]), len(b["log"]["approx_error"]))
                x = a["x"][:length]
                mask = (x <= budget + 1e-9) & (np.arange(length) >= 1)
                ea = np.maximum(np.asarray(a["log"]["approx_error"][:length], dtype=np.float64), APPROX_FLOOR)
                eb = np.maximum(np.asarray(b["log"]["approx_error"][:length], dtype=np.float64), APPROX_FLOOR)
                ratio = (np.log10(ea) - np.log10(eb))[mask]
                ratio = ratio[np.isfinite(ratio)]
                if ratio.size:
                    medians.append(float(np.median(ratio)))
                    fractions.append(float(np.mean(ratio < 0.0)))
                floors.append(_log10_ratio(tail_median(a, "approx_error"), tail_median(b, "approx_error"),
                                           APPROX_FLOOR))
            lines.append(f"| {SAMPLING_LABELS[sampling]} | {rho:g} | {_fmt(medians, digits=3)} | "
                         f"{_fmt(fractions, digits=3)} | {_fmt(floors, digits=3)} | {len(pairs)} |")
    return "\n".join(lines) + "\n"


def random_point_sections(runs: list) -> str:
    """
    概要: ランダム点の SVRG・NFG-SVRG（実験計画 3.4節）の集計の節を作る．ランダム点の試行が無い場合は空文字列を返す．
    引数: runs (list of dict)．
    戻り値: text (str)．Markdown．
    """
    if not any(r["method"] in RANDOM_POINT_PAIRS for r in runs):
        return ""
    parts = [
        "\n## ランダム点の SVRG・NFG-SVRG（実験計画 3.4節）\n",
        "ランダム点の手法はスナップショットを $ \\{\\mathbf w_s^k\\}_{k=0}^{K-1} $ から一様ランダムに選ぶ．"
        "添字の列は最終点の手法・ASA-SVRG と同一であり（選択用の乱数生成器を分けた），差はスナップショットの選び方のみによる．"
        "目的関数誤差は選ばれた点 $ \\mathbf z_s $ で評価する．\n",
        "### random reshuffling の実測の縮小率 r̂\n",
        hat_r_comparison_table(runs),
        "\n### 独立一様サンプリングの誤差床（最後の20外側ループの中央値）\n",
        "SVRG は全勾配を用いるため近似誤差は0である．\n",
        floor_comparison_table(runs),
        "\n### ランダム点と最終点の対応のある差（同じシード，同じ添字の列）\n",
        "r̂ の差は random reshuffling，誤差床の比は独立一様サンプリングで示す．比は log10(ランダム点／最終点) であり，"
        "負はランダム点の方が小さいことを表す．目的関数誤差は $ 10^{-16} $ で切り上げてから比をとる．\n",
        paired_difference_table(runs),
        "\n### ASA-SVRG とランダム点の NFG-SVRG の近似誤差（同じシード）\n",
        approx_error_ratio_table(runs),
    ]
    return "\n".join(parts)


def plot_random_point_seed_curves(experiment_dir: str, runs: list) -> list:
    """
    概要: ランダム点の手法について，平均 ± 標準誤差（対数の平均）に各シードの曲線を薄く重ねた確認用の図を，
        (sampling, 手法, 指標) ごとに作成する．$ \\rho $ ごとに色を分ける．
    引数:
        experiment_dir (str)．保存先の実験ディレクトリ．
        runs (list of dict)．`load_runs` の戻り値．
    戻り値: paths (list of str)．保存した図のパス．
    """
    paths = []
    for sampling, method in itertools_product(SAMPLINGS, RANDOM_POINT_PAIRS):
        keys = [("objective_gap", r"$f(\mathbf{z}_s)-f(\mathbf{w}^*)$", LOSS_FLOOR)]
        if method in APPROX_METHODS:
            keys.append(("approx_error", r"$\|\mathbf{e}_s\|^2$", APPROX_FLOOR))
        for key, ylabel, floor in keys:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            drawn = False
            for rho in rhos_for_sampling(sampling):
                selected = select_runs(runs, method, sampling, rho)
                if not selected:
                    continue
                drawn = True
                for r in selected:
                    y = np.maximum(np.asarray(r["log"][key], dtype=np.float64), floor)
                    ax.plot(r["x"][:len(y)], y, "-", color=RHO_COLORS[rho], lw=0.5, alpha=0.3)
                x, mean, _ = mean_and_se(selected, key, None, True, floor)
                ax.plot(x, 10 ** mean, "-", color=RHO_COLORS[rho], lw=1.4, label=rf"$\rho={rho:g}$")
            if not drawn:
                plt.close(fig)
                continue
            ax.set_yscale("log")
            ax.set_xlabel(r"$\#\mathrm{grad}/N$")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{METHOD_LABELS[method]}, {SAMPLING_LABELS[sampling]} (each seed)", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6.5, loc="best")
            fig.tight_layout()
            save_dir = os.path.join(experiment_dir, "figures")
            os.makedirs(save_dir, exist_ok=True)
            name = "loss" if key == "objective_gap" else "approx_error"
            path = os.path.join(save_dir, f"e1_seeds_{name}_{sampling}_{method}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    return paths


def plot_approx_error_comparison(experiment_dir: str, runs: list) -> list:
    """
    概要: $ \\rho \\in \\{0.5, 1.0, 298.4\\} $ の各 (sampling, $ \\rho $) について，ASA-SVRG，ランダム点の
        NFG-SVRG，最終点の NFG-SVRG の近似誤差（平均 ± 標準誤差，対数の平均）を重ねた図を作成する．
    引数:
        experiment_dir (str)．保存先の実験ディレクトリ．
        runs (list of dict)．`load_runs` の戻り値．
    戻り値: paths (list of str)．保存した図のパス．
    """
    styles = {"ASAI_SVRG": ("tab:red", "-"), "NFG_SVRG_random": ("tab:blue", "-"), "NFG_SVRG": ("tab:blue", "--")}
    labels = {"ASAI_SVRG": "ASA-SVRG", "NFG_SVRG_random": "NFG-SVRG (random)", "NFG_SVRG": "NFG-SVRG (final)"}
    paths = []
    for sampling, rho in itertools_product(SAMPLINGS, OVERLAY_RHOS):
        if not select_runs(runs, "NFG_SVRG_random", sampling, rho):
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        for method, (color, style) in styles.items():
            selected = select_runs(runs, method, sampling, rho)
            if not selected:
                continue
            x, mean, se = mean_and_se(selected, "approx_error", None, True, APPROX_FLOOR)
            ax.plot(x, 10 ** mean, style, color=color, lw=1.4, label=labels[method])
            ax.fill_between(x, 10 ** (mean - se), 10 ** (mean + se), color=color, alpha=0.15, lw=0)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\#\mathrm{grad}/N$")
        ax.set_ylabel(r"$\|\mathbf{e}_s\|^2$")
        ax.set_title(rf"{SAMPLING_LABELS[sampling]}, $\rho={rho:g}$", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        save_dir = os.path.join(experiment_dir, "figures")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"e1_approx_error_compare_{sampling}_rho{rho:g}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def write_cross_pc_consistency(this_pc_root: str, experiment_dir: str = EXPERIMENT_DIR) -> str:
    """
    概要: 本PCで再実行した最終点の試行（`this_pc_root`）と既存のログ（別のPCで実行）を，同じ条件・シードで
        比べる．各試行について，整数の項目と精度の一致，目的関数誤差の最大絶対差，$ \\hat{r} $ の差，誤差床の比を
        集計し，Markdown に保存する．
    引数:
        this_pc_root (str)．本PCで再実行した試行のルート．
        experiment_dir (str) = EXPERIMENT_DIR．既存の試行のルート．
    戻り値: path (str)．保存したMarkdownのパス．
    """
    this_pc_runs = load_runs(this_pc_root)
    existing = {(r["method"], r["condition"], r["seed"]): r for r in load_runs(experiment_dir)}
    lines = ["# E1 の最終点の試行の PC 間の差（本PCの再実行 ↔ 既存のログ）\n",
             f"本PCの再実行：`{os.path.relpath(this_pc_root, REVIEW_ROOT)}`，既存のログ："
             f"`{os.path.relpath(experiment_dir, REVIEW_ROOT)}`．\n",
             "| 手法 | sampling | ρ | n | 整数の項目と精度が一致 | f* の差の最大 | 目的関数誤差の最大絶対差 | "
             "r̂ の差の最大絶対値 | log10 誤差床の比の最大絶対値 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    groups = {}
    for r in this_pc_runs:
        key = (r["method"], r["condition"], r["seed"])
        if key in existing:
            groups.setdefault((r["method"], r["sampling"], r["target_rho"]), []).append((r, existing[key]))
    for (method, sampling, rho), pairs in sorted(groups.items()):
        n_equal, f_star_diff, gap_diff, hat_r_diff, floor_ratio = 0, 0.0, 0.0, 0.0, 0.0
        for a, b in pairs:
            equal = all(a["log"][k] == b["log"][k] for k in ("epoch", "oracle_calls", "test_correct", "terminated"))
            n_equal += int(equal)
            f_star_diff = max(f_star_diff, abs(a["config"]["f_star"] - b["config"]["f_star"]))
            ga = np.asarray(a["log"]["objective_gap"], dtype=np.float64)
            gb = np.asarray(b["log"]["objective_gap"], dtype=np.float64)
            length = min(ga.size, gb.size)
            gap_diff = max(gap_diff, float(np.nanmax(np.abs(ga[:length] - gb[:length]))))
            ra, rb = compute_hat_r(a)["hat_r"], compute_hat_r(b)["hat_r"]
            if ra is not None and rb is not None:
                hat_r_diff = max(hat_r_diff, abs(ra - rb))
            ratio = _log10_ratio(tail_median(a, "objective_gap"), tail_median(b, "objective_gap"))
            if np.isfinite(ratio):
                floor_ratio = max(floor_ratio, abs(ratio))
        lines.append(f"| {METHOD_LABELS[method]} | {SAMPLING_LABELS[sampling]} | {rho:g} | {len(pairs)} | "
                     f"{n_equal}/{len(pairs)} | {f_star_diff:.1e} | {gap_diff:.1e} | {hat_r_diff:.1e} | "
                     f"{floor_ratio:.3f} |")
    path = os.path.join(experiment_dir, "consistency_final_this_pc.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_summary(experiment_dir: str = EXPERIMENT_DIR) -> str:
    """
    概要: 全ての図と `outputs/e1_summary.md` を作成する．
    引数: experiment_dir (str) = EXPERIMENT_DIR．
    戻り値: path (str)．保存したMarkdownのパス．
    """
    runs = load_runs(experiment_dir)
    sections = ["# E1：Mushroomにおける ρ の異なる学習率での収束挙動とサンプリング方式の影響（集計結果）\n",
                "値は5シードの平均 ± 標準誤差である．最終値は100エポック終了時（打ち切りの場合は打ち切り直前）の"
                "評価点の値，誤差床は最後の20外側ループにおける中央値である．\n"]
    figure_paths = []
    for sampling in SAMPLINGS:
        for method in METHODS:
            for budget, suffix in ((BUDGET_SHORT, "budget31"), (None, "full")):
                p = plot_rho_overlay(experiment_dir, runs, sampling, method, "objective_gap", budget, True,
                                     r"$f(\mathbf{z}_s)-f(\mathbf{w}^*)$", LOSS_FLOOR, suffix)
                if p:
                    figure_paths.append(p)
            if method in APPROX_METHODS:
                for budget, suffix in ((BUDGET_SHORT, "budget31"), (None, "full")):
                    p = plot_rho_overlay(experiment_dir, runs, sampling, method, "approx_error", budget, True,
                                         r"$\|\mathbf{e}_s\|^2$", APPROX_FLOOR, suffix)
                    if p:
                        figure_paths.append(p)
    for method in METHODS:
        p = plot_sampling_overlay(experiment_dir, runs, method, "objective_gap", True,
                                  r"$f(\mathbf{z}_s)-f(\mathbf{w}^*)$", LOSS_FLOOR)
        if p:
            figure_paths.append(p)
        if method in APPROX_METHODS:
            p = plot_sampling_overlay(experiment_dir, runs, method, "approx_error", True, r"$\|\mathbf{e}_s\|^2$",
                                      APPROX_FLOOR)
            if p:
                figure_paths.append(p)
    figure_paths += plot_random_point_seed_curves(experiment_dir, runs)
    figure_paths += plot_approx_error_comparison(experiment_dir, runs)
    sections.append("## 表\n")
    sections.append(summary_table(runs))
    sections.append(random_point_sections(runs))
    sections.append("\n## 図\n")
    sections.append("\n".join(f"- `{os.path.relpath(p, OUTPUTS_DIR)}`" for p in figure_paths) + "\n")
    path = os.path.join(OUTPUTS_DIR, "e1_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(sections))
    return path


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--this-pc-root", default=None,
                        help="本PCで再実行した最終点の試行のルート（指定した場合のみ PC 間の差を集計する）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(write_summary())
    if args.this_pc_root:
        print(write_cross_pc_consistency(os.path.abspath(args.this_pc_root)))
