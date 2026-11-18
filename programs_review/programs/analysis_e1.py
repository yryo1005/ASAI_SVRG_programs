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

使い方:
    python analysis_e1.py            # 図と表を outputs/ 以下に保存する
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_PROGRAMS_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
OUTPUTS_DIR = os.path.join(REVIEW_ROOT, "outputs")
EXPERIMENT = "ex006_e1_mushroom_rho_sampling"
EXPERIMENT_DIR = os.path.join(OUTPUTS_DIR, EXPERIMENT)

METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
METHOD_LABELS = {"SGD": "SGD", "SVRG": "SVRG", "NFG_SVRG": "NFG-SVRG", "ASAI_SVRG": "ASA-SVRG"}
APPROX_METHODS = ["NFG_SVRG", "ASAI_SVRG"]
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
    sections.append("## 表\n")
    sections.append(summary_table(runs))
    sections.append("\n## 図\n")
    sections.append("\n".join(f"- `{os.path.relpath(p, OUTPUTS_DIR)}`" for p in figure_paths) + "\n")
    path = os.path.join(OUTPUTS_DIR, "e1_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(sections))
    return path


if __name__ == "__main__":
    print(write_summary())
