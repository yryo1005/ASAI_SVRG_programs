"""
E0（内側ループの開始点の影響確認）の結果を集計し，図と表を作成するモジュール．

実験計画2.3節の出力を作成する．
- 各問題について，`inner_start` の2条件を重ねた図（学習損失または目的関数誤差，テスト精度，近似誤差）．
  横軸は $ \\#\\mathrm{grad}/N = \\mathrm{oracle\\_calls}/N_{\\mathrm{train}} $ で，既存の図と同じ予算までを描く．
- 2条件の最終精度（平均 ± 標準誤差）と，学習後半の近似誤差の平均の表．
- シードごとの対応のある差（`snapshot` − `last_iterate`）の表．

最終精度は「予算以下の最後の評価点の精度（打ち切られた試行は打ち切り直前の評価点）」とする
（論文4.4節 Table IIIと同じ定義）．学習後半は予算の後半 $ B/2 < \\#\\mathrm{grad}/N \\le B $ とする．

使い方:
    python analysis_e0.py            # 全ての問題の図と表を outputs/ 以下に保存する
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

INNER_STARTS = ("last_iterate", "snapshot")
INNER_START_LABELS = {"last_iterate": r"$\mathbf{w}_s^0=\mathbf{w}_{s-1}^K$ (last iterate)",
                      "snapshot": r"$\mathbf{w}_s^0=\mathbf{z}_s$ (snapshot)"}
INNER_START_STYLES = {"last_iterate": "-", "snapshot": "--"}
METHOD_LABELS = {"SVRG": "SVRG", "NFG_SVRG": "NFG-SVRG", "ASAI_SVRG": "ASA-SVRG"}
METHOD_COLORS = {"SVRG": "#1f77b4", "NFG_SVRG": "#2ca02c", "ASAI_SVRG": "#d62728"}

# 問題ごとの設定．`loss_key` は縦軸の損失（Mushroomでは目的関数誤差），`budget` は既存の図の横軸の上限．
# `loss_floor` は対数で平均する際の下限である．Mushroomでは $ f({\mathbf w}^{*}) \approx 0.55 $ であり，倍精度の
# 丸め誤差（約 $ 10^{-16} $）より小さい目的関数誤差（0以下の値を含む）は意味を持たないため $ 10^{-16} $ で切り上げる．
# `loss_threshold` は「目的関数誤差が初めてこの値を下回る $ \#\mathrm{grad}/N $」を表に載せるための閾値である．
DEFAULT_LOG_FLOOR = 1e-300
PROBLEMS = {
    "ex001_e0_mushroom_logistic": {
        "title": "Mushroom (logistic regression)",
        "loss_key": "objective_gap", "loss_label": r"$f(\mathbf{z}_s)-f(\mathbf{w}^*)$",
        "accuracy_key": "accuracy", "budget": 31, "log_loss": True, "loss_floor": 1e-16, "loss_threshold": 1e-10,
        "methods": ["SVRG", "NFG_SVRG", "ASAI_SVRG"],
    },
    "ex002_e0_cifar10_alexnet_layernorm": {
        "title": "CIFAR-10 (AlexNet, LayerNorm, batch size 128)",
        "loss_key": "train_loss", "loss_label": "Training loss",
        "accuracy_key": "test_accuracy", "budget": 90, "log_loss": False,
        "methods": ["SVRG", "NFG_SVRG", "ASAI_SVRG"],
    },
    "ex003_e0_imagewoof_resnet18_layernorm": {
        "title": "ImageWoof (ResNet18, LayerNorm, batch size 128)",
        "loss_key": "train_loss", "loss_label": "Training loss",
        "accuracy_key": "test_accuracy", "budget": 200, "log_loss": False,
        "methods": ["SVRG", "NFG_SVRG", "ASAI_SVRG"],
    },
}


def load_runs(experiment_dir: str) -> list:
    """
    概要: 実験ディレクトリ以下の全ての試行（`log.json` と `config.json` の組）を読み込む．
    引数: experiment_dir (str)．`outputs/{実験名}`．
    戻り値: runs (list of dict)．各要素は `method`，`inner_start`，`seed`，`condition`（条件のディレクトリ名），
        `config`，`log`，`x`（$ \\#\\mathrm{grad}/N $ の numpy.ndarray）を持つ．
    """
    runs = []
    if not os.path.isdir(experiment_dir):
        return runs
    for method in sorted(os.listdir(experiment_dir)):
        method_dir = os.path.join(experiment_dir, method)
        if not os.path.isdir(method_dir) or method not in METHOD_LABELS:
            continue
        for condition in sorted(os.listdir(method_dir)):
            for seed in sorted(os.listdir(os.path.join(method_dir, condition))):
                run_dir = os.path.join(method_dir, condition, seed)
                log_path, config_path = os.path.join(run_dir, "log.json"), os.path.join(run_dir, "config.json")
                if not (os.path.exists(log_path) and os.path.exists(config_path)):
                    continue
                with open(log_path) as f:
                    log = json.load(f)
                with open(config_path) as f:
                    config = json.load(f)
                n_train = config.get("N_train") or (config.get("K") or 0) * config.get("batch_size", 1)
                runs.append({
                    "method": method, "inner_start": config["inner_start"], "seed": int(seed),
                    "condition": condition, "config": config, "log": log,
                    "x": np.asarray(log["oracle_calls"], dtype=np.float64) / _n_train(config, log, n_train),
                })
    return runs


def _n_train(config: dict, log: dict, fallback: float) -> float:
    """
    概要: 学習データ数 $ N_{\\mathrm{train}} $ を求める．画像の実験の `config.json` には記録していないため，
        SVRG系手法の1エポックあたりの勾配評価回数（NFG-SVRG・ASA-SVRGは $ 2N $，SVRGは $ 3N $）から逆算する．
    引数:
        config (dict)．`config.json` の内容．
        log (dict)．`log.json` の内容．
        fallback (float)．`config.json` から得た値（あれば優先する）．
    戻り値: n_train (float)．
    """
    if "N_train" in config:
        return float(config["N_train"])
    calls = log["oracle_calls"]
    per_epoch = 3 if config["method"] == "SVRG" else 2
    if len(calls) >= 3:
        return float(calls[2] - calls[1]) / per_epoch
    return float(fallback)


def select_runs(runs: list, method: str, inner_start: str) -> list:
    """
    概要: 手法と内側ループの開始点で試行を絞り込み，シード順に並べる．
    引数:
        runs (list of dict)．`load_runs` の戻り値．
        method (str)．手法名．
        inner_start (str)．内側ループの開始点．
    戻り値: selected (list of dict)．
    """
    selected = [r for r in runs if r["method"] == method and r["inner_start"] == inner_start]
    return sorted(selected, key=lambda r: r["seed"])


def mean_and_se(runs: list, key: str, budget: float, log_scale: bool = False, floor: float = DEFAULT_LOG_FLOOR):
    """
    概要: 全シードの記録が揃っている評価点（最も早く打ち切られた試行まで）について，指標の平均と
        標準誤差を計算する．`log_scale` の場合は常用対数の平均（幾何平均）とその標準誤差を返す．
    引数:
        runs (list of dict)．同一条件の試行．
        key (str)．指標名．
        budget (float)．横軸の上限．
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
    mask = x <= budget + 1e-9
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(mean)
    return x[mask], mean[mask], se[mask]


def final_value(run: dict, key: str, budget: float) -> float:
    """
    概要: 予算以下の最後の評価点の値を返す（打ち切られた試行は打ち切り直前の有限の評価点）．
    引数:
        run (dict)．試行．
        key (str)．指標名．
        budget (float)．横軸の上限．
    戻り値: value (float)．
    """
    values = np.asarray(run["log"][key], dtype=np.float64)
    loss = np.asarray(run["log"].get("train_loss", run["log"].get("objective_gap")), dtype=np.float64)
    valid = (run["x"][:len(values)] <= budget + 1e-9) & np.isfinite(loss[:len(values)])
    return float(values[np.where(valid)[0][-1]])


def first_crossing(run: dict, key: str, threshold: float) -> float:
    """
    概要: 指標が初めて `threshold` 以下になる評価点の $ \\#\\mathrm{grad}/N $ を返す．
    引数:
        run (dict)．試行．
        key (str)．指標名．
        threshold (float)．閾値．
    戻り値: x (float)．到達しない場合はNaN．
    """
    values = np.asarray(run["log"][key], dtype=np.float64)
    hit = np.where(values <= threshold)[0]
    return float(run["x"][hit[0]]) if hit.size else float("nan")


def latter_half_mean(run: dict, key: str, budget: float) -> float:
    """
    概要: 予算の後半 $ B/2 < \\#\\mathrm{grad}/N \\le B $ の評価点における指標の平均を返す．
    引数:
        run (dict)．試行．
        key (str)．指標名．
        budget (float)．横軸の上限 $ B $．
    戻り値: value (float)．該当する評価点がない場合はNaN．
    """
    values = np.asarray(run["log"][key], dtype=np.float64)
    x = run["x"][:len(values)]
    mask = (x > budget / 2) & (x <= budget + 1e-9) & np.isfinite(values)
    return float(values[mask].mean()) if mask.any() else float("nan")


def _plot_metric(ax, runs, methods, key, budget, log_scale, ylabel, floor=DEFAULT_LOG_FLOOR):
    """
    概要: 1つの指標について，手法ごと・開始点ごとの平均 ± 標準誤差を描く．
    引数:
        ax (matplotlib.axes.Axes)．描画先．
        runs (list of dict)．全試行．
        methods (list of str)．描く手法．
        key (str)．指標名．
        budget (float)．横軸の上限．
        log_scale (bool)．縦軸を対数にするか（幾何平均で描く）．
        ylabel (str)．縦軸のラベル．
        floor (float)．対数で平均する際の下限．
    戻り値: なし
    """
    for method, inner_start in ((m, s) for m in methods for s in INNER_STARTS):
        selected = select_runs(runs, method, inner_start)
        if not selected or key not in selected[0]["log"]:
            continue
        x, mean, se = mean_and_se(selected, key, budget, log_scale, floor)
        if log_scale:
            y, lower, upper = 10 ** mean, 10 ** (mean - se), 10 ** (mean + se)
        else:
            y, lower, upper = mean, mean - se, mean + se
        ax.plot(x, y, INNER_START_STYLES[inner_start], color=METHOD_COLORS[method], lw=1.4,
                label=f"{METHOD_LABELS[method]}, {inner_start} (n={len(selected)})")
        ax.fill_between(x, lower, upper, color=METHOD_COLORS[method], alpha=0.15, lw=0)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\#\mathrm{grad}/N$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def plot_problem(experiment: str, runs: list, methods=None, save_dir=None) -> list:
    """
    概要: 1つの問題について，損失（目的関数誤差），テスト精度，近似誤差の3枚の図を作成して保存する．
    引数:
        experiment (str)．実験名（`PROBLEMS` のキー）．
        runs (list of dict)．`load_runs` の戻り値．
        methods (list of str) = None．描く手法．`None` の場合は `PROBLEMS` の既定．
        save_dir (str) = None．保存先．`None` の場合は `outputs/{実験名}/figures`．
    戻り値: paths (list of str)．保存した図のパス．
    """
    spec = PROBLEMS[experiment]
    methods = methods or spec["methods"]
    save_dir = save_dir or os.path.join(OUTPUTS_DIR, experiment, "figures")
    os.makedirs(save_dir, exist_ok=True)
    panels = [
        ("loss", spec["loss_key"], spec["log_loss"], spec["loss_label"], methods, spec.get("loss_floor", DEFAULT_LOG_FLOOR)),
        ("accuracy", spec["accuracy_key"], False, "Test accuracy", methods, DEFAULT_LOG_FLOOR),
        ("approx_error", "approx_error", True, r"$\|\mathbf{e}_s\|^2$", [m for m in methods if m != "SVRG"],
         DEFAULT_LOG_FLOOR),
    ]
    paths = []
    for name, key, log_scale, ylabel, panel_methods, floor in panels:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        _plot_metric(ax, runs, panel_methods, key, spec["budget"], log_scale, ylabel, floor)
        ax.set_title(spec["title"], fontsize=9)
        ax.legend(fontsize=6.5, loc="best")
        fig.tight_layout()
        path = os.path.join(save_dir, f"e0_{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def _fmt(values: list, digits: int = 4, sci: bool = False) -> str:
    """
    概要: 値の列を「平均 ± 標準誤差」の文字列にする．
    引数:
        values (list of float)．シードごとの値．
        digits (int) = 4．小数点以下の桁数．
        sci (bool) = False．指数表記にするか．
    戻り値: text (str)．
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "—"
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
    if sci:
        return f"{mean:.3e} ± {se:.1e}"
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def summary_table(experiment: str, runs: list, methods=None) -> str:
    """
    概要: 手法・開始点ごとの最終精度，最終損失，学習後半の近似誤差の平均，打ち切り数の表（Markdown）を作る．
    引数:
        experiment (str)．実験名．
        runs (list of dict)．`load_runs` の戻り値．
        methods (list of str) = None．表に載せる手法．
    戻り値: text (str)．
    """
    spec = PROBLEMS[experiment]
    methods = methods or spec["methods"]
    budget = spec["budget"]
    loss_header = (f"{spec['loss_key']} ≤ {spec['loss_threshold']:.0e} に初めて到達する #grad/N"
                   if "loss_threshold" in spec else f"最終の{spec['loss_key']}")
    lines = [f"| 手法 | 内側ループの開始点 | 試行数 | 打ち切り | 最終テスト精度 | {loss_header} | "
             f"学習後半の近似誤差の平均 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        for inner_start in INNER_STARTS:
            selected = select_runs(runs, method, inner_start)
            if not selected:
                continue
            acc = [final_value(r, spec["accuracy_key"], budget) for r in selected]
            if "loss_threshold" in spec:
                loss = [first_crossing(r, spec["loss_key"], spec["loss_threshold"]) for r in selected]
            else:
                loss = [final_value(r, spec["loss_key"], budget) for r in selected]
            err = [latter_half_mean(r, "approx_error", budget) for r in selected]
            n_term = sum(bool(r["log"].get("terminated", [False])[-1]) for r in selected)
            err_text = "（全勾配）" if method == "SVRG" else _fmt(err, sci=True)
            lines.append(f"| {METHOD_LABELS[method]} | {inner_start} | {len(selected)} | {n_term} | "
                         f"{_fmt(acc)} | {_fmt(loss, digits=2 if 'loss_threshold' in spec else 4)} | {err_text} |")
    return "\n".join(lines) + "\n"


def paired_difference_table(experiment: str, runs: list, methods=None) -> str:
    """
    概要: 同じシードの2条件の差（`snapshot` − `last_iterate`）を，シードごとと平均 ± 標準誤差で示す表を作る．
        2条件の初期値とデータの順序は同じであるため，対応のある比較となる．
    引数:
        experiment (str)．実験名．
        runs (list of dict)．`load_runs` の戻り値．
        methods (list of str) = None．表に載せる手法．
    戻り値: text (str)．
    """
    spec = PROBLEMS[experiment]
    methods = methods or spec["methods"]
    budget = spec["budget"]
    lines = ["| 手法 | 指標 | シードごとの差（snapshot − last_iterate） | 平均 ± 標準誤差 |", "|---|---|---|---:|"]
    for method in methods:
        last = {r["seed"]: r for r in select_runs(runs, method, "last_iterate")}
        snap = {r["seed"]: r for r in select_runs(runs, method, "snapshot")}
        seeds = sorted(set(last) & set(snap))
        if not seeds:
            continue
        metrics = [("最終テスト精度", spec["accuracy_key"], final_value, False)]
        if "loss_threshold" in spec:
            metrics.append((f"{spec['loss_key']} ≤ {spec['loss_threshold']:.0e} に到達する #grad/N", spec["loss_key"],
                            lambda r, k, b: first_crossing(r, k, spec["loss_threshold"]), False))
        else:
            metrics.append((f"最終の{spec['loss_key']}", spec["loss_key"], final_value, False))
        if method != "SVRG":
            metrics.append(("学習後半の log10 近似誤差", "approx_error",
                            lambda r, k, b: np.log10(latter_half_mean(r, k, b)), False))
        for label, key, func, _ in metrics:
            diffs = [func(snap[s], key, budget) - func(last[s], key, budget) for s in seeds]
            per_seed = ", ".join(f"{s}: {d:+.4f}" for s, d in zip(seeds, diffs))
            lines.append(f"| {METHOD_LABELS[method]} | {label} | {per_seed} | {_fmt(diffs)} |")
    return "\n".join(lines) + "\n"


def identical_runs_check(runs: list, methods: list) -> str:
    """
    概要: 最終点をスナップショットとする手法など，2条件が同一になるはずの手法について，全ての指標が
        完全に一致するかを確認する．
    引数:
        runs (list of dict)．`load_runs` の戻り値．
        methods (list of str)．確認する手法．
    戻り値: text (str)．Markdownの箇条書き．
    """
    lines = []
    for method in methods:
        last = {r["seed"]: r for r in select_runs(runs, method, "last_iterate")}
        snap = {r["seed"]: r for r in select_runs(runs, method, "snapshot")}
        seeds = sorted(set(last) & set(snap))
        identical = all(
            all(np.array_equal(np.asarray(last[s]["log"][k], dtype=np.float64),
                               np.asarray(snap[s]["log"][k], dtype=np.float64), equal_nan=True)
                for k in last[s]["log"] if k != "elapsed_time")
            for s in seeds
        )
        lines.append(f"- {METHOD_LABELS[method]}：{len(seeds)}シードで2条件の全ての記録（`elapsed_time` を除く）が"
                     f"{'完全に一致した' if identical else '一致しなかった'}．")
    return "\n".join(lines) + "\n"


def write_summary(experiments=None) -> str:
    """
    概要: 全ての問題について図と表を作成し，`outputs/e0_summary.md` に保存する．
    引数: experiments (list of str) = None．対象の実験名．`None` の場合は `PROBLEMS` の全て．
    戻り値: path (str)．保存したMarkdownのパス．
    """
    experiments = experiments or list(PROBLEMS)
    sections = ["# E0：内側ループの開始点の影響（集計結果）\n",
                "最終精度は予算以下の最後の評価点（打ち切りの場合は打ち切り直前）の値，学習後半は予算の後半"
                " $ B/2 < \\#\\mathrm{grad}/N \\le B $ である．値は平均 ± 標準誤差（シード間）である．\n"]
    for experiment in experiments:
        runs = load_runs(os.path.join(OUTPUTS_DIR, experiment))
        if not runs:
            continue
        spec = PROBLEMS[experiment]
        paths = plot_problem(experiment, runs)
        sections.append(f"## {spec['title']}（予算 $ \\#\\mathrm{{grad}}/N \\le {spec['budget']} $）\n")
        sections.append(summary_table(experiment, runs))
        sections.append("\n対応のある差：\n\n" + paired_difference_table(experiment, runs))
        if experiment == "ex001_e0_mushroom_logistic":
            sections.append("\n最終点をスナップショットとする手法の確認：\n\n"
                            + identical_runs_check(runs, ["SVRG", "NFG_SVRG"]))
        sections.append("\n図：" + "，".join(f"`{os.path.relpath(p, OUTPUTS_DIR)}`" for p in paths) + "\n")
    path = os.path.join(OUTPUTS_DIR, "e0_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(sections))
    return path


if __name__ == "__main__":
    print(write_summary())
