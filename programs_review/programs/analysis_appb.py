"""
付録B（バッチサイズ32・512）を `inner_start="snapshot"` で再実施した結果を集計し，既存の実験の
`last_iterate`（既存の実装）とSGDの結果と比較する図と表を作成するモジュール．

本ディレクトリの外のファイルを参照しないため，既存の実験の結果のディレクトリはコマンドライン引数で受け取る．
既存の実験の各手法のディレクトリから，同じバッチサイズの条件を探して用いる．SGDは同じバッチサイズの条件のうち
エポック数が最大のものを用いる（予算までの記録を得るため）．

出力:
- `outputs/{実験名}/figures/appb_bs{B}_{loss,accuracy,approx_error}.png`
  - 実線は `last_iterate`（既存のログ），破線は `snapshot`（新しい実行），黒はSGD．
  - 各評価点で，打ち切られていない試行の平均 ± 標準誤差を描く．
- `outputs/appb_summary.md`：最終値，打ち切り数，SGDとの差，学習後半の近似誤差，揺らぎ，開始点の対応のある差の表．

使い方:
    python analysis_appb.py \\
        --existing ex004_appb_cifar10_alexnet_layernorm=<既存のoutputs>/ex0024_cifar10_alexnet_layernorm_longrun \\
                   ex005_appb_imagewoof_resnet18_layernorm=<既存のoutputs>/ex0051_imagewoof_resnet18_bs128_longrun
"""

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis_e0 import (INNER_START_STYLES, METHOD_COLORS, METHOD_LABELS, OUTPUTS_DIR, _fmt,  # noqa: E402
                         final_value, latter_half_mean)
from analysis_e0_with_sgd import SGD_COLOR, roughness  # noqa: E402

METHODS = ["SVRG", "NFG_SVRG", "ASAI_SVRG"]
PROBLEMS = {
    "ex004_appb_cifar10_alexnet_layernorm": {
        "title": "CIFAR-10 (AlexNet, LayerNorm)", "budgets": {32: 24, 512: 384},
    },
    "ex005_appb_imagewoof_resnet18_layernorm": {
        "title": "ImageWoof (ResNet18, LayerNorm)", "budgets": {32: 33, 512: 1010},
    },
}


def _read_run(run_dir: str, method: str, inner_start):
    """
    概要: 1つの試行の `log.json` と `config.json` を読み込む．
    引数: run_dir (str)，method (str)，inner_start (str or None)．
    戻り値: run (dict) または None（ファイルがない場合）．`terminated` は学習損失が非有限で終わったか．
    """
    log_path, config_path = os.path.join(run_dir, "log.json"), os.path.join(run_dir, "config.json")
    if not (os.path.exists(log_path) and os.path.exists(config_path)):
        return None
    with open(log_path) as f:
        log = json.load(f)
    with open(config_path) as f:
        config = json.load(f)
    loss = np.asarray(log["train_loss"], dtype=np.float64)
    return {"method": method, "inner_start": inner_start, "seed": int(os.path.basename(run_dir)),
            "config": config, "log": log, "terminated": bool(loss.size and not np.isfinite(loss[-1])),
            "x": np.asarray(log["oracle_calls"], dtype=np.float64) / float(config["N_train"])}


def _condition_dirs(method_dir: str, batch_size: int) -> list:
    """
    概要: 手法のディレクトリから，指定したバッチサイズの条件のディレクトリを (エポック数, パス) の組で返す．
    引数: method_dir (str)，batch_size (int)．
    戻り値: list of (int, str)．エポック数の昇順．
    """
    if not os.path.isdir(method_dir):
        return []
    found = []
    for name in os.listdir(method_dir):
        match_bs, match_ep = re.search(r"_bs(\d+)_", name), re.search(r"epochs(\d+)$", name)
        if match_bs and match_ep and int(match_bs.group(1)) == batch_size:
            found.append((int(match_ep.group(1)), os.path.join(method_dir, name)))
    return sorted(found)


def _load_condition(condition_dir: str, method: str, inner_start) -> list:
    """
    概要: 条件のディレクトリから全シードの試行を読み込む．
    引数: condition_dir (str)，method (str)，inner_start (str or None)．
    戻り値: runs (list of dict)．シード順．
    """
    runs = [_read_run(os.path.join(condition_dir, s), method, inner_start)
            for s in sorted(os.listdir(condition_dir)) if s.isdigit()]
    return sorted([r for r in runs if r is not None], key=lambda r: r["seed"])


def load_all(experiment: str, existing_root: str, batch_size: int) -> dict:
    """
    概要: 1つの (問題, バッチサイズ) について，新しい `snapshot` の試行，既存の `last_iterate` の試行，SGDを読み込む．
    引数: experiment (str)，existing_root (str)．既存の実験の結果のディレクトリ．batch_size (int)．
    戻り値: dict．キーは (手法, 開始点) の組と "SGD"，値は試行のリスト．
    """
    groups = {}
    for method in METHODS:
        new_dirs = [d for _, d in _condition_dirs(os.path.join(OUTPUTS_DIR, experiment, method), batch_size)
                    if os.path.basename(d).startswith("inner_snapshot_")]
        old_dirs = _condition_dirs(os.path.join(existing_root, method), batch_size)
        groups[(method, "snapshot")] = _load_condition(new_dirs[-1], method, "snapshot") if new_dirs else []
        groups[(method, "last_iterate")] = _load_condition(old_dirs[-1][1], method, "last_iterate") if old_dirs else []
    sgd_dirs = _condition_dirs(os.path.join(existing_root, "SGD"), batch_size)
    groups["SGD"] = _load_condition(sgd_dirs[-1][1], "SGD", None) if sgd_dirs else []
    return groups


def _curve(runs: list, key: str, budget: float, log_scale: bool):
    """
    概要: 評価点ごとに，有限の値を持つ試行の平均 ± 標準誤差を計算する（打ち切られた試行は以降を除く）．
    引数: runs (list of dict)，key (str)，budget (float)，log_scale (bool)．対数で平均するか．
    戻り値: (x, mean, se) の組（numpy.ndarray）．
    """
    length = max(len(r["log"][key]) for r in runs)
    longest = max(runs, key=lambda r: len(r["log"][key]))
    values = np.full((len(runs), length), np.nan)
    for i, r in enumerate(runs):
        v = np.asarray(r["log"][key], dtype=np.float64)
        values[i, :len(v)] = v
    values[~np.isfinite(values)] = np.nan
    if log_scale:
        values = np.log10(np.where(values > 0, values, np.nan))
    x = longest["x"][:length]
    count = np.sum(np.isfinite(values), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(np.where(count > 0, values, 0.0), axis=0)
        mean[count == 0] = np.nan
        se = np.where(count > 1, np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.maximum(count, 1)), 0.0)
    mask = x <= budget + 1e-9
    return x[mask], mean[mask], se[mask]


def plot_batch_size(experiment: str, batch_size: int, groups: dict) -> list:
    """
    概要: 学習損失，テスト精度，近似誤差の図を保存する．
    引数: experiment (str)，batch_size (int)，groups (dict)．`load_all` の戻り値．
    戻り値: paths (list of str)．
    """
    spec = PROBLEMS[experiment]
    budget = spec["budgets"][batch_size]
    save_dir = os.path.join(OUTPUTS_DIR, experiment, "figures")
    os.makedirs(save_dir, exist_ok=True)
    panels = [("loss", "train_loss", True, "Training loss"), ("accuracy", "test_accuracy", False, "Test accuracy"),
              ("approx_error", "approx_error", True, r"$\|\mathbf{e}_s\|^2$")]
    paths = []
    for name, key, log_scale, ylabel in panels:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        series = [] if key == "approx_error" else [("SGD", None, groups["SGD"])]
        series += [(m, s, groups[(m, s)]) for m in METHODS for s in ("last_iterate", "snapshot")
                   if not (key == "approx_error" and m == "SVRG")]
        for method, inner_start, runs in series:
            if not runs:
                continue
            x, mean, se = _curve(runs, key, budget, log_scale)
            y, lower, upper = (10 ** mean, 10 ** (mean - se), 10 ** (mean + se)) if log_scale else \
                (mean, mean - se, mean + se)
            n_term = sum(r["terminated"] for r in runs)
            if method == "SGD":
                color, style, label = SGD_COLOR, "-", f"SGD (n={len(runs)})"
            else:
                color, style = METHOD_COLORS[method], INNER_START_STYLES[inner_start]
                label = f"{METHOD_LABELS[method]}, {inner_start} (n={len(runs)}"
                label += f", {n_term} terminated)" if n_term else ")"
            ax.plot(x, y, style, color=color, lw=1.0 if method == "SGD" else 1.4, label=label)
            ax.fill_between(x, lower, upper, color=color, alpha=0.12, lw=0)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel(r"$\#\mathrm{grad}/N$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_title(f"{spec['title']}, batch size {batch_size}\nsolid: last_iterate (existing), dashed: snapshot",
                     fontsize=9)
        ax.legend(fontsize=6.5, loc="best")
        fig.tight_layout()
        path = os.path.join(save_dir, f"appb_bs{batch_size}_{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def _mean_se(values):
    """
    概要: 有限の値の平均と標準誤差を返す．
    引数: values (list of float)．
    戻り値: (mean, se, n) の組．
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0, int(arr.size)


def summary_batch_size(experiment: str, batch_size: int, groups: dict) -> str:
    """
    概要: 1つの (問題, バッチサイズ) の表（Markdown）を作る．
    引数: experiment (str)，batch_size (int)，groups (dict)．
    戻り値: text (str)．
    """
    spec = PROBLEMS[experiment]
    budget = spec["budgets"][batch_size]
    sgd = groups["SGD"]
    sgd_acc = [final_value(r, "test_accuracy", budget) for r in sgd]
    sgd_mean, sgd_se, _ = _mean_se(sgd_acc)
    lines = [f"## {spec['title']}，バッチサイズ {batch_size}（予算 $ \\#\\mathrm{{grad}}/N \\le {budget} $）", "",
             "| 手法 | 開始点 | 試行数 | 打ち切り | 最終テスト精度 | SGDとの差 | 最終の学習損失 | 学習後半の近似誤差の平均 | 精度の揺らぎ（後半） |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append(f"| SGD | － | {len(sgd)} | {sum(r['terminated'] for r in sgd)} | {_fmt(sgd_acc)} | － | "
                 f"{_fmt([final_value(r, 'train_loss', budget) for r in sgd], 3)} | － | "
                 f"{_fmt([roughness(r, 'test_accuracy', budget) for r in sgd])}（間隔2：{_fmt([roughness(r, 'test_accuracy', budget, 2) for r in sgd])}） |")
    for method in METHODS:
        for inner_start in ("last_iterate", "snapshot"):
            runs = groups[(method, inner_start)]
            if not runs:
                lines.append(f"| {METHOD_LABELS[method]} | {inner_start} | 0 | － | － | － | － | － | － |")
                continue
            acc = [final_value(r, "test_accuracy", budget) for r in runs]
            mean, se, _ = _mean_se(acc)
            approx = "（全勾配）" if method == "SVRG" else \
                _fmt([latter_half_mean(r, "approx_error", budget) for r in runs], sci=True)
            lines.append(
                f"| {METHOD_LABELS[method]} | {inner_start} | {len(runs)} | {sum(r['terminated'] for r in runs)} | "
                f"{_fmt(acc)} | {mean - sgd_mean:+.4f} ± {np.hypot(se, sgd_se):.4f} | "
                f"{_fmt([final_value(r, 'train_loss', budget) for r in runs], 3)} | {approx} | "
                f"{_fmt([roughness(r, 'test_accuracy', budget) for r in runs])} |")
    lines += ["", "開始点の対応のある差（snapshot − last_iterate，同じシードどうし）：", "",
              "| 手法 | 最終テスト精度 | 最終の学習損失 |", "|---|---:|---:|"]
    for method in METHODS:
        old = {r["seed"]: r for r in groups[(method, "last_iterate")]}
        new = {r["seed"]: r for r in groups[(method, "snapshot")]}
        seeds = sorted(set(old) & set(new))
        if not seeds:
            continue
        cells = []
        for key in ("test_accuracy", "train_loss"):
            diffs = [final_value(new[s], key, budget) - final_value(old[s], key, budget) for s in seeds]
            m, e, n = _mean_se(diffs)
            per_seed = ", ".join(f"{s}: {d:+.3f}" for s, d in zip(seeds, diffs))
            cells.append(f"{m:+.4f} ± {e:.4f}（{per_seed}）")
        lines.append(f"| {METHOD_LABELS[method]} | {cells[0]} | {cells[1]} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．`existing` は「実験名=既存の実験の結果のディレクトリ」の文字列のリスト．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--existing", nargs="+", required=True, help="実験名=既存の実験の結果のディレクトリ")
    parser.add_argument("--output", default=os.path.join(OUTPUTS_DIR, "appb_summary.md"))
    return parser.parse_args()


def main() -> None:
    """
    概要: 全ての (問題, バッチサイズ) について図と表を作成して保存する．
    引数: なし
    戻り値: なし
    """
    args = parse_args()
    existing = dict(item.split("=", 1) for item in args.existing)
    sections = ["# 付録B（バッチサイズ32・512）：内側ループの開始点を snapshot とした再実施", "",
                "`last_iterate` とSGDは既存の実験のログ，`snapshot` は新しい実行である．最終値は予算以下の最後の評価点",
                "（打ち切られた試行は打ち切り直前の有限の評価点）の値，学習後半は $ B/2 < \\#\\mathrm{grad}/N \\le B $ である．",
                "揺らぎは学習後半の隣り合う評価点の間のテスト精度の変化の標準偏差である．値は平均 ± 標準誤差（シード間）である．", ""]
    for experiment, spec in PROBLEMS.items():
        if experiment not in existing:
            continue
        for batch_size in spec["budgets"]:
            groups = load_all(experiment, existing[experiment], batch_size)
            paths = plot_batch_size(experiment, batch_size, groups)
            sections.append(summary_batch_size(experiment, batch_size, groups))
            sections.append("図：" + "，".join(f"`{os.path.relpath(p, OUTPUTS_DIR)}`" for p in paths) + "\n")
    with open(args.output, "w") as f:
        f.write("\n".join(sections))
    print("\n".join(sections))


if __name__ == "__main__":
    main()
