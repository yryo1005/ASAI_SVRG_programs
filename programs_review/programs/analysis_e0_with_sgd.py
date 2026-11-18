"""
E0（内側ループの開始点の影響）の結果を SGD と比較する図と表を作成するモジュール．

SGDは内側ループを持たないため開始点の影響を受けず，学習済みのログ（既存の実験の結果）をそのまま用いる．
本ディレクトリの外のファイルを参照しないため，SGDのログのパスはコマンドライン引数で受け取る．

出力:
- `outputs/{実験名}/figures/e0_with_sgd_{loss,accuracy}.png`：SGD（黒）と3手法 × 2条件を重ねた図．
- `outputs/e0_with_sgd_summary.md`：最終値，SGDとの差，学習曲線の揺らぎの表．

学習曲線の揺らぎは，予算の後半 $ B/2 < \\#\\mathrm{grad}/N \\le B $ における，隣り合う評価点の間の
テスト精度の変化の標準偏差とする（シードごとに計算し，シード間で平均する）．評価の間隔は手法ごとに異なる
（SGDは $ \\#\\mathrm{grad}/N = 1 $ ごと，NFG-SVRG・ASA-SVRGは2ごと，SVRGは3ごと）．間隔の違いの影響を除くため，
SGDは評価点を間引いて間隔2と3にそろえた値も示す．

使い方:
    python analysis_e0_with_sgd.py \\
        --sgd ex001_e0_mushroom_logistic=<既存のoutputs>/ex001_mushroom_logistic/SGD/eta1.049767e-02_lambda9.155984e-01_K7311_epochs100 \\
        --sgd ex002_e0_cifar10_alexnet_layernorm=<既存のoutputs>/ex0024_cifar10_alexnet_layernorm_longrun/SGD/lr0.001_bs128_normlayernorm_lambda0.0005_epochs96 \\
        --sgd ex003_e0_imagewoof_resnet18_layernorm=<既存のoutputs>/ex0051_imagewoof_resnet18_bs128_longrun/SGD/lr0.001_bs128_lambda0.0005_epochs384
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis_e0 import (DEFAULT_LOG_FLOOR, INNER_STARTS, METHOD_LABELS, OUTPUTS_DIR, PROBLEMS,  # noqa: E402
                         _fmt, _plot_metric, final_value, load_runs, mean_and_se, select_runs)

SGD_COLOR = "black"


def load_sgd_runs(condition_dir: str) -> list:
    """
    概要: 既存の実験のSGDの1条件のディレクトリから，全シードの試行を読み込む．
    引数: condition_dir (str)．`{手法}/{条件}` のディレクトリ（直下にシードのディレクトリがある）．
    戻り値: runs (list of dict)．`analysis_e0.load_runs` と同じ形式（`method="SGD"`，`inner_start=None`）．
    """
    runs = []
    for seed in sorted(os.listdir(condition_dir), key=lambda s: (not s.isdigit(), s)):
        run_dir = os.path.join(condition_dir, seed)
        log_path, config_path = os.path.join(run_dir, "log.json"), os.path.join(run_dir, "config.json")
        if not (seed.isdigit() and os.path.exists(log_path) and os.path.exists(config_path)):
            continue
        with open(log_path) as f:
            log = json.load(f)
        with open(config_path) as f:
            config = json.load(f)
        runs.append({"method": "SGD", "inner_start": None, "seed": int(seed), "config": config, "log": log,
                     "x": np.asarray(log["oracle_calls"], dtype=np.float64) / float(config["N_train"])})
    return runs


def roughness(run: dict, key: str, budget: float, step: float = None) -> float:
    """
    概要: 予算の後半における，隣り合う評価点の間の指標の変化の標準偏差を返す．
    引数:
        run (dict)．試行．
        key (str)．指標名．
        budget (float)．横軸の上限 $ B $．
        step (float) = None．指定した場合，$ \\#\\mathrm{grad}/N $ がこの値の整数倍の評価点だけを用いる（間隔をそろえる）．
    戻り値: value (float)．
    """
    values = np.asarray(run["log"][key], dtype=np.float64)
    x = run["x"][:len(values)]
    mask = (x > budget / 2) & (x <= budget + 1e-9) & np.isfinite(values)
    if step is not None:
        mask &= np.isclose(np.mod(x + 1e-9, step), 0.0, atol=1e-6)
    return float(np.std(np.diff(values[mask]))) if mask.sum() > 2 else float("nan")


def _plot_sgd(ax, sgd_runs, key, budget, log_scale, floor):
    """
    概要: SGDの平均 ± 標準誤差を描く．
    引数: ax，sgd_runs (list of dict)，key (str)，budget (float)，log_scale (bool)，floor (float)．
    戻り値: なし
    """
    x, mean, se = mean_and_se(sgd_runs, key, budget, log_scale, floor)
    if log_scale:
        y, lower, upper = 10 ** mean, 10 ** (mean - se), 10 ** (mean + se)
    else:
        y, lower, upper = mean, mean - se, mean + se
    ax.plot(x, y, "-", color=SGD_COLOR, lw=1.0, label=f"SGD (n={len(sgd_runs)})")
    ax.fill_between(x, lower, upper, color=SGD_COLOR, alpha=0.12, lw=0)


def plot_with_sgd(experiment: str, runs: list, sgd_runs: list) -> list:
    """
    概要: 損失（目的関数誤差）とテスト精度について，SGDと3手法 × 2条件を重ねた図を保存する．
    引数: experiment (str)，runs (list of dict)．E0の試行．sgd_runs (list of dict)．SGDの試行．
    戻り値: paths (list of str)．
    """
    spec = PROBLEMS[experiment]
    save_dir = os.path.join(OUTPUTS_DIR, experiment, "figures")
    os.makedirs(save_dir, exist_ok=True)
    panels = [("loss", spec["loss_key"], spec["log_loss"], spec["loss_label"], spec.get("loss_floor", DEFAULT_LOG_FLOOR)),
              ("accuracy", spec["accuracy_key"], False, "Test accuracy", DEFAULT_LOG_FLOOR)]
    paths = []
    for name, key, log_scale, ylabel, floor in panels:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        _plot_sgd(ax, sgd_runs, key, spec["budget"], log_scale, floor)
        _plot_metric(ax, runs, spec["methods"], key, spec["budget"], log_scale, ylabel, floor)
        ax.set_title(spec["title"] + "\nsolid: last_iterate, dashed: snapshot", fontsize=9)
        ax.legend(fontsize=6.5, loc="best")
        fig.tight_layout()
        path = os.path.join(save_dir, f"e0_with_sgd_{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def _mean_se(values):
    """
    概要: 平均と標準誤差を返す．
    引数: values (list of float)．
    戻り値: (mean, se) の組．
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0


def summary_with_sgd(experiment: str, runs: list, sgd_runs: list) -> str:
    """
    概要: SGDと3手法 × 2条件の最終値，SGDとの精度の差，揺らぎの表（Markdown）を作る．
    引数: experiment (str)，runs (list of dict)，sgd_runs (list of dict)．
    戻り値: text (str)．
    """
    spec = PROBLEMS[experiment]
    budget, acc_key, loss_key = spec["budget"], spec["accuracy_key"], spec["loss_key"]
    loss_floor = spec.get("loss_floor", DEFAULT_LOG_FLOOR)

    def loss_text(selected):
        finals = [final_value(r, loss_key, budget) for r in selected]
        if spec["log_loss"]:
            logs = np.log10(np.maximum(finals, loss_floor))
            mean, se = _mean_se(logs)
            return f"{10 ** mean:.2e}（log10: {mean:.2f} ± {se:.2f}）"
        return _fmt(finals, 3)

    sgd_acc = [final_value(r, acc_key, budget) for r in sgd_runs]
    sgd_mean, sgd_se = _mean_se(sgd_acc)
    lines = [f"## {spec['title']}（予算 $ \\#\\mathrm{{grad}}/N \\le {budget} $）", "",
             "| 手法 | 開始点 | 試行数 | 最終テスト精度 | SGDとの差 | 最終の損失 | テスト精度の揺らぎ（後半） |",
             "|---|---|---:|---:|---:|---:|---:|"]
    rough_sgd = [roughness(r, acc_key, budget) for r in sgd_runs]
    lines.append(f"| SGD | － | {len(sgd_runs)} | {_fmt(sgd_acc)} | － | {loss_text(sgd_runs)} | "
                 f"{_fmt(rough_sgd)}（間隔2：{_fmt([roughness(r, acc_key, budget, 2) for r in sgd_runs])}，"
                 f"間隔3：{_fmt([roughness(r, acc_key, budget, 3) for r in sgd_runs])}） |")
    for method in spec["methods"]:
        for inner_start in INNER_STARTS:
            selected = select_runs(runs, method, inner_start)
            if not selected:
                continue
            acc = [final_value(r, acc_key, budget) for r in selected]
            mean, se = _mean_se(acc)
            diff = f"{mean - sgd_mean:+.4f} ± {np.hypot(se, sgd_se):.4f}"
            lines.append(f"| {METHOD_LABELS[method]} | {inner_start} | {len(selected)} | {_fmt(acc)} | {diff} | "
                         f"{loss_text(selected)} | {_fmt([roughness(r, acc_key, budget) for r in selected])} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．`sgd` は「実験名=SGDの条件のディレクトリ」の文字列のリスト．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sgd", nargs="+", required=True, help="実験名=SGDの条件のディレクトリ")
    parser.add_argument("--output", default=os.path.join(OUTPUTS_DIR, "e0_with_sgd_summary.md"))
    return parser.parse_args()


def main() -> None:
    """
    概要: 全ての問題について図と表を作成して保存する．
    引数: なし
    戻り値: なし
    """
    args = parse_args()
    sgd_dirs = dict(item.split("=", 1) for item in args.sgd)
    sections = ["# E0：内側ループの開始点の影響（SGDとの比較）", "",
                "最終値は予算以下の最後の評価点の値，SGDとの差は「平均の差 ± 標準誤差の二乗和の平方根」である．",
                "Mushroomの損失は目的関数誤差（$ 10^{-16} $ で切り上げた幾何平均），画像は学習損失である．",
                "揺らぎは予算の後半の，隣り合う評価点の間のテスト精度の変化の標準偏差である．評価の間隔は SGD が1，",
                "NFG-SVRG・ASA-SVRG が2，SVRG が3（$ \\#\\mathrm{grad}/N $）であり，SGDは間隔をそろえた値も示す．", ""]
    for experiment in PROBLEMS:
        if experiment not in sgd_dirs:
            continue
        runs = load_runs(os.path.join(OUTPUTS_DIR, experiment))
        sgd_runs = load_sgd_runs(sgd_dirs[experiment])
        paths = plot_with_sgd(experiment, runs, sgd_runs)
        sections.append(summary_with_sgd(experiment, runs, sgd_runs))
        sections.append("図：" + "，".join(f"`{os.path.relpath(p, OUTPUTS_DIR)}`" for p in paths) + "\n")
    with open(args.output, "w") as f:
        f.write("\n".join(sections))
    print("\n".join(sections))


if __name__ == "__main__":
    main()
