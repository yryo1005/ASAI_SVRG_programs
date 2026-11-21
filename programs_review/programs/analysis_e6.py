"""
E6（最終点のスナップショットの SVRG・NFG-SVRG，CIFAR-10，バッチサイズ32・512）の結果を集計するモジュール．

- 値は5シードの平均 ± 標準誤差（`ddof=1`）である．横軸 $ \\#\\mathrm{grad}/N $ は `oracle_calls / 50000` である．
- 予算は付録B と同じく，バッチサイズ32で $ \\#\\mathrm{grad}/N \\le 20 $，512で $ \\le 350 $ とする．
  各試行で予算以下の最後の評価点の値を用いる（打ち切られた試行は打ち切り直前の有限の評価点．`analysis_e0.final_value`）．
- 近似誤差は，予算の後半（予算/2 < $ \\#\\mathrm{grad}/N $ ≤ 予算）の評価点の中央値をシードごとに求めて平均する
  （`NFG_SVRG_final` のみ．`SVRG_final` は全勾配のため0である）．
- 参考として，同じバッチサイズの付録B（`ex004`）の `SVRG`・`NFG_SVRG`・`ASAI_SVRG`（ランダム点または平均点，reset）を
  同じ指標で並べる．

出力:
- `outputs/e6_summary.md`
- `outputs/ex011_e6_cifar10_final_bs32_512/figures/e6_bs{32,512}_{train_loss,test_accuracy,approx_error}.png`

使い方（`programs_review/programs` から）:
    python analysis_e6.py
    python analysis_e6.py --e6-root ../outputs/ex011_e6_cifar10_final_bs32_512 \\
        --reference-root ../outputs/ex004_appb_cifar10_alexnet_layernorm
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis_e0 import OUTPUTS_DIR, _fmt, final_value  # noqa: E402

EXPERIMENT_NAME = "ex011_e6_cifar10_final_bs32_512"
REFERENCE_NAME = "ex004_appb_cifar10_alexnet_layernorm"
N_TRAIN = 50000
BUDGETS = {32: 20, 512: 350}
EPOCHS = {32: 12, 512: 192}
E6_METHODS = ["SVRG_final", "NFG_SVRG_final"]
REFERENCE_METHODS = ["SVRG", "NFG_SVRG", "ASAI_SVRG"]
LABELS = {"SVRG_final": "SVRG (final point)", "NFG_SVRG_final": "NFG-SVRG (final point)",
          "SVRG": "SVRG (random point, reset)", "NFG_SVRG": "NFG-SVRG (random point, reset)",
          "ASAI_SVRG": "ASA-SVRG (average point, reset)"}
COLORS = {"SVRG_final": "#1f77b4", "NFG_SVRG_final": "#2ca02c",
          "SVRG": "#1f77b4", "NFG_SVRG": "#2ca02c", "ASAI_SVRG": "#d62728"}
FULL_GRADIENT_METHODS = {"SVRG_final", "SVRG"}
PLOT_KEYS = {"train_loss": ("Training loss", False), "test_accuracy": ("Test accuracy", False),
             "approx_error": ("Approximation error", True)}


def condition_name(batch_size: int) -> str:
    """
    概要: バッチサイズに対応する条件のディレクトリ名を返す（E6 と ex004 で同じ名前である）．
    引数: batch_size (int)．
    戻り値: name (str)．
    """
    return f"inner_snapshot_lr0.001_bs{batch_size}_lambda0.0005_epochs{EPOCHS[batch_size]}"


def read_run(run_dir: str, method: str):
    """
    概要: 1つの試行の `log.json` を読み込む．
    引数: run_dir (str)，method (str)．
    戻り値: run (dict) または None（`log.json` がない場合）．`x` は $ \\#\\mathrm{grad}/N $（numpy.ndarray）．
    """
    log_path = os.path.join(run_dir, "log.json")
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        log = json.load(f)
    return {"method": method, "seed": int(os.path.basename(run_dir)), "log": log,
            "x": np.asarray(log["oracle_calls"], dtype=np.float64) / N_TRAIN}


def load_runs(root: str, method: str, batch_size: int) -> list:
    """
    概要: 1つの (手法, バッチサイズ) の全シードの試行を読み込む．
    引数: root (str)．実験の結果のディレクトリ．method (str)．batch_size (int)．
    戻り値: runs (list of dict)．シード順．ディレクトリがない場合は空のリスト．
    """
    condition_dir = os.path.join(root, method, condition_name(batch_size))
    if not os.path.isdir(condition_dir):
        return []
    runs = [read_run(os.path.join(condition_dir, s), method) for s in sorted(os.listdir(condition_dir)) if s.isdigit()]
    return [r for r in runs if r is not None]


def terminated_epoch(run: dict):
    """
    概要: 学習損失が非有限になって打ち切られたエポックを返す．
    引数: run (dict)．
    戻り値: epoch (int) または None（打ち切られていない場合）．
    """
    flags = run["log"].get("terminated", [])
    for epoch, flag in zip(run["log"]["epoch"], flags):
        if flag:
            return int(epoch)
    return None


def latter_half_median(run: dict, key: str, budget: float) -> float:
    """
    概要: 予算の後半（$ B/2 < \\#\\mathrm{grad}/N \\le B $）の有限の評価点における指標の中央値を返す．
    引数: run (dict)，key (str)，budget (float)．
    戻り値: value (float)．該当する評価点がない場合は NaN．
    """
    values = np.asarray(run["log"][key], dtype=np.float64)
    x = run["x"][:len(values)]
    mask = (x > budget / 2) & (x <= budget + 1e-9) & np.isfinite(values)
    return float(np.median(values[mask])) if mask.any() else float("nan")


def row_values(runs: list, method: str, budget: float) -> dict:
    """
    概要: 1つの (手法, バッチサイズ) の表の1行の値（文字列）を計算する．
    引数: runs (list of dict)，method (str)，budget (float)．
    戻り値: cells (dict)．
    """
    stops = [(r["seed"], terminated_epoch(r)) for r in runs]
    stopped = [f"seed {s}: {e}" for s, e in stops if e is not None]
    if method in FULL_GRADIENT_METHODS:
        approx = "0（全勾配）"
    else:
        approx = _fmt([latter_half_median(r, "approx_error", budget) for r in runs], sci=True)
    return {
        "n": str(len(runs)),
        "loss_budget": _fmt([final_value(r, "train_loss", budget) for r in runs]),
        "acc_budget": _fmt([final_value(r, "test_accuracy", budget) for r in runs]),
        "loss_final": _fmt([final_value(r, "train_loss", np.inf) for r in runs]),
        "acc_final": _fmt([final_value(r, "test_accuracy", np.inf) for r in runs]),
        "approx": approx,
        "stopped": f"{len(stopped)}" + (f"（{'，'.join(stopped)}）" if stopped else ""),
    }


def summary_table(groups: dict, methods: list, batch_sizes) -> str:
    """
    概要: 手法 × バッチサイズの表（Markdown）を作る．
    引数: groups (dict)．(手法, バッチサイズ) -> 試行のリスト．methods (list of str)．batch_sizes (iterable of int)．
    戻り値: text (str)．
    """
    lines = ["| 手法 | バッチサイズ | 予算 | シード数 | 予算での学習損失 | 予算でのテスト精度 | 最終エポックの学習損失 | "
             "最終エポックのテスト精度 | 近似誤差（予算の後半の中央値） | 打ち切り（エポック） |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for method in methods:
        for batch_size in batch_sizes:
            runs = groups.get((method, batch_size), [])
            if not runs:
                lines.append(f"| {LABELS[method]} | {batch_size} | {BUDGETS[batch_size]} | 0 | — | — | — | — | — | — |")
                continue
            c = row_values(runs, method, BUDGETS[batch_size])
            lines.append(f"| {LABELS[method]} | {batch_size} | {BUDGETS[batch_size]} | {c['n']} | {c['loss_budget']} | "
                         f"{c['acc_budget']} | {c['loss_final']} | {c['acc_final']} | {c['approx']} | {c['stopped']} |")
    return "\n".join(lines)


def mean_se_curve(runs: list, key: str, log_scale: bool):
    """
    概要: 評価点ごとに，有限の値を持つ試行の平均 ± 標準誤差（ddof=1）を計算する．
    引数: runs (list of dict)，key (str)，log_scale (bool)．正の値の常用対数で平均するか．
    戻り値: (x, mean, se) の組（numpy.ndarray）．`log_scale` の場合は対数の値である．
    """
    length = max(len(r["log"][key]) for r in runs)
    values = np.full((len(runs), length), np.nan)
    for i, r in enumerate(runs):
        v = np.asarray(r["log"][key], dtype=np.float64)
        values[i, :len(v)] = v
    values[~np.isfinite(values)] = np.nan
    if log_scale:
        values = np.log10(np.where(values > 0, values, np.nan))
    x = max(runs, key=lambda r: len(r["log"][key]))["x"][:length]
    count = np.sum(np.isfinite(values), axis=0)
    mean = np.full(length, np.nan)
    se = np.zeros(length)
    for j in np.where(count > 0)[0]:
        column = values[:, j][np.isfinite(values[:, j])]
        mean[j] = column.mean()
        se[j] = column.std(ddof=1) / np.sqrt(column.size) if column.size > 1 else 0.0
    return x, mean, se


def plot_metric(groups: dict, batch_size: int, key: str, output_dir: str) -> str:
    """
    概要: 1つのバッチサイズ・指標について，E6 の2手法（実線）と付録B の3手法（破線）の推移を描いて保存する．
    引数: groups (dict)，batch_size (int)，key (str)．`PLOT_KEYS` のいずれか．output_dir (str)．
    戻り値: path (str)．保存した図のパス．
    """
    ylabel, log_scale = PLOT_KEYS[key]
    budget = BUDGETS[batch_size]
    fig, ax = plt.subplots(figsize=(5, 5))
    for method, style in [(m, "-") for m in E6_METHODS] + [(m, "--") for m in REFERENCE_METHODS]:
        runs = groups.get((method, batch_size), [])
        if not runs or (key == "approx_error" and method in FULL_GRADIENT_METHODS):
            continue
        x, mean, se = mean_se_curve(runs, key, log_scale)
        mask = x <= budget + 1e-9
        ax.plot(x[mask], mean[mask], style, color=COLORS[method], label=LABELS[method], linewidth=1.2)
        ax.fill_between(x[mask], (mean - se)[mask], (mean + se)[mask], color=COLORS[method], alpha=0.2)
    ax.axvspan(budget / 2, budget, color="0.9", zorder=0)
    ax.set_xlabel(r"#grad/$N$")
    ax.set_ylabel(f"log10({ylabel})" if log_scale else ylabel)
    ax.set_title(f"CIFAR-10, batch size {batch_size}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(output_dir, f"e6_bs{batch_size}_{key}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_groups(e6_root: str, reference_root: str) -> dict:
    """
    概要: E6 と付録B（ex004）の全ての (手法, バッチサイズ) の試行を読み込む．
    引数: e6_root (str)，reference_root (str)．
    戻り値: groups (dict)．(手法, バッチサイズ) -> 試行のリスト．
    """
    groups = {}
    for batch_size in BUDGETS:
        for method in E6_METHODS:
            groups[(method, batch_size)] = load_runs(e6_root, method, batch_size)
        for method in REFERENCE_METHODS:
            groups[(method, batch_size)] = load_runs(reference_root, method, batch_size)
    return groups


def write_summary(e6_root: str, reference_root: str, output: str, figure_dir: str) -> str:
    """
    概要: 表と図を作成し，Markdown を保存する．
    引数: e6_root (str)，reference_root (str)，output (str)．Markdown の保存先．figure_dir (str)．図の保存先．
    戻り値: text (str)．保存した Markdown．
    """
    groups = load_groups(e6_root, reference_root)
    os.makedirs(figure_dir, exist_ok=True)
    figures = [plot_metric(groups, b, k, figure_dir) for b in BUDGETS for k in PLOT_KEYS]
    text = "\n".join([
        "# E6：最終点のスナップショットの SVRG・NFG-SVRG（CIFAR-10，バッチサイズ32・512）", "",
        "値は5シードの平均 ± 標準誤差（`ddof=1`）である．横軸は $ \\#\\mathrm{grad}/N $（`oracle_calls / 50000`）である．"
        "予算での値は予算以下の最後の評価点の値（打ち切られた試行は打ち切り直前の有限の評価点の値）である．"
        "近似誤差は予算の後半（予算/2 < $ \\#\\mathrm{grad}/N $ ≤ 予算）の中央値をシードごとに求めて平均した．", "",
        "## E6 の結果", "", summary_table(groups, E6_METHODS, BUDGETS), "",
        "## 参考：付録B（ex004．ランダム点または平均点，reset）", "",
        summary_table(groups, REFERENCE_METHODS, BUDGETS), "",
        "## 図", "", *[f"- `{os.path.relpath(p, os.path.dirname(os.path.abspath(output)))}`" for p in figures], ""])
    with open(output, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e6-root", default=os.path.join(OUTPUTS_DIR, EXPERIMENT_NAME))
    parser.add_argument("--reference-root", default=os.path.join(OUTPUTS_DIR, REFERENCE_NAME),
                        help="付録B（ex004）の結果のディレクトリ")
    parser.add_argument("--output", default=os.path.join(OUTPUTS_DIR, "e6_summary.md"))
    parser.add_argument("--figure-dir", default=None, help="既定は {e6-root}/figures")
    return parser.parse_args()


def main() -> None:
    """
    概要: 集計を実行し，結果を表示する．
    引数: なし
    戻り値: なし
    """
    args = parse_args()
    figure_dir = args.figure_dir or os.path.join(args.e6_root, "figures")
    print(write_summary(os.path.abspath(args.e6_root), os.path.abspath(args.reference_root),
                        os.path.abspath(args.output), os.path.abspath(figure_dir)))


if __name__ == "__main__":
    main()
