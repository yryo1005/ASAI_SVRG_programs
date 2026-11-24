"""
`ex016_imagewoof_bs128_walltime`（ImageWoof・バッチサイズ128 の5手法の実時間の取り直し，`.orders/order_083.md`）の集計．

出力：
- `outputs/ex016_summary.md`：
  1. 1エポックの時間：シードごとに2エポック目以降の `elapsed_time` の増分の中央値を求め，5シード（5周）の平均（最小〜最大）と
     (最大 − 最小)/平均 を示す．SVRG は初期の全勾配の時間（`elapsed_time[0]`）も示す．
  2. 予算（$ \\#\\mathrm{grad}/N \\le 200 $）までの実時間と，予算でのテスト精度・学習損失（5シードの平均 ± 標準誤差）．
  3. 手法間の比（ASA-SVRG ÷ SVRG（最終点）など）．
  4. 実行の順序（`config.json` の `started_at`）．
- `outputs/ex016_imagewoof_bs128_walltime/figures/`：横軸を実時間［s］とした学習損失・テスト精度・近似誤差の図と，
  横軸を $ \\#\\mathrm{grad}/N $ とした同じ図（確認用）．

実時間は `elapsed_time`（$ = $ `time_optimizer + time_svrg_full_grad` の累積値．評価と診断の時間を除く）である．
図の横軸は，各エポックの実時間の5シードの平均である．

実行例（`programs_review/programs` から）:
    python analysis_ex016.py
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "outputs")
EXPERIMENT_NAME = "ex016_imagewoof_bs128_walltime"
DEFAULT_ROOT = os.path.join(OUTPUTS_DIR, EXPERIMENT_NAME)
N_TRAIN = 9025
BUDGET = 200.0

# 条件名 → (手法のディレクトリ, 条件のディレクトリ名の先頭, 表示名, 色, 線種)．並びは実行順．
CONDITIONS = {
    "SGD": ("SGD", "inner_none_", "SGD", "tab:gray", "-"),
    "SVRG_final": ("SVRG_final", "inner_snapshot_", "SVRG (final)", "tab:blue", "-"),
    "NFG_SVRG_final": ("NFG_SVRG_final", "inner_snapshot_", "NFG-SVRG (final)", "tab:orange", "-"),
    "ASAI_SVRG_continue": ("ASAI_SVRG", "inner_last_iterate_", "ASA-SVRG (continue)", "tab:red", "-"),
    "ASAI_SVRG_reset": ("ASAI_SVRG", "inner_snapshot_", "ASA-SVRG (reset)", "tab:red", "--"),
}
LABELS_JA = {"SGD": "SGD", "SVRG_final": "SVRG（最終点）", "NFG_SVRG_final": "NFG-SVRG（最終点）",
             "ASAI_SVRG_continue": "ASA-SVRG（continue）", "ASAI_SVRG_reset": "ASA-SVRG（reset）"}
RATIO_PAIRS = [("ASAI_SVRG_continue", "SVRG_final"), ("ASAI_SVRG_reset", "SVRG_final"),
               ("ASAI_SVRG_continue", "NFG_SVRG_final"), ("ASAI_SVRG_reset", "ASAI_SVRG_continue"),
               ("NFG_SVRG_final", "SVRG_final"), ("ASAI_SVRG_continue", "SGD")]


# ---------------------------------------------------------------------------
# 読み込みと1試行の量
# ---------------------------------------------------------------------------

def _as_float_array(values: list) -> np.ndarray:
    """
    概要: `None` を含みうるリストを，`None` を NaN とした float の配列に変換する．
    引数: values (list)．長さ T．
    戻り値: array (np.ndarray)．形状 (T,)．
    """
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


def load_condition(root: str, condition: str) -> dict:
    """
    概要: 1つの条件のシードごとの `log.json` と `config.json` を読み込む．
    引数: root (str)．結果のルート．condition (str)．`CONDITIONS` のキー．
    戻り値: runs (dict)．キーはシード（int），値は {"log": dict, "config": dict}．
    """
    method_dir, prefix = CONDITIONS[condition][:2]
    runs = {}
    for log_path in sorted(glob.glob(os.path.join(root, method_dir, prefix + "*", "*", "log.json"))):
        run_dir = os.path.dirname(log_path)
        with open(log_path, encoding="utf-8") as file:
            log = json.load(file)
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as file:
            config = json.load(file)
        runs[int(os.path.basename(run_dir))] = {"log": log, "config": config}
    return runs


def epoch_time(log: dict) -> float:
    """
    概要: 1エポックの時間（2エポック目以降の `elapsed_time` の増分の中央値）を返す．1エポック目は準備運転として除く．
    引数: log (dict)．`elapsed_time` を含む．
    戻り値: seconds (float)．エポックが2未満なら NaN．
    """
    increments = np.diff(_as_float_array(log["elapsed_time"]))
    return float(np.median(increments[1:])) if increments.size >= 2 else float("nan")


def budget_index(log: dict, budget: float = BUDGET, n_train: int = N_TRAIN) -> int:
    """
    概要: $ \\#\\mathrm{grad}/N \\le $ budget である最後の評価点の添字を返す．
    引数: log (dict)．budget (float)．n_train (int)．
    戻り値: index (int)．
    """
    x = np.asarray(log["oracle_calls"], dtype=np.float64) / n_train
    return int(np.nonzero(x <= budget + 1e-9)[0][-1])


def summarize_run(log: dict) -> dict:
    """
    概要: 1試行の1エポックの時間，初期の時間（SVRG の初期の全勾配），予算までの実時間，予算での値を求める．
    引数: log (dict)．
    戻り値: row (dict)．キーは `epoch_time`，`initial_time`，`time_to_budget`，`test_accuracy`，`train_loss`，`epochs`．
    """
    index = budget_index(log)
    elapsed = _as_float_array(log["elapsed_time"])
    return {
        "epoch_time": epoch_time(log),
        "initial_time": float(elapsed[0]),
        "time_to_budget": float(elapsed[index]),
        "test_accuracy": float(log["test_accuracy"][index]),
        "train_loss": float(log["train_loss"][index]),
        "epochs": len(log["epoch"]) - 1,
    }


def mean_and_standard_error(values) -> tuple:
    """
    概要: 有限の値の平均と標準誤差（`ddof=1`）を返す．
    引数: values (iterable of float)．
    戻り値: (mean, standard_error) (tuple of float)．値が1つなら標準誤差は NaN．
    """
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float("nan")
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(array.size))


def spread(values) -> float:
    """
    概要: (最大 − 最小)/平均 を返す（周回の間の揺らぎの大きさ）．
    引数: values (iterable of float)．
    戻り値: ratio (float)．
    """
    array = np.asarray(list(values), dtype=np.float64)
    return float((array.max() - array.min()) / array.mean()) if array.size > 0 else float("nan")


# ---------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------

def _pm(values, digits: int) -> str:
    """
    概要: 平均 ± 標準誤差の文字列を作る．
    引数: values (iterable of float)．digits (int)．
    戻り値: text (str)．
    """
    mean, standard_error = mean_and_standard_error(values)
    se_text = "—" if not np.isfinite(standard_error) else f"{standard_error:.{digits}f}"
    return f"{mean:.{digits}f} ± {se_text}"


def time_table(rows: dict) -> list:
    """
    概要: 1エポックの時間と予算までの実時間の表の行を作る．
    引数: rows (dict)．キーは条件，値はシードごとの `summarize_run` の戻り値の dict．
    戻り値: lines (list of str)．
    """
    lines = ["| 手法 | シード数 | エポック数 | 1エポックの時間［s］（最小〜最大） | 5周の (最大−最小)/平均 | 初期の時間［s］ | "
             "予算までの実時間［s］（平均 ± 標準誤差） | 予算でのテスト精度［%］ | 予算での学習損失 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition, per_seed in rows.items():
        if not per_seed:
            continue
        times = [r["epoch_time"] for r in per_seed.values()]
        initial = [r["initial_time"] for r in per_seed.values()]
        lines.append(
            f"| {LABELS_JA[condition]} | {len(per_seed)} | {next(iter(per_seed.values()))['epochs']} | "
            f"{np.mean(times):.3f}（{np.min(times):.3f}〜{np.max(times):.3f}） | {100 * spread(times):.2f}% | "
            f"{np.mean(initial):.2f} | {_pm([r['time_to_budget'] for r in per_seed.values()], 1)} | "
            f"{_pm([100 * r['test_accuracy'] for r in per_seed.values()], 2)} | "
            f"{_pm([r['train_loss'] for r in per_seed.values()], 4)} |")
    return lines


def ratio_table(rows: dict) -> list:
    """
    概要: 手法間の1エポックの時間と予算までの実時間の比の表の行を作る．比は5シードの平均の比である．
    引数: rows (dict)．`time_table` と同じ．
    戻り値: lines (list of str)．
    """
    lines = ["| 比 | 1エポックの時間の比 | 予算までの実時間の比 |", "|---|---:|---:|"]
    for numerator, denominator in RATIO_PAIRS:
        if not rows.get(numerator) or not rows.get(denominator):
            continue
        a, b = rows[numerator].values(), rows[denominator].values()
        epoch_ratio = np.mean([r["epoch_time"] for r in a]) / np.mean([r["epoch_time"] for r in b])
        budget_ratio = np.mean([r["time_to_budget"] for r in a]) / np.mean([r["time_to_budget"] for r in b])
        lines.append(f"| {LABELS_JA[numerator]} ÷ {LABELS_JA[denominator]} | {epoch_ratio:.3f} | {budget_ratio:.3f} |")
    return lines


def order_table(runs: dict) -> list:
    """
    概要: 実行の順序（`config.json` の `started_at` の昇順）の表の行を作り，同じ手法が連続した箇所を数える．
    引数: runs (dict)．キーは条件，値は `load_condition` の戻り値．
    戻り値: lines (list of str)．
    """
    entries = sorted((run["config"].get("started_at", ""), seed, condition)
                     for condition, per_seed in runs.items() for seed, run in per_seed.items())
    consecutive = sum(1 for a, b in zip(entries, entries[1:]) if a[2] == b[2])
    lines = [f"同じ手法が連続した箇所：{consecutive}．", "", "| 開始時刻 | 終了時刻 | シード | 手法 |", "|---|---|---:|---|"]
    for started_at, seed, condition in entries:
        finished_at = runs[condition][seed]["config"].get("finished_at", "")
        lines.append(f"| {started_at} | {finished_at} | {seed} | {LABELS_JA[condition]} |")
    return lines


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------

def mean_curve(per_seed: dict, key: str, x_key: str) -> tuple:
    """
    概要: シード間で共通の長さに切りそろえ，横軸と値の平均・標準誤差を返す．
    引数: per_seed (dict)．`load_condition` の戻り値．key (str)．値の項目．x_key (str)．"elapsed_time" または "grad_per_n"．
    戻り値: (x, mean, standard_error) (tuple of np.ndarray)．
    """
    logs = [run["log"] for run in per_seed.values()]
    length = min(len(log["epoch"]) for log in logs)
    if x_key == "grad_per_n":
        xs = np.array([np.asarray(log["oracle_calls"][:length], dtype=np.float64) / N_TRAIN for log in logs])
    else:
        xs = np.array([_as_float_array(log["elapsed_time"][:length]) for log in logs])
    values = np.array([_as_float_array(log[key][:length]) for log in logs])
    count = np.sum(np.isfinite(values), axis=0)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1) if len(logs) > 1 else np.zeros(length)
    return np.nanmean(xs, axis=0), mean, np.where(count > 1, std / np.sqrt(np.maximum(count, 1)), 0.0)


def plot_metric(runs: dict, key: str, ylabel: str, x_key: str, path: str, log_scale: bool = False) -> None:
    """
    概要: 5手法の値の推移（平均 ± 標準誤差）を1枚の図に描いて保存する．
    引数: runs (dict)．key (str)．ylabel (str)．x_key (str)．path (str)．log_scale (bool)．
    戻り値: なし
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    for condition, per_seed in runs.items():
        if not per_seed or (key == "approx_error" and condition in ("SGD", "SVRG_final")):
            continue
        _, _, label, color, style = CONDITIONS[condition]
        x, mean, standard_error = mean_curve(per_seed, key, x_key)
        ax.plot(x, mean, style, color=color, label=label, lw=1.5)
        ax.fill_between(x, mean - standard_error, mean + standard_error, color=color, alpha=0.2, lw=0)
    ax.set_xlabel("Wall-clock time [s]" if x_key == "elapsed_time" else "#grad/N")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("ImageWoof, batch size 128")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_figures(runs: dict, figure_dir: str) -> list:
    """
    概要: 学習損失・テスト精度・近似誤差を，横軸を実時間と $ \\#\\mathrm{grad}/N $ の2通りで描く．
    引数: runs (dict)．figure_dir (str)．
    戻り値: paths (list of str)．
    """
    os.makedirs(figure_dir, exist_ok=True)
    paths = []
    for x_key, suffix in (("elapsed_time", "walltime"), ("grad_per_n", "gradn")):
        for key, ylabel, log_scale in (("train_loss", "Training loss", False), ("test_accuracy", "Test accuracy", False),
                                       ("approx_error", "Approximation error", True)):
            path = os.path.join(figure_dir, f"ex016_{key}_{suffix}.png")
            plot_metric(runs, key, ylabel, x_key, path, log_scale)
            paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def build_summary(runs: dict, rows: dict, figure_paths: list) -> str:
    """
    概要: 表と図の一覧を Markdown にまとめる．
    引数: runs (dict)．rows (dict)．figure_paths (list of str)．
    戻り値: text (str)．
    """
    lines = ["# ex016 の集計結果（ImageWoof・バッチサイズ128，5手法の実時間）", "",
             "実時間は `elapsed_time`（$ = $ `time_optimizer + time_svrg_full_grad` の累積．評価と診断の時間を除く）である．"
             "1エポックの時間は，シードごとの2エポック目以降の増分の中央値であり，表には5シード（5周）の平均（最小〜最大）を示す．"
             f"予算は $ \\#\\mathrm{{grad}}/N \\le {BUDGET:.0f} $ であり，予算での値は予算以下の最後の評価点の値である．"
             "初期の時間は外側ループの前の `elapsed_time`（SVRG の初期の全勾配）である．", "",
             "## 1. 1エポックの時間と予算までの実時間", ""]
    lines += time_table(rows)
    lines += ["", "## 2. 手法間の比", ""] + ratio_table(rows)
    lines += ["", "## 3. 実行の順序", ""] + order_table(runs)
    lines += ["", "## 4. 図", ""] + [f"- `{os.path.relpath(p, OUTPUTS_DIR)}`" for p in figure_paths] + [""]
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: argv (list of str or None)．
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output", default=os.path.join(OUTPUTS_DIR, "ex016_summary.md"))
    parser.add_argument("--figure-dir", default=None, help="既定は {root}/figures")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """
    概要: 結果を読み込み，表と図を作る．
    引数: argv (list of str or None)．
    戻り値: なし
    """
    args = parse_args(argv)
    runs = {condition: load_condition(args.root, condition) for condition in CONDITIONS}
    rows = {condition: {seed: summarize_run(run["log"]) for seed, run in per_seed.items()}
            for condition, per_seed in runs.items()}
    figure_paths = write_figures(runs, args.figure_dir or os.path.join(args.root, "figures"))
    text = build_summary(runs, rows, figure_paths)
    with open(args.output, "w", encoding="utf-8") as file:
        file.write(text)
    print(args.output)


if __name__ == "__main__":
    main()
