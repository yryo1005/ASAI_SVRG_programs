"""
E7 の集計．

- B（`ex013_e7_epoch_time`）：1エポックの時間の表（データセット × バッチサイズの6つの組，各組4手法）．
- A（`ex012_e7_imagewoof_final`）：予算での精度・学習損失，近似誤差，揺らぎ，打ち切りの表と確認用の図．

1エポックの時間の定義：`time_optimizer + time_svrg_full_grad` の累積値 $ T_{e} $ の差分．1つの試行の値は
2〜4エポック目の差分の平均 $ (T_{4} - T_{1})/3 $（1エポック目は準備運転として除く）．SVRG の初期の全勾配
（`epoch = 0` の記録）は含めない．1つの手法の値は3周の平均とし，最小値と最大値も示す．

実行例:
    python analysis_e7.py
"""

import argparse
import glob
import json
import math
import os

import numpy as np

_PROGRAMS_DIR = os.path.dirname(os.path.abspath(__file__))
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
OUTPUTS_DIR = os.path.join(_REVIEW_ROOT, "outputs")
B_ROOT = os.path.join(OUTPUTS_DIR, "ex013_e7_epoch_time")
A_ROOT = os.path.join(OUTPUTS_DIR, "ex012_e7_imagewoof_final")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "e7_summary.md")

B_METHODS = ["ASAI_SVRG", "NFG_SVRG_final", "SVRG_final", "SGD"]
METHOD_LABELS = {"ASAI_SVRG": "ASA-SVRG（continue）", "NFG_SVRG_final": "NFG-SVRG（最終点）",
                 "SVRG_final": "SVRG（最終点）", "SGD": "SGD", "SVRG": "SVRG（ランダム点）",
                 "NFG_SVRG": "NFG-SVRG（ランダム点）"}
# 1エポックあたりの #grad/N．SVRG は初期の全勾配 N を別に数える．
GRAD_PER_EPOCH = {"SGD": 1, "ASAI_SVRG": 2, "NFG_SVRG_final": 2, "SVRG_final": 3, "NFG_SVRG": 2, "SVRG": 3}
BUDGETS = {("cifar10", 32): 20, ("cifar10", 128): 90, ("cifar10", 512): 350,
           ("imagewoof", 32): 30, ("imagewoof", 128): 200, ("imagewoof", 512): 1000}
DATASET_LABELS = {"imagewoof": "ImageWoof", "cifar10": "CIFAR-10"}
FIRST_MEASURED_EPOCH = 1  # 1エポック目（T_1 - T_0）を除き，T_1 から最後までの差分を使う


# ---- B：1エポックの時間 ----

def _load_json(path: str) -> dict:
    """
    概要: JSON ファイルを読み込む．
    引数: path (str)．
    戻り値: data (dict)．
    """
    with open(path) as f:
        return json.load(f)


def cumulative_wall_time(log: dict) -> np.ndarray:
    """
    概要: 実時間の累積値 $ T_{e} = $ `time_optimizer + time_svrg_full_grad` を返す．
    引数: log (dict)．`log.json` の内容．
    戻り値: T (np.ndarray)．形状 (エポック数 + 1,)．[s]．
    """
    return np.asarray(log["time_optimizer"], dtype=float) + np.asarray(log["time_svrg_full_grad"], dtype=float)


def epoch_time_of_run(log: dict) -> float:
    """
    概要: 1つの試行の1エポックの時間 $ (T_{E} - T_{1})/(E - 1) $ を返す（$ E $ は最後のエポック．既定の4エポックでは
        2〜4エポック目の平均）．
    引数: log (dict)．`log.json` の内容．
    戻り値: seconds (float)．
    """
    T = cumulative_wall_time(log)
    last = len(T) - 1
    if last <= FIRST_MEASURED_EPOCH:
        raise ValueError("1エポック目を除いた計測のエポックがない．")
    return float((T[last] - T[FIRST_MEASURED_EPOCH]) / (last - FIRST_MEASURED_EPOCH))


def initial_full_grad_time(log: dict) -> float:
    """
    概要: SVRG の初期の全勾配の時間（`epoch = 0` の `time_svrg_full_grad`）を返す．
    引数: log (dict)．
    戻り値: seconds (float)．
    """
    return float(log["time_svrg_full_grad"][0])


def summarize_repeats(values) -> dict:
    """
    概要: 3周の値の平均，最小，最大と，(最大 - 最小)/平均 を返す．
    引数: values (list of float)．
    戻り値: summary (dict)．`mean`，`min`，`max`，`spread`（割合），`n`．
    """
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    return {"mean": mean, "min": float(values.min()), "max": float(values.max()),
            "spread": float((values.max() - values.min()) / mean), "n": int(values.size)}


def load_b_runs(root: str = B_ROOT) -> dict:
    """
    概要: B の全ての試行を読み込む．
    引数: root (str)．`outputs/ex013_e7_epoch_time`．
    戻り値: runs (dict)．{(データセット, バッチサイズ, 手法): [(周回, log, config), ...]}（周回の順）．
    """
    runs = {}
    for log_path in sorted(glob.glob(os.path.join(root, "*", "bs*", "*", "rep*", "log.json"))):
        run_dir = os.path.dirname(log_path)
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_path):
            continue
        parts = os.path.relpath(run_dir, root).split(os.sep)
        key = (parts[0], int(parts[1][2:]), parts[2])
        runs.setdefault(key, []).append((int(parts[3][3:]), _load_json(log_path), _load_json(config_path)))
    for value in runs.values():
        value.sort(key=lambda item: item[0])
    return runs


def epoch_time_table(runs: dict) -> dict:
    """
    概要: 組と手法ごとに1エポックの時間（3周の要約）と，SVRG の初期の全勾配の時間を求める．
    引数: runs (dict)．`load_b_runs` の戻り値．
    戻り値: table (dict)．{(データセット, バッチサイズ, 手法): {"epoch": summary, "initial": summary or None}}．
    """
    table = {}
    for key, items in runs.items():
        logs = [log for _, log, _ in items]
        initial = summarize_repeats([initial_full_grad_time(log) for log in logs]) if key[2] == "SVRG_final" else None
        table[key] = {"epoch": summarize_repeats([epoch_time_of_run(log) for log in logs]), "initial": initial}
    return table


def epochs_to_budget(method: str, budget: float) -> int:
    """
    概要: 予算 $ \\#\\mathrm{grad}/N $ に届く最小のエポック数を返す（SVRG は初期の全勾配 N を含めて数える）．
    引数: method (str)．budget (float)．
    戻り値: epochs (int)．
    """
    offset = 1 if method in ("SVRG_final", "SVRG") else 0
    return int(math.ceil((budget - offset) / GRAD_PER_EPOCH[method]))


def time_to_budget(method: str, budget: float, epoch_time: float, initial_time: float = 0.0) -> float:
    """
    概要: 予算までの推定時間（1エポックの時間 × 予算に届く最小のエポック数．SVRG は初期の全勾配の時間を加える）．
    引数: method (str)．budget (float)．epoch_time (float)．initial_time (float) = 0．
    戻り値: seconds (float)．
    """
    return epochs_to_budget(method, budget) * epoch_time + initial_time


def b_table_lines(table: dict) -> list:
    """
    概要: B の表（Markdown）を作る．
    引数: table (dict)．`epoch_time_table` の戻り値．
    戻り値: lines (list of str)．
    """
    lines = ["| データセット | バッチサイズ | 手法 | 試行数 | 1エポックの時間［s］（最小〜最大） | "
             "#grad/N=1 あたり［s］ | SGD に対する比 | 予算 | 予算までのエポック数 | 予算までの推定時間［s］ | "
             "3周の (最大−最小)/平均 |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("imagewoof", "cifar10"):
        for batch_size in (32, 128, 512):
            sgd = table.get((dataset, batch_size, "SGD"))
            for method in B_METHODS:
                entry = table.get((dataset, batch_size, method))
                if entry is None:
                    continue
                e = entry["epoch"]
                budget = BUDGETS[(dataset, batch_size)]
                initial = entry["initial"]["mean"] if entry["initial"] else 0.0
                ratio = f"{e['mean'] / sgd['epoch']['mean']:.3f}" if sgd else "—"
                lines.append(
                    f"| {DATASET_LABELS[dataset]} | {batch_size} | {METHOD_LABELS[method]} | {e['n']} | "
                    f"{e['mean']:.3f}（{e['min']:.3f}〜{e['max']:.3f}） | {e['mean'] / GRAD_PER_EPOCH[method]:.3f} | "
                    f"{ratio} | {budget} | {epochs_to_budget(method, budget)} | "
                    f"{time_to_budget(method, budget, e['mean'], initial):.1f} | {e['spread'] * 100:.2f}% |")
    return lines


def ratio_lines(table: dict) -> list:
    """
    概要: 組ごとの1エポックの時間の比（ASA-SVRG ÷ SVRG（最終点），ASA-SVRG ÷ NFG-SVRG（最終点），NFG-SVRG ÷ SVRG）の表を作る．
    引数: table (dict)．
    戻り値: lines (list of str)．
    """
    lines = ["| データセット | バッチサイズ | ASA-SVRG ÷ SVRG（最終点） | ASA-SVRG ÷ NFG-SVRG（最終点） | "
             "NFG-SVRG（最終点） ÷ SVRG（最終点） | SVRG の初期の全勾配［s］ |",
             "|---|---:|---:|---:|---:|---:|"]
    for dataset in ("imagewoof", "cifar10"):
        for batch_size in (32, 128, 512):
            get = {m: table.get((dataset, batch_size, m)) for m in B_METHODS}
            if any(v is None for v in get.values()):
                continue
            t = {m: v["epoch"]["mean"] for m, v in get.items()}
            lines.append(f"| {DATASET_LABELS[dataset]} | {batch_size} | {t['ASAI_SVRG'] / t['SVRG_final']:.3f} | "
                         f"{t['ASAI_SVRG'] / t['NFG_SVRG_final']:.3f} | {t['NFG_SVRG_final'] / t['SVRG_final']:.3f} | "
                         f"{get['SVRG_final']['initial']['mean']:.3f} |")
    return lines


def execution_order_lines(runs: dict) -> list:
    """
    概要: 各試行の開始時刻の一覧（実行の順序の確認用）を作る．
    引数: runs (dict)．
    戻り値: lines (list of str)．
    """
    rows = [(cfg.get("started_at", ""), key[0], key[1], rep, key[2]) for key, items in runs.items()
            for rep, _, cfg in items]
    lines = ["| 開始時刻 | データセット | バッチサイズ | 周回 | 手法 |", "|---|---|---:|---:|---|"]
    for started, dataset, batch_size, rep, method in sorted(rows):
        lines.append(f"| {started} | {DATASET_LABELS[dataset]} | {batch_size} | {rep} | {METHOD_LABELS[method]} |")
    return lines


def check_alternation(runs: dict) -> list:
    """
    概要: 開始時刻の順に並べたとき，同じ組の中で同じ手法が連続していないかを調べる．
    引数: runs (dict)．
    戻り値: problems (list of str)．連続していた箇所．空なら問題なし．
    """
    rows = sorted((cfg.get("started_at", ""), key) for key, items in runs.items() for _, _, cfg in items)
    problems = []
    for (_, a), (_, b) in zip(rows, rows[1:]):
        if a == b:
            problems.append(f"{a} が連続している")
    return problems


# ---- A：精度の実験 ----

N_IMAGEWOOF = 9025
A_METHODS = ["SVRG_final", "NFG_SVRG_final"]
A_EPOCHS = {("SVRG_final", 32): 11, ("NFG_SVRG_final", 32): 16, ("SVRG_final", 128): 67,
            ("NFG_SVRG_final", 128): 100, ("SVRG_final", 512): 334, ("NFG_SVRG_final", 512): 500}
A_BUDGETS = {32: 30, 128: 200, 512: 1000}
# 既存の SVRG・NFG-SVRG（ランダム点，内側ループの開始点は snapshot）の条件．ルートはコマンドライン引数で与える．
REFERENCE_CONDITIONS = {32: "inner_snapshot_lr0.001_bs32_lambda0.0005_epochs33",
                        128: "inner_snapshot_lr0.001_bs128_lambda0.0005_epochs128",
                        512: "inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505"}
DEFAULT_REFERENCE_ROOTS = {32: os.path.join(OUTPUTS_DIR, "ex005_appb_imagewoof_resnet18_layernorm"),
                           128: os.path.join(OUTPUTS_DIR, "ex003_e0_imagewoof_resnet18_layernorm"),
                           512: os.path.join(OUTPUTS_DIR, "ex005_appb_imagewoof_resnet18_layernorm")}
REFERENCE_METHODS = ["SVRG", "NFG_SVRG"]
A_FIGURE_DIR = os.path.join(A_ROOT, "figures")
COLORS = {"SVRG_final": "tab:blue", "NFG_SVRG_final": "tab:orange", "SVRG": "tab:blue", "NFG_SVRG": "tab:orange"}


def a_condition_name(method: str, batch_size: int) -> str:
    """
    概要: A の条件のディレクトリ名を返す．
    引数: method (str)．batch_size (int)．
    戻り値: name (str)．
    """
    return f"inner_snapshot_lr0.001_bs{batch_size}_lambda0.0005_epochs{A_EPOCHS[(method, batch_size)]}"


def load_condition(condition_dir: str) -> dict:
    """
    概要: 1つの条件の全シードの `log.json` を読み込む．
    引数: condition_dir (str)．`{手法}/{条件}` のディレクトリ．
    戻り値: logs (dict)．{シード: log}．
    """
    logs = {}
    for log_path in sorted(glob.glob(os.path.join(condition_dir, "*", "log.json"))):
        seed = os.path.basename(os.path.dirname(log_path))
        if seed.isdigit():
            logs[int(seed)] = _load_json(log_path)
    return logs


def grad_axis(log: dict, n_train: int = N_IMAGEWOOF) -> np.ndarray:
    """
    概要: 横軸 $ \\#\\mathrm{grad}/N $ を返す．
    引数: log (dict)．n_train (int)．
    戻り値: x (np.ndarray)．
    """
    return np.asarray(log["oracle_calls"], dtype=float) / n_train


def terminated_epoch(log: dict):
    """
    概要: 打ち切られたエポックを返す（打ち切られていなければ None）．
    引数: log (dict)．
    戻り値: epoch (int or None)．
    """
    for epoch, flag in zip(log["epoch"], log["terminated"]):
        if flag:
            return int(epoch)
    return None


def value_at_budget(log: dict, key: str, budget: float) -> float:
    """
    概要: 予算以下の最後の評価点の値を返す．打ち切られた試行で予算に届かない場合は NaN．
    引数: log (dict)．key (str)．budget (float)．
    戻り値: value (float)．
    """
    x = grad_axis(log)
    index = np.where(x <= budget + 1e-9)[0]
    if terminated_epoch(log) is not None and x[-1] < budget * 0.999:
        return float("nan")
    return float(log[key][index[-1]])


def latter_half_mask(log: dict, budget: float) -> np.ndarray:
    """
    概要: 予算の後半（$ \\#\\mathrm{grad}/N $ が予算の1/2以上・予算以下）の評価点を表すマスクを返す．
    引数: log (dict)．budget (float)．
    戻り値: mask (np.ndarray of bool)．
    """
    x = grad_axis(log)
    return (x >= budget / 2) & (x <= budget + 1e-9)


def latter_half_median(log: dict, key: str, budget: float) -> float:
    """
    概要: 予算の後半における値の中央値を返す（非有限の値を除く）．
    引数: log (dict)．key (str)．budget (float)．
    戻り値: median (float)．該当する点がなければ NaN．
    """
    values = np.asarray(log[key], dtype=float)[latter_half_mask(log, budget)]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def latter_half_fluctuation(log: dict, budget: float) -> float:
    """
    概要: 予算の後半における隣り合う評価点の間のテスト精度の変化 $ |a_{s} - a_{s-1}| $ の平均を返す．
    引数: log (dict)．budget (float)．
    戻り値: fluctuation (float)．該当する点が2つ未満なら NaN．
    """
    values = np.asarray(log["test_accuracy"], dtype=float)[latter_half_mask(log, budget)]
    return float(np.mean(np.abs(np.diff(values)))) if values.size >= 2 else float("nan")


def mean_se(values) -> tuple:
    """
    概要: 有限の値の平均と標準誤差（ddof=1）を返す．
    引数: values (list of float)．
    戻り値: (mean, se, n) の組．
    """
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), 0
    se = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan")
    return float(values.mean()), se, int(values.size)


def _fmt(ms: tuple, scale: float = 1.0, digits: int = 4) -> str:
    """
    概要: (mean, se, n) を「平均 ± 標準誤差」の文字列にする．
    引数: ms (tuple)．scale (float)．digits (int)．
    戻り値: text (str)．
    """
    mean, se, n = ms
    if n == 0:
        return "—"
    se_text = f"{se * scale:.{digits}f}" if np.isfinite(se) else "—"
    return f"{mean * scale:.{digits}f} ± {se_text}"


def a_row(logs: dict, budget: float, with_approx: bool) -> dict:
    """
    概要: 1つの条件の集計値を求める．
    引数: logs (dict)．{シード: log}．budget (float)．with_approx (bool)．近似誤差の中央値を求めるか．
    戻り値: row (dict)．
    """
    stopped = {seed: terminated_epoch(log) for seed, log in logs.items() if terminated_epoch(log) is not None}
    return {
        "n": len(logs),
        "acc": mean_se([value_at_budget(log, "test_accuracy", budget) for log in logs.values()]),
        "loss": mean_se([value_at_budget(log, "train_loss", budget) for log in logs.values()]),
        "approx": (mean_se([latter_half_median(log, "approx_error", budget) for log in logs.values()])
                   if with_approx else None),
        "fluct": mean_se([latter_half_fluctuation(log, budget) for log in logs.values()]),
        "stopped": stopped,
    }


def a_groups(a_root: str = A_ROOT, reference_roots: dict = None) -> dict:
    """
    概要: A の結果と既存の SVRG・NFG-SVRG（ランダム点）の結果を読み込む．
    引数: a_root (str)．reference_roots (dict) = None．{バッチサイズ: 既存の実験の出力ルート}．
    戻り値: groups (dict)．{(バッチサイズ, 手法): {シード: log}}．
    """
    reference_roots = reference_roots or DEFAULT_REFERENCE_ROOTS
    groups = {}
    for batch_size in (32, 128, 512):
        for method in A_METHODS:
            groups[(batch_size, method)] = load_condition(os.path.join(a_root, method, a_condition_name(method, batch_size)))
        for method in REFERENCE_METHODS:
            groups[(batch_size, method)] = load_condition(
                os.path.join(reference_roots[batch_size], method, REFERENCE_CONDITIONS[batch_size]))
    return groups


def a_table_lines(groups: dict) -> list:
    """
    概要: A の表（手法 × バッチサイズ，既存のランダム点を参考として併記）を作る．
    引数: groups (dict)．`a_groups` の戻り値．
    戻り値: lines (list of str)．
    """
    lines = ["| バッチサイズ（予算） | 手法 | シード数 | 予算でのテスト精度［%］ | 予算での学習損失 | "
             "予算の後半の近似誤差の中央値 | 予算の後半の揺らぎ［%pt］ | 打ち切り |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for batch_size in (32, 128, 512):
        budget = A_BUDGETS[batch_size]
        for method in A_METHODS + REFERENCE_METHODS:
            logs = groups.get((batch_size, method), {})
            if not logs:
                continue
            row = a_row(logs, budget, with_approx=method.startswith("NFG"))
            approx = _fmt(row["approx"], digits=4) if row["approx"] else "—（全勾配）"
            stopped = (f"{len(row['stopped'])}（" + "，".join(f"seed {s}: {e}" for s, e in sorted(row["stopped"].items()))
                       + "）") if row["stopped"] else "0"
            label = METHOD_LABELS[method] + ("（既存）" if method in REFERENCE_METHODS else "")
            lines.append(f"| {batch_size}（{budget}） | {label} | {row['n']} | {_fmt(row['acc'], 100, 2)} | "
                         f"{_fmt(row['loss'])} | {approx} | {_fmt(row['fluct'], 100, 2)} | {stopped} |")
    return lines


def mean_se_curve(logs: dict, key: str):
    """
    概要: シードの間で評価点の数をそろえ，平均と標準誤差の曲線を返す（打ち切られた試行は打ち切りの前の点までで短くなる）．
    引数: logs (dict)．key (str)．
    戻り値: (x, mean, se) の組（np.ndarray）．
    """
    series = []
    for log in logs.values():
        values = np.asarray(log[key], dtype=float)
        series.append((grad_axis(log), np.where(np.isfinite(values), values, np.nan)))
    length = min(len(x) for x, _ in series)
    x = series[0][0][:length]
    stack = np.stack([v[:length] for _, v in series])
    count = np.sum(np.isfinite(stack), axis=0)
    mean = np.nanmean(stack, axis=0)
    se = np.where(count > 1, np.nanstd(stack, axis=0, ddof=1) / np.sqrt(np.maximum(count, 1)), 0.0)
    return x, mean, se


def plot_a(groups: dict, batch_size: int, key: str, figure_dir: str = A_FIGURE_DIR) -> str:
    """
    概要: A の確認用の図（E7 の2手法を実線，既存のランダム点を破線）を保存する．
    引数: groups (dict)．batch_size (int)．key (str)．"train_loss"，"test_accuracy"，"approx_error"．
        figure_dir (str)．
    戻り値: path (str)．
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"SVRG_final": "SVRG (final)", "NFG_SVRG_final": "NFG-SVRG (final)",
              "SVRG": "SVRG (random, existing)", "NFG_SVRG": "NFG-SVRG (random, existing)"}
    budget = A_BUDGETS[batch_size]
    fig, ax = plt.subplots(figsize=(5, 5))
    for method in A_METHODS + REFERENCE_METHODS:
        logs = groups.get((batch_size, method), {})
        if not logs or (key == "approx_error" and method in ("SVRG_final", "SVRG")):
            continue
        x, mean, se = mean_se_curve(logs, key)
        keep = x <= budget * 1.01
        style = "-" if method in A_METHODS else "--"
        ax.plot(x[keep], mean[keep], style, color=COLORS[method], label=labels[method])
        ax.fill_between(x[keep], (mean - se)[keep], (mean + se)[keep], color=COLORS[method], alpha=0.2)
    ax.axvspan(budget / 2, budget, color="gray", alpha=0.1)
    if key in ("train_loss", "approx_error"):
        ax.set_yscale("log")
    ax.set_xlabel("#grad/N")
    ax.set_ylabel({"train_loss": "Training loss", "test_accuracy": "Test accuracy",
                   "approx_error": "Approximation error"}[key])
    ax.set_title(f"ImageWoof, batch size {batch_size}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    path = os.path.join(figure_dir, f"e7_bs{batch_size}_{key}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def a_section_lines(a_root: str = A_ROOT, reference_roots: dict = None, figure_dir: str = A_FIGURE_DIR) -> list:
    """
    概要: A の節（表と図の一覧）を作る．A の結果がない場合は「結果なし」とする．
    引数: a_root (str)．reference_roots (dict)．figure_dir (str)．
    戻り値: lines (list of str)．
    """
    groups = a_groups(a_root, reference_roots)
    if not any(groups[(bs, m)] for bs in (32, 128, 512) for m in A_METHODS):
        return ["## A：精度の実験（`ex012_e7_imagewoof_final`）", "", "結果なし．", ""]
    figures = [plot_a(groups, bs, key, figure_dir) for bs in (32, 128, 512)
               for key in ("train_loss", "test_accuracy", "approx_error")
               if any(groups[(bs, m)] for m in A_METHODS)]
    return ["## A：精度の実験（`ex012_e7_imagewoof_final`）", "",
            "値は平均 ± 標準誤差（ddof=1）である．予算での値は，$ \\#\\mathrm{grad}/N $ が予算以下の最後の評価点の値である．"
            "近似誤差の中央値と揺らぎ（隣り合う評価点の間のテスト精度の変化の平均）は，予算の1/2以上・予算以下の区間で求めた．"
            "打ち切りは「試行数（シード: 打ち切りのエポック）」である．打ち切られて予算に届かない試行は，予算での値の平均から除く．"
            "「既存」は，同じバッチサイズの既存の SVRG・NFG-SVRG（ランダム点，内側ループの開始点は snapshot）である．", "",
            *a_table_lines(groups), "", "### 図", "",
            *[f"- `{os.path.relpath(path, OUTPUTS_DIR)}`" for path in figures], ""]


def write_summary(output: str = SUMMARY_PATH, b_root: str = B_ROOT, a_root: str = A_ROOT,
                  reference_roots: dict = None, figure_dir: str = A_FIGURE_DIR) -> str:
    """
    概要: 集計結果を Markdown に書き出す．
    引数: output (str)．b_root (str)．a_root (str)．reference_roots (dict) = None．figure_dir (str)．
    戻り値: output (str)．
    """
    runs = load_b_runs(b_root)
    table = epoch_time_table(runs)
    problems = check_alternation(runs)
    lines = ["# E7 の集計結果", "",
             "## B：1エポックの時間（`ex013_e7_epoch_time`）", "",
             "1エポックの時間は `time_optimizer + time_svrg_full_grad` の2〜4エポック目の平均（1エポック目は準備運転として"
             "除く）であり，評価の時間と近似誤差の評価のためだけの全勾配の時間を含めない．値は3周の平均（最小〜最大）である．"
             "予算までの推定時間は，1エポックの時間 × 予算に届く最小のエポック数であり，SVRG は初期の全勾配の時間を加えた．", "",
             *b_table_lines(table), "", "### 手法間の比", "", *ratio_lines(table), "",
             "### 実行の順序", "",
             "同じ組の中で同じ手法が連続した箇所：" + ("なし．" if not problems else "；".join(problems)), "",
             *execution_order_lines(runs), "",
             *a_section_lines(a_root, reference_roots, figure_dir)]
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write("\n".join(lines))
    return output


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b-root", default=B_ROOT)
    parser.add_argument("--a-root", default=A_ROOT)
    for batch_size in (32, 128, 512):
        parser.add_argument(f"--reference-root-bs{batch_size}", default=DEFAULT_REFERENCE_ROOTS[batch_size],
                            help=f"既存の SVRG・NFG-SVRG（ランダム点）のバッチサイズ{batch_size}の出力ルート")
    parser.add_argument("--output", default=SUMMARY_PATH)
    return parser.parse_args()


def main() -> None:
    """
    概要: 集計を実行する．
    引数: なし
    戻り値: なし
    """
    args = parse_args()
    reference_roots = {bs: getattr(args, f"reference_root_bs{bs}") for bs in (32, 128, 512)}
    print(write_summary(args.output, args.b_root, args.a_root, reference_roots))


if __name__ == "__main__":
    main()
