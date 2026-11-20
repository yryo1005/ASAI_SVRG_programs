"""
E3（実験計画5章）：ImageWoof における実時間の比較の集計．

`outputs/ex009_e3_imagewoof_walltime` の試行を読み込み，次を出力する．

- `outputs/e3_summary.md`：
  - 手法ごとの1エポックあたりの平均時間（`time_optimizer`，`time_svrg_full_grad`，参考として `time_diagnostic`，`time_evaluation`）
  - $ \\#\\mathrm{grad}/N \\approx 200 $ に到達するまでの実時間
  - テスト精度40%に初めて到達するまでの実時間
  - 最終の学習損失とテスト精度
  - 計測環境の一覧
- `outputs/ex009_e3_imagewoof_walltime/figures/e3_walltime_{train_loss,test_accuracy}.png`：横軸は実時間 $ t $［s］．
- `outputs/ex009_e3_imagewoof_walltime/figures/e3_grad_{train_loss,test_accuracy}.png`：横軸は $ \\#\\mathrm{grad}/N $（参考）．

- 各手法に用いる条件（`PAPER_CONDITIONS`）：実験計画5.6節により，NFG-SVRG は restart（`inner_snapshot_...`），
  SVRG・ASA-SVRG は `inner_last_iterate_...`，SGD は `inner_none_...` を用いる．当初の NFG-SVRG（continue，
  `inner_last_iterate_...`）は確認用の節と図（`e3_check_*.png`）にのみ用いる．
- 確認用の節：restart と continue の NFG-SVRG の時間の比（計算機の速度の変動の確認）と対応のある差，
  restart の NFG-SVRG と E0（`ex003_e0_imagewoof_resnet18_layernorm`，`snapshot`）のログの一致．

実時間は $ t = \\mathrm{time\\_optimizer} + \\mathrm{time\\_svrg\\_full\\_grad} $ である（`elapsed_time` に記録）．
曲線は，同じ評価点（エポック）における $ t $ と指標の値をシードの間で平均して描く．

実行例:
    python analysis_e3.py
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis_e0 import _fmt  # noqa: E402

_PROGRAMS_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
OUTPUTS_DIR = os.path.join(REVIEW_ROOT, "outputs")
E3_DIR = os.path.join(OUTPUTS_DIR, "ex009_e3_imagewoof_walltime")
FIGURE_DIR = os.path.join(E3_DIR, "figures")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "e3_summary.md")

N_TRAIN = 9025
ACCURACY_THRESHOLD = 0.40
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
LABELS = {"SGD": "SGD", "SVRG": "SVRG", "NFG_SVRG": "NFG-SVRG", "ASAI_SVRG": "ASA-SVRG",
          "NFG_SVRG_continue": "NFG-SVRG (continue)"}
COLORS = {"SGD": "#7f7f7f", "SVRG": "#1f77b4", "NFG_SVRG": "#2ca02c", "ASAI_SVRG": "#d62728",
          "NFG_SVRG_continue": "#98df8a"}
_HP = "lr0.001_bs128_lambda0.0005"
# 系列名 → (手法のディレクトリ，条件名)．論文用の図と表に用いる条件（実験計画5.6節）．
PAPER_CONDITIONS = {
    "SGD": ("SGD", f"inner_none_{_HP}_epochs200"),
    "SVRG": ("SVRG", f"inner_last_iterate_{_HP}_epochs67"),
    "NFG_SVRG": ("NFG_SVRG", f"inner_snapshot_{_HP}_epochs100"),
    "ASAI_SVRG": ("ASAI_SVRG", f"inner_last_iterate_{_HP}_epochs100"),
}
# 確認用の系列：当初の計測の NFG-SVRG（continue）．
CONTINUE_CONDITION = {"NFG_SVRG_continue": ("NFG_SVRG", f"inner_last_iterate_{_HP}_epochs100")}
# 当初の NFG-SVRG（continue）の time_optimizer／エポック（report_051，5シード）．実験計画5.6節の確認2の基準．
CONTINUE_TIME_OPTIMIZER_REFERENCE = 10.708
SPEED_TOLERANCE = 0.03
E0_DIR = os.path.join(OUTPUTS_DIR, "ex003_e0_imagewoof_resnet18_layernorm")
E0_RESTART_CONDITION = ("NFG_SVRG", f"inner_snapshot_{_HP}_epochs128")
E0_COMPARED_KEYS = ("epoch", "oracle_calls", "train_loss", "test_accuracy", "test_correct", "approx_error")
TIME_KEYS = ("time_optimizer", "time_svrg_full_grad", "time_diagnostic", "time_evaluation")


def load_runs(experiment_dir: str = E3_DIR, conditions: dict = None) -> dict:
    """
    概要: 完了した試行を系列ごとに読み込む．系列ごとに1つの条件（手法のディレクトリと条件名）だけを読む．
    引数: experiment_dir (str)．conditions (dict) = None．系列名 → (手法のディレクトリ，条件名)．
        None の場合は `PAPER_CONDITIONS`．
    戻り値: runs (dict)．系列名 → list of dict（`seed`，`log`，`config`，`x`（$ \\#\\mathrm{grad}/N $））．シード順．
    """
    runs = {}
    for series, (method, condition) in (conditions or PAPER_CONDITIONS).items():
        condition_dir = os.path.join(experiment_dir, method, condition)
        if not os.path.isdir(condition_dir):
            continue
        for seed in sorted(os.listdir(condition_dir), key=int):
            run_dir = os.path.join(condition_dir, seed)
            log_path, config_path = os.path.join(run_dir, "log.json"), os.path.join(run_dir, "config.json")
            if not (os.path.exists(log_path) and os.path.exists(config_path)):
                continue
            with open(log_path) as f:
                log = json.load(f)
            with open(config_path) as f:
                config = json.load(f)
            runs.setdefault(series, []).append({
                "seed": int(seed), "log": log, "config": config,
                "x": np.asarray(log["oracle_calls"], dtype=np.float64) / N_TRAIN})
    return {m: sorted(r, key=lambda run: run["seed"]) for m, r in runs.items()}


def _array(run: dict, key: str) -> np.ndarray:
    """
    概要: 指標の記録を float64 の配列として返す．
    引数: run (dict)，key (str)．
    戻り値: values (numpy.ndarray)．
    """
    return np.asarray([np.nan if v is None else v for v in run["log"][key]], dtype=np.float64)


def per_epoch_time(run: dict, key: str) -> float:
    """
    概要: 1エポックあたりの平均時間を返す．0エポック目（初期状態）の時間は除く．
    引数: run (dict)，key (str)．`TIME_KEYS` のいずれか．
    戻り値: seconds (float)．
    """
    values = _array(run, key)
    epochs = len(values) - 1
    return float((values[-1] - values[0]) / epochs) if epochs > 0 else float("nan")


def time_to_accuracy(run: dict, threshold: float = ACCURACY_THRESHOLD) -> float:
    """
    概要: テスト精度が初めて `threshold` 以上になる評価点の実時間を返す．
    引数: run (dict)，threshold (float)．
    戻り値: t (float)．到達しない場合は NaN．
    """
    hit = np.where(_array(run, "test_accuracy") >= threshold)[0]
    return float(_array(run, "elapsed_time")[hit[0]]) if hit.size else float("nan")


def timing_table(runs: dict) -> str:
    """
    概要: 手法ごとの時間の表（実験計画5.4節の表）を Markdown で作る．
    引数: runs (dict)．`load_runs` の戻り値．
    戻り値: text (str)．
    """
    lines = [
        "| 手法 | シード数 | エポック数 | 最終の $ \\#\\mathrm{grad}/N $ | time_optimizer／エポック［s］ | "
        "time_svrg_full_grad／エポック［s］ | t／エポック［s］ | time_diagnostic／エポック［s］（参考） | "
        "time_evaluation／エポック［s］（参考） | 最終の t［s］ | 精度40%到達の t［s］（到達数） | 最終の学習損失 | 最終のテスト精度 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, method_runs in runs.items():
        opt = [per_epoch_time(r, "time_optimizer") for r in method_runs]
        full = [per_epoch_time(r, "time_svrg_full_grad") for r in method_runs]
        wall = [per_epoch_time(r, "elapsed_time") for r in method_runs]
        diag = [per_epoch_time(r, "time_diagnostic") for r in method_runs]
        evaluation = [per_epoch_time(r, "time_evaluation") for r in method_runs]
        final_t = [_array(r, "elapsed_time")[-1] for r in method_runs]
        to_acc = [time_to_accuracy(r) for r in method_runs]
        reached = int(np.isfinite(to_acc).sum())
        loss = [_array(r, "train_loss")[-1] for r in method_runs]
        acc = [_array(r, "test_accuracy")[-1] for r in method_runs]
        epochs = len(method_runs[0]["log"]["epoch"]) - 1
        lines.append(
            f"| {LABELS[method]} | {len(method_runs)} | {epochs} | {method_runs[0]['x'][-1]:.1f} | {_fmt(opt, 3)} | "
            f"{_fmt(full, 3)} | {_fmt(wall, 3)} | {_fmt(diag, 3)} | {_fmt(evaluation, 3)} | {_fmt(final_t, 1)} | "
            f"{_fmt(to_acc, 1)}（{reached}/{len(method_runs)}） | {_fmt(loss)} | {_fmt(acc)} |")
    return "\n".join(lines)


def comparison_lines(runs: dict) -> str:
    """
    概要: 実験計画5.5節の論点に関する比を計算する．
    引数: runs (dict)．
    戻り値: text (str)．箇条書き．
    """
    def mean_of(method, key):
        if method not in runs:
            return float("nan")
        return float(np.mean([per_epoch_time(r, key) for r in runs[method]]))

    svrg_opt = mean_of("SVRG", "time_optimizer")
    svrg_full = mean_of("SVRG", "time_svrg_full_grad")
    nfg_opt, asa_opt = mean_of("NFG_SVRG", "time_optimizer"), mean_of("ASAI_SVRG", "time_optimizer")
    sgd_opt = mean_of("SGD", "time_optimizer")
    return "\n".join([
        f"- SVRG の1エポックあたりの時間のうち，全勾配の計算を除いた部分（time_optimizer）：{svrg_opt:.3f} s．"
        f"全勾配の計算：{svrg_full:.3f} s（1エポックあたりの t の {100 * svrg_full / (svrg_opt + svrg_full):.1f}%）．",
        f"- NFG-SVRG の time_optimizer／SVRG の time_optimizer：{nfg_opt / svrg_opt:.3f}．",
        f"- ASA-SVRG の time_optimizer／SVRG の time_optimizer：{asa_opt / svrg_opt:.3f}．",
        f"- ASA-SVRG の time_optimizer／NFG-SVRG の time_optimizer：{asa_opt / nfg_opt:.3f}"
        f"（平均パラメータの逐次更新のオーバーヘッド {1000 * (asa_opt - nfg_opt):.1f} ms／エポック）．",
        f"- SGD の time_optimizer／ASA-SVRG の time_optimizer：{sgd_opt / asa_opt:.3f}．",
    ])


def full_pass_time(run: dict) -> float:
    """
    概要: 1エポックあたりの全勾配の計算（1回）の時間を返す．SVRG は `time_svrg_full_grad`，
        NFG-SVRG・ASA-SVRG は `time_diagnostic` である．両者は同じ計算（訓練データ全体の勾配）である．
    引数: run (dict)．
    戻り値: seconds (float)．SGD では NaN．
    """
    value = per_epoch_time(run, "time_svrg_full_grad") + per_epoch_time(run, "time_diagnostic")
    return value if value > 0.0 else float("nan")


def per_seed_table(runs: dict) -> str:
    """
    概要: 試行ごとの1エポックあたりの time_optimizer と全勾配の計算の時間，およびその比を Markdown の表にする．
        計測の間に計算機の速度が変化した場合に，その影響を確認するための表である．
    引数: runs (dict)．
    戻り値: text (str)．
    """
    seeds = sorted({r["seed"] for method_runs in runs.values() for r in method_runs})
    by_seed = {m: {r["seed"]: r for r in method_runs} for m, method_runs in runs.items()}
    header = "| シード | " + " | ".join(f"{LABELS[m]} time_optimizer［s］ | {LABELS[m]} 全勾配［s］ | "
                                      f"{LABELS[m]} 比" for m in runs if m != "SGD")
    header = header + (" | SGD time_optimizer［s］ |" if "SGD" in runs else " |")
    columns = sum(3 for m in runs if m != "SGD") + (1 if "SGD" in runs else 0)
    lines = [header, "|---:|" + "---:|" * columns]
    for seed in seeds:
        cells = []
        for method in runs:
            if method == "SGD":
                continue
            run = by_seed[method].get(seed)
            if run is None:
                cells += ["—"] * 3
                continue
            opt, full = per_epoch_time(run, "time_optimizer"), full_pass_time(run)
            cells += [f"{opt:.3f}", f"{full:.3f}", f"{opt / full:.4f}"]
        if "SGD" in runs:
            run = by_seed["SGD"].get(seed)
            cells.append(f"{per_epoch_time(run, 'time_optimizer'):.3f}" if run else "—")
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def normalized_comparison_lines(runs: dict) -> str:
    """
    概要: 計算機の速度の変化の影響を除いた比を計算する．
        - SVRG系の手法の間：各試行の time_optimizer を，同じ試行の全勾配の計算の時間（`full_pass_time`）で割った値を
          手法ごとに平均し，その比をとる．全勾配の計算は全手法で同じ計算であるため，試行ごとの速度の違いが打ち消される．
        - SGD と ASA-SVRG：同じシードの試行の time_optimizer の比の中央値をとる（SGD には全勾配の計算がないため）．
    引数: runs (dict)．
    戻り値: text (str)．箇条書き．
    """
    def normalized(method):
        if method not in runs:
            return float("nan")
        return float(np.mean([per_epoch_time(r, "time_optimizer") / full_pass_time(r) for r in runs[method]]))

    svrg, nfg, asa = normalized("SVRG"), normalized("NFG_SVRG"), normalized("ASAI_SVRG")
    lines = [
        f"- time_optimizer／全勾配の計算（試行ごとの比の平均）：SVRG {svrg:.4f}，NFG-SVRG {nfg:.4f}，ASA-SVRG {asa:.4f}．",
        f"- NFG-SVRG／SVRG：{nfg / svrg:.4f}．ASA-SVRG／SVRG：{asa / svrg:.4f}．",
        f"- ASA-SVRG／NFG-SVRG：{asa / nfg:.4f}（平均パラメータの逐次更新のオーバーヘッド {100 * (asa / nfg - 1):.2f}%）．",
    ]
    if "SGD" in runs and "ASAI_SVRG" in runs:
        sgd_by_seed = {r["seed"]: per_epoch_time(r, "time_optimizer") for r in runs["SGD"]}
        ratios = [sgd_by_seed[r["seed"]] / per_epoch_time(r, "time_optimizer")
                  for r in runs["ASAI_SVRG"] if r["seed"] in sgd_by_seed]
        lines.append(f"- SGD／ASA-SVRG の time_optimizer（同じシードの比）：中央値 {np.median(ratios):.3f}"
                     f"（範囲 {min(ratios):.3f}〜{max(ratios):.3f}）．")
    return "\n".join(lines)


def environment_table(runs: dict) -> str:
    """
    概要: 計測環境の一覧を作る（全試行で同一であることも確認する）．
    引数: runs (dict)．
    戻り値: text (str)．
    """
    environments = [json.dumps(r["config"]["environment"], sort_keys=True) for rs in runs.values() for r in rs]
    if not environments:
        return "（試行なし）"
    env = json.loads(environments[0])
    lines = ["| 項目 | 値 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(env.items())]
    same = len(set(environments)) == 1
    lines.append(f"\n全ての試行で計測環境が同一：{'はい' if same else 'いいえ（試行ごとの config.json を参照）'}")
    return "\n".join(lines)


def _mean_curve(method_runs: list, key: str, x_key: str):
    """
    概要: 全シードに共通する評価点について，横軸と指標の平均と標準誤差を求める．
    引数: method_runs (list)，key (str)，x_key (str)．"elapsed_time" または "grad"．
    戻り値: (x, mean, se) の組．
    """
    length = min(len(r["log"]["epoch"]) for r in method_runs)
    if x_key == "grad":
        x = method_runs[0]["x"][:length]
    else:
        x = np.mean([_array(r, x_key)[:length] for r in method_runs], axis=0)
    values = np.array([_array(r, key)[:length] for r in method_runs])
    mean = np.nanmean(values, axis=0)
    se = np.nanstd(values, axis=0, ddof=1) / np.sqrt(len(method_runs)) if len(method_runs) > 1 else np.zeros_like(mean)
    return x, mean, se


def plot_curves(runs: dict, name_prefix: str = "e3") -> list:
    """
    概要: 学習損失とテスト精度を，横軸を実時間と $ \\#\\mathrm{grad}/N $ の2通りで描く．
        系列 `NFG_SVRG_continue` は破線で描く．
    引数: runs (dict)．name_prefix (str) = "e3"．ファイル名の接頭辞（確認用の図は "e3_check"）．
    戻り値: paths (list of str)．
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    paths = []
    for x_key, kind, xlabel in (("elapsed_time", "walltime", "Wall-clock time $t$ [s]"),
                                ("grad", "grad", r"$\#\mathrm{grad}/N$")):
        prefix = f"{name_prefix}_{kind}"
        for key, ylabel in (("train_loss", "Training loss"), ("test_accuracy", "Test accuracy")):
            fig, ax = plt.subplots(figsize=(5, 5))
            for method, method_runs in runs.items():
                x, mean, se = _mean_curve(method_runs, key, x_key)
                style = "--" if method == "NFG_SVRG_continue" else "-"
                ax.plot(x, mean, style, color=COLORS[method], label=LABELS[method], lw=1.5)
                ax.fill_between(x, mean - se, mean + se, color=COLORS[method], alpha=0.2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = os.path.join(FIGURE_DIR, f"{prefix}_{key}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    return paths


def restart_speed_check_lines(runs: dict, continue_runs: dict) -> str:
    """
    概要: 実験計画5.6節の確認2：restart の NFG-SVRG の time_optimizer／エポックを，当初の continue の計測と比べる．
        restart と continue は1エポックの計算量が同じであるため，差は計算機の速度の変動を表す．
        全勾配の計算（time_diagnostic）で割った比も示す．
    引数: runs (dict)．論文用の系列．continue_runs (dict)．`NFG_SVRG_continue` の系列．
    戻り値: text (str)．箇条書き．
    """
    if "NFG_SVRG" not in runs or "NFG_SVRG_continue" not in continue_runs:
        return "（restart または continue の試行なし）"
    restart = [per_epoch_time(r, "time_optimizer") for r in runs["NFG_SVRG"]]
    cont = [per_epoch_time(r, "time_optimizer") for r in continue_runs["NFG_SVRG_continue"]]
    restart_norm = [per_epoch_time(r, "time_optimizer") / full_pass_time(r) for r in runs["NFG_SVRG"]]
    cont_norm = [per_epoch_time(r, "time_optimizer") / full_pass_time(r) for r in continue_runs["NFG_SVRG_continue"]]
    ratio = float(np.mean(restart)) / float(np.mean(cont))
    within = abs(ratio - 1.0) <= SPEED_TOLERANCE
    return "\n".join([
        f"- time_optimizer／エポック：restart {_fmt(restart, 3)} s，continue {_fmt(cont, 3)} s"
        f"（当初の報告値 {CONTINUE_TIME_OPTIMIZER_REFERENCE:.3f} s）．",
        f"- restart／continue：{ratio:.4f}（差 {100 * (ratio - 1.0):+.2f}%）．"
        f"判定（差が{100 * SPEED_TOLERANCE:.0f}%以内）：{'以内．SGD・SVRG・ASA-SVRG の計測をそのまま用いる' if within else '超過'}．",
        f"- time_optimizer／全勾配の計算：restart {_fmt(restart_norm, 4)}，continue {_fmt(cont_norm, 4)}"
        f"（比 {float(np.mean(restart_norm)) / float(np.mean(cont_norm)):.4f}）．",
    ])


def _reach_text(t: float) -> str:
    """
    概要: 精度の到達時間を表示用の文字列にする．到達しない場合（NaN）は「到達せず」とする．
    引数: t (float)．
    戻り値: text (str)．
    """
    return "到達せず" if np.isnan(t) else f"{t:.1f}"


def continue_vs_restart_table(runs: dict, continue_runs: dict) -> str:
    """
    概要: NFG-SVRG の restart と continue の，同じシードの最終の学習損失とテスト精度，その差を表にする．
    引数: runs (dict)．continue_runs (dict)．
    戻り値: text (str)．Markdown の表．
    """
    if "NFG_SVRG" not in runs or "NFG_SVRG_continue" not in continue_runs:
        return "（restart または continue の試行なし）"
    cont_by_seed = {r["seed"]: r for r in continue_runs["NFG_SVRG_continue"]}
    lines = ["| シード | 学習損失 restart | 学習損失 continue | 差（restart − continue） | テスト精度 restart | "
             "テスト精度 continue | 差（restart − continue） | 精度40%到達の t［s］ restart | continue |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    d_loss, d_acc = [], []
    for run in runs["NFG_SVRG"]:
        other = cont_by_seed.get(run["seed"])
        if other is None:
            continue
        lr, lc = _array(run, "train_loss")[-1], _array(other, "train_loss")[-1]
        ar, ac = _array(run, "test_accuracy")[-1], _array(other, "test_accuracy")[-1]
        d_loss.append(lr - lc)
        d_acc.append(ar - ac)
        lines.append(f"| {run['seed']} | {lr:.4f} | {lc:.4f} | {lr - lc:+.4f} | {ar:.4f} | {ac:.4f} | {ar - ac:+.4f} | "
                     f"{_reach_text(time_to_accuracy(run))} | {_reach_text(time_to_accuracy(other))} |")
    lines.append(f"| 平均 ± 標準誤差 | | | {_fmt(d_loss)} | | | {_fmt(d_acc)} | | |")
    return "\n".join(lines)


def e0_consistency_table(runs: dict, e0_dir: str = E0_DIR) -> str:
    """
    概要: restart の NFG-SVRG の全評価点（0〜100エポック）を，E0 の ImageWoof の NFG-SVRG（`snapshot`，128エポック）
        の先頭の同じ数の評価点と比べる（実験計画5.6節の確認1）．
    引数: runs (dict)．e0_dir (str) = E0_DIR．
    戻り値: text (str)．Markdown の表．
    """
    if "NFG_SVRG" not in runs:
        return "（restart の試行なし）"
    method, condition = E0_RESTART_CONDITION
    lines = ["| シード | 比較した点数 | " + " | ".join(E0_COMPARED_KEYS) + " |",
             "|---:|---:|" + "---|" * len(E0_COMPARED_KEYS)]
    for run in runs["NFG_SVRG"]:
        path = os.path.join(e0_dir, method, condition, str(run["seed"]), "log.json")
        if not os.path.exists(path):
            lines.append(f"| {run['seed']} | — | " + " | ".join("E0 のログなし" for _ in E0_COMPARED_KEYS) + " |")
            continue
        with open(path) as f:
            e0_log = json.load(f)
        n = len(run["log"]["epoch"])
        cells = []
        for key in E0_COMPARED_KEYS:
            same = run["log"][key] == e0_log[key][:n]
            cells.append("完全一致" if same else "不一致")
        lines.append(f"| {run['seed']} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def environment_match_lines(runs: dict, continue_runs: dict) -> str:
    """
    概要: restart の NFG-SVRG の計測環境（`config.json` の `environment`）が，他の手法の当初の計測と同じかを確認する
        （実験計画5.6節の確認3）．
    引数: runs (dict)．continue_runs (dict)．
    戻り値: text (str)．箇条書き．
    """
    if "NFG_SVRG" not in runs:
        return "（restart の試行なし）"
    new = {json.dumps(r["config"]["environment"], sort_keys=True) for r in runs["NFG_SVRG"]}
    old = {json.dumps(r["config"]["environment"], sort_keys=True)
           for series, rs in list(runs.items()) + list(continue_runs.items()) if series != "NFG_SVRG" for r in rs}
    if new == old:
        return "- restart の NFG-SVRG と当初の計測の計測環境（機器，ソフトウェアのバージョン等）は同一である．"
    diff = []
    for env_text in new | old:
        env = json.loads(env_text)
        diff.append(f"  - {'restart' if env_text in new else '当初'}：{env}")
    return "- 計測環境が異なる：\n" + "\n".join(diff)


def write_summary() -> str:
    """
    概要: 表と図を作り，`outputs/e3_summary.md` に保存する．
    引数: なし
    戻り値: text (str)．
    """
    runs = load_runs()
    continue_runs = load_runs(conditions=CONTINUE_CONDITION)
    figures = plot_curves(runs) if runs else []
    if "NFG_SVRG_continue" in continue_runs:
        figures += plot_curves({**runs, **continue_runs}, "e3_check")
    text = "\n\n".join([
        "# E3：ImageWoof における実時間の比較（集計結果）",
        "値はシードの平均 ± 標準誤差である．実時間 $ t $ は time_optimizer + time_svrg_full_grad であり，"
        "近似誤差の診断（time_diagnostic）と評価（time_evaluation）の時間は含めない．全ての試行は逐次に1つずつ実行した．",
        "## 時間の表", timing_table(runs) if runs else "（試行なし）",
        "## 比（実験計画5.5節）", comparison_lines(runs) if runs else "（試行なし）",
        "## 計算機の速度の変化を除いた比",
        "同じ計算でも，試行によって計算機の速度が数%異なる場合がある（下の試行ごとの表を参照）．"
        "そのため，各試行の time_optimizer を同じ試行の全勾配の計算の時間で割った比でも比較する．",
        normalized_comparison_lines(runs) if runs else "（試行なし）",
        "### 試行ごとの1エポックあたりの時間", per_seed_table(runs) if runs else "（試行なし）",
        "## NFG-SVRG の restart での再計測の確認（実験計画5.6節）",
        "NFG-SVRG は restart（`inner_start=\"snapshot\"`）の結果を上の表と図に用いた．当初の continue "
        "（`last_iterate`）の計測は，この節と確認用の図（`e3_check_*.png`）にのみ用いる．",
        "### 計算機の速度の変動（確認2）", restart_speed_check_lines(runs, continue_runs),
        "### E0 のログとの一致（確認1）", e0_consistency_table(runs),
        "### 計測環境（確認3）", environment_match_lines(runs, continue_runs),
        "### restart と continue の対応のある差（参考）", continue_vs_restart_table(runs, continue_runs),
        "## 計測環境", environment_table(runs),
        "## 図", "\n".join(f"- `{os.path.relpath(p, OUTPUTS_DIR)}`" for p in figures),
    ]) + "\n"
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print(write_summary())
