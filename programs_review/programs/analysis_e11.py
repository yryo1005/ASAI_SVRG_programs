"""
E11（`実験指示E11.md`．Mushroom における全てのスナップショットと内側ループの開始点の構成）の集計．

`outputs/ex006_e1_mushroom_rho_sampling/` の10構成（スナップショット × 全勾配の推定 × 内側ループの開始点）を，
`config.json` の `inner_start` で区別して読み込み，`outputs/e11_summary.md` に次をまとめる（図は作らない）．

1. random reshuffling の全構成 × 全 $ \\rho $ の表：$ \\hat{r} $，$ f({\\mathbf z}_{s}) - f({\\mathbf w}^{*}) \\le 10^{-12} $ に
   初めて到達する $ \\#\\mathrm{grad}/N $，最後の20外側ループの $ \\|{\\mathbf e}_{s}\\|^{2} $ の中央値，
   最後の外側ループのテスト精度と正解数（5シードの平均 ± 標準誤差，`ddof=1`）．
2. 一様サンプリング（$ \\rho = 0.5, 1.0, 298.4 $）の目的関数誤差と近似誤差の誤差床（最後の20外側ループの中央値）．
3. 打ち切られた試行の一覧．
4. 実行の記録（試行の数，所要時間，並列数，PC の構成）．

$ \\hat{r} $ は指示書6節の定義（`make_report090_e1.hat_r_of`，`compute_hat_r_lsq` と同じ計算）である．
シード $ c $ ごとに，$ \\Delta_{s} > 10^{-12} $ が $ s = 1 $ から続く最後の外側ループを $ s_{\\mathrm{end}} $ とし，
$ s = 1, \\dots, s_{\\mathrm{end}} $ の点 $ (s, \\log \\Delta_{s}) $ に最小二乗で直線をあてはめた傾き $ a_{c} $ から
$ \\hat{r} = \\frac{1}{C}\\sum_{c}\\exp(a_{c}) $ とする．$ s_{\\mathrm{end}} < 3 $ のシードは除く（参照先と同じ）．
`programs_review/` の外は import しないため，同じ計算をここに書いた．

使い方:
    python analysis_e11.py
    python analysis_e11.py --workers 32 --run-logs ../outputs/_run_logs/ex006_e11_continue.log ../outputs/_run_logs/ex006_e11_reset.log
"""

import argparse
import datetime
import json
import os
import platform

import numpy as np

_PROGRAMS_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
OUTPUTS_DIR = os.path.join(REVIEW_ROOT, "outputs")
EXPERIMENT_DIR = os.path.join(OUTPUTS_DIR, "ex006_e1_mushroom_rho_sampling")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "e11_summary.md")

# (表示名, ex006 の手法名, inner_start, E11 で新たに実行したか)．指示書1節の表の順．
CONFIGS = [
    ("SVRG / random / reset", "SVRG_random", "snapshot", False),
    ("SVRG / random / continue", "SVRG_random", "last_iterate", True),
    ("SVRG / final / reset = continue", "SVRG", "snapshot", False),
    ("SVRG / average / reset", "SVRG_average", "snapshot", True),
    ("SVRG / average / continue", "SVRG_average", "last_iterate", True),
    ("NFG-SVRG / random / reset", "NFG_SVRG_random", "snapshot", False),
    ("NFG-SVRG / random / continue", "NFG_SVRG_random", "last_iterate", True),
    ("NFG-SVRG / final / reset = continue", "NFG_SVRG", "snapshot", False),
    ("ASA-SVRG / average / reset", "ASAI_SVRG", "snapshot", False),
    ("ASA-SVRG / average / continue", "ASAI_SVRG", "last_iterate", True),
]
RHO_CONDITIONS = [0.5, 0.64, 0.8, 1.0, 2.0, 10.0, 298.4]
UNIFORM_RHOS = [0.5, 1.0, 298.4]
SAMPLINGS = ("random_reshuffle", "uniform_with_replacement")
SEEDS = [0, 1, 2, 3, 4]
HAT_R_THRESHOLD = 1e-12
# 到達の閾値は $ \\hat{r} $ の回帰範囲の閾値にそろえる（指示書6節の $ 10^{-10} $ から変更．ユーザーの指示，order_084 の追加指示）．
REACH_THRESHOLD = HAT_R_THRESHOLD
TAIL_WINDOW = 20


def load_runs(experiment_dir: str = EXPERIMENT_DIR) -> list:
    """
    概要: `CONFIGS` に含まれる手法の全ての試行を読み込む．構成は `config.json` の `inner_start` で区別する．
    引数: experiment_dir (str)．`ex006_e1_mushroom_rho_sampling` ディレクトリ．
    戻り値: runs (list of dict)．各要素は `method`，`inner_start`，`sampling`，`target_rho`，`seed`，`config`，`log`，
        `x`（$ \\#\\mathrm{grad}/N $，numpy.ndarray 形状 (T,)），`run_dir` を持つ．
    """
    methods = sorted({method for _, method, _, _ in CONFIGS})
    runs = []
    for method in methods:
        method_dir = os.path.join(experiment_dir, method)
        if not os.path.isdir(method_dir):
            continue
        for condition in sorted(os.listdir(method_dir)):
            condition_dir = os.path.join(method_dir, condition)
            if not os.path.isdir(condition_dir):
                continue
            for seed in sorted(os.listdir(condition_dir)):
                run = _load_run(os.path.join(condition_dir, seed))
                if run is not None:
                    runs.append(run)
    return runs


def _load_run(run_dir: str):
    """
    概要: 1試行の `log.json` と `config.json` を読み込む．
    引数: run_dir (str)．
    戻り値: run (dict) または None（ファイルが無い場合）．
    """
    log_path, config_path = os.path.join(run_dir, "log.json"), os.path.join(run_dir, "config.json")
    if not (os.path.exists(log_path) and os.path.exists(config_path)):
        return None
    with open(log_path) as f:
        log = json.load(f)
    with open(config_path) as f:
        config = json.load(f)
    return {
        "method": config["method"], "inner_start": config["inner_start"], "sampling": config["sampling"],
        "target_rho": float(config["target_rho"]), "seed": int(config["seed"]), "config": config, "log": log,
        "x": np.asarray(log["oracle_calls"], dtype=np.float64) / float(config["N_train"]), "run_dir": run_dir,
    }


def select_runs(runs: list, method: str, inner_start: str, sampling: str, target_rho: float) -> list:
    """
    概要: 手法・内側ループの開始点・サンプリング・$ \\rho $ で試行を絞り込み，シード順に並べる．
    引数: runs (list of dict)．method (str)．inner_start (str)．sampling (str)．target_rho (float)．
    戻り値: selected (list of dict)．
    """
    selected = [r for r in runs if r["method"] == method and r["inner_start"] == inner_start
                and r["sampling"] == sampling and abs(r["target_rho"] - target_rho) < 1e-9]
    return sorted(selected, key=lambda r: r["seed"])


def hat_r_lsq(run: dict):
    """
    概要: 1シードの実測の縮小率 $ \\exp(a_{c}) $ を最小二乗で求める（`make_report090_e1.compute_hat_r_lsq` と同じ）．
    引数: run (dict)．`log` に `objective_gap` を持つ．
    戻り値: hat_r (float) または None（$ s_{\\mathrm{end}} < 3 $ の場合）．
    """
    delta = np.asarray(run["log"]["objective_gap"], dtype=np.float64)
    s_end = 0
    for s in range(1, len(delta)):
        if not (np.isfinite(delta[s]) and delta[s] > HAT_R_THRESHOLD):
            break
        s_end = s
    if s_end < 3:
        return None
    s = np.arange(1, s_end + 1)
    slope = np.polyfit(s, np.log(delta[1:s_end + 1]), 1)[0]
    return float(np.exp(slope))


def reach_grad_n(run: dict, threshold: float = REACH_THRESHOLD):
    """
    概要: 目的関数誤差が初めて `threshold` 以下になる評価点の $ \\#\\mathrm{grad}/N $ を返す．
    引数: run (dict)．threshold (float)．
    戻り値: value (float) または None（到達しない場合）．
    """
    gap = np.asarray(run["log"]["objective_gap"], dtype=np.float64)
    hit = np.nonzero(np.isfinite(gap) & (gap <= threshold))[0]
    return float(run["x"][hit[0]]) if hit.size else None


def tail_median(run: dict, key: str, window: int = TAIL_WINDOW) -> float:
    """
    概要: 最後の `window` 外側ループ（有限の値）の中央値を返す（`make_e1_paper_assets.tail_median` と同じ）．
    引数: run (dict)．key (str)．window (int)．
    戻り値: value (float)．有限の値が無い場合は NaN．
    """
    values = np.asarray(run["log"][key], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values[-window:])) if values.size else float("nan")


def mean_se(values) -> tuple:
    """
    概要: None と非有限の値を除いた平均と標準誤差（`ddof=1`）を返す．
    引数: values (iterable)．
    戻り値: (mean, se, n) の組．値が無ければ (NaN, NaN, 0)，1つなら標準誤差は NaN．
    """
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
    return float(arr.mean()), se, int(arr.size)


def fmt(values, digits: int = 4, sci: bool = False, total: int = None) -> str:
    """
    概要: 値の列を「平均 ± 標準誤差」の文字列にする．None を除いた数が `total` より少なければ，その数を添える．
    引数: values (iterable)．digits (int)．sci (bool)．total (int, optional)．
    戻り値: text (str)．
    """
    mean, se, n = mean_se(values)
    if n == 0:
        return "—"
    spec = f".{digits - 1}e" if sci else f".{digits}f"
    text = f"{mean:{spec}}" if not np.isfinite(se) else f"{mean:{spec}} ± {se:{spec}}"
    if total is not None and n < total:
        text += f"（{n}/{total}）"
    return text


def reach_text(selected: list) -> str:
    """
    概要: $ 10^{-12} $ への到達の $ \\#\\mathrm{grad}/N $ の文字列（未到達のシードの数を添える）．
    引数: selected (list of dict)．
    戻り値: text (str)．
    """
    values = [reach_grad_n(r) for r in selected]
    n_miss = sum(v is None for v in values)
    if n_miss == len(values):
        return "未到達"
    text = fmt(values, digits=1)
    return text + (f"（未到達 {n_miss}）" if n_miss else "")


def hat_r_text(selected: list) -> str:
    """
    概要: $ \\hat{r} $ の文字列．全てのシードで $ s_{\\mathrm{end}} < 3 $（3外側ループ以内に $ 10^{-12} $ 以下）なら，その旨を示す．
    引数: selected (list of dict)．
    戻り値: text (str)．
    """
    values = [hat_r_lsq(r) for r in selected]
    if all(v is None for v in values):
        return "—（全シードが3外側ループ以内に $10^{-12}$ 以下）"
    return fmt(values, digits=4, total=len(selected))


def final_value(run: dict, key: str) -> float:
    """
    概要: 最後の外側ループ（打ち切られた場合は最後に記録された点）の値を返す．
    引数: run (dict)．key (str)．
    戻り値: value (float)．
    """
    return float(run["log"][key][-1])


def random_reshuffle_table(runs: list) -> list:
    """
    概要: random reshuffling の 10構成 × 7 $ \\rho $ の表の行を作る．
    引数: runs (list of dict)．
    戻り値: lines (list of str)．
    """
    lines = ["| $\\rho$ | 構成 | 新規 | シード数 | $\\hat{r}$ | $10^{-12}$ 到達の $\\#\\mathrm{grad}/N$ | "
             "$\\|{\\mathbf e}_s\\|^2$（最後の20の中央値） | テスト精度［%］ | 正解数（/813） |",
             "|---:|---|:---:|---:|---:|---:|---:|---:|---:|"]
    for rho in RHO_CONDITIONS:
        for label, method, inner_start, is_new in CONFIGS:
            selected = select_runs(runs, method, inner_start, "random_reshuffle", rho)
            if not selected:
                lines.append(f"| {rho:g} | {label} | {'○' if is_new else ''} | 0 | — | — | — | — | — |")
                continue
            lines.append(
                f"| {rho:g} | {label} | {'○' if is_new else ''} | {len(selected)} | "
                f"{hat_r_text(selected)} | {reach_text(selected)} | "
                f"{fmt([tail_median(r, 'approx_error') for r in selected], digits=3, sci=True)} | "
                f"{fmt([100.0 * final_value(r, 'accuracy') for r in selected], digits=2)} | "
                f"{fmt([final_value(r, 'test_correct') for r in selected], digits=1)} |")
    return lines


def uniform_floor_table(runs: list) -> list:
    """
    概要: 一様サンプリングの誤差床（目的関数誤差，近似誤差）の表の行を作る．
    引数: runs (list of dict)．
    戻り値: lines (list of str)．
    """
    lines = ["| $\\rho$ | 構成 | 新規 | シード数 | 目的関数誤差の誤差床 | 近似誤差の誤差床 |",
             "|---:|---|:---:|---:|---:|---:|"]
    for rho in UNIFORM_RHOS:
        for label, method, inner_start, is_new in CONFIGS:
            selected = select_runs(runs, method, inner_start, "uniform_with_replacement", rho)
            lines.append(
                f"| {rho:g} | {label} | {'○' if is_new else ''} | {len(selected)} | "
                f"{fmt([tail_median(r, 'objective_gap') for r in selected], digits=3, sci=True)} | "
                f"{fmt([tail_median(r, 'approx_error') for r in selected], digits=3, sci=True)} |")
    return lines


def terminated_table(runs: list) -> list:
    """
    概要: 打ち切られた試行（目的関数誤差が非有限）の一覧の行を作る．
    引数: runs (list of dict)．
    戻り値: lines (list of str)．
    """
    labels = {(method, inner_start): label for label, method, inner_start, _ in CONFIGS}
    rows = []
    for run in runs:
        key = (run["method"], run["inner_start"])
        if key in labels and bool(run["log"]["terminated"][-1]):
            rows.append((labels[key], run["target_rho"], run["sampling"], run["seed"], int(run["log"]["epoch"][-1])))
    if not rows:
        return ["打ち切られた試行は無い．"]
    lines = ["| 構成 | $\\rho$ | サンプリング | シード | 外側ループ |", "|---|---:|---|---:|---:|"]
    lines += [f"| {a} | {b:g} | {c} | {d} | {e} |" for a, b, c, d, e in sorted(rows)]
    return lines


def completeness_lines(runs: list) -> list:
    """
    概要: 構成ごとの試行の数（期待値 7×5 + 3×5 = 50）と，$ f^{*} $ の最大値（シードによりデータの分割が異なる）を示す．
        この値は実行した PC によって最後の桁が異なる（本PC 0.552022386099835，別のPC 0.5520223860998349）ため，PC の区別に使う．
    引数: runs (list of dict)．
    戻り値: lines (list of str)．
    """
    lines = ["| 構成 | 新規 | 試行の数（random reshuffling / 一様） | 外側ループ100まで記録した数 | $f^*$ の最大値 |",
             "|---|:---:|---:|---:|---|"]
    for label, method, inner_start, is_new in CONFIGS:
        selected = [r for r in runs if r["method"] == method and r["inner_start"] == inner_start]
        n_rr = sum(r["sampling"] == "random_reshuffle" for r in selected)
        n_full = sum(len(r["log"]["epoch"]) == r["config"]["epochs"] + 1 for r in selected)
        f_star_max = repr(max(r["config"]["f_star"] for r in selected)) if selected else ""
        lines.append(f"| {label} | {'○' if is_new else ''} | {n_rr} / {len(selected) - n_rr} | {n_full} | "
                     f"{f_star_max} |")
    return lines


def execution_lines(runs: list, workers: int, run_logs: list) -> list:
    """
    概要: 新たに実行した試行の数，所要時間（`config.json` の更新時刻の範囲），並列数，PC の構成を示す．
    引数: runs (list of dict)．workers (int)．run_logs (list of str)．実行ログのパス．
    戻り値: lines (list of str)．
    """
    new_keys = {(method, inner_start) for _, method, inner_start, is_new in CONFIGS if is_new}
    new_runs = [r for r in runs if (r["method"], r["inner_start"]) in new_keys]
    mtimes = [os.path.getmtime(os.path.join(r["run_dir"], "config.json")) for r in new_runs]
    lines = [f"- 新たに実行した試行の数：{len(new_runs)}（期待値 250）．"]
    if mtimes:
        first = datetime.datetime.fromtimestamp(min(mtimes)).isoformat(timespec="seconds")
        last = datetime.datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds")
        lines.append(f"- 試行の終了時刻（`config.json` の更新時刻）：{first} 〜 {last}．")
        per_loop = [float(np.median(np.diff(r["log"]["elapsed_time"])[1:])) for r in new_runs
                    if len(r["log"]["elapsed_time"]) > 2]
        lines.append(f"- 1外側ループの時間（`elapsed_time` の増分の中央値，評価を除く）："
                     f"{np.median(per_loop):.2f} s（試行の間の最小 {min(per_loop):.2f} s，最大 {max(per_loop):.2f} s）．")
    lines.append(f"- 並列数：{workers}（`--workers`）．各プロセスは1スレッドである．")
    for path in run_logs:
        if os.path.exists(path):
            with open(path, errors="replace") as f:
                text = f.read()
            lines.append(f"- `{os.path.basename(path)}`：`[done]` {text.count('[done]')} 件，`[skip]` "
                         f"{text.count('[skip]')} 件，終了の表示 {'有り' if '全ての学習が終了しました' in text else '無し'}．")
    lines.append(f"- PC：{_cpu_name()}，メモリ {_mem_total_gb():.0f} GB，{platform.platform()}．GPU は使わない．")
    return lines


def _cpu_name() -> str:
    """
    概要: `/proc/cpuinfo` から CPU の名前と論理コア数を返す．
    引数: なし
    戻り値: name (str)．
    """
    try:
        with open("/proc/cpuinfo") as f:
            names = [line.split(":", 1)[1].strip() for line in f if line.startswith("model name")]
        return f"{names[0]}（{len(names)}論理コア）"
    except (OSError, IndexError):
        return platform.processor()


def _mem_total_gb() -> float:
    """
    概要: `/proc/meminfo` から総メモリ［GB］を返す．
    引数: なし
    戻り値: total (float)．取得できなければ NaN．
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return float("nan")


def write_summary(runs: list, workers: int, run_logs: list, path: str = SUMMARY_PATH) -> str:
    """
    概要: `e11_summary.md` を書く．
    引数: runs (list of dict)．workers (int)．run_logs (list of str)．path (str)．
    戻り値: path (str)．
    """
    parts = [
        "# E11 の集計結果（Mushroom，全てのスナップショットと内側ループの開始点の構成）", "",
        "実験：`outputs/ex006_e1_mushroom_rho_sampling/`．構成は `config.json` の `inner_start` で区別した"
        "（reset = `snapshot`，continue = `last_iterate`）．値は5シードの平均 ± 標準誤差（`ddof=1`）である．"
        "「新規」は E11 で実行した構成である．final の reset と continue は同じ計算のため1行とした．", "",
        "## 0. 試行の数", "", *completeness_lines(runs), "",
        "$f^*$ の値は実行した PC によって最後の桁が異なる（2.2e-16 以下．"
        "`consistency_final_this_pc.md`）．", "",
        "## 1. random reshuffling の全構成 × 全 $\\rho$", "",
        f"- $\\hat{{r}}$：シードごとに，$\\Delta_s > 10^{{-12}}$ が $s = 1$ から続く範囲で $\\log \\Delta_s$ を $s$ に最小二乗で"
        "あてはめた傾き $a_c$ から $\\exp(a_c)$ を求め，シードの間で平均した．$s_{\\mathrm{end}} < 3$ のシードは除いた"
        "（括弧は値のあるシードの数）．",
        f"- 到達：$f({{\\mathbf z}}_s) - f({{\\mathbf w}}^*) \\le 10^{{-12}}$ に初めてなる評価点の $\\#\\mathrm{{grad}}/N$．"
        "未到達のシードを除いて平均し，その数を添えた．"
        "閾値は $\\hat{r}$ の回帰範囲の閾値（$10^{-12}$）にそろえた（指示書6節の $10^{-10}$ から変更．ユーザーの指示）．",
        f"- $\\|{{\\mathbf e}}_s\\|^2$：最後の{TAIL_WINDOW}外側ループの中央値（シードごと）．全勾配の構成（SVRG）は0の水準である．",
        "- テスト精度と正解数：最後の外側ループの値．", "",
        *random_reshuffle_table(runs), "",
        "## 2. 一様サンプリングの誤差床（$\\rho = 0.5, 1.0, 298.4$）", "",
        f"最後の{TAIL_WINDOW}外側ループの中央値（シードごと）の平均 ± 標準誤差である（表III の誤差床と同じ定義）．", "",
        *uniform_floor_table(runs), "",
        "## 3. 打ち切り", "", *terminated_table(runs), "",
        "## 4. 実行の記録", "", *execution_lines(runs, workers, run_logs), "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(parts))
    return path


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", default=EXPERIMENT_DIR)
    parser.add_argument("--output", default=SUMMARY_PATH)
    parser.add_argument("--workers", type=int, default=32, help="本実行の並列数（記録のため）")
    parser.add_argument("--run-logs", nargs="*", default=[
        os.path.join(OUTPUTS_DIR, "_run_logs", "ex006_e11_continue.log"),
        os.path.join(OUTPUTS_DIR, "_run_logs", "ex006_e11_reset.log")])
    return parser.parse_args()


def main() -> None:
    """
    概要: 集計して `e11_summary.md` を書く．
    引数: なし
    戻り値: なし
    """
    args = parse_args()
    runs = load_runs(args.experiment_dir)
    print(write_summary(runs, args.workers, args.run_logs, args.output))


if __name__ == "__main__":
    main()
