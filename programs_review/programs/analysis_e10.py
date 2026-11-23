"""
E10（付録B.1，a9a の BCE の独立一様サンプリング）の集計．`outputs/e10_summary.md` を作る．

定義は E1 の表III（`analysis_e1.py` の `tail_median`）と同一である．

- 誤差床：シードごとに，有限の値だけを残した系列の最後の20点（打ち切りが無ければ外側ループ81〜100）の中央値．
  目的関数誤差 `objective_gap` は4手法，近似誤差 `approx_error` は `NFG_SVRG_random`・`ASAI_SVRG` について求める．
- $ \\sigma^{2}/K $：シードごとに，本実験と同じ分割の学習データで最適解 $ {\\mathbf w}^{*} $（切片を含む）を
  `compute_optimal_solution` で求め，$ \\sigma^{2} = \\frac{1}{N}\\sum_{n=1}^{N}\\|\\nabla f_{n}({\\mathbf w}^{*}) - \\nabla f({\\mathbf w}^{*})\\|^{2} $
  を計算して $ K = N $ で割る．$ \\nabla f_{n} $ は1サンプルの BCE の勾配（重みと切片）であり，正則化項の勾配は
  $ f_{n} $ と $ f $ で共通なので差では打ち消し合う．得られた $ f({\\mathbf w}^{*}) $ が `config.json` の `f_star` と
  一致すること（相対差 $ 10^{-10} $ 以下）も確かめる．
- 参考：random reshuffling の値は，E4（`ex008_e4_a9a`：SGD，ASA-SVRG）と E8（`ex014_e8_a9a_supplement`：
  SVRG・NFG-SVRG のランダム点）の同じ4手法の誤差床である．

実行例（`programs_review/programs` から）:
    python analysis_e10.py
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "outputs")
EXPERIMENT_DIR = os.path.join(_THIS_DIR, "ex015_e10_a9a_bce_uniform")

DEFAULT_E10_ROOT = os.path.join(OUTPUTS_DIR, "ex015_e10_a9a_bce_uniform")
DEFAULT_E4_ROOT = os.path.join(OUTPUTS_DIR, "ex008_e4_a9a")
DEFAULT_E8_ROOT = os.path.join(OUTPUTS_DIR, "ex014_e8_a9a_supplement")

METHODS = ("SGD", "SVRG_random", "NFG_SVRG_random", "ASAI_SVRG")
METHOD_LABELS = {"SGD": "SGD", "SVRG_random": "SVRG（ランダム点）", "NFG_SVRG_random": "NFG-SVRG（ランダム点）",
                 "ASAI_SVRG": "ASA-SVRG（reset）"}
APPROX_METHODS = ("NFG_SVRG_random", "ASAI_SVRG")
FULL_GRADIENT_METHODS = ("SVRG_random",)
# random reshuffling の参考値の取得元（E4 または E8）．
REFERENCE_EXPERIMENT = {"SGD": "E4", "ASAI_SVRG": "E4", "SVRG_random": "E8", "NFG_SVRG_random": "E8"}
SAMPLING = "uniform_with_replacement"
SEEDS = (0, 1, 2, 3, 4)
TAIL_WINDOW = 20
F_STAR_REL_TOL = 1e-10


# ---------------------------------------------------------------------------
# 統計量
# ---------------------------------------------------------------------------

def _as_float_array(values: list) -> np.ndarray:
    """
    概要: `None` を含みうるリストを，`None` を NaN とした float の配列に変換する．
    引数: values (list)．長さ T．
    戻り値: array (np.ndarray)．形状 (T,)．
    """
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


def tail_median(values: list, window: int = TAIL_WINDOW) -> float:
    """
    概要: 有限の値だけを残した系列の最後の `window` 点の中央値（誤差床）を返す（`analysis_e1.tail_median` と同一）．
    引数:
        values (list of float)．長さ T（外側ループ0〜）．
        window (int) = TAIL_WINDOW．
    戻り値: median (float)．有限の値が無い場合は NaN．
    """
    array = _as_float_array(values)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.median(array[-window:]))


def mean_and_standard_error(values: list) -> tuple:
    """
    概要: 有限の値の平均と標準誤差（`ddof=1`）を返す．
    引数: values (list of float)．
    戻り値: (mean, standard_error) (tuple of float)．値が1つなら標準誤差は NaN，値が無ければ両方 NaN．
    """
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float("nan")
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(array.size))


def gradient_variance(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    概要: BCE の1サンプルの勾配の分散 $ \\sigma^{2} = \\frac{1}{N}\\sum_{n}\\|\\nabla f_{n} - \\nabla f\\|^{2} $ を計算する．
        $ \\nabla f_{n}({\\mathbf w}, b) = (p_{n} - y_{n})[{\\mathbf x}_{n}; 1] + \\lambda[{\\mathbf w}; b] $，
        $ p_{n} = \\sigma({\\mathbf w}^{\\top}{\\mathbf x}_{n} + b) $ であり，正則化項は差で打ち消し合うため用いない．
    引数:
        X (np.ndarray)．形状 (N, d)．特徴量．
        y (np.ndarray)．形状 (N,)．0/1 の教師信号．
        w (np.ndarray)．形状 (d,)．重み．
        b (float)．切片．
    戻り値: sigma_sq (float)．
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    residual = 1.0 / (1.0 + np.exp(-(X @ w + b))) - y
    per_sample = np.hstack([residual[:, None] * X, residual[:, None]])
    centered = per_sample - per_sample.mean(axis=0)
    return float(np.mean(np.sum(centered ** 2, axis=1)))


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------

def find_run_dir(root: str, method: str, seed: int, sampling: str = None) -> str:
    """
    概要: `{root}/bce/{method}/inner_*/{seed}/` のうち `log.json` を持ち，外側ループ数が100の
        ディレクトリを1つ返す．sampling を指定した場合は，名前に `_{sampling}_` を含むものに限る．
    引数: root (str)，method (str)，seed (int)，sampling (str or None)．
    戻り値: run_dir (str or None)．見つからない場合は None．
    例外: 2つ以上見つかった場合は RuntimeError．
    """
    pattern = os.path.join(root, "bce", method, "inner_*_epochs100", str(seed), "log.json")
    candidates = sorted(glob.glob(pattern))
    if sampling is not None:
        candidates = [c for c in candidates if f"_{sampling}_" in c]
    if len(candidates) > 1:
        raise RuntimeError(f"{root} の {method} シード{seed} の結果が複数ある: {candidates}")
    return os.path.dirname(candidates[0]) if candidates else None


def load_run(run_dir: str) -> dict:
    """
    概要: `log.json` と `config.json` を読み込む．
    引数: run_dir (str)．
    戻り値: run (dict)．キーは `log`，`config`．
    """
    with open(os.path.join(run_dir, "log.json"), encoding="utf-8") as file:
        log = json.load(file)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as file:
        config = json.load(file)
    return {"log": log, "config": config}


def load_condition(root: str, method: str, sampling: str = None, seeds=SEEDS) -> dict:
    """
    概要: 1つの条件のシードごとの結果を読み込む．
    引数: root (str)，method (str)，sampling (str or None)，seeds (iterable of int)．
    戻り値: runs (dict)．キーはシード，値は `load_run` の戻り値．
    """
    runs = {}
    for seed in seeds:
        run_dir = find_run_dir(root, method, seed, sampling)
        if run_dir is not None:
            runs[seed] = load_run(run_dir)
    return runs


def summarize_run(run: dict, method: str) -> dict:
    """
    概要: 1つの試行の誤差床，打ち切りの有無，最終外側ループのテスト精度を求める．
    引数: run (dict)．`load_run` の戻り値．method (str)．
    戻り値: row (dict)．キーは `objective_floor`，`approx_floor`（近似誤差を持たない手法は NaN），
        `terminated`，`final_accuracy`，`recorded_epochs`．
    """
    log = run["log"]
    terminated = log.get("terminated", [])
    return {
        "objective_floor": tail_median(log["objective_gap"]),
        "approx_floor": tail_median(log["approx_error"]) if method in APPROX_METHODS else float("nan"),
        "terminated": bool(terminated[-1]) if terminated else False,
        "final_accuracy": float(log["accuracy"][-1]),
        "recorded_epochs": len(log["epoch"]) - 1,
    }


# ---------------------------------------------------------------------------
# σ²/K
# ---------------------------------------------------------------------------

def compute_sigma_sq_over_k(seed: int, reg_lambda: float, f_star_config: float = None) -> dict:
    """
    概要: シードの学習データで $ {\\mathbf w}^{*} $ を求め，$ \\sigma^{2} $，$ \\sigma^{2}/K $ と
        $ f({\\mathbf w}^{*}) $ の `config.json` の値との相対差を返す．
    引数:
        seed (int)．データの分割のシード．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
        f_star_config (float or None)．`config.json` の `f_star`．
    戻り値: result (dict)．キーは `sigma_sq`，`K`，`sigma_sq_over_k`，`f_star`，`f_star_rel_diff`．
    """
    if EXPERIMENT_DIR not in sys.path:
        sys.path.insert(0, EXPERIMENT_DIR)
    from data import load_dataloader  # noqa: E402
    from train import BATCH_SIZE, compute_optimal_solution  # noqa: E402

    train_dataloader, _ = load_dataloader(seed=seed, batch_size=BATCH_SIZE)
    X_train, y_train = (t.numpy().astype(np.float64) for t in train_dataloader.dataset.tensors)
    y_train = y_train.reshape(-1)
    x_star, f_star = compute_optimal_solution(X_train, y_train, "bce", reg_lambda)
    sigma_sq = gradient_variance(X_train, y_train, x_star[:-1], float(x_star[-1]))
    K = X_train.shape[0]
    rel_diff = float("nan") if f_star_config is None else abs(f_star - f_star_config) / abs(f_star_config)
    return {"sigma_sq": sigma_sq, "K": K, "sigma_sq_over_k": sigma_sq / K, "f_star": f_star,
            "f_star_rel_diff": rel_diff}


# ---------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------

def _fmt_sci(value: float) -> str:
    """
    概要: 数値を指数表記にする．NaN は「—」．
    引数: value (float)．
    戻り値: text (str)．
    """
    return "—" if value is None or not np.isfinite(value) else f"{value:.3e}"


def _fmt_mean_se(values: list) -> str:
    """
    概要: 平均 ± 標準誤差を指数表記の文字列にする．
    引数: values (list of float)．
    戻り値: text (str)．
    """
    mean, standard_error = mean_and_standard_error(values)
    if not np.isfinite(mean):
        return "—"
    return f"{mean:.3e} ± {standard_error:.1e}" if np.isfinite(standard_error) else f"{mean:.3e}"


def floor_table(uniform_rows: dict, rr_rows: dict) -> list:
    """
    概要: 表1（一様サンプリングと random reshuffling の誤差床，平均 ± 標準誤差）の行を作る．
    引数:
        uniform_rows (dict)．キーは手法，値はシードごとの `summarize_run` の戻り値の dict．
        rr_rows (dict)．同じ形式の random reshuffling の値．
    戻り値: lines (list of str)．
    """
    lines = ["| 手法 | 指標 | 一様サンプリング（E10） | 試行数 | 参考：random reshuffling（E4・E8） | 試行数 |",
             "|---|---|---:|---:|---:|---:|"]
    for method in METHODS:
        for key, label in (("objective_floor", "目的関数誤差"), ("approx_floor", "近似誤差")):
            if key == "approx_floor" and method not in APPROX_METHODS:
                note = "（0．全勾配）" if method in FULL_GRADIENT_METHODS else "（定義されない）"
                lines.append(f"| {METHOD_LABELS[method]} | {label} | {note} | — | {note} | — |")
                continue
            u = [row[key] for row in uniform_rows.get(method, {}).values()]
            r = [row[key] for row in rr_rows.get(method, {}).values()]
            lines.append(f"| {METHOD_LABELS[method]} | {label} | {_fmt_mean_se(u)} | {len(u)} | "
                         f"{_fmt_mean_se(r)}（{REFERENCE_EXPERIMENT[method]}） | {len(r)} |")
    return lines


def sigma_table(sigma_rows: dict, uniform_rows: dict) -> list:
    """
    概要: 表2（シードごとの $ \\sigma^{2}/K $ と近似誤差の床，その比）の行を作る．
    引数:
        sigma_rows (dict)．キーはシード，値は `compute_sigma_sq_over_k` の戻り値．
        uniform_rows (dict)．`floor_table` と同じ．
    戻り値: lines (list of str)．
    """
    lines = ["| シード | $ \\sigma^{2} $ | $ \\sigma^{2}/K $ | $ f({\\mathbf w}^{*}) $ と `f_star` の相対差 | "
             "NFG-SVRG の近似誤差の床 | ASA-SVRG の近似誤差の床 | NFG-SVRG の床 ÷ $ \\sigma^{2}/K $ | "
             "ASA-SVRG の床 ÷ $ \\sigma^{2}/K $ |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    columns = {"sigma_sq": [], "sigma_sq_over_k": [], "nfg": [], "asa": [], "nfg_ratio": [], "asa_ratio": []}
    for seed, sigma in sorted(sigma_rows.items()):
        nfg = uniform_rows.get("NFG_SVRG_random", {}).get(seed, {}).get("approx_floor", float("nan"))
        asa = uniform_rows.get("ASAI_SVRG", {}).get(seed, {}).get("approx_floor", float("nan"))
        values = {"sigma_sq": sigma["sigma_sq"], "sigma_sq_over_k": sigma["sigma_sq_over_k"], "nfg": nfg, "asa": asa,
                  "nfg_ratio": nfg / sigma["sigma_sq_over_k"], "asa_ratio": asa / sigma["sigma_sq_over_k"]}
        for key, value in values.items():
            columns[key].append(value)
        lines.append(f"| {seed} | {_fmt_sci(values['sigma_sq'])} | {_fmt_sci(values['sigma_sq_over_k'])} | "
                     f"{_fmt_sci(sigma['f_star_rel_diff'])} | {_fmt_sci(nfg)} | {_fmt_sci(asa)} | "
                     f"{values['nfg_ratio']:.3f} | {values['asa_ratio']:.3f} |")
    lines.append(f"| 平均 ± 標準誤差 | {_fmt_mean_se(columns['sigma_sq'])} | {_fmt_mean_se(columns['sigma_sq_over_k'])} | — | "
                 f"{_fmt_mean_se(columns['nfg'])} | {_fmt_mean_se(columns['asa'])} | "
                 f"{_fmt_ratio(columns['nfg_ratio'])} | {_fmt_ratio(columns['asa_ratio'])} |")
    return lines


def _fmt_ratio(values: list) -> str:
    """
    概要: 比の平均 ± 標準誤差を固定小数点の文字列にする．
    引数: values (list of float)．
    戻り値: text (str)．
    """
    mean, standard_error = mean_and_standard_error(values)
    if not np.isfinite(mean):
        return "—"
    return f"{mean:.3f} ± {standard_error:.3f}" if np.isfinite(standard_error) else f"{mean:.3f}"


def per_seed_table(uniform_rows: dict) -> list:
    """
    概要: 表3（シードごとの4手法の床，打ち切りの有無，最終外側ループのテスト精度）の行を作る．
    引数: uniform_rows (dict)．`floor_table` と同じ．
    戻り値: lines (list of str)．
    """
    lines = ["| 手法 | シード | 記録の外側ループ数 | 目的関数誤差の床 | 近似誤差の床 | 打ち切り | 最終外側ループのテスト精度 |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for method in METHODS:
        for seed, row in sorted(uniform_rows.get(method, {}).items()):
            lines.append(f"| {METHOD_LABELS[method]} | {seed} | {row['recorded_epochs']} | "
                         f"{_fmt_sci(row['objective_floor'])} | {_fmt_sci(row['approx_floor'])} | "
                         f"{'あり' if row['terminated'] else 'なし'} | {row['final_accuracy']:.4f} |")
    return lines


def build_summary(uniform_rows: dict, rr_rows: dict, sigma_rows: dict, uniform_runs: dict) -> str:
    """
    概要: 表1〜3と説明を Markdown にまとめる．
    引数:
        uniform_rows (dict)，rr_rows (dict)．`floor_table` と同じ．
        sigma_rows (dict)．`sigma_table` と同じ．
        uniform_runs (dict)．キーは手法，値は `load_condition` の戻り値（条件の説明に用いる）．
    戻り値: text (str)．
    """
    config = next(iter(next(iter(uniform_runs.values())).values()))["config"]
    lines = [
        "# E10 の集計結果（a9a の BCE，独立一様サンプリング）",
        "",
        f"条件：$ \\eta = {config['eta']:.6e} $（$ 0.99/(3L) $，$ \\rho = {config['rho']:.2f} $），$ \\lambda = {config['reg_lambda']:.6g} $，"
        f"$ K = N = {config['K']} $，ミニバッチサイズ1，外側ループ数100，シード0〜4．サンプリングは `uniform_with_replacement`．",
        "誤差床は，有限の値だけを残した系列の最後の20点（打ち切りが無ければ外側ループ81〜100）の中央値である（E1 の表IIIと同じ定義）．"
        "値は5シードの平均 ± 標準誤差（`ddof=1`）である．",
        "",
        "## 表1：誤差床（一様サンプリングと random reshuffling）",
        "",
        "random reshuffling の値は，SGD・ASA-SVRG が E4（`ex008_e4_a9a`），SVRG・NFG-SVRG のランダム点が E8（`ex014_e8_a9a_supplement`）の結果である．",
        "",
    ]
    lines += floor_table(uniform_rows, rr_rows)
    lines += ["", "## 表2：$ \\sigma^{2}/K $ と近似誤差の床", "",
              "$ \\sigma^{2} = \\frac{1}{N}\\sum_{n}\\|\\nabla f_{n}({\\mathbf w}^{*}) - \\nabla f({\\mathbf w}^{*})\\|^{2} $（重みと切片），"
              "$ K = N $．$ {\\mathbf w}^{*} $ はシードごとの学習データで L-BFGS-B により求めた．", ""]
    lines += sigma_table(sigma_rows, uniform_rows)
    lines += ["", "## 表3：シードごとの値（一様サンプリング）", ""]
    lines += per_seed_table(uniform_rows)
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: argv (list of str or None)．
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e10-root", default=DEFAULT_E10_ROOT)
    parser.add_argument("--e4-root", default=DEFAULT_E4_ROOT)
    parser.add_argument("--e8-root", default=DEFAULT_E8_ROOT)
    parser.add_argument("--output", default=os.path.join(OUTPUTS_DIR, "e10_summary.md"))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """
    概要: 結果を読み込み，誤差床と $ \\sigma^{2}/K $ を求めて `e10_summary.md` に書く．
    引数: argv (list of str or None)．
    戻り値: なし
    """
    args = parse_args(argv)
    uniform_runs = {m: load_condition(args.e10_root, m, SAMPLING) for m in METHODS}
    rr_runs = {m: load_condition(args.e4_root if REFERENCE_EXPERIMENT[m] == "E4" else args.e8_root, m)
               for m in METHODS}
    uniform_rows = {m: {s: summarize_run(r, m) for s, r in runs.items()} for m, runs in uniform_runs.items()}
    rr_rows = {m: {s: summarize_run(r, m) for s, r in runs.items()} for m, runs in rr_runs.items()}

    sigma_rows = {}
    for seed in SEEDS:
        run = next((uniform_runs[m][seed] for m in METHODS if seed in uniform_runs[m]), None)
        if run is None:
            continue
        sigma_rows[seed] = compute_sigma_sq_over_k(seed, run["config"]["reg_lambda"], run["config"]["f_star"])
        if not sigma_rows[seed]["f_star_rel_diff"] <= F_STAR_REL_TOL:
            print(f"[warning] シード{seed}の f(w*) と config.json の f_star の相対差が "
                  f"{sigma_rows[seed]['f_star_rel_diff']:.1e} である．", flush=True)

    text = build_summary(uniform_rows, rr_rows, sigma_rows, uniform_runs)
    with open(args.output, "w", encoding="utf-8") as file:
        file.write(text)
    print(args.output)


if __name__ == "__main__":
    main()
