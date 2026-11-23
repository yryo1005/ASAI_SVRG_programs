"""
E9（ImageWoof・バッチサイズ512 の ASA-SVRG（reset，`inner_start="snapshot"`），シード0〜4）の集計．

`outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/{seed}/log.json`
を読み，シードごとの表と，5シードの平均 ± 標準誤差（`ddof=1`）の表を `outputs/e9_summary.md` に書く．

定義（$ N = 9025 $，横軸 $ \\#\\mathrm{grad}/N = $ `oracle_calls / N`，予算 $ B = 1000 $）：

- 予算での値：$ \\#\\mathrm{grad}/N \\le B $ である最後の評価点の値．
- 予算の後半の揺らぎ：隣り合う評価点 $ (x_{j-1}, x_{j}) $ のうち，$ x_{j-1} \\ge B/2 $ かつ $ x_{j} \\le B $ である組についての，
  テスト精度の変化の絶対値 $ |a_{j} - a_{j-1}| $ の平均［ポイント］．
- 予算の後半の近似誤差の中央値：$ B/2 < x_{j} \\le B $ である評価点の `approx_error` の中央値．

実行例（リポジトリのルートから）:
    python programs_review/programs/analysis_e9.py
"""

import argparse
import json
import os

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "outputs")
DEFAULT_RUN_ROOT = os.path.join(OUTPUTS_DIR, "ex005_appb_imagewoof_resnet18_layernorm", "ASAI_SVRG",
                                "inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505")
N_TRAIN = 9025
BUDGET = 1000.0
EPOCHS = 505
SEEDS = (0, 1, 2, 3, 4)


def _as_float_array(values: list) -> np.ndarray:
    """
    概要: `None` を含みうるリストを，`None` を NaN とした float の配列に変換する．
    引数: values (list)．長さ T．
    戻り値: array (np.ndarray)．形状 (T,)，dtype float64．
    """
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def grad_per_n(log: dict, n_train: int = N_TRAIN) -> np.ndarray:
    """
    概要: 各評価点の $ \\#\\mathrm{grad}/N $ を返す．
    引数:
        log (dict)．`log.json` の内容（`oracle_calls` を含む）．
        n_train (int)．学習データ数 $ N $．
    戻り値: x (np.ndarray)．形状 (T,)．
    """
    return np.asarray(log["oracle_calls"], dtype=float) / n_train


def value_at_budget(x: np.ndarray, values: np.ndarray, budget: float = BUDGET) -> float:
    """
    概要: $ x \\le $ budget である最後の評価点の値を返す．
    引数:
        x (np.ndarray)．形状 (T,)．評価点の $ \\#\\mathrm{grad}/N $（単調増加）．
        values (np.ndarray)．形状 (T,)．評価点の値．
        budget (float)．予算．
    戻り値: value (float)．該当する評価点が無い場合は NaN．
    """
    indices = np.nonzero(x <= budget)[0]
    return float(values[indices[-1]]) if indices.size > 0 else float("nan")


def latter_half_fluctuation(x: np.ndarray, accuracy: np.ndarray, budget: float = BUDGET) -> float:
    """
    概要: 予算の後半の，隣り合う評価点の間のテスト精度の変化の絶対値の平均を返す．
        対象は $ x_{j-1} \\ge $ budget/2 かつ $ x_{j} \\le $ budget の組である．
    引数:
        x (np.ndarray)．形状 (T,)．
        accuracy (np.ndarray)．形状 (T,)．テスト精度（値の単位のまま）．
        budget (float)．予算．
    戻り値: fluctuation (float)．accuracy と同じ単位．対象の組が無い場合は NaN．
    """
    mask = (x[:-1] >= budget / 2) & (x[1:] <= budget)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(np.diff(accuracy))[mask]))


def latter_half_median(x: np.ndarray, values: np.ndarray, budget: float = BUDGET) -> float:
    """
    概要: $ $ budget/2 $ < x \\le $ budget である評価点の値の中央値を返す（NaN は除く）．
    引数:
        x (np.ndarray)．形状 (T,)．
        values (np.ndarray)．形状 (T,)．
        budget (float)．予算．
    戻り値: median (float)．該当する値が無い場合は NaN．
    """
    selected = values[(x > budget / 2) & (x <= budget)]
    selected = selected[np.isfinite(selected)]
    return float(np.median(selected)) if selected.size > 0 else float("nan")


def summarize_run(log: dict, budget: float = BUDGET, n_train: int = N_TRAIN) -> dict:
    """
    概要: 1つの試行のログから，シードごとの表の値を求める．
    引数:
        log (dict)．`log.json` の内容（`epoch`，`oracle_calls`，`train_loss`，`test_accuracy`，
            `approx_error`，`terminated`）．
        budget (float)．予算．
        n_train (int)．学習データ数．
    戻り値: row (dict)．キーは `recorded_epochs`（int），`terminated`（bool），`max_grad_per_n`（float），
        `test_accuracy_pct`（予算でのテスト精度［%］），`train_loss`（予算での学習損失），
        `fluctuation_pt`（予算の後半の揺らぎ［ポイント］），`approx_error_median`．
    """
    x = grad_per_n(log, n_train)
    accuracy = _as_float_array(log["test_accuracy"])
    terminated = log.get("terminated", [])
    return {
        "recorded_epochs": len(log["epoch"]) - 1,
        "terminated": bool(terminated[-1]) if len(terminated) > 0 else False,
        "max_grad_per_n": float(x[-1]),
        "test_accuracy_pct": 100.0 * value_at_budget(x, accuracy, budget),
        "train_loss": value_at_budget(x, _as_float_array(log["train_loss"]), budget),
        "fluctuation_pt": 100.0 * latter_half_fluctuation(x, accuracy, budget),
        "approx_error_median": latter_half_median(x, _as_float_array(log["approx_error"]), budget),
    }


def mean_and_standard_error(values: list) -> tuple:
    """
    概要: NaN を除いた値の平均と標準誤差（`ddof=1`）を返す．
    引数: values (list of float)．
    戻り値: (mean, standard_error) (tuple of float)．値が1つの場合，標準誤差は NaN．
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float("nan")
    return float(np.mean(array)), float(np.std(array, ddof=1) / np.sqrt(array.size))


def load_runs(run_root: str, seeds=SEEDS) -> dict:
    """
    概要: シードごとの `log.json` を読み込む．存在しないシードは含めない．
    引数:
        run_root (str)．`{seed}/log.json` を含むディレクトリ．
        seeds (iterable of int)．
    戻り値: logs (dict)．キーはシード（int），値は `log.json` の内容（dict）．
    """
    logs = {}
    for seed in seeds:
        path = os.path.join(run_root, str(seed), "log.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as file:
                logs[seed] = json.load(file)
    return logs


def _fmt(value: float, digits: int) -> str:
    """
    概要: 数値を固定小数点の文字列にする．NaN は「—」とする．
    引数: value (float)，digits (int)．小数点以下の桁数．
    戻り値: text (str)．
    """
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


def _fmt_mean_se(values: list, digits: int) -> str:
    """
    概要: 平均 ± 標準誤差の文字列を作る．
    引数: values (list of float)，digits (int)．
    戻り値: text (str)．
    """
    mean, standard_error = mean_and_standard_error(values)
    return f"{_fmt(mean, digits)} ± {_fmt(standard_error, digits)}"


def build_summary(rows: dict, run_root: str) -> str:
    """
    概要: シードごとの表と平均 ± 標準誤差の表を Markdown にする．
    引数:
        rows (dict)．キーはシード，値は `summarize_run` の戻り値．
        run_root (str)．結果のディレクトリ（表の説明に用いる）．
    戻り値: text (str)．
    """
    relative_root = os.path.relpath(run_root, OUTPUTS_DIR)
    lines = [
        "# E9 の集計結果（ImageWoof・バッチサイズ512 の ASA-SVRG（reset））",
        "",
        f"結果：`outputs/{relative_root}/{{seed}}/log.json`．横軸 $ \\#\\mathrm{{grad}}/N $ は `oracle_calls / {N_TRAIN}`，"
        f"予算は $ \\#\\mathrm{{grad}}/N \\le {BUDGET:.0f} $ である．予算での値は予算以下の最後の評価点の値である．"
        "揺らぎは，隣り合う評価点の両方が予算の後半（$ 500 \\le \\#\\mathrm{grad}/N $，かつ予算以下）にある組の，"
        "テスト精度の変化の絶対値の平均である．近似誤差の中央値は $ 500 < \\#\\mathrm{grad}/N \\le 1000 $ の評価点で求めた．",
        "",
        "## シードごとの値",
        "",
        "| シード | 記録エポック数 | 打ち切り | 最後の #grad/N | 予算でのテスト精度［%］ | 予算での学習損失 | 予算の後半の揺らぎ［%pt］ | 予算の後半の近似誤差の中央値 |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for seed, row in sorted(rows.items()):
        lines.append(f"| {seed} | {row['recorded_epochs']} | {'あり' if row['terminated'] else 'なし'} | "
                     f"{_fmt(row['max_grad_per_n'], 1)} | {_fmt(row['test_accuracy_pct'], 2)} | "
                     f"{_fmt(row['train_loss'], 4)} | {_fmt(row['fluctuation_pt'], 2)} | "
                     f"{_fmt(row['approx_error_median'], 4)} |")
    values = list(rows.values())
    lines += [
        "",
        "## 平均 ± 標準誤差（`ddof=1`）",
        "",
        "| 試行数 | 打ち切り | 予算でのテスト精度［%］ | 予算での学習損失 | 予算の後半の揺らぎ［%pt］ | 予算の後半の近似誤差の中央値 |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {len(values)} | {sum(row['terminated'] for row in values)} | "
        f"{_fmt_mean_se([row['test_accuracy_pct'] for row in values], 2)} | "
        f"{_fmt_mean_se([row['train_loss'] for row in values], 4)} | "
        f"{_fmt_mean_se([row['fluctuation_pt'] for row in values], 2)} | "
        f"{_fmt_mean_se([row['approx_error_median'] for row in values], 4)} |",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    """
    概要: コマンドライン引数を解析する．
    引数: argv (list of str or None)．
    戻り値: args (argparse.Namespace)．`run_root`，`output`．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="{seed}/log.json を含むディレクトリ")
    parser.add_argument("--output", default=os.path.join(OUTPUTS_DIR, "e9_summary.md"))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """
    概要: ログを読み込み，`e9_summary.md` を書く．
    引数: argv (list of str or None)．
    戻り値: なし
    """
    args = parse_args(argv)
    logs = load_runs(args.run_root)
    rows = {seed: summarize_run(log) for seed, log in logs.items()}
    text = build_summary(rows, args.run_root)
    with open(args.output, "w", encoding="utf-8") as file:
        file.write(text)
    print(args.output)


if __name__ == "__main__":
    main()
