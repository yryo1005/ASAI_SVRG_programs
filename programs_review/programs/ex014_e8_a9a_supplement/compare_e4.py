"""
E4 の新しいプログラムの結果が既存の a9a の実験（`ex000_a9a_least_squares`，`ex0001_a9a_logistic`）のログと
一致するかを確認するスクリプト（実験指示E4.md 3節）．

`programs_review/programs/compare_with_existing_logs.py` の `COMPARED_KEYS` には `objective_value`（MSE）と
`full_grad_computations` が含まれないため，既存のログの全てのキー（`elapsed_time` を除く）を比べる処理を
本ファイルに置く．`config.json` の定数（$ \\eta $，$ L $，$ \\lambda $，$ f({\\mathbf w}^{*}) $，データ数）も比べる．

判定の規則：
- 整数の量（`epoch`，`oracle_calls`，`full_grad_computations`，`accuracy`）は完全一致を要する．
- 実数の量は，既存の値の絶対値が `SMALL_VALUE`（$ 10^{-12} $）以上の点では相対誤差 $ \\le $ `REL_TOL`（$ 10^{-12} $）を，
  それ未満の点（倍精度の丸め誤差の水準まで小さくなった区間）では絶対誤差 $ \\le $ `ABS_TOL`（$ 10^{-15} $）を要する．
- NaN 同士は一致とみなす．

既存の結果は `programs_review` の外にあるため，パスはコマンドライン引数で与える．

実行例:
    python ex008_e4_a9a/compare_e4.py --new-root ../outputs/_verify/ex008 \\
        --existing mse=../../outputs/ex000_a9a_least_squares bce=../../outputs/ex0001_a9a_logistic \\
        --output ../outputs/ex008_e4_a9a/consistency_with_existing.md
"""

import argparse
import json
import os

import numpy as np

EXCLUDED_KEYS = ("elapsed_time",)
EXACT_KEYS = ("epoch", "oracle_calls", "full_grad_computations", "accuracy")
CONFIG_KEYS = ("eta", "L", "L_individual", "L_average", "reg_lambda", "mu", "rho", "f_star", "f_star_reference",
               "N_train", "N_test", "d", "K", "epochs")
REL_TOL = 1e-12
ABS_TOL = 1e-15
SMALL_VALUE = 1e-12


def strip_inner_prefix(condition: str) -> str:
    """
    概要: 新しい条件名から `inner_{none,snapshot,last_iterate}_` を除き，既存の条件名にする．
    引数: condition (str)．例：`inner_last_iterate_eta1.530030e-01_K29304_epochs100`．
    戻り値: old_condition (str)．例：`eta1.530030e-01_K29304_epochs100`．
    """
    for start in ("none", "snapshot", "last_iterate"):
        prefix = f"inner_{start}_"
        if condition.startswith(prefix):
            return condition[len(prefix):]
    raise ValueError(f"条件名が inner_ で始まらない: {condition}")


def load_json(path: str) -> dict:
    """
    概要: JSON ファイルを読み込む．
    引数: path (str)．
    戻り値: content (dict)．
    """
    with open(path) as f:
        return json.load(f)


def compare_series(key: str, new_values: list, old_values: list) -> dict:
    """
    概要: 2つの数値の列を比較し，完全一致か，最大の絶対誤差・相対誤差，丸め誤差の区間の最大の絶対誤差，
        判定（モジュールの説明の規則）を返す．長さが異なる場合は不一致とする．
    引数:
        key (str)．指標名．
        new_values (list of float)．新しいプログラムの値．
        old_values (list of float)．既存のログの値．
    戻り値: result (dict)．`length`，`same_length`，`exact`，`max_abs`，`max_rel`（有意な区間），
        `max_abs_small`（丸め誤差の区間），`n_small`，`passed`．
    """
    length = min(len(new_values), len(old_values))
    new_arr = np.asarray(new_values[:length], dtype=np.float64)
    old_arr = np.asarray(old_values[:length], dtype=np.float64)
    both_nan = np.isnan(new_arr) & np.isnan(old_arr)
    diff = np.where(both_nan, 0.0, np.abs(new_arr - old_arr))
    exact = bool(np.all((new_arr == old_arr) | both_nan))
    small = np.abs(old_arr) < SMALL_VALUE
    significant = ~small & ~both_nan
    rel = np.zeros_like(diff)
    rel[significant] = diff[significant] / np.abs(old_arr[significant])
    max_rel = float(np.nanmax(rel[significant])) if np.any(significant) else 0.0
    max_abs_small = float(np.nanmax(diff[small])) if np.any(small) else 0.0
    same_length = len(new_values) == len(old_values)
    if key in EXACT_KEYS:
        passed = exact and same_length
    else:
        passed = same_length and not np.any(np.isnan(diff)) and max_rel <= REL_TOL and max_abs_small <= ABS_TOL
    return {"length": length, "same_length": same_length, "exact": exact,
            "max_abs": float(np.nanmax(diff)) if length else float("nan"), "max_rel": max_rel,
            "max_abs_small": max_abs_small, "n_small": int(np.sum(small & ~both_nan)), "passed": bool(passed)}


def compare_configs(new_config: dict, old_config: dict) -> list:
    """
    概要: `config.json` の共通する定数を比較する．
    引数:
        new_config (dict)．新しいプログラムの `config.json`．
        old_config (dict)．既存の `config.json`．
    戻り値: rows (list of tuple)．(キー, 新しい値, 既存の値, 相対誤差) の組のリスト．
    """
    rows = []
    for key in CONFIG_KEYS:
        if key not in new_config or key not in old_config:
            continue
        new_value, old_value = new_config[key], old_config[key]
        if new_value is None or old_value is None:
            rel = 0.0 if new_value == old_value else float("nan")
        else:
            rel = abs(new_value - old_value) / max(abs(old_value), np.finfo(np.float64).tiny)
        rows.append((key, new_value, old_value, rel))
    return rows


def iterate_pairs(new_root: str, existing: dict):
    """
    概要: 新しい結果と既存の結果の対応する試行のディレクトリの組を列挙する．新しい条件名から
        `inner_{inner_start}_` を除いたものが既存の条件名である．
    引数:
        new_root (str)．新しい結果のルート（`{loss}/{手法}/{条件}/{seed}` を含む）．
        existing (dict)．損失から既存の実験ディレクトリへの対応．
    戻り値: (loss, method, condition, seed, new_dir, old_dir) の組を返すイテレータ．
    """
    for loss in sorted(existing):
        loss_dir = os.path.join(new_root, loss)
        if not os.path.isdir(loss_dir):
            continue
        for method in sorted(os.listdir(loss_dir)):
            for condition in sorted(os.listdir(os.path.join(loss_dir, method))):
                old_condition = strip_inner_prefix(condition)
                for seed in sorted(os.listdir(os.path.join(loss_dir, method, condition))):
                    new_dir = os.path.join(loss_dir, method, condition, seed)
                    old_dir = os.path.join(existing[loss], method, old_condition, seed)
                    if os.path.exists(os.path.join(new_dir, "log.json")) and \
                            os.path.exists(os.path.join(old_dir, "log.json")):
                        yield loss, method, condition, seed, new_dir, old_dir


def compare_all(new_root: str, existing: dict) -> list:
    """
    概要: 全ての対応する試行について，既存のログの全てのキー（`elapsed_time` を除く）と定数を比較する．
    引数:
        new_root (str)．新しい結果のルート．
        existing (dict)．損失から既存の実験ディレクトリへの対応．
    戻り値: results (list of dict)．各試行の `loss`，`method`，`condition`，`seed`，`old_dir`，`series`
        （キーから `compare_series` の結果への辞書），`config`（`compare_configs` の結果），`missing_keys`．
    """
    results = []
    for loss, method, condition, seed, new_dir, old_dir in iterate_pairs(new_root, existing):
        new_log, old_log = load_json(os.path.join(new_dir, "log.json")), load_json(os.path.join(old_dir, "log.json"))
        keys = [k for k in old_log if k not in EXCLUDED_KEYS]
        series = {k: compare_series(k, new_log[k], old_log[k]) for k in keys if k in new_log}
        config = compare_configs(load_json(os.path.join(new_dir, "config.json")),
                                 load_json(os.path.join(old_dir, "config.json")))
        results.append({"loss": loss, "method": method, "condition": condition, "seed": seed, "old_dir": old_dir,
                        "series": series, "config": config, "missing_keys": [k for k in keys if k not in new_log]})
    return results


def format_markdown(results: list) -> str:
    """
    概要: 比較の結果を Markdown にする．
    引数: results (list of dict)．`compare_all` の戻り値．
    戻り値: text (str)．
    """
    lines = ["# E4：既存のログとの一致の確認（シード0）\n",
             f"判定：整数の量（{', '.join(EXACT_KEYS)}）は完全一致．実数の量は，既存の値の絶対値が $ 10^{{-12}} $ 以上の点で"
             f"相対誤差 $ \\le {REL_TOL:g} $，それ未満の点（丸め誤差の区間）で絶対誤差 $ \\le {ABS_TOL:g} $．"
             "`elapsed_time` は比較しない．\n",
             "## ログ\n",
             "| 損失 | 手法 | 新しい条件 | シード | 指標 | 点数 | 完全一致 | 最大絶対誤差 | 最大相対誤差（有意な区間） | "
             "丸め誤差の区間の点数 | 同区間の最大絶対誤差 | 判定 |",
             "|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---|"]
    for r in results:
        for key, s in r["series"].items():
            lines.append(f"| {r['loss']} | {r['method']} | `{r['condition']}` | {r['seed']} | {key} | {s['length']} | "
                         f"{'○' if s['exact'] else '×'} | {s['max_abs']:.3e} | {s['max_rel']:.3e} | {s['n_small']} | "
                         f"{s['max_abs_small']:.3e} | {'合格' if s['passed'] else '**不合格**'} |")
    lines += ["\n## config.json の定数\n", "| 損失 | 手法 | キー | 新しい値 | 既存の値 | 相対誤差 |",
              "|---|---|---|---:|---:|---:|"]
    for r in results:
        for key, new_value, old_value, rel in r["config"]:
            lines.append(f"| {r['loss']} | {r['method']} | {key} | {new_value!r} | {old_value!r} | {rel:.3e} |")
    n_series = sum(len(r["series"]) for r in results)
    n_passed = sum(s["passed"] for r in results for s in r["series"].values())
    n_exact = sum(s["exact"] for r in results for s in r["series"].values())
    missing = [(r["method"], k) for r in results for k in r["missing_keys"]]
    lines.append(f"\n比較した試行数: {len(results)}，系列数: {n_series}，完全一致: {n_exact}，合格: {n_passed}，"
                 f"新しいログに無いキー: {missing if missing else 'なし'}\n")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--existing", nargs="+", required=True, help="損失=既存の実験ディレクトリ")
    parser.add_argument("--output", default=None, help="結果の Markdown の保存先")
    return parser.parse_args()


def main() -> None:
    """
    概要: 比較を実行し，結果を表示・保存する．
    引数: なし
    戻り値: なし
    """
    args = parse_args()
    existing = dict(item.split("=", 1) for item in args.existing)
    text = format_markdown(compare_all(args.new_root, existing))
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
