"""
新しい実装（`inner_start="last_iterate"`）の結果が，既存の実装の結果と一致するかを確認するスクリプト．

実験計画2.2節に従い，`train_loss`（Mushroomでは `objective_gap`），`test_accuracy`（Mushroomでは
`accuracy`），`approx_error` の差が浮動小数点誤差の範囲内であることを確認する．新しい結果の
エポック数が既存より短い場合（動作確認のための短い実行）は，共通する先頭のエポックだけを比較する．

既存の結果は `programs_review` の外にあるため，パスはコマンドライン引数で与える（コードには
埋め込まない）．

実行例:
    python compare_with_existing_logs.py \\
        --new-root ../outputs/ex002_e0_cifar10_alexnet_layernorm \\
        --existing-root /path/to/outputs/ex0024_cifar10_alexnet_layernorm_longrun \\
        --hp-map inner_last_iterate_lr0.001_bs128_lambda0.0005_epochs48=lr0.001_bs128_normlayernorm_lambda0.0005_epochs48 \\
        --output ../outputs/ex002_e0_cifar10_alexnet_layernorm/consistency_with_existing.md
"""

import argparse
import json
import os

import numpy as np

COMPARED_KEYS = ("oracle_calls", "train_loss", "objective_gap", "test_accuracy", "accuracy", "approx_error",
                 "grad_norm_sq")


def load_log(path: str) -> dict:
    """
    概要: `log.json` を読み込む．
    引数: path (str)．`log.json` のパス．
    戻り値: history (dict)．キーが指標名，値がリストの辞書．
    """
    with open(path) as f:
        return json.load(f)


def compare_series(new_values, old_values) -> dict:
    """
    概要: 2つの数値列の共通する先頭部分について，完全一致か，最大の絶対誤差と相対誤差を計算する．
    引数:
        new_values (list of float)．新しい実装の値．
        old_values (list of float)．既存の実装の値．
    戻り値: result (dict)．`length`，`exact`（全て完全一致か），`max_abs`，`max_rel`．
    """
    length = min(len(new_values), len(old_values))
    new_arr = np.asarray(new_values[:length], dtype=np.float64)
    old_arr = np.asarray(old_values[:length], dtype=np.float64)
    both_nan = np.isnan(new_arr) & np.isnan(old_arr)
    diff = np.where(both_nan, 0.0, np.abs(new_arr - old_arr))
    scale = np.maximum(np.abs(old_arr), np.finfo(np.float64).tiny)
    rel = np.where(both_nan, 0.0, diff / scale)
    exact = bool(np.all((new_arr == old_arr) | both_nan))
    return {
        "length": length,
        "exact": exact,
        "max_abs": float(np.nanmax(diff)) if length else float("nan"),
        "max_rel": float(np.nanmax(rel)) if length else float("nan"),
    }


def compare_run(new_dir: str, old_dir: str) -> dict:
    """
    概要: 1つの試行（1シード）について，共通するキーを全て比較する．
    引数:
        new_dir (str)．新しい実装の結果のディレクトリ（`log.json` を含む）．
        old_dir (str)．既存の実装の結果のディレクトリ（`log.json` を含む）．
    戻り値: results (dict)．キーが指標名，値が `compare_series` の結果の辞書．
    """
    new_log = load_log(os.path.join(new_dir, "log.json"))
    old_log = load_log(os.path.join(old_dir, "log.json"))
    return {
        key: compare_series(new_log[key], old_log[key])
        for key in COMPARED_KEYS
        if key in new_log and key in old_log
    }


def iterate_pairs(new_root: str, existing_root: str, hp_map: dict):
    """
    概要: 新しい結果と既存の結果の対応する (手法, 条件, シード) のディレクトリの組を列挙する．
    引数:
        new_root (str)．新しい結果の実験ディレクトリ．
        existing_root (str)．既存の結果の実験ディレクトリ．
        hp_map (dict)．新しい条件名から既存の条件名への対応．
    戻り値: (method, new_hp, seed, new_dir, old_dir) の組を返すイテレータ．
    """
    for method in sorted(os.listdir(new_root)):
        for new_hp, old_hp in hp_map.items():
            new_hp_dir = os.path.join(new_root, method, new_hp)
            if not os.path.isdir(new_hp_dir):
                continue
            for seed in sorted(os.listdir(new_hp_dir)):
                new_dir = os.path.join(new_hp_dir, seed)
                old_dir = os.path.join(existing_root, method, old_hp, seed)
                if os.path.exists(os.path.join(new_dir, "log.json")) and os.path.exists(os.path.join(old_dir, "log.json")):
                    yield method, new_hp, seed, new_dir, old_dir


def format_markdown(rows: list) -> str:
    """
    概要: 比較結果をMarkdownの表にする．
    引数: rows (list of tuple)．(method, seed, key, result) の組のリスト．
    戻り値: text (str)．
    """
    lines = ["| 手法 | シード | 指標 | 比較した点数 | 完全一致 | 最大絶対誤差 | 最大相対誤差 |",
             "|---|---:|---|---:|---|---:|---:|"]
    for method, seed, key, result in rows:
        lines.append(f"| {method} | {seed} | {key} | {result['length']} | {'○' if result['exact'] else '×'} | "
                     f"{result['max_abs']:.3e} | {result['max_rel']:.3e} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--existing-root", required=True)
    parser.add_argument("--hp-map", nargs="+", required=True, help="新しい条件名=既存の条件名")
    parser.add_argument("--output", default=None, help="結果のMarkdownの保存先")
    return parser.parse_args()


def main() -> None:
    """
    概要: 全ての対応する試行を比較し，結果を表示・保存する．
    引数: なし
    戻り値: なし
    """
    args = parse_args()
    hp_map = dict(item.split("=", 1) for item in args.hp_map)
    rows = []
    for method, _, seed, new_dir, old_dir in iterate_pairs(args.new_root, args.existing_root, hp_map):
        for key, result in compare_run(new_dir, old_dir).items():
            rows.append((method, seed, key, result))
    text = format_markdown(rows)
    all_exact = all(result["exact"] for *_, result in rows)
    summary = f"\n比較した系列数: {len(rows)}，全て完全一致: {all_exact}\n"
    print(text + summary)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text + summary)


if __name__ == "__main__":
    main()
