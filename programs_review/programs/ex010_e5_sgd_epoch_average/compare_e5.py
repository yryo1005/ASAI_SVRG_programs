"""
E5 の最終点の記録が，既存の SGD のログと一致するかを確認するスクリプト（実験計画8.4節）．

| データセット・バッチサイズ | 比較対象 | 期待 |
|---|---|---|
| ImageWoof・128 | E3 の SGD（`outputs/ex009_e3_imagewoof_walltime/SGD/inner_none_lr0.001_bs128_lambda0.0005_epochs200`） | 完全一致（同じ実装） |
| CIFAR-10・128，512 | 既存ログ ex0024（`reference/` に複写したもの） | 完全一致（E0 で確認済みの同等性） |
| ImageWoof・512 | 既存ログ ex0051（`reference/` に複写したもの） | 参考（ImageWoof の既存ログは前処理の実装が異なり，E0 でも完全には一致しない） |

共通する先頭の評価点（`--max-points` で上限を指定できる）について，`oracle_calls`，`train_loss`，`test_accuracy`，
`test_correct`（両方にある場合）を比べる．

実行例:
    python compare_e5.py --new-root ../../outputs/_verify/ex010 --max-points 6 \\
        --output ../../outputs/ex010_e5_sgd_epoch_average/consistency_check.md
    python compare_e5.py --output ../../outputs/ex010_e5_sgd_epoch_average/consistency_full_runs.md
"""

import argparse
import glob
import json
import os

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUTS = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), "outputs")
E5_ROOT = os.path.join(_OUTPUTS, "ex010_e5_sgd_epoch_average")
REFERENCE_ROOT = os.path.join(E5_ROOT, "reference")

# (データセット, バッチサイズ) -> (比較対象の条件のディレクトリ, 完全一致を期待するか)
REFERENCES = {
    ("imagewoof", 128): (os.path.join(_OUTPUTS, "ex009_e3_imagewoof_walltime", "SGD",
                                      "inner_none_lr0.001_bs128_lambda0.0005_epochs200"), True),
    ("cifar10", 128): (os.path.join(REFERENCE_ROOT, "ex0024_cifar10_alexnet_layernorm_longrun", "SGD",
                                    "lr0.001_bs128_normlayernorm_lambda0.0005_epochs96"), True),
    ("cifar10", 512): (os.path.join(REFERENCE_ROOT, "ex0024_cifar10_alexnet_layernorm_longrun", "SGD",
                                    "lr0.001_bs512_normlayernorm_lambda0.0005_epochs384"), True),
    ("imagewoof", 512): (os.path.join(REFERENCE_ROOT, "ex0051_imagewoof_resnet18_bs128_longrun", "SGD",
                                      "lr0.001_bs512_lambda0.0005_epochs1010"), False),
}
COMPARED_KEYS = ("oracle_calls", "train_loss", "test_accuracy", "test_correct")


def compare_series(new_values, ref_values) -> dict:
    """
    概要: 2つの数値列（同じ長さ）が完全一致するか，最大の絶対誤差と相対誤差を計算する．
    引数: new_values (list of float)．ref_values (list of float)．
    戻り値: result (dict)．`exact`，`max_abs`，`max_rel`．
    """
    new_arr = np.asarray(new_values, dtype=np.float64)
    ref_arr = np.asarray(ref_values, dtype=np.float64)
    diff = np.abs(new_arr - ref_arr)
    rel = diff / np.maximum(np.abs(ref_arr), np.finfo(np.float64).tiny)
    return {"exact": bool(np.array_equal(new_arr, ref_arr)),
            "max_abs": float(diff.max()) if diff.size else float("nan"),
            "max_rel": float(rel.max()) if rel.size else float("nan")}


def compare_logs(new_log: dict, ref_log: dict, max_points: int = None) -> dict:
    """
    概要: 1シードのログを，共通する先頭の評価点について項目ごとに比べる．
    引数: new_log (dict)．ref_log (dict)．max_points (int) = None．比べる評価点の数の上限．
    戻り値: results (dict)．{"points": int, 項目名: `compare_series` の結果}．
    """
    points = min(len(new_log["epoch"]), len(ref_log["epoch"]))
    if max_points is not None:
        points = min(points, max_points)
    results = {"points": points}
    for key in COMPARED_KEYS:
        if key in new_log and key in ref_log:
            results[key] = compare_series(new_log[key][:points], ref_log[key][:points])
    return results


def _load(path: str) -> dict:
    """
    概要: `log.json` を読み込む．
    引数: path (str)．
    戻り値: log (dict)．
    """
    with open(path) as f:
        return json.load(f)


def collect_rows(new_root: str, max_points: int = None) -> list:
    """
    概要: `new_root` の下にある E5 の全試行を，対応する比較対象と比べる．
    引数: new_root (str)．E5 の出力のルート（`{dataset}/SGD/{条件}/{seed}/log.json` を含む）．max_points (int) = None．
    戻り値: rows (list of dict)．`dataset`，`batch_size`，`seed`，`expect_exact`，`results`．
    """
    rows = []
    for log_path in sorted(glob.glob(os.path.join(new_root, "*", "SGD", "*", "*", "log.json"))):
        seed_dir = os.path.dirname(log_path)
        dataset = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(seed_dir))))
        condition = os.path.basename(os.path.dirname(seed_dir))
        batch_size = int(condition.split("_bs")[1].split("_")[0])
        if (dataset, batch_size) not in REFERENCES:
            continue
        ref_dir, expect_exact = REFERENCES[(dataset, batch_size)]
        ref_path = os.path.join(ref_dir, os.path.basename(seed_dir), "log.json")
        if not os.path.exists(ref_path):
            continue
        rows.append({"dataset": dataset, "batch_size": batch_size, "seed": int(os.path.basename(seed_dir)),
                     "expect_exact": expect_exact,
                     "results": compare_logs(_load(log_path), _load(ref_path), max_points)})
    return rows


def _cell(result: dict) -> str:
    """
    概要: 1項目の比較結果を表のセルの文字列にする．
    引数: result (dict)．`compare_series` の結果．存在しない場合は None．
    戻り値: text (str)．
    """
    if result is None:
        return "—"
    return "完全一致" if result["exact"] else f"相対 {result['max_rel']:.2e}"


def to_markdown(rows: list) -> str:
    """
    概要: 比較結果を Markdown の表にする．
    引数: rows (list of dict)．`collect_rows` の戻り値．
    戻り値: text (str)．
    """
    lines = ["| データセット | バッチサイズ | シード | 比較した点数 | 期待 | " + " | ".join(COMPARED_KEYS) + " | 判定 |",
             "|---|---:|---:|---:|---|" + "---|" * len(COMPARED_KEYS) + "---|"]
    for row in rows:
        res = row["results"]
        all_exact = all(res[k]["exact"] for k in COMPARED_KEYS if k in res)
        verdict = "一致" if all_exact else ("不一致" if row["expect_exact"] else "参考（差あり）")
        cells = " | ".join(_cell(res.get(k)) for k in COMPARED_KEYS)
        lines.append(f"| {row['dataset']} | {row['batch_size']} | {row['seed']} | {res['points']} | "
                     f"{'完全一致' if row['expect_exact'] else '参考'} | {cells} | {verdict} |")
    return "\n".join(lines)


def main() -> None:
    """
    概要: 比較結果を表示し，Markdown に保存する．
    引数: なし（コマンドライン引数を用いる）
    戻り値: なし
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new-root", default=E5_ROOT)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    text = to_markdown(collect_rows(os.path.abspath(args.new_root), args.max_points))
    header = (f"# E5 の最終点と既存の SGD のログの比較\n\n比較した E5 の結果：`{os.path.relpath(os.path.abspath(args.new_root), _OUTPUTS)}`"
              f"（`outputs/` からの相対パス）．\n\n")
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(header + text + "\n")
    print(text)


if __name__ == "__main__":
    main()
