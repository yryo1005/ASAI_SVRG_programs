"""
E10 の一致の確認（実験指示E10.md 3.1節）を行い，結果を Markdown の表にして書き出すスクリプト．

(a) random reshuffling（`--samplings random_reshuffle`）の結果を，既存の結果の最初の点と項目ごとに比べる．
    `SGD`・`ASAI_SVRG` は E4（`ex008_e4_a9a`），`SVRG_random`・`NFG_SVRG_random` は E8（`ex014_e8_a9a_supplement`）と比べる．
    E4 の本実行は別のPCで行われたため，`--e4-this-pc-root` に本PCで E4 のプログラム（無変更）を実行した結果を
    与えると，`SGD`・`ASAI_SVRG` をそれとも比べる（判定にはこちらを用いる）．
(b) 一様サンプリング（`uniform_with_replacement`）の結果について，記録の点数，`config.json` の `sampling`，
    (a) の結果と比べて値が異なり始める外側ループを確かめる．`SGD`・`SVRG_random` は外側ループ1から異なる．
    `NFG_SVRG_random`・`ASAI_SVRG` は $ {\\mathbf g}_{0} = {\\mathbf 0} $ のため1回目の内側ループで点が動かず，
    外側ループ1は (a) と一致し，外側ループ2から異なる．

実行例（`programs_review/programs` から）:
    python ex015_e10_a9a_bce_uniform/compare_e10.py --rr-root ../outputs/_verify/ex015_rr \\
        --uniform-root ../outputs/_verify/ex015_uniform \\
        --e4-this-pc-root ../outputs/_verify/ex014_e4_ref_ex008_this_pc \\
        --output ../outputs/ex015_e10_a9a_bce_uniform/consistency.md
"""

import argparse
import glob
import json
import os

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), "outputs")
DEFAULT_E4_ROOT = os.path.join(_OUTPUTS_DIR, "ex008_e4_a9a")
DEFAULT_E8_ROOT = os.path.join(_OUTPUTS_DIR, "ex014_e8_a9a_supplement")

METHODS = ("SGD", "SVRG_random", "NFG_SVRG_random", "ASAI_SVRG")
REFERENCE_EXPERIMENT = {"SGD": "E4", "ASAI_SVRG": "E4", "SVRG_random": "E8", "NFG_SVRG_random": "E8"}
COMPARED_KEYS = ("oracle_calls", "objective_gap", "grad_norm_sq", "accuracy", "approx_error")
# g_0 = 0 のため1回目の内側ループで点が動かない手法（外側ループ2から値が異なる）．
ZERO_INITIAL_GRADIENT_METHODS = ("NFG_SVRG_random", "ASAI_SVRG")


def _as_float_array(values: list) -> np.ndarray:
    """
    概要: `None` を含みうるリストを，`None` を NaN とした float の配列に変換する．
    引数: values (list)．長さ T．
    戻り値: array (np.ndarray)．形状 (T,)．
    """
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def find_run_dir(root: str, method: str, seed: int, sampling: str = None) -> str:
    """
    概要: `{root}/bce/{method}/inner_*/{seed}/` のうち `log.json` を持つディレクトリを1つ返す．
        sampling を指定した場合は，名前に `_{sampling}_` を含むものに限る．
    引数:
        root (str)．結果のルート．
        method (str)．手法名．
        seed (int)．シード．
        sampling (str or None)．サンプリング方式．
    戻り値: run_dir (str)．
    例外: 該当が1つでない場合は RuntimeError．
    """
    candidates = sorted(glob.glob(os.path.join(root, "bce", method, "inner_*", str(seed), "log.json")))
    if sampling is not None:
        candidates = [c for c in candidates if f"_{sampling}_" in c]
    if len(candidates) != 1:
        raise RuntimeError(f"{root} の {method} シード{seed}（{sampling}）の結果が1つに定まらない: {candidates}")
    return os.path.dirname(candidates[0])


def load_run(run_dir: str) -> tuple:
    """
    概要: `log.json` と `config.json` を読み込む．
    引数: run_dir (str)．
    戻り値: (log, config) (tuple of dict)．
    """
    with open(os.path.join(run_dir, "log.json"), encoding="utf-8") as file:
        log = json.load(file)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as file:
        config = json.load(file)
    return log, config


def compare_series(new: list, reference: list) -> tuple:
    """
    概要: 新しい系列と既存の系列の先頭（新しい系列の長さ）を比べる．NaN 同士は一致とみなす．
    引数:
        new (list)．長さ T．
        reference (list)．長さ T 以上．
    戻り値:
        bitwise (bool)．全ての点がビット単位で一致するか．
        max_relative (float)．最大の相対差 $ |a - b| / |b| $（$ b = 0 $ の点は $ |a - b| $）．
    """
    a = _as_float_array(new)
    b = _as_float_array(reference[:len(new)])
    if a.shape != b.shape:
        return False, float("inf")
    both_nan = np.isnan(a) & np.isnan(b)
    bitwise = bool(np.all((a == b) | both_nan))
    diff = np.where(both_nan, 0.0, np.abs(a - b))
    scale = np.where(np.abs(b) > 0.0, np.abs(b), 1.0)
    return bitwise, float(np.nanmax(diff / scale)) if diff.size > 0 else 0.0


def compare_random_reshuffle(rr_root: str, e4_root: str, e8_root: str, seed: int = 0, methods=METHODS) -> list:
    """
    概要: (a) の比較を行う．
    引数:
        rr_root (str)．`--samplings random_reshuffle` の結果のルート．
        e4_root (str)，e8_root (str)．既存の結果のルート．
        seed (int)．シード．
        methods (iterable of str)．比べる手法．
    戻り値: rows (list of dict)．キーは `method`，`reference`，`points`，`key`，`bitwise`，`max_relative`．
    """
    rows = []
    for method in methods:
        new_log, _ = load_run(find_run_dir(rr_root, method, seed, "random_reshuffle"))
        reference_root = e4_root if REFERENCE_EXPERIMENT[method] == "E4" else e8_root
        reference_dir = find_run_dir(reference_root, method, seed)
        reference_log, _ = load_run(reference_dir)
        keys = [k for k in COMPARED_KEYS if not (method == "SGD" and k == "approx_error")]
        for key in keys:
            bitwise, max_relative = compare_series(new_log[key], reference_log[key])
            rows.append({"method": method, "reference": os.path.relpath(reference_dir, _OUTPUTS_DIR),
                         "points": len(new_log[key]), "key": key, "bitwise": bitwise, "max_relative": max_relative})
    return rows


def check_uniform(uniform_root: str, rr_root: str, epochs: int, seed: int = 0) -> list:
    """
    概要: (b) の確認を行う．
    引数:
        uniform_root (str)．`uniform_with_replacement` の結果のルート．
        rr_root (str)．(a) の結果のルート．
        epochs (int)．外側ループ数（確認の実行で指定した値）．
        seed (int)．シード．
    戻り値: rows (list of dict)．キーは `method`，`points`，`expected_points`，`sampling`，`first_diff_epoch`，
        `expected_first_diff_epoch`，`differ_after`（最初に異なった外側ループ以降が全て異なるか），
        `gap_uniform`，`gap_rr`．
    """
    rows = []
    for method in METHODS:
        log, config = load_run(find_run_dir(uniform_root, method, seed, "uniform_with_replacement"))
        rr_log, _ = load_run(find_run_dir(rr_root, method, seed, "random_reshuffle"))
        uniform_gap = _as_float_array(log["objective_gap"])
        rr_gap = _as_float_array(rr_log["objective_gap"])
        differs = uniform_gap != rr_gap[:len(uniform_gap)]
        first_diff = int(np.argmax(differs)) if np.any(differs) else None
        rows.append({
            "method": method,
            "points": len(log["epoch"]),
            "expected_points": epochs + 1,
            "sampling": config.get("sampling"),
            "first_diff_epoch": first_diff,
            "expected_first_diff_epoch": 2 if method in ZERO_INITIAL_GRADIENT_METHODS else 1,
            "differ_after": bool(first_diff is not None and np.all(differs[first_diff:])),
            "gap_uniform": uniform_gap.tolist(),
            "gap_rr": rr_gap[:len(uniform_gap)].tolist(),
        })
    return rows


def _rr_table(rows: list) -> list:
    """
    概要: (a) の比較の行を Markdown の表の行にする．
    引数: rows (list of dict)．`compare_random_reshuffle` の戻り値．
    戻り値: lines (list of str)．
    """
    lines = ["| 手法 | 比較先 | 点数 | 項目 | ビット単位で一致 | 最大の相対差 |", "|---|---|---:|---|---|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | `{row['reference']}` | {row['points']} | `{row['key']}` | "
                     f"{'はい' if row['bitwise'] else 'いいえ'} | {row['max_relative']:.1e} |")
    return lines


def build_markdown(rr_rows: list, uniform_rows: list, this_pc_rows: list = None) -> str:
    """
    概要: (a)・(b) の結果を Markdown の文章にする．
    引数:
        rr_rows (list of dict)．既存の結果（E4 の本実行，E8）との比較．
        uniform_rows (list of dict or None)．(b) の確認．
        this_pc_rows (list of dict or None)．本PCで実行した E4 のプログラムの結果との比較（SGD・ASAI_SVRG）．
    戻り値: text (str)．
    """
    lines = ["## (a) random reshuffling での E4・E8 の再現", "", "### (a-1) 既存の結果（E4 の本実行，E8）との比較", ""]
    lines += _rr_table(rr_rows)
    if this_pc_rows is None:
        judged = rr_rows
    else:
        # E4 の手法は本PCでの E4 のプログラムの結果で判定し，E8 の手法は E8 の結果（本PC）で判定する．
        judged = [r for r in rr_rows if REFERENCE_EXPERIMENT[r["method"]] == "E8"] + this_pc_rows
        lines += ["", "### (a-2) 本PCで E4 のプログラム（`ex008_e4_a9a/train.py`，無変更）を実行した結果との比較", ""]
        lines += _rr_table(this_pc_rows)
    all_bitwise = all(row["bitwise"] for row in judged)
    basis = "E8 の手法は (a-1)，E4 の手法は (a-2) で判定" if this_pc_rows is not None else "(a-1) で判定"
    lines += ["", f"**判定（{basis}）：{'合格（全ての項目がビット単位で一致）' if all_bitwise else '不合格（一致しない項目がある）'}．**",
              ""]
    if uniform_rows is not None:
        lines += [
            "## (b) 一様サンプリングの動作",
            "",
            "| 手法 | 記録の点数（期待値） | `config.json` の `sampling` | (a) と異なり始める外側ループ（期待値） | それ以降も全て異なる | 目的関数誤差（一様，外側ループ0〜） | 目的関数誤差（(a)，外側ループ0〜） |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in uniform_rows:
            gaps_u = "，".join(f"{v:.6e}" for v in row["gap_uniform"])
            gaps_r = "，".join(f"{v:.6e}" for v in row["gap_rr"])
            lines.append(f"| {row['method']} | {row['points']}（{row['expected_points']}） | `{row['sampling']}` | "
                         f"{row['first_diff_epoch']}（{row['expected_first_diff_epoch']}） | "
                         f"{'はい' if row['differ_after'] else 'いいえ'} | {gaps_u} | {gaps_r} |")
        ok = all(r["points"] == r["expected_points"] and r["sampling"] == "uniform_with_replacement"
                 and r["first_diff_epoch"] == r["expected_first_diff_epoch"] and r["differ_after"]
                 for r in uniform_rows)
        lines += ["", f"**判定：{'合格' if ok else '不合格'}．**", ""]
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: argv (list of str or None)．
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rr-root", required=True, help="(a) の結果のルート")
    parser.add_argument("--uniform-root", default=None, help="(b) の結果のルート（省略すると (a) のみ）")
    parser.add_argument("--e4-root", default=DEFAULT_E4_ROOT)
    parser.add_argument("--e8-root", default=DEFAULT_E8_ROOT)
    parser.add_argument("--e4-this-pc-root", default=None,
                        help="本PCで E4 のプログラムを実行した結果のルート（SGD・ASAI_SVRG の判定に用いる）")
    parser.add_argument("--epochs", type=int, default=3, help="確認の実行の外側ループ数")
    parser.add_argument("--output", required=True, help="書き出す Markdown のパス")
    parser.add_argument("--header", default="", help="表の前に書く文章（ファイルのパス）")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """
    概要: (a)・(b) の確認を行い，Markdown に書き出す．
    引数: argv (list of str or None)．
    戻り値: なし
    """
    args = parse_args(argv)
    rr_rows = compare_random_reshuffle(args.rr_root, args.e4_root, args.e8_root)
    uniform_rows = check_uniform(args.uniform_root, args.rr_root, args.epochs) if args.uniform_root else None
    this_pc_rows = None
    if args.e4_this_pc_root:
        e4_methods = [m for m in METHODS if REFERENCE_EXPERIMENT[m] == "E4"]
        this_pc_rows = compare_random_reshuffle(args.rr_root, args.e4_this_pc_root, args.e8_root, methods=e4_methods)
    header = ""
    if args.header:
        with open(args.header, encoding="utf-8") as file:
            header = file.read().rstrip("\n") + "\n\n"
    text = header + build_markdown(rr_rows, uniform_rows, this_pc_rows)
    with open(args.output, "w", encoding="utf-8") as file:
        file.write(text)
    print(text)


if __name__ == "__main__":
    main()
