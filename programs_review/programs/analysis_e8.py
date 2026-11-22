"""
E8（付録B.1，a9a の追加実験．`ex014_e8_a9a_supplement`）の結果を集計し，表と確認用の図を作成するモジュール．

`実験指示E8.md` 6節の出力を作成する．値は5シードの平均 ± 標準誤差（ddof=1）である．
- BCE の表：SGD，SVRG（ランダム点，reset），NFG-SVRG（ランダム点，reset），ASA-SVRG（reset）の4行と，参考の
  SVRG・NFG-SVRG（最終点）．SGD・ASA-SVRG・最終点は E4（`ex008_e4_a9a`）の結果である．
  項目は最終（100外側ループ目）の目的関数の誤差，テスト精度，近似誤差の後半（51〜100外側ループ）の中央値．
- MSE の表：SGD，SVRG（最終点），NFG-SVRG（最終点），ASA-SVRG（reset），ASA-SVRG（continue）の5行．
  continue 以外は E4 の結果である．項目は最終の $ \\|\\nabla f({\\mathbf z}_{s})\\|^{2} $，目的関数の値，テスト精度，
  近似誤差の後半の中央値．
- 確認用の図（`outputs/ex014_e8_a9a_supplement/figures/`）：横軸 $ \\#\\mathrm{grad}/N $（`full_grad_computations`），
  全区間（100外側ループ）．BCE：(A) 目的関数の誤差，(B) テスト精度，(C) 近似誤差．MSE：(A) $ \\|\\nabla f\\|^{2} $，
  (B) テスト精度，(C) 近似誤差．対数軸の量の平均は `analysis_e4.py` と同じく幾何平均である．

読み込み・統計量・曲線の描画は `analysis_e4.py` の関数を用いる．

使い方:
    python analysis_e8.py
    python analysis_e8.py --e4-root ../outputs/ex008_e4_a9a --e8-root ../outputs/ex014_e8_a9a_supplement
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import analysis_e4 as e4  # noqa: E402

OUTPUTS_DIR = e4.OUTPUTS_DIR
E4_ROOT = os.path.join(OUTPUTS_DIR, "ex008_e4_a9a")
E8_ROOT = os.path.join(OUTPUTS_DIR, "ex014_e8_a9a_supplement")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "e8_summary.md")
LATTER_HALF = (51, 100)  # 近似誤差の中央値を求める外側ループの区間（両端を含む）

# 行の定義：(キー, 表示名, 結果のルート（"e4" または "e8"）, 手法, 内側ループの開始点, 参考の行か)
BCE_ROWS = [
    ("SGD", "SGD", "e4", "SGD", "none", False),
    ("SVRG_random", "SVRG（ランダム点，reset）", "e8", "SVRG_random", "snapshot", False),
    ("NFG_SVRG_random", "NFG-SVRG（ランダム点，reset）", "e8", "NFG_SVRG_random", "snapshot", False),
    ("ASAI_SVRG", "ASA-SVRG（reset）", "e4", "ASAI_SVRG", "snapshot", False),
    ("SVRG_final", "参考：SVRG（最終点）", "e4", "SVRG", "snapshot", True),
    ("NFG_SVRG_final", "参考：NFG-SVRG（最終点）", "e4", "NFG_SVRG", "snapshot", True),
]
MSE_ROWS = [
    ("SGD", "SGD", "e4", "SGD", "none", False),
    ("SVRG_final", "SVRG（最終点）", "e4", "SVRG", "snapshot", False),
    ("NFG_SVRG_final", "NFG-SVRG（最終点）", "e4", "NFG_SVRG", "snapshot", False),
    ("ASAI_SVRG", "ASA-SVRG（reset）", "e4", "ASAI_SVRG", "snapshot", False),
    ("ASAI_SVRG_continue", "ASA-SVRG（continue）", "e8", "ASAI_SVRG", "last_iterate", False),
]
ROWS = {"bce": BCE_ROWS, "mse": MSE_ROWS}
PLOT_LABELS = {"SGD": "SGD", "SVRG_random": "SVRG (random)", "NFG_SVRG_random": "NFG-SVRG (random)",
               "ASAI_SVRG": "ASA-SVRG (reset)", "SVRG_final": "SVRG (final)", "NFG_SVRG_final": "NFG-SVRG (final)",
               "ASAI_SVRG_continue": "ASA-SVRG (continue)"}
PLOT_COLORS = {"SGD": "tab:gray", "SVRG_random": "tab:blue", "NFG_SVRG_random": "tab:green",
               "ASAI_SVRG": "tab:red", "SVRG_final": "tab:blue", "NFG_SVRG_final": "tab:green",
               "ASAI_SVRG_continue": "tab:purple"}
# SGD は近似誤差を持たず（NaN），全勾配を用いる SVRG は近似誤差が0のため，(C) には描かない．
NO_APPROX_KEYS = ("SGD", "SVRG_random", "SVRG_final")


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------

def load_condition_runs(root: str, loss: str, method: str, inner_start: str) -> list:
    """
    概要: `{root}/{loss}/{手法}/inner_{inner_start}_*/{seed}/` の試行をシード順に読み込む．
    引数: root (str)．loss (str)．method (str)．inner_start (str)．"none"，"snapshot"，"last_iterate"．
    戻り値: runs (list of dict)．`analysis_e4._read_run` の戻り値に `seed` を加えたもの．
    """
    method_dir = os.path.join(root, loss, method)
    if not os.path.isdir(method_dir):
        return []
    runs = []
    for condition in sorted(os.listdir(method_dir)):
        if not condition.startswith(f"inner_{inner_start}_eta"):
            continue
        condition_dir = os.path.join(method_dir, condition)
        for seed in sorted(os.listdir(condition_dir), key=int):
            run = e4._read_run(os.path.join(condition_dir, seed))
            if run is not None:
                run["seed"] = int(seed)
                runs.append(run)
    return runs


def load_rows(loss: str, roots: dict) -> dict:
    """
    概要: 1つの損失の全ての行の試行を読み込む．
    引数: loss (str)．roots (dict)．{"e4": E4 のルート, "e8": E8 のルート}．
    戻り値: rows (dict)．{行のキー: runs}．
    """
    return {key: load_condition_runs(roots[source], loss, method, inner_start)
            for key, _, source, method, inner_start, _ in ROWS[loss]}


# ---------------------------------------------------------------------------
# 統計量
# ---------------------------------------------------------------------------

def latter_half_median(run: dict, key: str = "approx_error", interval=LATTER_HALF) -> float:
    """
    概要: 外側ループ `interval`（両端を含む）における値の中央値を返す（非有限の値を除く）．
    引数: run (dict)．key (str) = "approx_error"．interval (tuple of int) = (51, 100)．
    戻り値: median (float)．該当する点が無ければ NaN．
    """
    epochs = np.asarray(run["log"]["epoch"])
    values = np.asarray(run["log"][key], dtype=np.float64)
    mask = (epochs >= interval[0]) & (epochs <= interval[1])
    values = values[mask][np.isfinite(values[mask])]
    return float(np.median(values)) if values.size else float("nan")


def final_epoch(run: dict) -> int:
    """
    概要: 最終の評価点（打ち切られた試行は打ち切り直前の有限の点）の外側ループの番号を返す．
    引数: run (dict)．
    戻り値: epoch (int)．
    """
    objective = np.asarray(run["log"][e4.objective_key(run)], dtype=np.float64)
    idx = np.where(np.isfinite(objective))[0]
    return int(run["log"]["epoch"][idx[-1]])


def _sci(values: list) -> str:
    """
    概要: 値の列を指数表記の「平均 ± 標準誤差」にする（`analysis_e4._fmt` の sci=True）．
    引数: values (list of float)．
    戻り値: text (str)．
    """
    return e4._fmt(values, sci=True)


def table_lines(loss: str, rows: dict) -> list:
    """
    概要: 1つの損失の表（Markdown）の行を作る．
    引数: loss (str)．rows (dict)．`load_rows` の戻り値．
    戻り値: lines (list of str)．
    """
    if loss == "bce":
        header = ["| 手法 | 結果 | 試行数 | 最終の外側ループ | $ f(\\mathbf z_s)-f(\\mathbf w^*) $（最終） | 同（幾何平均） "
                  "| テスト精度（最終） | $ \\|\\mathbf e_s\\|^2 $ の中央値（51〜100） |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
    else:
        header = ["| 手法 | 結果 | 試行数 | 最終の外側ループ | $ \\|\\nabla f(\\mathbf z_s)\\|^2 $（最終） | 同（幾何平均） "
                  "| $ f(\\mathbf z_s) $（最終） | テスト精度（最終） | $ \\|\\mathbf e_s\\|^2 $ の中央値（51〜100） |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines = list(header)
    for key, label, source, method, inner_start, _ in ROWS[loss]:
        runs = rows[key]
        if not runs:
            lines.append(f"| {label} | {source.upper()} | 0 |" + " — |" * (len(header[1].split("|")) - 5))
            continue
        epochs = sorted({final_epoch(r) for r in runs})
        primary_key = "objective_gap" if loss == "bce" else "grad_norm_sq"
        primary = [e4.final_value(r, primary_key) for r in runs]
        floor = e4.LOSS_FLOOR if loss == "bce" else e4.SMALL_FLOOR
        approx = "—（SGD）" if key == "SGD" else _sci([latter_half_median(r) for r in runs])
        cells = [label, source.upper(), str(len(runs)), "，".join(map(str, epochs)), _sci(primary),
                 e4._fmt_geo(primary, floor)]
        if loss == "mse":
            cells.append(e4._fmt([e4.final_value(r, "objective_value") for r in runs], digits=5))
        cells += [e4._fmt([e4.final_value(r, "accuracy") for r in runs], digits=4), approx]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------

def plot_panel(loss: str, rows: dict, panel: str, figure_dir: str) -> str:
    """
    概要: 1つの損失の1つの図（(A) 主な評価量，(B) テスト精度，(C) 近似誤差）を作成して保存する．
        参考の行（BCE の最終点の SVRG・NFG-SVRG）は細い破線で描く．
    引数: loss (str)．rows (dict)．panel (str)．"A"，"B"，"C"．figure_dir (str)．
    戻り値: path (str)．保存した図のパス（描く試行が無い場合は None）．
    """
    key, log_scale, floor, ylabel = e4.PANELS[panel][loss]
    fig, ax = plt.subplots(figsize=(5, 5))
    drawn = False
    for row_key, _, _, _, _, reference in ROWS[loss]:
        if panel == "C" and row_key in NO_APPROX_KEYS:
            continue
        runs = rows[row_key]
        if reference:
            x, mean, se = e4.mean_and_se(runs, key, None, log_scale, floor)
            if x.size:
                y = 10 ** mean if log_scale else mean
                ax.plot(x, y, "--", color=PLOT_COLORS[row_key], lw=0.8, alpha=0.8,
                        label=f"{PLOT_LABELS[row_key]} (n={len(runs)})")
                drawn = True
            continue
        drawn |= e4._draw(ax, runs, key, None, log_scale, floor, PLOT_COLORS[row_key], PLOT_LABELS[row_key])
    if not drawn:
        plt.close(fig)
        return None
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\#\mathrm{grad}/N$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"({panel}) {e4.LOSS_LABELS[loss]}", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    path = os.path.join(figure_dir, f"e8_{loss}_{panel}_{e4.PANEL_NAMES[panel]}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# まとめ
# ---------------------------------------------------------------------------

def write_summary(e4_root: str = E4_ROOT, e8_root: str = E8_ROOT, output: str = SUMMARY_PATH) -> str:
    """
    概要: 表と図を作成し，`e8_summary.md` に書き出す．
    引数: e4_root (str)．e8_root (str)．output (str)．
    戻り値: output (str)．
    """
    roots = {"e4": e4_root, "e8": e8_root}
    figure_dir = os.path.join(e8_root, "figures")
    lines = ["# E8 の集計結果", "",
             "値は5シードの平均 ± 標準誤差（ddof=1）である．「最終」は100外側ループ目（打ち切られた試行は打ち切り直前の有限の点）の値である．"
             "「幾何平均」は下限（目的関数の誤差は $ 10^{-16} $，$ \\|\\nabla f\\|^2 $ は $ 10^{-300} $）で切り上げた値の幾何平均である．"
             "近似誤差の中央値は，各試行の51〜100外側ループの中央値の平均 ± 標準誤差である．"
             "「結果」は，E4（`ex008_e4_a9a`）と E8（`ex014_e8_a9a_supplement`）のどちらの結果かを示す．", ""]
    for loss, title in (("bce", "## 1. BCE（強凸）"), ("mse", "## 2. MSE（非凸）")):
        rows = load_rows(loss, roots)
        figures = [plot_panel(loss, rows, panel, figure_dir) for panel in ("A", "B", "C")]
        lines += [title, "", *table_lines(loss, rows), "", "図：", "",
                  *[f"- `{os.path.relpath(p, OUTPUTS_DIR)}`" for p in figures if p], ""]
        if loss == "bce":
            lines += ["SVRG の近似誤差は全勾配を用いるため0である（図 (C) には描かない）．"
                      "参考の最終点の SVRG・NFG-SVRG は，図では細い破線で描いた．", ""]
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
    parser.add_argument("--e4-root", default=E4_ROOT, help="E4 の結果（ex008_e4_a9a）のディレクトリ")
    parser.add_argument("--e8-root", default=E8_ROOT, help="E8 の結果（ex014_e8_a9a_supplement）のディレクトリ")
    parser.add_argument("--output", default=SUMMARY_PATH)
    return parser.parse_args()


def main() -> None:
    """
    概要: 集計を実行する．
    引数: なし
    戻り値: なし
    """
    args = parse_args()
    print(write_summary(args.e4_root, args.e8_root, args.output))


if __name__ == "__main__":
    main()
