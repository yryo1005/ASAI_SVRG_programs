"""
E2（実験計画4章）：平均勾配を用いる手法のスナップショットと初期値のアブレーション（CIFAR-10）の集計．

E0（`outputs/ex002_e0_cifar10_alexnet_layernorm`）の6条件，本PCの E2 の条件（`outputs/ex007_e2_cifar10_ablation`），
別のPCの旧版の E2 の最終点・SVRG-average（restart）の条件（`outputs/ex007_e2_cifar10_snapshot_gradient_factorial`）を合わせて，
次を出力する．最終点の手法では2つの初期値が同じ計算となるため，`snapshot` の結果を両方の初期値の結果として扱う．

- `outputs/e2_summary.md`：アブレーションの表（実験計画4.6節），最良の構成の選択（4.5節），対応のある差．
- `outputs/ex007_e2_cifar10_ablation/figures/`：
  - `e2_average_gradient_{train_loss,test_accuracy,approx_error}.png`：平均勾配の5パターン（横軸 $ \\#\\mathrm{grad}/N \\le 90 $）．
  - `e2_asa_vs_svrg_average_{train_loss,test_accuracy}_{grad,epoch}.png`：ASA-SVRG と SVRG-average（ともに `last_iterate`）．
  - `e2_omega.png`：$ \\omega({\\mathbf z}_{s})/K $ と $ \\omega({\\mathbf z}_{s})/\\omega_{\\min} $（記録のある条件のみ）．

実行例:
    python analysis_e2.py
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
E0_DIR = os.path.join(OUTPUTS_DIR, "ex002_e0_cifar10_alexnet_layernorm")
E2_DIR = os.path.join(OUTPUTS_DIR, "ex007_e2_cifar10_ablation")
# 別のPCで実施した旧版の E2（6手法の要因配置，全て `inner_start="snapshot"`，order_060）．
# 最終点の手法（NFG-final，SVRG-final）は2つの初期値が同じ計算となるため，この結果を用いる（order_051，order_052）．
E2_FACTORIAL_DIR = os.path.join(OUTPUTS_DIR, "ex007_e2_cifar10_snapshot_gradient_factorial")
FIGURE_DIR = os.path.join(E2_DIR, "figures")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "e2_summary.md")

N_TRAIN = 50000
K_INNER = 391
BUDGET = 90.0
HP_SUFFIX = "lr0.001_bs128_lambda0.0005_epochs48"

# (条件の識別子, g_s, スナップショット, inner_start, 実験ディレクトリ, 手法のディレクトリ名, 表示名)
CONDITIONS = [
    ("avg_random_snapshot", "平均勾配", "ランダム点", "snapshot", E0_DIR, "NFG_SVRG", "NFG-SVRG (restart)"),
    ("avg_random_last", "平均勾配", "ランダム点", "last_iterate", E0_DIR, "NFG_SVRG", "NFG-SVRG (continue)"),
    ("avg_final", "平均勾配", "最終点", "snapshot", E2_FACTORIAL_DIR, "NFG_SVRG_final", "NFG-final"),
    ("avg_average_snapshot", "平均勾配", "平均点", "snapshot", E0_DIR, "ASAI_SVRG", "ASA-SVRG (restart)"),
    ("avg_average_last", "平均勾配", "平均点", "last_iterate", E0_DIR, "ASAI_SVRG", "ASA-SVRG (continue)"),
    ("full_random_snapshot", "全勾配", "ランダム点", "snapshot", E0_DIR, "SVRG", "SVRG (restart)"),
    ("full_random_last", "全勾配", "ランダム点", "last_iterate", E0_DIR, "SVRG", "SVRG (continue)"),
    ("full_average_last", "全勾配", "平均点", "last_iterate", E2_DIR, "SVRG_average", "SVRG-average (continue)"),
    ("full_average_snapshot", "全勾配", "平均点", "snapshot", E2_FACTORIAL_DIR, "SVRG_average", "SVRG-average (restart)"),
    ("full_final", "全勾配", "最終点", "snapshot", E2_FACTORIAL_DIR, "SVRG_final", "SVRG-final"),
]
AVERAGE_GRADIENT_IDS = ["avg_random_snapshot", "avg_random_last", "avg_final",
                        "avg_average_snapshot", "avg_average_last"]
COLORS = {"avg_random_snapshot": "#98df8a", "avg_random_last": "#2ca02c", "avg_final": "#9467bd",
          "avg_average_snapshot": "#ff9896", "avg_average_last": "#d62728",
          "full_average_last": "#1f77b4"}
STYLES = {"avg_random_snapshot": "--", "avg_random_last": "-", "avg_final": "-",
          "avg_average_snapshot": "--", "avg_average_last": "-", "full_average_last": "-"}
PAIRS = [
    ("近似誤差の効果（平均点，continue）", "avg_average_last", "full_average_last"),
    ("近似誤差の効果（ランダム点，continue）", "avg_random_last", "full_random_last"),
    ("近似誤差の効果（ランダム点，restart）", "avg_random_snapshot", "full_random_snapshot"),
    ("近似誤差の効果（最終点）", "avg_final", "full_final"),
    ("近似誤差の効果（平均点，restart）", "avg_average_snapshot", "full_average_snapshot"),
    ("スナップショットの効果（平均勾配，平均点 − 最終点）", "avg_average_last", "avg_final"),
    ("スナップショットの効果（平均勾配，平均点 − ランダム点，continue）", "avg_average_last", "avg_random_last"),
    ("スナップショットの効果（全勾配，平均点 − ランダム点，continue）", "full_average_last", "full_random_last"),
]


def load_condition(experiment_dir: str, method_dir: str, inner_start: str) -> list:
    """
    概要: 1つの条件の全シードの試行を読み込む．
    引数:
        experiment_dir (str)．実験の出力ディレクトリ．
        method_dir (str)．手法のディレクトリ名．
        inner_start (str)．内側ループの開始点．
    戻り値: runs (list of dict)．各要素は `seed`，`log`，`x`（$ \\#\\mathrm{grad}/N $），`epoch` を持つ．シード順．
    """
    condition_dir = os.path.join(experiment_dir, method_dir, f"inner_{inner_start}_{HP_SUFFIX}")
    runs = []
    if not os.path.isdir(condition_dir):
        return runs
    for seed in sorted(os.listdir(condition_dir), key=int):
        log_path = os.path.join(condition_dir, seed, "log.json")
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            log = json.load(f)
        if len(log["epoch"]) < 49 and not any(log.get("terminated", [])):
            continue  # 実行中の試行は集計に含めない
        runs.append({"seed": int(seed), "log": log,
                     "x": np.asarray(log["oracle_calls"], dtype=np.float64) / N_TRAIN,
                     "epoch": np.asarray(log["epoch"], dtype=np.float64)})
    return runs


def load_all() -> dict:
    """
    概要: `CONDITIONS` の全ての条件を読み込む．試行がない条件は含めない．
    引数: なし
    戻り値: data (dict)．条件の識別子 → {"info": CONDITIONS の要素, "runs": list}．
    """
    data = {}
    for info in CONDITIONS:
        runs = load_condition(info[4], info[5], info[3])
        if runs:
            data[info[0]] = {"info": info, "runs": runs}
    return data


def _values(run: dict, key: str) -> np.ndarray:
    """
    概要: 指標の記録を float64 の配列として返す（記録のない項目は NaN の配列）．
    引数: run (dict)，key (str)．
    戻り値: values (numpy.ndarray)，形状 (T,)．
    """
    if key not in run["log"]:
        return np.full(len(run["x"]), np.nan)
    return np.asarray([np.nan if v is None else v for v in run["log"][key]], dtype=np.float64)


def value_at(run: dict, key: str, budget: float = None, epoch: int = None) -> float:
    """
    概要: 予算以下の最後の評価点，または指定したエポックの値を返す．
    引数: run (dict)，key (str)，budget (float) = None，epoch (int) = None．
    戻り値: value (float)．該当する評価点がなければ NaN．
    """
    values = _values(run, key)
    if epoch is not None:
        hit = np.where(run["epoch"][:len(values)] == epoch)[0]
        return float(values[hit[0]]) if hit.size else float("nan")
    idx = np.where(run["x"][:len(values)] <= budget + 1e-9)[0]
    return float(values[idx[-1]]) if idx.size else float("nan")


def _latter_mask(run: dict, budget: float) -> np.ndarray:
    """
    概要: 予算の後半 $ B/2 < \\#\\mathrm{grad}/N \\le B $ の評価点を表すマスクを返す．
    引数: run (dict)，budget (float)．
    戻り値: mask (numpy.ndarray of bool)．
    """
    return (run["x"] > budget / 2) & (run["x"] <= budget + 1e-9)


def accuracy_variation(run: dict, budget: float = BUDGET) -> float:
    """
    概要: 精度のエポック間変化量．予算の後半における $ |a_{s} - a_{s-1}| $ の平均（論文4.4節 Table IV と同じ定義）．
    引数: run (dict)，budget (float)．
    戻り値: value (float)．
    """
    acc = _values(run, "test_accuracy")
    mask = _latter_mask(run, budget)[:len(acc)]
    idx = np.where(mask)[0]
    idx = idx[idx >= 1]
    diffs = np.abs(acc[idx] - acc[idx - 1])
    return float(np.nanmean(diffs)) if diffs.size else float("nan")


def latter_median(run: dict, key: str, budget: float = BUDGET, transform=None) -> float:
    """
    概要: 予算の後半における指標の中央値を返す．
    引数: run (dict)，key (str)，budget (float)，transform (callable) = None．run を受け取り配列を返す関数（指定時は key より優先）．
    戻り値: value (float)．
    """
    values = transform(run) if transform is not None else _values(run, key)
    mask = _latter_mask(run, budget)[:len(values)] & np.isfinite(values)
    return float(np.median(values[mask])) if mask.any() else float("nan")


def omega_ratio(run: dict) -> np.ndarray:
    """
    概要: $ \\omega({\\mathbf z}_{s})/\\omega_{\\min} $ の系列を返す．
    引数: run (dict)．
    戻り値: ratio (numpy.ndarray)，形状 (T,)．記録のない評価点は NaN．
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = _values(run, "omega_snapshot") / _values(run, "omega_average")
    return np.where(np.isfinite(ratio), ratio, np.nan)  # 1エポック目（ω=0）は NaN


def condition_metrics(runs: list) -> dict:
    """
    概要: 1つの条件について，表に載せる量をシードごとに計算する．
    引数: runs (list of dict)．
    戻り値: metrics (dict)．量の名前 → シードごとの値のリスト．
    """
    return {
        "loss_budget": [value_at(r, "train_loss", budget=BUDGET) for r in runs],
        "acc_budget": [value_at(r, "test_accuracy", budget=BUDGET) for r in runs],
        "loss_ep48": [value_at(r, "train_loss", epoch=48) for r in runs],
        "acc_ep48": [value_at(r, "test_accuracy", epoch=48) for r in runs],
        "acc_variation": [accuracy_variation(r) for r in runs],
        "approx_median": [latter_median(r, "approx_error") for r in runs],
        "omega_ratio_median": [latter_median(r, None, transform=omega_ratio) for r in runs],
        "terminated": [int(any(r["log"].get("terminated", []))) for r in runs],
    }


def ablation_table(data: dict) -> str:
    """
    概要: アブレーションの表（実験計画4.6節）を Markdown で作る．
    引数: data (dict)．`load_all` の戻り値．
    戻り値: text (str)．
    """
    lines = [
        "| $ g_s $ | スナップショット | 初期値 | シード数 | 学習損失（予算） | テスト精度（予算） | 学習損失（48エポック） | "
        "テスト精度（48エポック） | 精度の変化量 | 近似誤差の中央値（後半） | ω/ω_min の中央値（後半） | 打ち切り |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cid, entry in data.items():
        info, runs = entry["info"], entry["runs"]
        m = condition_metrics(runs)
        init = "最終点で同一" if info[2] == "最終点" else info[3]
        approx = _fmt(m["approx_median"], sci=True) if info[1] == "平均勾配" else "（全勾配）"
        lines.append(
            f"| {info[1]} | {info[2]} | {init} | {len(runs)} | {_fmt(m['loss_budget'])} | {_fmt(m['acc_budget'])} | "
            f"{_fmt(m['loss_ep48'])} | {_fmt(m['acc_ep48'])} | {_fmt(m['acc_variation'])} | {approx} | "
            f"{_fmt(m['omega_ratio_median'])} | {sum(m['terminated'])} |")
    return "\n".join(lines)


def best_configuration(data: dict) -> str:
    """
    概要: 実験計画4.5節の基準（平均勾配の5パターンの，予算 $ \\#\\mathrm{grad}/N \\le 90 $ の最後の評価点の
        学習損失の5シードの平均が最小）で最良の構成を選ぶ．
    引数: data (dict)．
    戻り値: text (str)．順位の表と結論．
    """
    ranking = []
    for cid in AVERAGE_GRADIENT_IDS:
        if cid in data:
            losses = condition_metrics(data[cid]["runs"])["loss_budget"]
            ranking.append((float(np.nanmean(losses)), cid, losses))
    ranking.sort()
    lines = ["| 順位 | 条件 | 学習損失（予算，平均 ± 標準誤差） | シード数 |", "|---:|---|---:|---:|"]
    for rank, (mean, cid, losses) in enumerate(ranking, start=1):
        lines.append(f"| {rank} | {data[cid]['info'][6]} | {_fmt(losses)} | {len(losses)} |")
    missing = [cid for cid in AVERAGE_GRADIENT_IDS if cid not in data]
    if missing:
        lines.append(f"\n未実施の条件：{', '.join(missing)}")
    if ranking:
        lines.append(f"\n最良の構成：**{data[ranking[0][1]]['info'][6]}**")
    return "\n".join(lines)


def paired_table(data: dict) -> str:
    """
    概要: 同じシードの2条件の差（平均 ± 標準誤差）を示す．
    引数: data (dict)．
    戻り値: text (str)．
    """
    lines = ["| 比較 | A − B | 学習損失（予算） | テスト精度（予算） | 学習損失（48エポック） | テスト精度（48エポック） |",
             "|---|---|---:|---:|---:|---:|"]
    for title, a, b in PAIRS:
        if a not in data or b not in data:
            continue
        ma, mb = condition_metrics(data[a]["runs"]), condition_metrics(data[b]["runs"])
        seeds_a = [r["seed"] for r in data[a]["runs"]]
        seeds_b = [r["seed"] for r in data[b]["runs"]]
        common = [s for s in seeds_a if s in seeds_b]
        cells = []
        for key in ("loss_budget", "acc_budget", "loss_ep48", "acc_ep48"):
            diffs = [ma[key][seeds_a.index(s)] - mb[key][seeds_b.index(s)] for s in common]
            cells.append(_fmt(diffs))
        lines.append(f"| {title} | {data[a]['info'][6]} − {data[b]['info'][6]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _curve(runs: list, key: str, axis: str, log_scale: bool, transform=None):
    """
    概要: 全シードに共通する評価点について，平均と標準誤差を求める（対数の場合は常用対数で平均）．
    引数: runs (list)，key (str)，axis (str)．"grad" または "epoch"．log_scale (bool)，transform (callable) = None．
    戻り値: (x, mean, se) の組．いずれも numpy.ndarray．
    """
    length = min(len(r["x"]) for r in runs)
    x = (runs[0]["x"] if axis == "grad" else runs[0]["epoch"])[:length]
    values = np.array([(transform(r) if transform else _values(r, key))[:length] for r in runs])
    if log_scale:
        values = np.log10(np.maximum(values, 1e-300))
    mean = np.nanmean(values, axis=0)
    se = np.nanstd(values, axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(mean)
    return x, mean, se


def _plot(ax, runs, key, axis, log_scale, label, color, style, xmax=None, transform=None):
    """
    概要: 1つの条件の平均 ± 標準誤差を描く．
    引数: ax，runs，key，axis，log_scale，label，color，style，xmax (float) = None，transform = None．
    戻り値: なし
    """
    x, mean, se = _curve(runs, key, axis, log_scale, transform)
    if xmax is not None:
        keep = x <= xmax + 1e-9
        x, mean, se = x[keep], mean[keep], se[keep]
    if log_scale:
        ax.plot(x, 10 ** mean, style, color=color, label=label, lw=1.5)
        ax.fill_between(x, 10 ** (mean - se), 10 ** (mean + se), color=color, alpha=0.2)
        ax.set_yscale("log")
    else:
        ax.plot(x, mean, style, color=color, label=label, lw=1.5)
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2)


METRIC_INFO = {"train_loss": ("Training loss", False), "test_accuracy": ("Test accuracy", False),
               "approx_error": (r"$\|\mathbf{e}_s\|^2$", True)}


def plot_average_gradient(data: dict) -> list:
    """
    概要: 平均勾配の5パターンの学習損失，テスト精度，近似誤差を描く（横軸 $ \\#\\mathrm{grad}/N \\le 90 $）．
    引数: data (dict)．
    戻り値: paths (list of str)．
    """
    paths = []
    for key, (ylabel, log_scale) in METRIC_INFO.items():
        fig, ax = plt.subplots(figsize=(5, 5))
        for cid in AVERAGE_GRADIENT_IDS:
            if cid in data:
                _plot(ax, data[cid]["runs"], key, "grad", log_scale, data[cid]["info"][6], COLORS[cid],
                      STYLES[cid], xmax=BUDGET)
        ax.set_xlabel(r"$\#\mathrm{grad}/N$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(FIGURE_DIR, f"e2_average_gradient_{key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_asa_vs_svrg_average(data: dict) -> list:
    """
    概要: ASA-SVRG と SVRG-average（ともに `last_iterate`）の学習損失とテスト精度を，
        横軸 $ \\#\\mathrm{grad}/N $ とエポック数の両方で描く．
    引数: data (dict)．
    戻り値: paths (list of str)．
    """
    paths = []
    if "avg_average_last" not in data or "full_average_last" not in data:
        return paths
    for key in ("train_loss", "test_accuracy"):
        for axis in ("grad", "epoch"):
            fig, ax = plt.subplots(figsize=(5, 5))
            for cid in ("avg_average_last", "full_average_last"):
                _plot(ax, data[cid]["runs"], key, axis, False, data[cid]["info"][6], COLORS[cid], STYLES[cid])
            ax.set_xlabel(r"$\#\mathrm{grad}/N$" if axis == "grad" else "Epoch")
            ax.set_ylabel(METRIC_INFO[key][0])
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = os.path.join(FIGURE_DIR, f"e2_asa_vs_svrg_average_{key}_{axis}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    return paths


def _from_epoch2(run: dict, values: np.ndarray) -> np.ndarray:
    """
    概要: 1エポック目以前の値を NaN にする．平均勾配の手法は1エポック目に $ {\\mathbf g}_{0} = {\\mathbf 0} $ のため
        パラメータがほぼ動かず，$ \\omega $ が丸め誤差の水準（$ 10^{-190} $ 程度）となり，対数軸の図が読めなくなるためである．
    引数: run (dict)，values (numpy.ndarray)，形状 (T,)．
    戻り値: values (numpy.ndarray)，形状 (T,)．
    """
    values = np.array(values, dtype=np.float64)
    values[run["epoch"][:len(values)] < 2] = np.nan
    return values


def plot_omega(data: dict) -> list:
    """
    概要: $ \\omega $ の記録がある条件について，$ \\omega({\\mathbf z}_{s})/K $ と $ \\omega({\\mathbf z}_{s})/\\omega_{\\min} $ を2エポック目から描く．
    引数: data (dict)．
    戻り値: paths (list of str)．
    """
    with_omega = {cid: e for cid, e in data.items()
                  if np.isfinite(_values(e["runs"][0], "omega_snapshot")).any()}
    if not with_omega:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    palette = plt.get_cmap("tab10")
    for i, (cid, entry) in enumerate(with_omega.items()):
        color = COLORS.get(cid, palette(i))
        label = entry["info"][6]
        _plot(axes[0], entry["runs"], None, "epoch", True, label, color, "-",
              transform=lambda r: _from_epoch2(r, _values(r, "omega_snapshot") / K_INNER))
        _plot(axes[1], entry["runs"], None, "epoch", False, label, color, "-",
              transform=lambda r: _from_epoch2(r, omega_ratio(r)))
    axes[0].set_ylabel(r"$\omega(\mathbf{z}_s)/K$")
    axes[1].set_ylabel(r"$\omega(\mathbf{z}_s)/\omega_{\min}$")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, "e2_omega.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def write_summary() -> str:
    """
    概要: 全ての表と図を作り，`outputs/e2_summary.md` に保存する．
    引数: なし
    戻り値: text (str)．保存した Markdown．
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    data = load_all()
    figures = plot_average_gradient(data) + plot_asa_vs_svrg_average(data) + plot_omega(data)
    text = "\n\n".join([
        "# E2：平均勾配を用いる手法のスナップショットと初期値のアブレーション（CIFAR-10，集計結果）",
        "値は5シードの平均 ± 標準誤差である．予算は $ \\#\\mathrm{grad}/N \\le 90 $ の最後の評価点"
        "（平均勾配の手法は45エポック，全勾配の手法は29エポック）．精度の変化量は予算の後半における "
        "$ |a_s - a_{s-1}| $ の平均，近似誤差と ω/ω_min は予算の後半の中央値である．"
        "restart は `inner_start=\"snapshot\"`，continue は `\"last_iterate\"` である．",
        "最終点（NFG-final，SVRG-final）と SVRG-average (restart) は別のPCの結果（`ex007_e2_cifar10_snapshot_gradient_factorial`），"
        "その他は本PCの結果である．GPU の違いによる丸め誤差のため，同じ条件でも PC 間で値が少し異なる．"
        "48エポック目のテスト精度の PC 間の差は，SVRG・ASA-SVRG（restart）で0.003以下，学習が不安定な "
        "NFG-SVRG（restart）で0.05（シード間の標準誤差0.04と同程度）である．"
        "このため，異なる PC の条件の対応のある差は，PC 間の差より十分大きい場合のみ解釈する．",
        "## アブレーションの表", ablation_table(data),
        "## 最良の構成の選択（実験計画4.5節：平均勾配の5パターン，予算での学習損失）", best_configuration(data),
        "## 対応のある差（同じシード）", paired_table(data),
        "## 図", "\n".join(f"- `{os.path.relpath(p, OUTPUTS_DIR)}`" for p in figures),
    ]) + "\n"
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する（現在は引数なし）．
    引数: なし
    戻り値: args (argparse.Namespace)．
    """
    return argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()


if __name__ == "__main__":
    parse_args()
    print(write_summary())
