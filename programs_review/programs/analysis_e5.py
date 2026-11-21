"""
E5（エポック内の平均点で評価した SGD，実験計画8章）の集計と可視化．

比べる3つの系列（実験計画8.6節）：
- SGD（最終点）：E5 の `test_accuracy`．既存の SGD と同じ評価点．
- SGD（平均点）：E5 の `avg_test_accuracy`．エポック内の反復の平均点 $ \\bar{\\mathbf w} $ での評価．
- ASA-SVRG：論文の表IVの元のログ（`ex010_e5_sgd_epoch_average/reference/` に複写した ex0024・ex0051）．

指標は論文の表IVと同じ定義である．
- 揺らぎ：$ \\#\\mathrm{grad}/N $ が予算の1/2以上・予算以下の区間における，隣り合う評価点の間の
  $ |a_{s} - a_{s-1}| $ の平均．SGD は評価点の間隔を ASA-SVRG と同じ2にそろえた値（1エポックおきに間引く）も示す．
- 予算での精度：$ \\#\\mathrm{grad}/N $ が予算以下となる最後の評価点のテスト精度．
- 5シードの平均 ± 標準誤差（表IVと同じく ddof=0）．

可視化する設定は `SETTINGS` で選べる．全シードの学習が終わっていない設定は，終わったシードだけで集計し，
シード数を表に示す．

実行例:
    python analysis_e5.py
"""

import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_PROGRAMS_DIR = os.path.dirname(os.path.abspath(__file__))
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
OUTPUTS = os.path.join(_REVIEW_ROOT, "outputs")
E5_DIR = os.path.join(OUTPUTS, "ex010_e5_sgd_epoch_average")
REFERENCE_DIR = os.path.join(E5_DIR, "reference")
FIGURE_DIR = os.path.join(E5_DIR, "figures")
SUMMARY_PATH = os.path.join(OUTPUTS, "e5_summary.md")

# (データセット, バッチサイズ) -> 設定．優先度の順．
SETTINGS = {
    ("imagewoof", 128): {"label": "ImageWoof", "n_train": 9025, "budget": 200, "epochs": 200,
                         "reference": "ex0051_imagewoof_resnet18_bs128_longrun",
                         "asa_condition": "lr0.001_bs128_lambda0.0005_epochs128",
                         "sgd_condition": "lr0.001_bs128_lambda0.0005_epochs384"},
    ("cifar10", 128): {"label": "CIFAR-10", "n_train": 50000, "budget": 90, "epochs": 90,
                       "reference": "ex0024_cifar10_alexnet_layernorm_longrun",
                       "asa_condition": "lr0.001_bs128_normlayernorm_lambda0.0005_epochs48",
                       "sgd_condition": "lr0.001_bs128_normlayernorm_lambda0.0005_epochs96"},
    ("cifar10", 512): {"label": "CIFAR-10", "n_train": 50000, "budget": 350, "epochs": 350,
                       "reference": "ex0024_cifar10_alexnet_layernorm_longrun",
                       "asa_condition": "lr0.001_bs512_normlayernorm_lambda0.0005_epochs192",
                       "sgd_condition": "lr0.001_bs512_normlayernorm_lambda0.0005_epochs384"},
    ("imagewoof", 512): {"label": "ImageWoof", "n_train": 9025, "budget": 1000, "epochs": 1000,
                         "reference": "ex0051_imagewoof_resnet18_bs128_longrun",
                         "asa_condition": "lr0.001_bs512_lambda0.0005_epochs505",
                         "sgd_condition": "lr0.001_bs512_lambda0.0005_epochs1010"},
}
SERIES = ("SGD_final", "SGD_average", "ASA_SVRG")
SERIES_LABELS = {"SGD_final": "SGD (final iterate)", "SGD_average": "SGD (epoch average)", "ASA_SVRG": "ASA-SVRG"}
SERIES_LABELS_JA = {"SGD_final": "SGD（最終点）", "SGD_average": "SGD（平均点）", "ASA_SVRG": "ASA-SVRG"}
SERIES_COLORS = {"SGD_final": "tab:gray", "SGD_average": "tab:purple", "ASA_SVRG": "tab:red"}


def _load_json(path: str) -> dict:
    """
    概要: JSON ファイルを読み込む．
    引数: path (str)．
    戻り値: data (dict)．
    """
    with open(path) as f:
        return json.load(f)


def e5_condition_dir(dataset: str, batch_size: int) -> str:
    """
    概要: E5 の条件のディレクトリを返す．
    引数: dataset (str)，batch_size (int)．
    戻り値: path (str)．
    """
    epochs = SETTINGS[(dataset, batch_size)]["epochs"]
    return os.path.join(E5_DIR, dataset, "SGD", f"lr0.001_bs{batch_size}_lambda0.0005_epochs{epochs}")


def load_completed_logs(condition_dir: str, epochs: int = None) -> dict:
    """
    概要: 条件のディレクトリから，学習が終わった全シードの `log.json` を読み込む．
    引数: condition_dir (str)．epochs (int) = None．指定した場合は評価点が epochs+1 個ある試行のみを用いる．
    戻り値: logs (dict)．{シード: ログ}．
    """
    logs = {}
    for path in sorted(glob.glob(os.path.join(condition_dir, "*", "log.json"))):
        log = _load_json(path)
        if epochs is not None and len(log["epoch"]) != epochs + 1:
            continue
        logs[int(os.path.basename(os.path.dirname(path)))] = log
    return logs


def series_view(log: dict, series: str) -> dict:
    """
    概要: 1つのログから，系列ごとの `oracle_calls`，`test_accuracy`，`train_loss` を取り出す．
    引数: log (dict)．series (str)．`SERIES` のいずれか．
    戻り値: view (dict)．
    """
    prefix = "avg_" if series == "SGD_average" else ""
    return {"oracle_calls": log["oracle_calls"], "test_accuracy": log[f"{prefix}test_accuracy"],
            "train_loss": log[f"{prefix}train_loss"]}


def load_setting(dataset: str, batch_size: int) -> dict:
    """
    概要: 1つの設定の3系列（SGD 最終点，SGD 平均点，ASA-SVRG）と，表IVの SGD の既存ログを読み込む．
    引数: dataset (str)，batch_size (int)．
    戻り値: data (dict)．{系列: {シード: view}}．`SGD_table_iv` に表IVの SGD の既存ログを含む．
    """
    setting = SETTINGS[(dataset, batch_size)]
    e5_logs = load_completed_logs(e5_condition_dir(dataset, batch_size), setting["epochs"])
    ref = os.path.join(REFERENCE_DIR, setting["reference"])
    asa_logs = load_completed_logs(os.path.join(ref, "ASAI_SVRG", setting["asa_condition"]))
    sgd_ref_logs = load_completed_logs(os.path.join(ref, "SGD", setting["sgd_condition"]))
    return {
        "SGD_final": {s: series_view(log, "SGD_final") for s, log in e5_logs.items()},
        "SGD_average": {s: series_view(log, "SGD_average") for s, log in e5_logs.items()},
        "ASA_SVRG": {s: series_view(log, "ASA_SVRG") for s, log in asa_logs.items()},
        "SGD_table_iv": {s: series_view(log, "SGD_final") for s, log in sgd_ref_logs.items()},
    }


def thin(view: dict, step: int) -> dict:
    """
    概要: 評価点を step 個おきに間引く（先頭の評価点を含む）．
    引数: view (dict)．step (int)．
    戻り値: thinned (dict)．
    """
    return {key: values[::step] for key, values in view.items()}


def mean_abs_accuracy_change(view: dict, n_train: int, budget: float) -> float:
    """
    概要: 表IVの揺らぎ：#grad/N が予算の1/2以上・予算以下の区間の |a_s - a_{s-1}| の平均．
        論文のリポジトリの `aggregate_existing_results.mean_abs_accuracy_change` と同じ定義である．
    引数: view (dict)．n_train (int)．budget (float)．
    戻り値: value (float)．対象の評価点の組がない場合は NaN．
    """
    grad_n = [c / n_train for c in view["oracle_calls"]]
    acc = view["test_accuracy"]
    pairs = [abs(acc[i] - acc[i - 1]) for i in range(1, len(acc))
             if budget / 2 <= grad_n[i - 1] and grad_n[i] <= budget + 1e-9]
    pairs = [p for p in pairs if math.isfinite(p)]
    return sum(pairs) / len(pairs) if pairs else float("nan")


def value_at_budget(view: dict, n_train: int, budget: float, key: str = "test_accuracy") -> float:
    """
    概要: #grad/N が予算以下となる最後の評価点の値を返す（表IVの精度と同じ定義）．
    引数: view (dict)．n_train (int)．budget (float)．key (str) = "test_accuracy"．
    戻り値: value (float)．
    """
    grad_n = [c / n_train for c in view["oracle_calls"]]
    index = max(i for i, g in enumerate(grad_n) if g <= budget + 1e-9)
    return float(view[key][index])


def mean_and_se(values) -> tuple:
    """
    概要: 平均と標準誤差（表IVと同じく ddof=0 の標準偏差 / sqrt(n)）を返す．
    引数: values (list of float)．
    戻り値: (mean, se) の組．値がない場合は (NaN, NaN)．
    """
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0) / math.sqrt(len(arr)))


def setting_statistics(data: dict, setting: dict) -> dict:
    """
    概要: 1つの設定の各系列の揺らぎ（間隔1・2）と予算での精度・学習損失を，シードごとの値から集計する．
    引数: data (dict)．`load_setting` の戻り値．setting (dict)．`SETTINGS` の値．
    戻り値: stats (dict)．{系列: {"n": int, "fluct1": (m, se), "fluct2": (m, se), "acc": (m, se), "loss": (m, se),
        "per_seed": {シード: {...}}}}．
    """
    n_train, budget = setting["n_train"], setting["budget"]
    stats = {}
    for series in SERIES + ("SGD_table_iv",):
        per_seed = {}
        for seed, view in data[series].items():
            per_seed[seed] = {
                "fluct1": mean_abs_accuracy_change(view, n_train, budget),
                "fluct2": mean_abs_accuracy_change(thin(view, 2), n_train, budget),
                "acc": value_at_budget(view, n_train, budget),
                "loss": value_at_budget(view, n_train, budget, "train_loss"),
            }
        stats[series] = {"n": len(per_seed), "per_seed": per_seed,
                         **{k: mean_and_se([v[k] for v in per_seed.values()]) for k in ("fluct1", "fluct2", "acc", "loss")}}
    return stats


def clearly_greater(a: tuple, b: tuple) -> bool:
    """
    概要: 平均 ± 標準誤差の区間が重ならず，a が b より大きいかを判定する（実験計画8.6節の「明確に」）．
    引数: a (tuple)．(mean, se)．b (tuple)．(mean, se)．
    戻り値: result (bool)．
    """
    return a[0] - a[1] > b[0] + b[1]


def judge(stats: dict) -> str:
    """
    概要: 実験計画8.6節の判断の目安1〜3に照らした判定を文で返す．揺らぎは SGD（平均点）の間隔2の値と
        ASA-SVRG の間隔2の値を比べる．ASA-SVRG の評価点はもともと間隔2であるため，その値は `fluct1`
        である（`fluct2` はさらに間引いた間隔4の値であり，比較に用いない）．揺らぎの条件（目安1）と
        精度の条件（目安3）が両方成り立つ場合は両方を示す．
    引数: stats (dict)．`setting_statistics` の戻り値．
    戻り値: text (str)．
    """
    avg, asa = stats["SGD_average"], stats["ASA_SVRG"]
    if avg["n"] == 0 or asa["n"] == 0:
        return "判定なし（結果がない）"
    parts = []
    if clearly_greater(avg["fluct2"], asa["fluct1"]):
        parts.append("目安1：SGD（平均点）の揺らぎが ASA-SVRG より明確に大きい（平均 ± 標準誤差が重ならない）")
    if clearly_greater(avg["acc"], asa["acc"]):
        parts.append("目安3：SGD（平均点）の精度が ASA-SVRG より明確に高い")
    if parts:
        return "．かつ，".join(parts)
    return "目安2に該当：SGD（平均点）の揺らぎは ASA-SVRG より明確には大きくなく，精度も明確には高くない"


def _pct(ms: tuple) -> str:
    """
    概要: (mean, se) の割合を百分率の点の文字列にする．
    引数: ms (tuple)．
    戻り値: text (str)．
    """
    return "—" if not math.isfinite(ms[0]) else f"{100 * ms[0]:.2f} ± {100 * ms[1]:.2f}"


def _num(ms: tuple, digits: int = 4) -> str:
    """
    概要: (mean, se) を文字列にする．
    引数: ms (tuple)．digits (int) = 4．
    戻り値: text (str)．
    """
    return "—" if not math.isfinite(ms[0]) else f"{ms[0]:.{digits}f} ± {ms[1]:.{digits}f}"


def summary_table(all_stats: dict) -> list:
    """
    概要: 全設定の揺らぎと予算での精度の表を作る．
    引数: all_stats (dict)．{(データセット, バッチサイズ): stats}．
    戻り値: lines (list of str)．
    """
    lines = ["| データセット | バッチサイズ | 予算 | 系列 | シード数 | 揺らぎ（間隔1）［%pt］ | 揺らぎ（間隔2）［%pt］ | "
             "予算での精度［%］ | 予算での学習損失 |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for (dataset, batch_size), stats in all_stats.items():
        setting = SETTINGS[(dataset, batch_size)]
        for series in SERIES + ("SGD_table_iv",):
            s = stats[series]
            name = SERIES_LABELS_JA.get(series, "SGD（表IVの既存ログ）")
            fluct1 = "—" if series == "ASA_SVRG" else _pct(s["fluct1"])
            fluct2 = _pct(s["fluct1"]) if series == "ASA_SVRG" else _pct(s["fluct2"])
            lines.append(f"| {setting['label']} | {batch_size} | {setting['budget']} | {name} | {s['n']} | {fluct1} | "
                         f"{fluct2} | {_pct(s['acc'])} | {_num(s['loss'])} |")
    return lines


def paired_difference_lines(all_stats: dict) -> list:
    """
    概要: 同じシードの SGD（平均点）− SGD（最終点）の差（揺らぎ・精度）を表にする．
    引数: all_stats (dict)．
    戻り値: lines (list of str)．
    """
    lines = ["| データセット | バッチサイズ | シード数 | 揺らぎ（間隔1）の差［%pt］ | 揺らぎ（間隔2）の差［%pt］ | "
             "精度の差［%pt］ | 学習損失の差 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for (dataset, batch_size), stats in all_stats.items():
        final, avg = stats["SGD_final"]["per_seed"], stats["SGD_average"]["per_seed"]
        seeds = sorted(set(final) & set(avg))
        if not seeds:
            continue
        diffs = {k: mean_and_se([avg[s][k] - final[s][k] for s in seeds]) for k in ("fluct1", "fluct2", "acc", "loss")}
        lines.append(f"| {SETTINGS[(dataset, batch_size)]['label']} | {batch_size} | {len(seeds)} | "
                     f"{_pct(diffs['fluct1'])} | {_pct(diffs['fluct2'])} | {_pct(diffs['acc'])} | {_num(diffs['loss'])} |")
    return lines


def _mean_curve(views: dict, key: str, n_train: int) -> tuple:
    """
    概要: シード間で共通する長さの曲線の平均と標準誤差を返す．
    引数: views (dict)．{シード: view}．key (str)．n_train (int)．
    戻り値: (x, mean, se) の組．
    """
    length = min(len(v[key]) for v in views.values())
    x = np.asarray(next(iter(views.values()))["oracle_calls"][:length], dtype=float) / n_train
    arr = np.asarray([v[key][:length] for v in views.values()], dtype=float)
    return x, arr.mean(axis=0), arr.std(axis=0, ddof=0) / math.sqrt(arr.shape[0])


def plot_setting(data: dict, dataset: str, batch_size: int, key: str, figure_dir: str = FIGURE_DIR) -> str:
    """
    概要: 1つの設定の3系列の推移（平均 ± 標準誤差）を，横軸 #grad/N で描く．
    引数: data (dict)．`load_setting` の戻り値．dataset (str)，batch_size (int)．
        key (str)．"test_accuracy" または "train_loss"．figure_dir (str) = FIGURE_DIR．
    戻り値: path (str)．保存した図のパス（系列がない場合は空文字列）．
    """
    setting = SETTINGS[(dataset, batch_size)]
    if not any(data[s] for s in SERIES):
        return ""
    fig, ax = plt.subplots(figsize=(5, 5))
    for series in SERIES:
        if not data[series]:
            continue
        x, mean, se = _mean_curve(data[series], key, setting["n_train"])
        keep = x <= setting["budget"] + 1e-9
        ax.plot(x[keep], mean[keep], color=SERIES_COLORS[series], label=SERIES_LABELS[series], lw=1.2)
        ax.fill_between(x[keep], (mean - se)[keep], (mean + se)[keep], color=SERIES_COLORS[series], alpha=0.25, lw=0)
    ax.axvspan(setting["budget"] / 2, setting["budget"], color="0.9", zorder=0)
    ax.set_xlabel(r"$\#\mathrm{grad}/N$")
    ax.set_ylabel("Test accuracy" if key == "test_accuracy" else "Training loss")
    ax.set_title(f"{setting['label']}, batch size {batch_size}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    path = os.path.join(figure_dir, f"e5_{dataset}_bs{batch_size}_{key}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_summary(settings=None, summary_path: str = SUMMARY_PATH) -> str:
    """
    概要: 全設定を集計し，図を保存して `e5_summary.md` を書く．
    引数: settings (list of tuple) = None．集計する (データセット, バッチサイズ)．None は全設定．
        summary_path (str) = SUMMARY_PATH．
    戻り値: text (str)．書いた Markdown．
    """
    settings = list(SETTINGS) if settings is None else settings
    all_stats, figures, judgements = {}, [], []
    for dataset, batch_size in settings:
        data = load_setting(dataset, batch_size)
        stats = setting_statistics(data, SETTINGS[(dataset, batch_size)])
        all_stats[(dataset, batch_size)] = stats
        figures += [p for p in (plot_setting(data, dataset, batch_size, k) for k in ("test_accuracy", "train_loss")) if p]
        judgements.append(f"- {SETTINGS[(dataset, batch_size)]['label']}・バッチサイズ{batch_size}"
                          f"（SGD {stats['SGD_average']['n']} シード）：{judge(stats)}．")
    lines = [
        "# E5：エポック内の平均点で評価した SGD（集計結果）",
        "",
        "値は5シードの平均 ± 標準誤差（表IVと同じく ddof=0）である．揺らぎは，$ \\#\\mathrm{grad}/N $ が予算の1/2以上・"
        "予算以下の区間における隣り合う評価点の間のテスト精度の変化 $ |a_{s} - a_{s-1}| $ の平均である（表IVと同じ定義）．"
        "SGD の評価点の間隔は1，ASA-SVRG は2である．SGD は1エポックおきに間引いて間隔を2にそろえた値も示す"
        "（ASA-SVRG は間隔2の列に示す）．ASA-SVRG と「SGD（表IVの既存ログ）」は，表IVの元の既存ログである．",
        "",
        "## 揺らぎと予算での精度",
        "",
        *summary_table(all_stats),
        "",
        "## 同じシードの差（SGD（平均点）− SGD（最終点））",
        "",
        *paired_difference_lines(all_stats),
        "",
        "## 実験計画8.6節の判断の目安に照らした判定",
        "",
        "判定は，SGD（平均点，間隔2）と ASA-SVRG の平均 ± 標準誤差の区間が重ならないかで行った．",
        "",
        *judgements,
        "",
        "## 図",
        "",
        *[f"- `{os.path.relpath(p, OUTPUTS)}`" for p in figures],
        "",
    ]
    text = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    print(write_summary())
