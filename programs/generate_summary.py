"""
`outputs/` 配下に保存済みの実験ログ（`log.json`，`config.json`）から，実験0，実験1，ex0023，
ex003（実施済みStageのみ）の全条件を横断的に一覧できる `summary.md` と，条件ごとの比較グラフ
（`outputs/{実験名}/summary_figures/`）を機械的に生成するスクリプト．

新たな学習は行わず，既存の `outputs/` 配下のログを読み込むのみである．選定（論文へ掲載する
条件・図の絞り込み）は行わず，実行時点で完了している条件をすべて掲載する．
"""

import json
import os
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# 日本語ラベルを正しく描画するため，IPAexゴシックフォントが存在すれば明示的に登録する
# （`visualize_result.ipynb` と同一の設定）．
_JP_FONT_PATH = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"
if os.path.exists(_JP_FONT_PATH):
    fm.fontManager.addfont(_JP_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_JP_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
SUMMARY_MD_PATH = PROJECT_ROOT / "summary.md"

METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
APPROX_ERROR_METHODS = ["NFG_SVRG", "ASAI_SVRG"]
METHOD_COLORS = {
    "SGD": "#2ca02c",
    "SVRG": "#1f77b4",
    "NFG_SVRG": "#ff7f0e",
    "ASAI_SVRG": "#d62728",
}


# ---------------------------------------------------------------------------
# 実験ごとの静的メタ情報（目的関数・モデル・データセットの説明文，指標キーの対応）
# ---------------------------------------------------------------------------

_ETA_PATTERN = r"eta([0-9.eE+-]+)"
_BS_PATTERN = r"bs(\d+)"
_LR_PATTERN = r"lr([0-9.eE+-]+)"

EXPERIMENT_SPECS = {
    "ex000_a9a_least_squares": {
        "title": "実験0：a9aデータセットを用いた非線形最小二乗回帰",
        "report_ref": ".reports/report_020.md",
        "purpose": (
            "NFG SVRG原論文（Medyakov et al., 2025）付録A.1の非凸実験設定を再現し，"
            "本実装（SVRG，NFG SVRG）が先行研究の挙動を正しく再現できることを確認する検証実験である．"
        ),
        "objective_formula": (
            r"$$ f(x) = \frac{1}{n}\sum_{i=1}^n (y_i - h_i)^2,\quad h_i = \sigma(A_i x) $$"
            "\n\nシグモイド出力に対する二乗和誤差（非線形最小二乗誤差，非凸）．正則化項は付加しない．"
        ),
        "model_desc": "線形結合にシグモイド関数を適用したモデル（ロジスティック関数出力の非線形最小二乗回帰）．",
        "dataset_desc": (
            "a9a（LIBSVM Data，UCI Adult所得予測データセットの二値分類向け前処理版）．"
            "特徴量の前処理は原データセットのLIBSVM形式をそのまま使用．"
        ),
        "objective_key": "objective_value",
        "objective_label_train_error": r"目的関数の値 $f(z_s)$",
        "objective_is_gap": False,
        "accuracy_key": "accuracy",
        "accuracy_label": "分類精度（検証用データ）",
        "condition_tag": lambda hp: f"eta{_extract(hp, r'eta([0-9.eE+-]+)')}",
        "condition_label": lambda cfg: f"$\\eta$ = {cfg.get('eta_rule', '?')} = {cfg.get('eta', float('nan')):.3g}",
    },
    "ex001_mushroom_logistic": {
        "title": "実験1：Mushroomデータセットを用いたL2正則化ロジスティック回帰",
        "report_ref": ".reports/report_021.md",
        "purpose": (
            "ASAI SVRG論文の理論解析の前提（L-平滑性，凸性，μ-強凸性，分散の一様有界性）を"
            "厳密に満たす設定において，定理1（収束特性）・定理2（誤差床の上界比較）を定量的に検証する．"
        ),
        "objective_formula": (
            r"$$ f(w, b) = \frac{1}{N}\sum_{n=1}^N \ell_{\mathrm{BCE}}(y_n, \hat{y}_n(w, b))"
            r" + \frac{\lambda}{2}(\|w\|^2 + b^2),\quad \hat{y}(w, b) = \sigma(w^\top x + b) $$"
            "\n\nL2正則化付き二値交差エントロピー損失．正則化は重み $ w $ と切片 $ b $ の両方に課す．"
        ),
        "model_desc": "切片項を含むロジスティック回帰モデル．",
        "dataset_desc": (
            "Mushroomデータセット（UCI Machine Learning Repository）．"
            "22種類のカテゴリ特徴量を順序符号化（OrdinalEncoder）し，9:1に分割後，学習用データの統計量で標準化．"
        ),
        "objective_key": "objective_gap",
        "objective_label_train_error": r"目的関数の差分 $f(z_s)-f(w^*)$（L-BFGS-Bによる最適値との差分）",
        "objective_is_gap": True,
        "accuracy_key": "accuracy",
        "accuracy_label": "分類精度（検証用データ）",
        "condition_tag": lambda hp: f"eta{_extract(hp, r'eta([0-9.eE+-]+)')}",
        "condition_label": lambda cfg: f"$\\eta$ = {cfg.get('eta_rule', '?')} = {cfg.get('eta', float('nan')):.3g}",
    },
    "ex0023_cifar10_alexnet_groupnorm_longrun": {
        "title": "ex0023：CIFAR-10・AlexNet（GroupNorm）長期学習による誤差床の収束観察",
        "report_ref": ".reports/report_026.md",
        "purpose": (
            "GroupNorm下・学習率0.001固定でエポック数を大幅に延長し，近似誤差 $\\|e_s\\|^2$ "
            "および分類精度がプラトーに達するまで観察することで，誤差床がtransientか恒久的かを"
            "バッチサイズごとに判定する．"
        ),
        "objective_formula": (
            r"$$ f(w) = \frac{1}{N}\sum_{n=1}^N \ell_{\mathrm{CCE}}(y_n, \hat{y}_n(w)) + \frac{\lambda}{2}\|w\|^2 $$"
            "\n\nカテゴリカルクロスエントロピー損失にL2正則化を加えた目的関数．"
        ),
        "model_desc": (
            "AlexNetをCIFAR-10向けに縮小した，5つの畳み込み層と3つの全結合層から成るCNN"
            "（`AlexNetCIFAR`，総パラメータ数約717万）．正規化層はGroupNorm固定．"
        ),
        "dataset_desc": "CIFAR-10．画素値はチャネルごとに標準化，データ拡張なし．",
        "objective_key": "train_loss",
        "objective_label_train_error": "目的関数の値（訓練誤差，train_loss）",
        "objective_is_gap": False,
        "accuracy_key": "test_accuracy",
        "accuracy_label": "分類精度（検証用データ，CIFAR-10）",
        "condition_tag": lambda hp: "bs" + _extract(hp, _BS_PATTERN),
        "condition_label": lambda cfg: f"バッチサイズ = {cfg.get('batch_size', '?')}（学習率0.001固定）",
    },
    "ex003_tinyshakespeare_transformer": {
        "title": "ex003：Tiny Shakespeareを用いた文字レベル次文字予測（Decoder-only Transformer，Stage A）",
        "report_ref": ".orders/order_027.md",
        "report_note": "（Stage Aはレポート未作成，実施中のため指示書へのリンクを示す）",
        "purpose": (
            "Transformer・系列データという新しい設定で，NFG SVRG・ASAI SVRGが発散・崩壊せずに"
            "学習が進む（バッチサイズ，学習率の）条件を特定するStage A（安定性探索）．"
            "この段階ではSGD・SVRGは対象に含めない．"
        ),
        "objective_formula": (
            r"$$ f(w) = \frac{1}{N}\sum_{n=1}^N \ell_{\mathrm{CCE}}(y_n, \hat{y}_n(w)) + \frac{\lambda}{2}\|w\|^2 $$"
            "\n\n次文字予測のカテゴリカルクロスエントロピー損失にL2正則化を加えた目的関数．"
        ),
        "model_desc": (
            "4層Decoder-only Transformer（`DecoderOnlyTransformer`，$ d_{\\text{model}}=128 $，"
            "Attention head数4，Feed Forward中間次元512）．Pre-LN構成のLayerNormを使用し，"
            "Dropout・BatchNormalizationは一切使用しない．位置エンコーディングは学習可能な埋め込み．"
        ),
        "dataset_desc": (
            "Tiny Shakespeare（文字レベル言語モデリング）．コーパスを系列長 $ T=128 $ の非重複チャンクに分割し，"
            "前方90%を学習用，後方10%を検証用とする．"
        ),
        "objective_key": "train_loss",
        "objective_label_train_error": "目的関数の値（訓練誤差，train_loss）",
        "objective_is_gap": False,
        "accuracy_key": "test_accuracy",
        "accuracy_label": "次文字予測精度（検証用データ）",
        "condition_tag": lambda hp: (
            "bs" + _extract(hp, _BS_PATTERN) + "_lr" + _extract(hp, _LR_PATTERN)
        ),
        "condition_label": lambda cfg: f"バッチサイズ = {cfg.get('batch_size', '?')}, 学習率 = {cfg.get('learning_rate', '?')}",
    },
}

EXPERIMENT_ORDER = [
    "ex000_a9a_least_squares",
    "ex001_mushroom_logistic",
    "ex0023_cifar10_alexnet_groupnorm_longrun",
    "ex003_tinyshakespeare_transformer",
]


def _extract(hp_name: str, pattern: str) -> str:
    """
    概要: ハイパーパラメータ条件名（ディレクトリ名）から正規表現で値を抽出する．
    引数: hp_name (str)．ディレクトリ名．pattern (str)．抽出用の正規表現（1グループ）．
    戻り値: value (str)．抽出した値．見つからない場合は "unknown"．
    """
    match = re.search(pattern, hp_name)
    return match.group(1) if match else "unknown"


# ---------------------------------------------------------------------------
# outputs/ からの読み込み
# ---------------------------------------------------------------------------


def is_seed_complete(seed_dir: Path) -> bool:
    """
    概要: 1つのSeedディレクトリが完了しているか（`log.json`・`config.json`が揃っているか）を判定する．
    引数: seed_dir (Path)．
    戻り値: complete (bool)．
    """
    return (seed_dir / "log.json").exists() and (seed_dir / "config.json").exists()


def load_method_condition(hp_dir: Path):
    """
    概要: 1つの(手法, ハイパーパラメータ条件)について，全Seedが完了している場合のみログを読み込む．
    引数: hp_dir (Path)．`outputs/{実験}/{手法}/{ハイパーパラメータ}/`．
    戻り値: data (dict) または None．
        全Seedが完了している場合 {"logs": [dict, ...], "configs": [dict, ...]}，
        1つでも未完了のSeedがある場合，またはSeedが存在しない場合は None．
    """
    seed_dirs = sorted(
        (d for d in hp_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if not seed_dirs or not all(is_seed_complete(d) for d in seed_dirs):
        return None
    logs = [json.loads((d / "log.json").read_text()) for d in seed_dirs]
    configs = [json.loads((d / "config.json").read_text()) for d in seed_dirs]
    return {"logs": logs, "configs": configs}


def discover_conditions(experiment: str) -> dict:
    """
    概要: 1つの実験について，実行時点で完了している(ハイパーパラメータ条件, 手法)の組をすべて発見する．
    引数: experiment (str)．実験ディレクトリ名．
    戻り値: conditions (dict)．{ハイパーパラメータ条件名: {手法名: data}}．
        一部の手法しか完了していない条件は，完了した手法のみを含む．
    """
    exp_dir = OUTPUTS_ROOT / experiment
    conditions = {}
    if not exp_dir.exists():
        return conditions
    for method in METHODS:
        method_dir = exp_dir / method
        if not method_dir.exists():
            continue
        for hp_dir in sorted(d for d in method_dir.iterdir() if d.is_dir()):
            data = load_method_condition(hp_dir)
            if data is not None:
                conditions.setdefault(hp_dir.name, {})[method] = data
    return conditions


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------


def aggregate_seeds(logs: list, key: str):
    """
    概要: 複数Seedのログから，指定した指標について全Seedの平均・標準偏差の軌跡を計算する．
    引数: logs (list of dict)．各Seedの `ResultLogger.history` に対応する辞書のリスト．
        key (str)．集計する指標名．
    戻り値: mean (numpy.ndarray)，std (numpy.ndarray)．いずれも形状 (T,)．
        Tは全Seed中の最短の系列長．発散（NaN）したSeedは `nanmean`／`nanstd` により自動的に除外される．
    """
    length = min(len(log[key]) for log in logs)
    values = np.array([log[key][:length] for log in logs], dtype=np.float64)
    return np.nanmean(values, axis=0), np.nanstd(values, axis=0)


def count_diverged_seeds(logs: list, key: str) -> int:
    """
    概要: 指定した指標にNaN・Infが1つでも含まれるSeedの数を数える．
    引数: logs (list of dict)．key (str)．発散判定に用いる指標名．
    戻り値: n_diverged (int)．
    """
    return sum(1 for log in logs if any(not np.isfinite(v) for v in log[key]))


def fmt_mean_std(mean: float, std: float, sci: bool = True) -> str:
    """
    概要: 平均値±標準偏差を文字列に整形する．
    引数: mean (float)．std (float)．sci (bool) = True．指数表記にするか．
    戻り値: text (str)．
    """
    if not np.isfinite(mean):
        return "NaN/Inf"
    if sci:
        return f"{mean:.4e} ± {std:.2e}"
    return f"{mean:.4f} ± {std:.4f}"


def fmt_scalar(value: float, sci: bool = True) -> str:
    """
    概要: 単一の値を文字列に整形する．
    引数: value (float)．sci (bool) = True．指数表記にするか．
    戻り値: text (str)．
    """
    if not np.isfinite(value):
        return "NaN/Inf"
    return f"{value:.4e}" if sci else f"{value:.4f}"


# ---------------------------------------------------------------------------
# グラフ生成
# ---------------------------------------------------------------------------


def plot_metric(figures_dir: Path, filename: str, methods_data: dict, methods: list,
                 x_key: str, x_label: str, y_key: str, y_label: str, log_y: bool):
    """
    概要: 1つの(横軸, 縦軸)の組み合わせについて，指定した手法群の平均±標準偏差の軌跡を1枚のグラフとして保存する．
    引数:
        figures_dir (Path)．保存先ディレクトリ．
        filename (str)．保存するファイル名（拡張子込み）．
        methods_data (dict)．{手法名: {"logs": [...], "configs": [...]}}．
        methods (list of str)．描画する手法名のリスト（実施済みのもののみ自動的に描画される）．
        x_key (str)．横軸の指標名（"epoch" または "oracle_calls_per_n"）．
        x_label (str)．横軸ラベル．
        y_key (str)．縦軸の指標名．
        y_label (str)．縦軸ラベル．
        log_y (bool)．縦軸を対数軸にするか．
    戻り値: なし
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    plotted = False
    for method in methods:
        if method not in methods_data:
            continue
        logs = methods_data[method]["logs"]
        y_mean, y_std = aggregate_seeds(logs, y_key)
        if x_key == "oracle_calls_per_n":
            n_train = methods_data[method]["configs"][0]["N_train"]
            oracle_mean, _ = aggregate_seeds(logs, "oracle_calls")
            x = oracle_mean[: len(y_mean)] / n_train
        else:
            epoch_mean, _ = aggregate_seeds(logs, "epoch")
            x = epoch_mean[: len(y_mean)]
        color = METHOD_COLORS[method]
        ax.plot(x, y_mean, label=method, color=color, linewidth=1.6)
        if log_y:
            lower = np.clip(y_mean - y_std, 1e-12, None)
        else:
            lower = y_mean - y_std
        ax.fill_between(x, lower, y_mean + y_std, color=color, alpha=0.15, linewidth=0)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_condition_figures(experiment: str, spec: dict, tag: str, methods_data: dict,
                                figures_dir: Path) -> list:
    """
    概要: 1条件につき6枚のグラフ（横軸2種×縦軸3種）を生成し，保存したファイル名のリストを返す．
    引数:
        experiment (str)．実験ディレクトリ名．
        spec (dict)．`EXPERIMENT_SPECS` の該当エントリ．
        tag (str)．条件を一意に識別するファイル名タグ．
        methods_data (dict)．{手法名: {"logs": [...], "configs": [...]}}．
        figures_dir (Path)．保存先ディレクトリ．
    戻り値: figure_files (list of str)．生成されたファイル名（生成できなかった組は含まない）のリスト．
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    x_axes = [
        ("epoch", "エポック数", "epoch"),
        ("oracle_calls_per_n", "#grad/N", "gradN"),
    ]
    y_axes = [
        (spec["objective_key"], spec["objective_label_train_error"], "train_error", True, METHODS),
        (spec["accuracy_key"], spec["accuracy_label"], "accuracy", False, METHODS),
        ("approx_error", r"近似誤差 $\|e_s\|^2$", "approx_error", True, APPROX_ERROR_METHODS),
    ]

    generated = []
    for x_key, x_label, x_tag in x_axes:
        for y_key, y_label, y_tag, log_y, methods in y_axes:
            filename = f"{experiment.split('_')[0]}_{tag}_{y_tag}_vs_{x_tag}.png"
            ok = plot_metric(figures_dir, filename, methods_data, methods,
                              x_key, x_label, y_key, y_label, log_y)
            if ok:
                generated.append(filename)
    return generated


# ---------------------------------------------------------------------------
# 数値サマリー表・条件仕様
# ---------------------------------------------------------------------------


def build_condition_spec_table(spec: dict, methods_data: dict) -> str:
    """
    概要: 1条件のハイパーパラメータ仕様一覧をMarkdownの箇条書きとして作成する．
    引数: spec (dict)．`EXPERIMENT_SPECS` の該当エントリ．methods_data (dict)．
    戻り値: text (str)．
    """
    any_config = next(iter(methods_data.values()))["configs"][0]
    batch_size = any_config.get("batch_size", 1)
    lr = any_config.get("learning_rate", any_config.get("eta"))
    reg_lambda = any_config.get("reg_lambda", 0)
    k = any_config.get("K")
    epochs = any_config.get("epochs")
    n_train = any_config.get("N_train")
    n_test = any_config.get("N_test")
    seed_counts = {m: len(d["logs"]) for m, d in methods_data.items()}
    missing = [m for m in METHODS if m not in methods_data]

    lines = [
        f"- モデル：{spec['model_desc']}",
        f"- データセット：{spec['dataset_desc']}（$ N_{{\\text{{train}}}}={n_train} $, $ N_{{\\text{{test}}}}={n_test} $）",
        f"- バッチサイズ：{batch_size}",
        f"- 学習率：{lr}",
        f"- 正則化係数 $ \\lambda $：{reg_lambda}",
        f"- 内部ループ長 $ K $：{k}",
        f"- エポック数：{epochs}",
        f"- Seed数：" + "，".join(f"{m}: {n}" for m, n in seed_counts.items()),
    ]
    if missing:
        lines.append(f"- 未実施の手法：{'，'.join(missing)}（実行時点で未完了のため対象外）")
    return "\n".join(lines)


def build_numeric_summary_table(spec: dict, methods_data: dict) -> str:
    """
    概要: 各手法について，最終値・最良値・Seed平均±標準偏差・発散Seed数をまとめたMarkdown表を作成する．
    引数: spec (dict)．`EXPERIMENT_SPECS` の該当エントリ．methods_data (dict)．
    戻り値: table (str)．
    """
    lines = [
        "| 手法 | 最終学習誤差(平均±std) | 最良学習誤差 | "
        "最終検証精度(平均±std) | 最良検証精度 | "
        "最終近似誤差(平均±std) | 最良近似誤差 | 発散Seed数 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    objective_key = spec["objective_key"]
    accuracy_key = spec["accuracy_key"]
    for method in METHODS:
        if method not in methods_data:
            continue
        logs = methods_data[method]["logs"]
        n_seeds = len(logs)

        obj_mean, obj_std = aggregate_seeds(logs, objective_key)
        acc_mean, acc_std = aggregate_seeds(logs, accuracy_key)
        obj_final, obj_final_std = obj_mean[-1], obj_std[-1]
        acc_final, acc_final_std = acc_mean[-1], acc_std[-1]
        obj_best = np.nanmin(obj_mean)
        acc_best = np.nanmax(acc_mean)
        n_diverged = count_diverged_seeds(logs, objective_key)

        if method == "SGD":
            approx_final = approx_best = "対象外"
        elif method == "SVRG":
            approx_final = "0（定義上）"
            approx_best = "0（定義上）"
        else:
            err_mean, err_std = aggregate_seeds(logs, "approx_error")
            approx_final = fmt_mean_std(err_mean[-1], err_std[-1])
            approx_best = fmt_scalar(np.nanmin(err_mean))

        lines.append(
            f"| {method} | {fmt_mean_std(obj_final, obj_final_std)} | {fmt_scalar(obj_best)} | "
            f"{fmt_mean_std(acc_final, acc_final_std, sci=False)} | {fmt_scalar(acc_best, sci=False)} | "
            f"{approx_final} | {approx_best} | {n_diverged}/{n_seeds} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# summary.md の組み立て
# ---------------------------------------------------------------------------


def sort_condition_names(experiment: str, hp_names: list) -> list:
    """
    概要: 条件名（ディレクトリ名）を，実験ごとに意味のある順序（学習率降順，バッチサイズ降順等）に並べ替える．
    引数: experiment (str)．hp_names (list of str)．
    戻り値: sorted_names (list of str)．
    """
    def key(hp_name: str):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", hp_name)
        return [-float(n) for n in numbers]

    return sorted(hp_names, key=key)


def build_experiment_section(experiment: str) -> str:
    """
    概要: 1つの実験について，全ての完了済み条件を含むsummary.mdのセクションを作成し，
        あわせて条件ごとのグラフを `outputs/{実験名}/summary_figures/` に保存する．
    引数: experiment (str)．実験ディレクトリ名．
    戻り値: section_text (str)．実験が存在しない，または完了条件が1つもない場合は空文字列．
    """
    spec = EXPERIMENT_SPECS[experiment]
    conditions = discover_conditions(experiment)
    if not conditions:
        return ""

    figures_dir = OUTPUTS_ROOT / experiment / "summary_figures"
    report_line = f"参照レポート：`{spec['report_ref']}`"
    if spec.get("report_note"):
        report_line += spec["report_note"]

    lines = [
        f"# {spec['title']}",
        "",
        report_line,
        "",
        spec["purpose"],
        "",
        "目的関数：",
        "",
        spec["objective_formula"],
        "",
    ]

    for hp_name in sort_condition_names(experiment, list(conditions.keys())):
        methods_data = conditions[hp_name]
        any_config = next(iter(methods_data.values()))["configs"][0]
        tag = spec["condition_tag"](hp_name)
        label = spec["condition_label"](any_config)

        lines.append(f"## {spec['title'].split('：')[0]} / {label}")
        lines.append("")
        lines.append(f"条件ディレクトリ：`outputs/{experiment}/{{手法}}/{hp_name}/`")
        lines.append("")
        lines.append("### 条件の仕様")
        lines.append("")
        lines.append(build_condition_spec_table(spec, methods_data))
        lines.append("")
        lines.append(
            f"学習誤差の指標：{spec['objective_label_train_error']}／"
            f"検証指標：{spec['accuracy_label']}"
        )
        lines.append("")

        figure_files = generate_condition_figures(experiment, spec, tag, methods_data, figures_dir)
        lines.append("### グラフ")
        lines.append("")
        for filename in figure_files:
            lines.append(f"![{filename}](outputs/{experiment}/summary_figures/{filename})")
            lines.append("")

        lines.append("### 数値サマリー")
        lines.append("")
        lines.append(build_numeric_summary_table(spec, methods_data))
        lines.append("")

    return "\n".join(lines)


def main():
    """
    概要: summary.mdおよび各実験のsummary_figures/を生成するメイン処理．
    引数: なし
    戻り値: なし
    """
    header = [
        "# summary",
        "",
        "本ファイルは，論文に掲載する実験（実験0，実験1，ex0023，ex003）の結果を横断的に一覧する"
        "ために機械的に生成したものであり，考察文書ではない．論文へ掲載する条件・図の選定は行っておらず，"
        "実行時点で完了している条件をすべて掲載している．",
        "",
        "比較対象はSGD，SVRG，NFG SVRG（NFG_SVRG），ASAI SVRG（ASAI_SVRG）の4手法である．"
        "ある条件で一部の手法しか完了していない場合は，完了した手法のみを描画し，"
        "条件の仕様に未実施の手法を明記する．",
        "",
        "フル勾配の近似誤差 $\\|e_s\\|^2$ のグラフは，SGD（近似誤差の概念がない）・"
        "SVRG（定義上常に0）を除き，NFG SVRG・ASAI SVRGの2手法のみを描画する．",
        "",
    ]

    sections = []
    for experiment in EXPERIMENT_ORDER:
        section = build_experiment_section(experiment)
        if section:
            sections.append(section)
            print(f"{experiment}: 生成完了")
        else:
            print(f"{experiment}: 完了済み条件なし（スキップ）")

    SUMMARY_MD_PATH.write_text("\n".join(header) + "\n\n" + "\n\n".join(sections) + "\n")
    print(f"summary.md を保存しました: {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()
