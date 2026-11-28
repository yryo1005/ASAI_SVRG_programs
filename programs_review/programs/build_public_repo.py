"""
論文（`main.pdf`，NOLTA 最終版）に掲載した実験のプログラムと結果から，公開版リポジトリ `../ASA-SVRG` を組み立てるスクリプト（order_085）．

- 実験のプログラムを `experiments/{データセット}/{実験名}/` にコピーし，ディレクトリの深さの変化に合わせてパスの計算を書き換える．
  学習の計算そのものは変更しない．
- 共通モジュール（`programs_review/programs/common/`）を `experiments/common/` にコピーする．
- 各試行の結果（`log.json`・`config.json`）を，実験の単位ではなく `results/{データセット}/{手法間で共通の条件}/{手法}/{シード}/`
  に並べ替えてコピーする（規則は公開版の `experiments/common/result_layout.py` と同じ．order_085 の追加の指示）．
  同じ条件・同じ手法の試行が複数の実験にある場合は，論文で使った1組だけを残す（`EXCLUDED_RUNS`）．
  `config.json` の提案手法の名前（開発時の `ASAI_SVRG`）は `ASA_SVRG` に置き換える（公開版の名前の統一）．
  重み（`*.pth`），図（`*.png`），作業記録（`*.md`，`*.log`）はコピーしない．
- 内部の検証用スクリプト（旧リポジトリの結果との比較）と，作業記録の README はコピーしない．
- 論文の図表を作るモジュール（論文のリポジトリの `programs/` と `analysis_e10.py`・`analysis_e11.py`）を `paper/` にコピーし，
  モジュール名，結果のパス，実験名を公開版に合わせて書き換える．

既存のファイルは読み取りのみで，変更しない．公開版の README，図表の再現スクリプト，テストは別に作成する．

プログラムのコピー（`--code`）は公開版を最初に作るときだけに使う．その後，公開版のプログラム（保存先の規則など）と
`paper/` は公開版で直接編集しているため，`--code` を付けて実行するとその変更は失われる．既定では結果の並べ替えだけを行う．

使い方（リポジトリのルートで）:
    python programs_review/programs/build_public_repo.py --target ../ASA-SVRG          # results/ を作り直す
    python programs_review/programs/build_public_repo.py --target ../ASA-SVRG --code   # （最初の作成のみ）プログラムもコピーする
"""

import argparse
import json
import os
import shutil

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_REVIEW_PROGRAMS = os.path.join(_REPO_ROOT, "programs_review", "programs")
_REVIEW_OUTPUTS = os.path.join(_REPO_ROOT, "programs_review", "outputs")

# (元のプログラムのディレクトリ, 元の結果のディレクトリ, 公開版の相対パス, 旧リポジトリの実験か)
EXPERIMENTS = [
    ("programs_review/programs/ex006_e1_mushroom_rho_sampling", "programs_review/outputs/ex006_e1_mushroom_rho_sampling",
     "mushroom/rho_sampling", False),
    ("programs_review/programs/ex008_e4_a9a", "programs_review/outputs/ex008_e4_a9a", "a9a/main", False),
    ("programs_review/programs/ex014_e8_a9a_supplement", "programs_review/outputs/ex014_e8_a9a_supplement",
     "a9a/supplement", False),
    ("programs_review/programs/ex015_e10_a9a_bce_uniform", "programs_review/outputs/ex015_e10_a9a_bce_uniform",
     "a9a/bce_uniform", False),
    ("programs/ex0024_cifar10_alexnet_layernorm_longrun", "outputs/ex0024_cifar10_alexnet_layernorm_longrun",
     "cifar10/batch_sizes", True),
    ("programs_review/programs/ex002_e0_cifar10_alexnet_layernorm", "programs_review/outputs/ex002_e0_cifar10_alexnet_layernorm",
     "cifar10/reset_continue_bs128", False),
    ("programs_review/programs/ex007_e2_cifar10_snapshot_gradient_factorial",
     "programs_review/outputs/ex007_e2_cifar10_snapshot_gradient_factorial", "cifar10/snapshot_factorial_bs128", False),
    ("programs_review/programs/ex007_e2_cifar10_ablation", "programs_review/outputs/ex007_e2_cifar10_ablation",
     "cifar10/svrg_average_continue_bs128", False),
    ("programs_review/programs/ex004_appb_cifar10_alexnet_layernorm",
     "programs_review/outputs/ex004_appb_cifar10_alexnet_layernorm", "cifar10/asa_reset_bs32_512", False),
    ("programs_review/programs/ex011_e6_cifar10_final_bs32_512", "programs_review/outputs/ex011_e6_cifar10_final_bs32_512",
     "cifar10/final_snapshot_bs32_512", False),
    ("programs/ex0051_imagewoof_resnet18_bs128_longrun", "outputs/ex0051_imagewoof_resnet18_bs128_longrun",
     "imagewoof/batch_sizes", True),
    ("programs_review/programs/ex005_appb_imagewoof_resnet18_layernorm",
     "programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm", "imagewoof/asa_reset_bs32_512", False),
    ("programs_review/programs/ex012_e7_imagewoof_final", "programs_review/outputs/ex012_e7_imagewoof_final",
     "imagewoof/final_snapshot", False),
    ("programs_review/programs/ex016_imagewoof_bs128_walltime", "programs_review/outputs/ex016_imagewoof_bs128_walltime",
     "imagewoof/walltime_bs128", False),
    ("programs_review/programs/ex013_e7_epoch_time", "programs_review/outputs/ex013_e7_epoch_time",
     "timing/epoch_time", False),
]

# 公開版にコピーしないファイル（旧リポジトリの結果との比較など，内部の検証用のスクリプトと作業記録）
EXCLUDED_FILES = {"compare_e4.py", "compare_e10.py", "check_consistency.py", "README.md"}

# programs_review の実験：experiments/{データセット}/{実験名}/ に置くと，programs の直下より1段深くなる
REVIEW_REPLACEMENTS = [
    ("_PROGRAMS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
     "_PROGRAMS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"),
    ("_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), \"common\")",
     "_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), \"common\")"),
    ("_PROGRAMS_DIR = os.path.dirname(_THIS_DIR)", "_PROGRAMS_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))"),
    ("_COMMON_DIR = os.path.join(os.path.dirname(_THIS_DIR), \"common\")",
     "_COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), \"common\")"),
]

# 旧リポジトリの実験：programs/{実験}/ から experiments/{データセット}/{実験名}/ に移し，
# `.ai/ai-dev-kit/src` と `programs/optimizers` の代わりに experiments/common を使う（内容は同一．README 参照）
LEGACY_REPLACEMENTS = [
    ("_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
     "_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))"),
    ("_PROGRAMS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
     "_PROGRAMS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"),
    ("sys.path.insert(0, os.path.join(_PROJECT_ROOT, \".ai\", \"ai-dev-kit\", \"src\"))",
     "sys.path.insert(0, os.path.join(_PROJECT_ROOT, \"experiments\", \"common\"))"),
    ("sys.path.insert(0, _PROGRAMS_DIR)\n", "sys.path.insert(0, os.path.join(_PROGRAMS_DIR, \"common\"))\n"),
]


# 旧リポジトリの実験の `__main__`：作業の途中で未実行のフェーズだけを実行するように書き換えてあったものを，
# 全てのフェーズを実行するように戻す（完了済みの条件は `is_run_completed` により飛ばされる）
MAIN_REPLACEMENTS = {
    "cifar10/batch_sizes": [(
        "    # 基本グリッド（60条件）は完了済みのため，SGD倍エポック追加学習（order_038）のみ実行する．\n"
        "    main(run_grid=False, run_sgd_double=True)",
        "    # 全てのフェーズ（基本グリッドと SGD のエポック数を倍にした学習）を実行する．完了済みの条件は飛ばす．\n"
        "    main(run_grid=True, run_sgd_double=True)")],
    "imagewoof/batch_sizes": [(
        "    # 基準条件・条件A・条件B・SGD倍エポック追加学習は全て完了済みのため，条件C\n"
        "    # （バッチサイズ32，order_040）のみ実行する。\n"
        "    main(run_bs128=False, run_bs512=False, run_bs32=True, run_sgd_double=False)",
        "    # 全てのフェーズ（バッチサイズ128・512・32 と SGD のエポック数を倍にした学習）を実行する．完了済みの条件は飛ばす．\n"
        "    main(run_bs128=True, run_bs512=True, run_bs32=True, run_sgd_double=True)")],
}


_PAPER_PROGRAMS = os.path.join(os.path.dirname(_REPO_ROOT), "ASAI_SVRG_Paper", "programs")

# 論文の図表を作るモジュール：(元のファイル, 公開版の paper/ の下の名前)．
# 元のファイル名は作業の報告の番号に由来するため，公開版では内容が分かる名前に変える．
PAPER_MODULES = [
    (os.path.join(_PAPER_PROGRAMS, "paper_figure_style.py"), "paper_figure_style.py"),
    (os.path.join(_PAPER_PROGRAMS, "grad_count.py"), "grad_count.py"),
    (os.path.join(_PAPER_PROGRAMS, "aggregate_existing_results.py"), "nonconvex_results.py"),
    (os.path.join(_PAPER_PROGRAMS, "make_e1_paper_assets.py"), "mushroom_assets.py"),
    (os.path.join(_PAPER_PROGRAMS, "make_report090_e1.py"), "fig_mushroom.py"),
    (os.path.join(_PAPER_PROGRAMS, "analyze_e1_gradient_variance.py"), "mushroom_gradient_variance.py"),
    (os.path.join(_PAPER_PROGRAMS, "make_report091_e2.py"), "fig_cifar10_bs128.py"),
    (os.path.join(_PAPER_PROGRAMS, "make_report098_e8.py"), "fig_a9a.py"),
    (os.path.join(_PAPER_PROGRAMS, "make_report100_e6_e7.py"), "fig_batch_sizes.py"),
    (os.path.join(_PAPER_PROGRAMS, "make_report114.py"), "fig_imagewoof_walltime.py"),
    (os.path.join(_REVIEW_PROGRAMS, "analysis_e11.py"), "mushroom_configs.py"),
    (os.path.join(_REVIEW_PROGRAMS, "analysis_e10.py"), "a9a_uniform_floor.py"),
]
PAPER_IMPORT_REPLACEMENTS = [
    ("import aggregate_existing_results as", "import nonconvex_results as"),
    ("import make_e1_paper_assets as", "import mushroom_assets as"),
    ("import make_report090_e1 as", "import fig_mushroom as"),
    ("import analyze_e1_gradient_variance as", "import mushroom_gradient_variance as"),
    ("import make_report091_e2 as", "import fig_cifar10_bs128 as"),
    ("import make_report098_e8 as", "import fig_a9a as"),
    ("import make_report100_e6_e7 as", "import fig_batch_sizes as"),
    ("import make_report114 as", "import fig_imagewoof_walltime as"),
    ("import analysis_e11 as", "import mushroom_configs as"),
    # TeX の表の組版（nolta.cls でのコンパイル）は公開版に含めない．図の関数はこのモジュールを使わない
    ("import make_report113_tables as t113  # noqa: E402\n", ""),
]
# 結果の場所：実験用リポジトリの programs_review/outputs と outputs は，公開版では results にまとめる
PAPER_PATH_REPLACEMENTS = [
    ("os.path.join(repo, \"programs_review\", \"outputs\")", "os.path.join(repo, \"results\")"),
    ("os.path.join(repo, \"outputs\", ", "os.path.join(repo, \"results\", "),
    ("OUTPUTS_DIR = os.path.join(REVIEW_ROOT, \"outputs\")", "OUTPUTS_DIR = os.path.join(REVIEW_ROOT, \"results\")"),
    ("OUTPUTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), \"outputs\")",
     "OUTPUTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), \"results\")"),
    ("EXPERIMENT_DIR = os.path.join(_THIS_DIR, \"ex015_e10_a9a_bce_uniform\")",
     "EXPERIMENT_DIR = os.path.join(os.path.dirname(_THIS_DIR), \"experiments\", \"a9a\", \"bce_uniform\")"),
]


def experiment_name_replacements() -> list:
    """
    概要: 元の実験名（結果のディレクトリ名）を公開版の相対パスに置き換える組を返す（長い名前から順に置き換える）．
    引数: なし．
    戻り値: replacements (list of tuple)．
    """
    pairs = [(f"\"{os.path.basename(src_out)}\"", f"\"{relative}\"") for _, src_out, relative, _ in EXPERIMENTS]
    return sorted(pairs, key=lambda p: -len(p[0]))


def build_paper(target: str) -> None:
    """
    概要: 論文の図表を作るモジュールを paper/ にコピーし，モジュール名・結果のパス・実験名を公開版に合わせて書き換える．
    引数: target (str)．公開版のルート．
    戻り値: なし．
    """
    dst_dir = os.path.join(target, "paper")
    os.makedirs(dst_dir, exist_ok=True)
    replacements = PAPER_IMPORT_REPLACEMENTS + PAPER_PATH_REPLACEMENTS + experiment_name_replacements()
    for src, name in PAPER_MODULES:
        with open(src, encoding="utf-8") as f:
            text, count = patch_source(f.read(), replacements)
        with open(os.path.join(dst_dir, name), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"paper/{name:32s} <- {os.path.basename(src):36s} [{count}]")


def output_replacements(relative: str, legacy: bool) -> list:
    """
    概要: 結果の保存先を `results/{データセット}/{実験名}` に書き換える置換の組を返す．
        旧リポジトリの実験は，生データの保存先も `datasets/{データセット}/raw` に書き換える．
    引数: relative (str)．公開版の相対パス（例: "mushroom/rho_sampling"）．legacy (bool)．旧リポジトリの実験か．
    戻り値: replacements (list of tuple)．(置換前, 置換後) の組のリスト．
    """
    parts = ", ".join(f"\"{p}\"" for p in relative.split("/"))
    if legacy:
        # 生データも他の実験と同じ datasets/{データセット}/raw に置く（同じファイルを重複してダウンロードしない）
        dataset = relative.split("/")[0]
        return [("os.path.join(_PROJECT_ROOT, \"outputs\", EXPERIMENT_NAME)",
                 f"os.path.join(_PROJECT_ROOT, \"results\", {parts})"),
                ("os.path.join(_PROJECT_ROOT, \"datasets\", EXPERIMENT_NAME, \"raw\")",
                 f"os.path.join(_PROJECT_ROOT, \"datasets\", \"{dataset}\", \"raw\")")]
    return [("os.path.join(_REVIEW_ROOT, \"outputs\", EXPERIMENT_NAME)", f"os.path.join(_REVIEW_ROOT, \"results\", {parts})")]


def patch_source(text: str, replacements: list) -> tuple:
    """
    概要: ソースに置換を順に適用し，適用した回数を数える．
    引数: text (str)．ソース．replacements (list of tuple)．(置換前, 置換後) の組のリスト．
    戻り値: (patched, count) の組．patched (str) は置換後のソース，count (int) は置換した箇所の数．
    """
    count = 0
    for old, new in replacements:
        count += text.count(old)
        text = text.replace(old, new)
    return text, count


def copy_program(src_dir: str, dst_dir: str, relative: str, legacy: bool) -> list:
    """
    概要: 1つの実験のプログラム（*.py）をコピーし，パスの計算を書き換える．
    引数: src_dir (str)．元のディレクトリ．dst_dir (str)．公開版のディレクトリ．relative (str)．公開版の相対パス．
        legacy (bool)．旧リポジトリの実験か．
    戻り値: records (list of tuple)．(ファイル名, 置換した箇所の数) のリスト．
    """
    os.makedirs(dst_dir, exist_ok=True)
    replacements = ((LEGACY_REPLACEMENTS if legacy else REVIEW_REPLACEMENTS) + output_replacements(relative, legacy)
                    + MAIN_REPLACEMENTS.get(relative, []))
    records = []
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".py") or name in EXCLUDED_FILES:
            continue
        with open(os.path.join(src_dir, name), encoding="utf-8") as f:
            text, count = patch_source(f.read(), replacements)
        with open(os.path.join(dst_dir, name), "w", encoding="utf-8") as f:
            f.write(text)
        records.append((name, count))
    return records


def copy_common(target: str) -> None:
    """
    概要: 共通モジュールを experiments/common にコピーする（__pycache__ は除く）．
    引数: target (str)．公開版のルート．
    戻り値: なし．
    """
    dst = os.path.join(target, "experiments", "common")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(_REVIEW_PROGRAMS, "common"), dst, ignore=shutil.ignore_patterns("__pycache__"))


# 結果の並べ替え：(元の実験の公開版の相対パス, 種類)．種類は保存先の決め方を表す
RESULT_KINDS = {
    "mushroom/rho_sampling": "mushroom", "a9a/main": "a9a", "a9a/supplement": "a9a", "a9a/bce_uniform": "a9a",
    "cifar10/batch_sizes": "legacy", "imagewoof/batch_sizes": "legacy", "timing/epoch_time": "timing",
}
# 論文で使った組と重複するため並べ替え後に残さない試行：(元の実験, 判定の関数)．
# 値が同じ重複（ビット一致または先頭部分）と，論文で使っていない別の実行（別の GPU）を除く．
EXCLUDED_RUNS = [
    # random スナップショット・reset：reset_continue_bs128 の試行を使う（こちらは RTX 3090 での別の実行）
    ("cifar10/snapshot_factorial_bs128", lambda c: c["method"] in ("SVRG", "NFG_SVRG", "ASAI_SVRG")),
    # ミニバッチ128 の分散縮小法（continue）：reset_continue_bs128 の試行とビット一致
    ("cifar10/batch_sizes", lambda c: c["batch_size"] == 128 and c["method"] != "SGD"),
    # SGD：論文はエポック数を倍にした試行を使う（短い試行はその先頭部分とビット一致）
    ("cifar10/batch_sizes", lambda c: c["method"] == "SGD" and c["epochs"] in (12, 48, 192)),
    # ImageWoof・ミニバッチ128：walltime_bs128 の試行を使う（final_snapshot の bs128 はビット一致．
    # batch_sizes の SGD・ASA-SVRG は論文で使っていない別の実行）
    ("imagewoof/final_snapshot", lambda c: c["batch_size"] == 128),
    ("imagewoof/batch_sizes", lambda c: c["batch_size"] == 128 and c["method"] in ("SGD", "ASAI_SVRG")),
    ("imagewoof/batch_sizes", lambda c: c["method"] == "SGD" and c["batch_size"] == 512 and c["epochs"] == 505),
]
IMAGE_METHODS = {"SGD": ("SGD", None), "SVRG": ("SVRG", "random"), "SVRG_final": ("SVRG", "final"),
                 "SVRG_average": ("SVRG", "average"), "NFG_SVRG": ("NFG-SVRG", "random"),
                 "NFG_SVRG_final": ("NFG-SVRG", "final"), "ASAI_SVRG": ("ASA-SVRG", "average")}
CONVEX_METHODS = {"SGD": ("SGD", None), "SVRG": ("SVRG", "final"), "NFG_SVRG": ("NFG-SVRG", "final"),
                  "SVRG_random": ("SVRG", "random"), "NFG_SVRG_random": ("NFG-SVRG", "random"),
                  "SVRG_average": ("SVRG", "average"), "ASAI_SVRG": ("ASA-SVRG", "average")}
START_NAMES = {"snapshot": "reset", "last_iterate": "continue"}


def method_label(algorithm: str, snapshot, inner_start) -> str:
    """
    概要: 手法のディレクトリ名を作る（公開版の `result_layout.method_label` と同じ規則）．
    引数: algorithm (str)．snapshot (str or None)．inner_start (str or None)．
    戻り値: label (str)．
    """
    if algorithm == "SGD":
        return "SGD"
    if snapshot == "final":
        return f"{algorithm}_final"
    if algorithm == "ASA-SVRG":
        return f"ASA-SVRG_{START_NAMES[inner_start]}"
    return f"{algorithm}_{snapshot}_{START_NAMES[inner_start]}"


def new_run_path(relative: str, run_rel: str, config: dict) -> str:
    """
    概要: 1つの試行の並べ替え後の相対パス（results の下）を返す．
    引数: relative (str)．元の実験の公開版の相対パス．run_rel (str)．元の実験の中の試行の相対パス．config (dict)．
    戻り値: path (str)．`{データセット}/{条件}/{手法}/{シード}` または `.../epoch_time/{手法}/rep{r}`．
    """
    kind = RESULT_KINDS.get(relative, "image")
    dataset = relative.split("/")[0]
    if kind == "mushroom":
        label = method_label(*CONVEX_METHODS[config["method"]], config["inner_start"])
        cond = f"{config['sampling']}_rho{config['target_rho']}_eta{config['eta']:.6e}"
        return f"mushroom/{cond}/{label}/{config['seed']}"
    if kind == "a9a":
        label = method_label(*CONVEX_METHODS[config["method"]], config["inner_start"])
        cond = f"{config['loss']}_{config['sampling']}_eta{config['eta']:.6e}"
        return f"a9a/{cond}/{label}/{config['seed']}"
    cond = f"bs{config['batch_size']}_lr{config['learning_rate']}_lambda{config['reg_lambda']}"
    if kind == "timing":
        label = method_label(*IMAGE_METHODS[config["method"]], config["inner_start"])
        return f"{config['dataset_key']}/{cond}/epoch_time/{label}/rep{config['repeat']}"
    # 旧リポジトリの実験は config に inner_start が無い．分散縮小法は全て continue である
    inner_start = config.get("inner_start", "last_iterate" if kind == "legacy" else None)
    label = method_label(*IMAGE_METHODS[config["method"]], inner_start)
    return f"{dataset}/{cond}/{label}/{config['seed']}"


def build_results(target: str) -> None:
    """
    概要: 全ての実験の結果を統一した配置に並べ替えてコピーする．重複を除き，保存先の衝突が無いことを確かめる．
    引数: target (str)．公開版のルート．
    戻り値: なし．
    """
    results_root = os.path.join(target, "results")
    if os.path.exists(results_root):
        shutil.rmtree(results_root)
    destinations = {}
    for _, src_out, relative, _ in EXPERIMENTS:
        src_root = os.path.join(_REPO_ROOT, src_out)
        n_kept = n_excluded = 0
        for root, _, files in os.walk(src_root):
            if "config.json" not in files or "log.json" not in files:
                continue
            with open(os.path.join(root, "config.json")) as f:
                config = json.load(f)
            if any(rel == relative and rule(config) for rel, rule in EXCLUDED_RUNS):
                n_excluded += 1
                continue
            new_rel = new_run_path(relative, os.path.relpath(root, src_root), config)
            if new_rel in destinations:
                raise RuntimeError(f"保存先が衝突します: {new_rel} <- {destinations[new_rel]} と {root}")
            destinations[new_rel] = root
            os.makedirs(os.path.join(results_root, new_rel))
            shutil.copy2(os.path.join(root, "log.json"), os.path.join(results_root, new_rel, "log.json"))
            with open(os.path.join(root, "config.json"), "rb") as f:
                data = f.read()
            # 公開版では提案手法の名前を ASA-SVRG に統一する（開発時の名前 ASAI_SVRG を ASA_SVRG に置き換える）．
            # 改行コードなど，それ以外のバイト列は元のまま保つ
            with open(os.path.join(results_root, new_rel, "config.json"), "wb") as f:
                f.write(data.replace(b"ASAI_SVRG", b"ASA_SVRG"))
            n_kept += 1
        print(f"{relative:40s} kept {n_kept:4d} runs, excluded {n_excluded:3d} duplicated runs")
    print(f"total {len(destinations)} runs")


def build_code(target: str) -> None:
    """
    概要: （最初の作成のみ）共通モジュールと各実験のプログラム，論文の図表のモジュールをコピーし，パスを書き換える．
    引数: target (str)．公開版のルート．
    戻り値: なし．
    """
    copy_common(target)
    for src_prog, _, relative, legacy in EXPERIMENTS:
        records = copy_program(os.path.join(_REPO_ROOT, src_prog), os.path.join(target, "experiments", relative),
                               relative, legacy)
        patched = ", ".join(f"{name}:{count}" for name, count in records)
        print(f"{relative:40s} <- {os.path.basename(src_prog):48s} [{patched}]")
    build_paper(target)


def main() -> None:
    """
    概要: コマンドライン引数を読み，公開版の結果（と，`--code` の場合はプログラム）を組み立てる．
    引数: なし．
    戻り値: なし．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default=os.path.join(os.path.dirname(_REPO_ROOT), "ASA-SVRG"))
    parser.add_argument("--code", action="store_true", help="プログラムもコピーする（最初の作成のみ）")
    args = parser.parse_args()
    target = os.path.abspath(args.target)
    if args.code:
        build_code(target)
    build_results(target)


if __name__ == "__main__":
    main()
