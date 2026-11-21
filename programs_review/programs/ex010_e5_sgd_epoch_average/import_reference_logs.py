"""
E5 の比較に用いる既存のログ（論文の表IVの元のログ）を，E5 の出力ディレクトリに複写するスクリプト．

既存のログは `programs_review` の外（実験用リポジトリの `outputs/`）にあるため，パスはコマンドライン引数で
与え，`log.json` と `config.json` だけを `outputs/ex010_e5_sgd_epoch_average/reference/` に複写する．
集計（`analysis_e5.py`）と一致の確認（`compare_e5.py`）は複写したログのみを読む．

| 実験 | 手法 | 条件 |
|---|---|---|
| `ex0024_cifar10_alexnet_layernorm_longrun` | SGD | `lr0.001_bs{128,512}_normlayernorm_lambda0.0005_epochs{96,384}` |
| 同上 | ASAI_SVRG | `lr0.001_bs{128,512}_normlayernorm_lambda0.0005_epochs{48,192}` |
| `ex0051_imagewoof_resnet18_bs128_longrun` | SGD | `lr0.001_bs{128,512}_lambda0.0005_epochs{384,1010}` |
| 同上 | ASAI_SVRG | `lr0.001_bs{128,512}_lambda0.0005_epochs{128,505}` |

使い方:
    python import_reference_logs.py --existing-outputs /path/to/ASAI_SVRG_programs/outputs
"""

import argparse
import json
import os
import shutil

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REVIEW_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
REFERENCE_ROOT = os.path.join(_REVIEW_ROOT, "outputs", "ex010_e5_sgd_epoch_average", "reference")

REFERENCE_CONDITIONS = [
    ("ex0024_cifar10_alexnet_layernorm_longrun", "SGD", "lr0.001_bs128_normlayernorm_lambda0.0005_epochs96"),
    ("ex0024_cifar10_alexnet_layernorm_longrun", "SGD", "lr0.001_bs512_normlayernorm_lambda0.0005_epochs384"),
    ("ex0024_cifar10_alexnet_layernorm_longrun", "ASAI_SVRG", "lr0.001_bs128_normlayernorm_lambda0.0005_epochs48"),
    ("ex0024_cifar10_alexnet_layernorm_longrun", "ASAI_SVRG", "lr0.001_bs512_normlayernorm_lambda0.0005_epochs192"),
    ("ex0051_imagewoof_resnet18_bs128_longrun", "SGD", "lr0.001_bs128_lambda0.0005_epochs384"),
    ("ex0051_imagewoof_resnet18_bs128_longrun", "SGD", "lr0.001_bs512_lambda0.0005_epochs1010"),
    ("ex0051_imagewoof_resnet18_bs128_longrun", "ASAI_SVRG", "lr0.001_bs128_lambda0.0005_epochs128"),
    ("ex0051_imagewoof_resnet18_bs128_longrun", "ASAI_SVRG", "lr0.001_bs512_lambda0.0005_epochs505"),
]
COPIED_FILES = ("log.json", "config.json")


def copy_condition(existing_outputs: str, experiment: str, method: str, condition: str, dest_root: str) -> list:
    """
    概要: 1つの条件の全シードの `log.json` と `config.json` を複写する．
    引数: existing_outputs (str)，experiment (str)，method (str)，condition (str)，dest_root (str)．
    戻り値: copied (list of str)．複写したファイルの，`dest_root` からの相対パス．
    """
    source_dir = os.path.join(existing_outputs, experiment, method, condition)
    copied = []
    for seed_name in sorted(os.listdir(source_dir)):
        for file_name in COPIED_FILES:
            source = os.path.join(source_dir, seed_name, file_name)
            if not os.path.isfile(source):
                continue
            relative = os.path.join(experiment, method, condition, seed_name, file_name)
            os.makedirs(os.path.dirname(os.path.join(dest_root, relative)), exist_ok=True)
            shutil.copy2(source, os.path.join(dest_root, relative))
            copied.append(relative)
    return copied


def main() -> None:
    """
    概要: 全ての参照条件を複写し，複写したファイルの一覧を `reference/manifest.json` に保存する．
    引数: なし（コマンドライン引数を用いる）
    戻り値: なし
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--existing-outputs", required=True, help="実験用リポジトリの outputs/ のパス")
    parser.add_argument("--dest-root", default=REFERENCE_ROOT)
    args = parser.parse_args()
    copied = []
    for experiment, method, condition in REFERENCE_CONDITIONS:
        copied += copy_condition(args.existing_outputs, experiment, method, condition, args.dest_root)
    with open(os.path.join(args.dest_root, "manifest.json"), "w") as f:
        json.dump({"source": "ASAI_SVRG_programs/outputs（実験用リポジトリの既存のログ）", "files": copied},
                  f, indent=2, ensure_ascii=False)
    print(f"{len(copied)} ファイルを {args.dest_root} に複写しました．")


if __name__ == "__main__":
    main()
