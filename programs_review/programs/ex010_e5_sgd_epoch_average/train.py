"""
E5（実験計画8章）：エポック内の平均点でも評価する SGD の参考実験．

| 項目 | 設定 |
|---|---|
| 手法 | SGD のみ．最終点 $ {\\mathbf w}^{K} $ と，エポック内の反復の平均点 $ \\bar{\\mathbf w} $ の両方で評価する（`e5_training.py`） |
| 学習率・正則化 | $ \\eta = 10^{-3} $，$ \\lambda = 5\\times10^{-4} $ |
| データ・モデル | ImageWoof・ResNet18（LayerNorm，`ex009_e3_imagewoof_walltime`），CIFAR-10・AlexNet（LayerNorm，`ex002_e0_cifar10_alexnet_layernorm`） |
| 設定（優先度の順） | 1. ImageWoof・128・200エポック，2. CIFAR-10・128・90エポック，3. CIFAR-10・512・350エポック，ImageWoof・512・1000エポック |
| シード | 0〜4 |

時間を計測する実験ではないため，複数の試行を並列に実行する（`--workers`）．タスクは優先度の順に並べ，
空いたプロセスから順に実行する．完了済みの試行は飛ばす．

出力：`outputs/ex010_e5_sgd_epoch_average/{dataset}/SGD/lr0.001_bs{B}_lambda0.0005_epochs{E}/{seed}/`．

実行例:
    python train.py                                              # 全設定（優先度1〜3）× 5シード
    python train.py --settings imagewoof:128 cifar10:128         # 優先度1・2のみ
    python train.py --settings imagewoof:128 cifar10:128 --seeds 0 --epochs-override 5 --workers 2 \\
        --output-root ../../outputs/_verify/ex010               # 一致の確認（実験計画8.4節）
"""

import argparse
import itertools
import multiprocessing
import os
import sys
import traceback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAMS_DIR = os.path.dirname(_THIS_DIR)
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))
sys.path.insert(0, _THIS_DIR)

from e0_runner import MAX_ROUNDS, _init_cuda  # noqa: E402

EXPERIMENT_NAME = "ex010_e5_sgd_epoch_average"
# (データセット, バッチサイズ) -> エポック数．並びが優先度の順である．
SETTINGS = {
    ("imagewoof", 128): 200,
    ("cifar10", 128): 90,
    ("cifar10", 512): 350,
    ("imagewoof", 512): 1000,
}
SPEC = {
    "experiment": EXPERIMENT_NAME,
    "seeds": [0, 1, 2, 3, 4],
    "learning_rate": 0.001,
    "reg_lambda": 5e-4,
    "output_root": os.path.join(_REVIEW_ROOT, "outputs", EXPERIMENT_NAME),
    "workers": 6,
}


def condition_name(batch_size: int, epochs: int) -> str:
    """
    概要: 条件のディレクトリ名を返す．
    引数: batch_size (int)，epochs (int)．
    戻り値: name (str)．例：`lr0.001_bs128_lambda0.0005_epochs200`．
    """
    return f"lr{SPEC['learning_rate']}_bs{batch_size}_lambda{SPEC['reg_lambda']}_epochs{epochs}"


def parse_setting(text: str) -> tuple:
    """
    概要: `imagewoof:128` の形の文字列を (データセット, バッチサイズ) に変換する．
    引数: text (str)．
    戻り値: setting (tuple of (str, int))．
    """
    dataset, batch_size = text.split(":")
    setting = (dataset, int(batch_size))
    if setting not in SETTINGS:
        raise argparse.ArgumentTypeError(f"未定義の設定: {text}（候補: {list(SETTINGS)}）")
    return setting


def build_tasks(settings, seeds, output_root, epochs_override=None) -> list:
    """
    概要: 設定（優先度の順）とシードの全ての組のタスクを作る．
    引数:
        settings (list of tuple)．(データセット, バッチサイズ) の列．
        seeds (list of int)．
        output_root (str)．
        epochs_override (int) = None．指定した場合は全設定のエポック数をこの値にする（確認用）．
    戻り値: tasks (list of dict)．`e5_training.run_single` の引数 `task`．
    """
    tasks = []
    for (dataset, batch_size), seed in itertools.product(settings, seeds):
        epochs = epochs_override or SETTINGS[(dataset, batch_size)]
        tasks.append({
            "experiment": EXPERIMENT_NAME,
            "dataset": dataset,
            "seed": seed,
            "run_config": {"learning_rate": SPEC["learning_rate"], "batch_size": batch_size, "epochs": epochs},
            "target_dir": os.path.join(output_root, dataset, "SGD", condition_name(batch_size, epochs), str(seed)),
        })
    return tasks


def run_task(task: dict) -> bool:
    """
    概要: 新しいプロセスで CUDA を初期化し，1つのタスクを実行する．例外は表示して False を返す．
    引数: task (dict)．
    戻り値: success (bool)．
    """
    from e5_training import run_single
    from problems import build_problem

    try:
        _init_cuda()
        run_single(task, build_problem(task["dataset"]))
        return True
    except Exception:  # noqa: BLE001  失敗したタスクは main が再実行する
        print(f"[failed] {task['target_dir']}\n{traceback.format_exc()}", flush=True)
        return False


def parse_args() -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: なし
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", nargs="+", type=parse_setting, default=list(SETTINGS),
                        help="dataset:batch_size の列（既定は全設定，優先度の順）")
    parser.add_argument("--seeds", nargs="+", type=int, default=SPEC["seeds"])
    parser.add_argument("--epochs-override", type=int, default=None, help="確認用：全設定のエポック数")
    parser.add_argument("--output-root", default=SPEC["output_root"])
    parser.add_argument("--workers", type=int, default=SPEC["workers"])
    return parser.parse_args()


def main() -> None:
    """
    概要: データのキャッシュを準備し，全タスクを優先度の順に並列に実行する．失敗したタスクは
        最大 `MAX_ROUNDS` 回まで再実行する．
    引数: なし
    戻り値: なし
    """
    from problems import build_problem
    from vr_image_training import is_run_completed

    args = parse_args()
    for dataset in dict.fromkeys(d for d, _ in args.settings):
        build_problem(dataset)["prepare_cache"]()
    tasks = build_tasks(args.settings, args.seeds, os.path.abspath(args.output_root), args.epochs_override)
    print(f"タスク数: {len(tasks)}，並列数: {args.workers}", flush=True)
    context = multiprocessing.get_context("spawn")
    remaining = tasks
    for round_index in range(MAX_ROUNDS):
        pending = [t for t in remaining if not is_run_completed(t["target_dir"], t["run_config"]["epochs"])]
        if not pending:
            break
        if round_index > 0:
            print(f"未完了のタスク {len(pending)} 件を再実行します（{round_index + 1}回目）．", flush=True)
        with context.Pool(processes=min(args.workers, len(pending)), maxtasksperchild=1) as pool:
            list(pool.imap(run_task, pending, chunksize=1))
        remaining = pending
    remaining = [t for t in tasks if not is_run_completed(t["target_dir"], t["run_config"]["epochs"])]
    if remaining:
        print(f"完了しなかったタスク: {[t['target_dir'] for t in remaining]}", flush=True)
    else:
        print("全ての学習が終了しました．", flush=True)


if __name__ == "__main__":
    main()
