"""
E7（B）：1エポックの時間の短い計測（ImageWoof・CIFAR-10，バッチサイズ32・128・512）．

`ex009_e3_imagewoof_walltime/train.py` を元にしたスクリプトである．`data.py`，`model.py`，`timed_training.py` は
ex009 から，`data_cifar10.py`，`model_cifar10.py` は `ex007_e2_cifar10_snapshot_gradient_factorial/` の
`data.py`，`model.py` から変更せずにコピーした．

| 項目 | 設定 |
|---|---|
| データセット・モデル | ImageWoof（ResNet18（LayerNorm）），CIFAR-10（AlexNet（LayerNorm）） |
| バッチサイズ | 32，128，512 |
| 手法 | ASA-SVRG（`last_iterate`，continue），NFG-SVRG（最終点），SVRG（最終点）（以上2つは `snapshot`），SGD |
| 学習率・正則化 | $ \\eta = 10^{-3} $，$ \\lambda = 5\\times10^{-4} $ |
| シード | 0 のみ |
| エポック数 | 4（1エポック目は準備運転．2〜4エポック目を計測に使う） |
| 繰り返し | 3周 |

試行の順序：データセットとバッチサイズの組ごとに，4手法を ASA-SVRG → NFG-SVRG → SVRG → SGD の順に交互に3周実行する．
1つの手法の3回の計測を，時間的に離れた3つの時点に分けるためである．失敗した試行は，同じ組の3周が終わった直後に
再実行する．

計測の規則（ex009 と同じ）：全ての試行を1つずつ逐次に，新しいプロセスで実行する．各試行の前に GPU のメモリ使用量が
`--idle-mib` 以下になるまで待つ．`torch.set_num_threads(1)`，ローダーのワーカー数0，cuDNN の決定論的設定．

実行例:
    python train.py                                                    # 72試行
    python train.py --datasets imagewoof --batch-sizes 128 --repeats 1 \\
        --output-root ../../outputs/_verify/ex013                       # 動作の確認
"""

import argparse
import datetime
import json
import multiprocessing
import os
import subprocess
import sys
import time
import traceback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAMS_DIR = os.path.dirname(_THIS_DIR)
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))
sys.path.insert(0, _THIS_DIR)

from e0_runner import MAX_ROUNDS, _init_cuda  # noqa: E402

EXPERIMENT_NAME = "ex013_e7_epoch_time"
METHODS = ["ASAI_SVRG", "NFG_SVRG_final", "SVRG_final", "SGD"]
INNER_START = {"ASAI_SVRG": "last_iterate", "NFG_SVRG_final": "snapshot", "SVRG_final": "snapshot", "SGD": None}
DATASETS = ["imagewoof", "cifar10"]
SPEC = {
    "experiment": EXPERIMENT_NAME,
    "seed": 0,
    "learning_rate": 0.001,
    "batch_sizes": [32, 128, 512],
    "reg_lambda": 5e-4,
    "epochs": 4,
    "repeats": 3,
    "output_root": os.path.join(_REVIEW_ROOT, "outputs", EXPERIMENT_NAME),
}


def _build_imagewoof_model(seed: int):
    """
    概要: 指定したシードで ImageWoof のモデル（ResNet18（LayerNorm））を初期化する．
    引数: seed (int)．
    戻り値: model (torch.nn.Module)．
    """
    from model import ResNet18LayerNorm, load_model

    return load_model(ResNet18LayerNorm, seed=seed)


def _build_cifar10_model(seed: int):
    """
    概要: 指定したシードで CIFAR-10 のモデル（AlexNet（LayerNorm））を初期化する．
    引数: seed (int)．
    戻り値: model (torch.nn.Module)．
    """
    from model_cifar10 import AlexNetCIFARNorm, load_model

    return load_model(AlexNetCIFARNorm, seed=seed, norm_type="layernorm")


def build_problem(dataset: str) -> dict:
    """
    概要: データセットごとの問題の定義（データ，モデル，損失，評価指標）を辞書にまとめる．各プロセスで呼ぶ．
    引数: dataset (str)．"imagewoof" または "cifar10"．
    戻り値: problem (dict)．
    """
    if dataset == "cifar10":
        from data_cifar10 import load_dataloader
        from model_cifar10 import loss_func, metrics_func

        build_model = _build_cifar10_model
        extra = {"dataset": "CIFAR-10", "model": "AlexNetCIFARNorm", "norm_type": "layernorm"}
    else:
        from data import load_dataloader
        from model import loss_func, metrics_func

        build_model = _build_imagewoof_model
        extra = {"dataset": "ImageWoof (imagewoof2-320)", "model": "ResNet18LayerNorm"}
    return {
        "load_dataloader": load_dataloader,
        "build_model": build_model,
        "loss_func": loss_func,
        "metrics_func": metrics_func,
        "reg_lambda": SPEC["reg_lambda"],
        "config_extra": extra,
    }


def prepare_dataset_cache(dataset: str) -> None:
    """
    概要: 指定したデータセットのキャッシュを準備する．
    引数: dataset (str)．"imagewoof" または "cifar10"．
    戻り値: なし
    """
    if dataset == "cifar10":
        from data_cifar10 import prepare_cache
    else:
        from data import prepare_cache
    prepare_cache()


def build_tasks(datasets, batch_sizes, methods, repeats: int, epochs: int, output_root: str) -> list:
    """
    概要: 試行を，データセットとバッチサイズの組ごとに手法を交互に `repeats` 周並べた順序で作る．
    引数: datasets (list of str)，batch_sizes (list of int)，methods (list of str)．この順に交互に並べる．
        repeats (int)．周回の数．epochs (int)．各試行のエポック数．output_root (str)．
    戻り値: tasks (list of dict)．`target_dir` は `{output_root}/{dataset}/bs{bs}/{手法}/rep{r}`．
    """
    tasks = []
    for dataset in datasets:
        for batch_size in batch_sizes:
            for repeat in range(repeats):
                for method in methods:
                    tasks.append({
                        "experiment": EXPERIMENT_NAME,
                        "dataset": dataset,
                        "method": method,
                        "inner_start": INNER_START[method],
                        "seed": SPEC["seed"],
                        "repeat": repeat,
                        "run_config": {"learning_rate": SPEC["learning_rate"], "batch_size": batch_size,
                                       "epochs": epochs, "record_omega": False},
                        "target_dir": os.path.join(output_root, dataset, f"bs{batch_size}", method, f"rep{repeat}"),
                    })
    return tasks


def group_tasks(tasks: list) -> list:
    """
    概要: タスクを (データセット, バッチサイズ) の組ごとに，元の順序を保って分ける．
    引数: tasks (list of dict)．`build_tasks` の戻り値．
    戻り値: groups (list of list of dict)．
    """
    groups, keys = [], []
    for task in tasks:
        key = (task["dataset"], task["run_config"]["batch_size"])
        if not keys or keys[-1] != key:
            keys.append(key)
            groups.append([])
        groups[-1].append(task)
    return groups


def gpu_memory_used_mib() -> int:
    """
    概要: `nvidia-smi` で GPU のメモリ使用量を取得する．
    引数: なし
    戻り値: used (int)．[MiB]．取得できない場合は -1．
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return -1


def wait_for_idle_gpu(idle_mib: int, poll_seconds: int = 60) -> None:
    """
    概要: GPU のメモリ使用量が `idle_mib` 以下になるまで待つ（他の学習が終わるのを待つ）．
    引数: idle_mib (int)．[MiB]．poll_seconds (int) = 60．
    戻り値: なし
    """
    while True:
        used = gpu_memory_used_mib()
        if 0 <= used <= idle_mib:
            return
        print(f"GPU のメモリ使用量 {used} MiB．他のプロセスの終了を待つ．", flush=True)
        time.sleep(poll_seconds)


def _now() -> str:
    """
    概要: 現在時刻を秒までの ISO 形式の文字列で返す．
    引数: なし
    戻り値: text (str)．
    """
    return datetime.datetime.now().isoformat(timespec="seconds")


def finalize_run(task: dict, started_at: str, finished_at: str) -> None:
    """
    概要: 試行の終了後に，`config.json` に周回の番号と開始・終了時刻を加え，`best_model.pth` を削除する．
    引数: task (dict)．started_at (str)．finished_at (str)．
    戻り値: なし
    """
    config_path = os.path.join(task["target_dir"], "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config.update({"dataset_key": task["dataset"], "repeat": task["repeat"],
                   "started_at": started_at, "finished_at": finished_at})
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    model_path = os.path.join(task["target_dir"], "best_model.pth")
    if os.path.exists(model_path):
        os.remove(model_path)


class _Worker:
    """
    概要: 新しいプロセスで1つの試行を実行する呼び出し可能オブジェクト．
    """

    def __call__(self, task: dict) -> bool:
        """
        概要: CUDA を初期化し，1つの試行を計測しながら実行する．例外は表示して False を返す．
        引数: task (dict)．`build_tasks` が返すタスク．
        戻り値: success (bool)．
        """
        from timed_training import run_single_timed

        try:
            _init_cuda()
            started_at = _now()
            run_single_timed(task, build_problem(task["dataset"]))
            finalize_run(task, started_at, _now())
            return True
        except Exception:  # noqa: BLE001  失敗した試行は main が再実行する
            print(f"[failed] {task['target_dir']}\n{traceback.format_exc()}", flush=True)
            return False


def run_task(task: dict, idle_mib: int, context) -> None:
    """
    概要: GPU が空くのを待ってから，1つの試行を新しいプロセスで実行する．
    引数: task (dict)．idle_mib (int)．context (multiprocessing.context.BaseContext)．
    戻り値: なし
    """
    wait_for_idle_gpu(idle_mib)
    print(f"[start] {_now()} {task['dataset']} bs{task['run_config']['batch_size']} rep{task['repeat']} "
          f"{task['method']}", flush=True)
    with context.Pool(processes=1, maxtasksperchild=1) as pool:
        pool.map(_Worker(), [task])


def run_group(group: list, idle_mib: int, context) -> list:
    """
    概要: 1つの組の試行を順に実行し，3周の後に失敗した試行を最大 `MAX_ROUNDS - 1` 回まで再実行する．
    引数: group (list of dict)．idle_mib (int)．context．
    戻り値: remaining (list of dict)．完了しなかった試行．
    """
    from vr_image_training import is_run_completed

    remaining = group
    for round_index in range(MAX_ROUNDS):
        for task in remaining:
            if not is_run_completed(task["target_dir"], task["run_config"]["epochs"]):
                run_task(task, idle_mib, context)
        remaining = [t for t in remaining if not is_run_completed(t["target_dir"], t["run_config"]["epochs"])]
        if not remaining:
            break
        print(f"[retry] 未完了の試行 {len(remaining)} 件を再実行します（{round_index + 2}回目）．", flush=True)
    return remaining


def parse_args(argv=None) -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: argv (list of str) = None．`None` の場合は `sys.argv` を用いる．
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=SPEC["batch_sizes"])
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--repeats", type=int, default=SPEC["repeats"])
    parser.add_argument("--epochs", type=int, default=SPEC["epochs"])
    parser.add_argument("--output-root", default=SPEC["output_root"])
    parser.add_argument("--idle-mib", type=int, default=2000, help="計測を開始してよい GPU のメモリ使用量の上限")
    return parser.parse_args(argv)


def main() -> None:
    """
    概要: データのキャッシュを準備し，組ごとに全ての試行を1つずつ新しいプロセスで逐次に実行する．
    引数: なし
    戻り値: なし
    """
    args = parse_args()
    for dataset in args.datasets:
        prepare_dataset_cache(dataset)
    tasks = build_tasks(args.datasets, args.batch_sizes, args.methods, args.repeats, args.epochs,
                        os.path.abspath(args.output_root))
    print(f"試行数: {len(tasks)}（逐次に実行）", flush=True)
    context = multiprocessing.get_context("spawn")
    unfinished = []
    for group in group_tasks(tasks):
        unfinished += run_group(group, args.idle_mib, context)
    if unfinished:
        print(f"完了しなかった試行: {[t['target_dir'] for t in unfinished]}", flush=True)
    else:
        print("全ての学習が終了しました．", flush=True)


if __name__ == "__main__":
    main()
