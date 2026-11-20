"""
E3（実験計画5章）：ImageWoof・ResNet18（LayerNorm）における実時間の計測．

| 項目 | 設定 |
|---|---|
| データ・モデル・目的関数 | E0 の ImageWoof（`ex003_e0_imagewoof_resnet18_layernorm` の `data.py`，`model.py`），$ \\lambda = 5\\times10^{-4} $ |
| 学習率・ミニバッチサイズ | $ \\eta = 10^{-3} $，128（$ K = 71 $） |
| 手法 | SGD，SVRG（ランダム点），NFG-SVRG（ランダム点），ASA-SVRG（平均点）．SVRG系は `inner_start="last_iterate"` |
| 計算予算 | $ \\#\\mathrm{grad}/N \\approx 200 $（SGD 200，NFG-SVRG・ASA-SVRG 100，SVRG 67エポック） |
| シード | 0〜4（既定）．当初は0〜2で実施し，他の実験と揃えるため3・4を追加した |

計測の規則（実験計画5.3節）：
- 全ての試行を1つずつ逐次に実行する．各試行は新しいプロセスで実行する．
- 各試行の開始前に GPU のメモリ使用量が `--idle-mib` 以下になるまで待つ．他のプロセスが GPU を使っていないことを確かめるためである．
- `torch.set_num_threads(1)`，ローダーのワーカー数0，cuDNN の決定論的設定（`set_seed`）を全手法で共通とし，`config.json` に記録する．
- 試行の順序はシードごとに4手法を並べる（SGD，SVRG，NFG-SVRG，ASA-SVRG の順）．時間とともに計測環境が変化した場合の偏りを手法間で揃えるためである．

実行例:
    python train.py                                    # 5シード × 4手法 = 20試行（完了済みの試行は飛ばす）
    python train.py --methods SVRG NFG_SVRG ASAI_SVRG --seeds 0 --epochs-override 2 \\
        --output-root ../../outputs/_verify/ex009      # E0 のログとの一致の確認
"""

import argparse
import itertools
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

EXPERIMENT_NAME = "ex009_e3_imagewoof_walltime"
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
EPOCHS = {"SGD": 200, "SVRG": 67, "NFG_SVRG": 100, "ASAI_SVRG": 100}
SPEC = {
    "experiment": EXPERIMENT_NAME,
    "seeds": [0, 1, 2, 3, 4],
    "learning_rate": 0.001,
    "batch_size": 128,
    "reg_lambda": 5e-4,
    "inner_start": "last_iterate",
    "output_root": os.path.join(_REVIEW_ROOT, "outputs", EXPERIMENT_NAME),
}


def _build_model(seed: int):
    """
    概要: 指定したシードでモデルを初期化する．
    引数: seed (int)．
    戻り値: model (torch.nn.Module)．
    """
    from model import ResNet18LayerNorm, load_model

    return load_model(ResNet18LayerNorm, seed=seed)


def build_problem() -> dict:
    """
    概要: 問題の定義（データ，モデル，損失，評価指標）を辞書にまとめる．各プロセスで呼ぶ．
    引数: なし
    戻り値: problem (dict)．
    """
    from data import load_dataloader
    from model import loss_func, metrics_func

    return {
        "load_dataloader": load_dataloader,
        "build_model": _build_model,
        "loss_func": loss_func,
        "metrics_func": metrics_func,
        "reg_lambda": SPEC["reg_lambda"],
        "config_extra": {"dataset": "ImageWoof (imagewoof2-320)", "model": "ResNet18LayerNorm"},
    }


def condition_name(method: str, epochs: int) -> str:
    """
    概要: 条件のディレクトリ名を返す．
    引数: method (str)，epochs (int)．
    戻り値: name (str)．例：`inner_last_iterate_lr0.001_bs128_lambda0.0005_epochs100`．SGD は `inner_none_...`．
    """
    inner = "none" if method == "SGD" else SPEC["inner_start"]
    return f"inner_{inner}_lr{SPEC['learning_rate']}_bs{SPEC['batch_size']}_lambda{SPEC['reg_lambda']}_epochs{epochs}"


def build_tasks(methods, seeds, output_root, epochs_override=None) -> list:
    """
    概要: 実行するタスクを，シードごとに手法を並べた順序で作る．
    引数: methods (list of str)，seeds (list of int)，output_root (str)，epochs_override (int) = None．
        指定した場合は全手法のエポック数をこの値にする（確認用）．
    戻り値: tasks (list of dict)．
    """
    tasks = []
    for seed, method in itertools.product(seeds, methods):
        epochs = epochs_override or EPOCHS[method]
        tasks.append({
            "experiment": EXPERIMENT_NAME,
            "method": method,
            "inner_start": None if method == "SGD" else SPEC["inner_start"],
            "seed": seed,
            "run_config": {"learning_rate": SPEC["learning_rate"], "batch_size": SPEC["batch_size"],
                           "epochs": epochs, "record_omega": False},
            "target_dir": os.path.join(output_root, method, condition_name(method, epochs), str(seed)),
        })
    return tasks


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
            print(f"GPU のメモリ使用量 {used} MiB（閾値 {idle_mib} MiB 以下）．計測を開始する．", flush=True)
            return
        print(f"GPU のメモリ使用量 {used} MiB．他のプロセスの終了を待つ．", flush=True)
        time.sleep(poll_seconds)


class _Worker:
    """
    概要: 新しいプロセスで1つのタスクを実行する呼び出し可能オブジェクト．
    """

    def __call__(self, task: dict) -> bool:
        """
        概要: CUDA を初期化し，1つのタスクを計測しながら実行する．例外は表示して False を返す．
        引数: task (dict)．
        戻り値: success (bool)．
        """
        from timed_training import run_single_timed

        try:
            _init_cuda()
            run_single_timed(task, build_problem())
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
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SPEC["seeds"])
    parser.add_argument("--epochs-override", type=int, default=None, help="確認用：全手法のエポック数")
    parser.add_argument("--output-root", default=SPEC["output_root"])
    parser.add_argument("--idle-mib", type=int, default=2000, help="計測を開始してよい GPU のメモリ使用量の上限")
    return parser.parse_args()


def main() -> None:
    """
    概要: データのキャッシュを準備し，全タスクを1つずつ新しいプロセスで逐次に実行する．
        失敗したタスクは最大 `MAX_ROUNDS` 回まで再実行する．
    引数: なし
    戻り値: なし
    """
    from data import prepare_cache
    from vr_image_training import is_run_completed

    args = parse_args()
    prepare_cache()
    tasks = build_tasks(args.methods, args.seeds, os.path.abspath(args.output_root), args.epochs_override)
    print(f"タスク数: {len(tasks)}（逐次に実行）", flush=True)
    context = multiprocessing.get_context("spawn")
    remaining = tasks
    for round_index in range(MAX_ROUNDS):
        for task in remaining:
            if is_run_completed(task["target_dir"], task["run_config"]["epochs"]):
                continue
            wait_for_idle_gpu(args.idle_mib)
            with context.Pool(processes=1, maxtasksperchild=1) as pool:
                pool.map(_Worker(), [task])
        remaining = [t for t in remaining if not is_run_completed(t["target_dir"], t["run_config"]["epochs"])]
        if not remaining:
            break
        print(f"未完了のタスク {len(remaining)} 件を再実行します（{round_index + 2}回目）．", flush=True)
    if remaining:
        print(f"完了しなかったタスク: {[t['target_dir'] for t in remaining]}", flush=True)
    else:
        print("全ての学習が終了しました．", flush=True)


if __name__ == "__main__":
    main()
