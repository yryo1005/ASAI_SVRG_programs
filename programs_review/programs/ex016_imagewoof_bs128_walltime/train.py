"""
ImageWoof・バッチサイズ128 の5手法（SGD，SVRG（最終点），NFG-SVRG（最終点），ASA-SVRG（continue），ASA-SVRG（reset））を，
全エポックの実時間を計測しながら学習し直すスクリプト（`.orders/order_083.md`）．

`ex013_e7_epoch_time/train.py`（E7 B）の逐次実行の仕組みを元にした．`data.py`，`model.py`，`timed_training.py` は
`ex009_e3_imagewoof_walltime/` と同一のファイル（`ex013` 経由のコピー．変更なし）である．

| 項目 | 設定 |
|---|---|
| データセット・モデル | ImageWoof（imagewoof2-320），ResNet18（LayerNorm） |
| バッチサイズ | 128（$ K = 71 $） |
| 学習率・正則化 | $ \\eta = 10^{-3} $，$ \\lambda = 5\\times10^{-4} $ |
| 手法とエポック数 | SGD 200，SVRG（最終点）67，NFG-SVRG（最終点）100，ASA-SVRG（continue：`last_iterate`）100，ASA-SVRG（reset：`snapshot`）100（予算 $ \\#\\mathrm{grad}/N = 200 $．ex009 と同じ） |
| 内側ループの開始点 | 最終点の2手法は `snapshot`（最終点では開始点によらず同じ計算） |
| シード | 0〜4 |

試行の順序：シードごとに SGD → SVRG（最終点）→ NFG-SVRG（最終点）→ ASA-SVRG（continue）→ ASA-SVRG（reset）の順に
実行し，これをシード0〜4の5周繰り返す．同じ手法の5回の計測を，時間的に離れた5つの時点に分けるためである．
失敗した試行は，全ての周回が終わった後に同じ順序で再実行する．

計測の規則（ex009・ex013 と同じ）：全ての試行を1つずつ逐次に，新しいプロセスで実行する．各試行の前に GPU のメモリ
使用量が `--idle-mib` 以下になるまで待つ．`torch.set_num_threads(1)`，ローダーのワーカー数0，cuDNN の決定論的設定．
実時間は `time_optimizer + time_svrg_full_grad`（評価と診断の時間を除く累積値）であり，`elapsed_time` に記録する．

実行例（`programs_review/programs` から）:
    python ex016_imagewoof_bs128_walltime/train.py                                  # 25試行
    python ex016_imagewoof_bs128_walltime/train.py --seeds 0 --epochs 2 \\
        --output-root ../outputs/_verify/ex016                                      # 動作の確認
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

EXPERIMENT_NAME = "ex016_imagewoof_bs128_walltime"
# 条件名 → (手法, 内側ループの開始点, エポック数)．並びが1周の中の実行順である．
CONDITIONS = {
    "SGD": ("SGD", "none", 200),
    "SVRG_final": ("SVRG_final", "snapshot", 67),
    "NFG_SVRG_final": ("NFG_SVRG_final", "snapshot", 100),
    "ASAI_SVRG_continue": ("ASAI_SVRG", "last_iterate", 100),
    "ASAI_SVRG_reset": ("ASAI_SVRG", "snapshot", 100),
}
SPEC = {
    "experiment": EXPERIMENT_NAME,
    "seeds": [0, 1, 2, 3, 4],
    "learning_rate": 0.001,
    "batch_size": 128,
    "reg_lambda": 5e-4,
    "output_root": os.path.join(_REVIEW_ROOT, "outputs", EXPERIMENT_NAME),
}


def _build_model(seed: int):
    """
    概要: 指定したシードで ResNet18（LayerNorm）を初期化する．
    引数: seed (int)．
    戻り値: model (torch.nn.Module)．
    """
    from model import ResNet18LayerNorm, load_model

    return load_model(ResNet18LayerNorm, seed=seed)


def build_problem() -> dict:
    """
    概要: 問題の定義（データ，モデル，損失，評価指標）を辞書にまとめる．各プロセスで呼ぶ．
    引数: なし
    戻り値: problem (dict)．`vr_image_training.run_variance_reduced` の引数 `problem`．
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


def condition_dir_name(condition: str, epochs: int) -> str:
    """
    概要: 条件のディレクトリ名 `inner_{inner_start}_lr{η}_bs{B}_lambda{λ}_epochs{E}` を返す（ex009 と同じ形式）．
    引数: condition (str)．`CONDITIONS` のキー．epochs (int)．エポック数．
    戻り値: name (str)．
    """
    _, inner_start, _ = CONDITIONS[condition]
    return (f"inner_{inner_start}_lr{SPEC['learning_rate']}_bs{SPEC['batch_size']}"
            f"_lambda{SPEC['reg_lambda']}_epochs{epochs}")


def build_tasks(conditions, seeds, output_root: str, epochs_override: int = None) -> list:
    """
    概要: 試行を，シードごとに条件を `conditions` の順に並べた順序で作る．
    引数:
        conditions (list of str)．`CONDITIONS` のキー．この順に1周の中で実行する．
        seeds (list of int)．周回（シード）の順．
        output_root (str)．結果の保存先のルート．
        epochs_override (int or None)．指定すると全ての条件のエポック数をこの値にする（確認用）．
    戻り値: tasks (list of dict)．`target_dir` は `{output_root}/{手法}/{条件のディレクトリ名}/{seed}`．
    """
    tasks = []
    for seed in seeds:
        for condition in conditions:
            method, inner_start, epochs = CONDITIONS[condition]
            epochs = epochs if epochs_override is None else epochs_override
            tasks.append({
                "experiment": EXPERIMENT_NAME,
                "condition": condition,
                "method": method,
                "inner_start": None if method == "SGD" else inner_start,
                "seed": seed,
                "run_config": {"learning_rate": SPEC["learning_rate"], "batch_size": SPEC["batch_size"],
                               "epochs": epochs, "record_omega": False},
                "target_dir": os.path.join(output_root, method, condition_dir_name(condition, epochs), str(seed)),
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
    概要: 試行の終了後に，`config.json` に条件名と開始・終了時刻を加え，`best_model.pth` を削除する（ex013 と同じ）．
    引数: task (dict)．started_at (str)．finished_at (str)．
    戻り値: なし
    """
    config_path = os.path.join(task["target_dir"], "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config.update({"condition": task["condition"], "started_at": started_at, "finished_at": finished_at})
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
            run_single_timed(task, build_problem())
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
    print(f"[start] {_now()} seed{task['seed']} {task['condition']}", flush=True)
    with context.Pool(processes=1, maxtasksperchild=1) as pool:
        pool.map(_Worker(), [task])


def run_all(tasks: list, idle_mib: int, context) -> list:
    """
    概要: 全ての試行を順に実行し，全周回の後に失敗した試行を最大 `MAX_ROUNDS - 1` 回まで同じ順序で再実行する．
    引数: tasks (list of dict)．idle_mib (int)．context．
    戻り値: remaining (list of dict)．完了しなかった試行．
    """
    from vr_image_training import is_run_completed

    remaining = tasks
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
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SPEC["seeds"])
    parser.add_argument("--epochs", type=int, default=None, help="確認用．全ての条件のエポック数を上書きする")
    parser.add_argument("--output-root", default=SPEC["output_root"])
    parser.add_argument("--idle-mib", type=int, default=2000, help="計測を開始してよい GPU のメモリ使用量の上限")
    return parser.parse_args(argv)


def main() -> None:
    """
    概要: データのキャッシュを準備し，全ての試行を1つずつ新しいプロセスで逐次に実行する．
    引数: なし
    戻り値: なし
    """
    from data import prepare_cache

    args = parse_args()
    prepare_cache()
    tasks = build_tasks(args.conditions, args.seeds, os.path.abspath(args.output_root), args.epochs)
    print(f"試行数: {len(tasks)}（逐次に実行）", flush=True)
    unfinished = run_all(tasks, args.idle_mib, multiprocessing.get_context("spawn"))
    if unfinished:
        print(f"完了しなかった試行: {[t['target_dir'] for t in unfinished]}", flush=True)
    else:
        print("全ての学習が終了しました．", flush=True)


if __name__ == "__main__":
    main()
