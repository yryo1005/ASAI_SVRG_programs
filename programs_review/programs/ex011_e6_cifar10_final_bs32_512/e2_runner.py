"""
E6 のタスクの構築とマルチプロセスでの並列実行（`ex007_e2_cifar10_snapshot_gradient_factorial/e2_runner.py` のコピー）．

ex007 からの変更点（2点のみ）：
1. `--batch-sizes`（既定は `None` で全バッチサイズ）を加え，`build_e2_tasks` の引数 `batch_sizes` から
   `common/e0_runner.build_tasks` に渡す．値を `epochs_by_batch_size` のキーに制限しない（`--epochs` と併用して
   バッチサイズ128で E2 との一致を確認するため）．
2. `--methods` の選択肢を `e2_training.METHODS`（6手法）とする．既定値は `spec["methods"]` のままである．

以下は ex007 の説明である．

E2 のタスクの構築とマルチプロセスでの並列実行．

`common/e0_runner.py` の `run_main` は `common/vr_image_training.run_single_experiment` を呼ぶため，
E2 の学習ループ（`e2_training.run_single_experiment`）を呼ぶ版をここに置く（`e0_runner.py` からのコピー）．
タスクの構築（`build_tasks`，`hp_name`）と CUDA の初期化の再試行（`_init_cuda`）は `common/e0_runner.py` のものを
そのまま使う．失敗したタスクを新しいプロセスで再実行する仕組みも `e0_runner.run_main` と同じである．

`e0_runner` からの変更点：
- 呼び出す学習の関数を `e2_training.run_single_experiment` とする．
- `--inner-starts` の選択肢を "snapshot" のみとする．
- 確認のための引数 `--no-omega`（$ \\omega $ を記録しない）を加える．
"""

import argparse
import multiprocessing
import os
import sys
import traceback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.join(os.path.dirname(_THIS_DIR), "common")
for _path in (_COMMON_DIR, _THIS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from e0_runner import MAX_ROUNDS, _init_cuda, build_tasks, hp_name  # noqa: E402,F401
from e2_training import INNER_STARTS, METHODS, is_run_completed, run_single_experiment  # noqa: E402


def parse_args(spec: dict, description: str, argv=None) -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．既定値は `spec` から取る．
    引数:
        spec (dict)．実験の定義（`methods`，`seeds`，`output_root`，`workers`）．
        description (str)．ヘルプに表示する説明．
        argv (list of str) = None．解釈する引数．`None` の場合は `sys.argv[1:]`．
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=spec["methods"], choices=list(METHODS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None, help="既定は spec の全てのバッチサイズ")
    parser.add_argument("--inner-starts", nargs="+", default=list(INNER_STARTS), choices=INNER_STARTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=spec["seeds"])
    parser.add_argument("--epochs", type=int, default=None, help="既定は spec のエポック数")
    parser.add_argument("--output-root", default=spec["output_root"])
    parser.add_argument("--workers", type=int, default=spec["workers"])
    parser.add_argument("--no-omega", action="store_true",
                        help="ω を記録しない（記録が学習に影響しないことの確認用）")
    return parser.parse_args(argv)


def build_e2_tasks(spec: dict, methods, seeds, epochs, output_root: str, record_omega: bool = True,
                   inner_starts=INNER_STARTS, batch_sizes=None) -> list:
    """
    概要: 全ての (手法, シード) の組のタスクを `e0_runner.build_tasks` で構築し，`record_omega` を加える．
    引数:
        spec (dict)．実験の定義（`experiment`，`learning_rate`，`reg_lambda`，`epochs_by_batch_size`）．
        methods (list of str)．手法名．
        seeds (list of int)．シード．
        epochs (int or None)．外側ループ数．`None` の場合は `spec["epochs"]`．
        output_root (str)．結果の保存先のルート．
        record_omega (bool) = True．$ \\omega $ を記録するか．
        inner_starts (tuple of str) = ("snapshot",)．内側ループの開始点．
        batch_sizes (list of int) = None．ミニバッチサイズ．`None` の場合は `spec["epochs_by_batch_size"]` の全て．
    戻り値: tasks (list of dict)．`e2_training.run_single_experiment` の引数 `task`．
        保存先は `{output_root}/{手法}/{条件}/{seed}` である．
    """
    tasks = build_tasks(spec, methods, inner_starts, seeds, epochs, output_root, batch_sizes=batch_sizes)
    for task in tasks:
        task["record_omega"] = record_omega
    return tasks


class _Worker:
    """
    概要: 並列実行の各プロセスで，問題の定義を構築してから1つのタスクを実行する呼び出し可能オブジェクト
        （`e0_runner._Worker` のコピー．E2 の学習を呼ぶ）．
    """

    def __init__(self, build_problem):
        """
        概要: 初期化する．
        引数: build_problem (callable)．引数なしで問題の定義（dict）を返す，モジュールのトップレベルの関数．
        戻り値: なし
        """
        self.build_problem = build_problem

    def __call__(self, task: dict) -> bool:
        """
        概要: 1つのタスクを実行する．例外は送出せずに表示し，他のタスクの実行を続ける．
        引数: task (dict)．`build_e2_tasks` が返すタスク．
        戻り値: success (bool)．
        """
        try:
            _init_cuda()
            run_single_experiment(task, self.build_problem())
            return True
        except Exception:  # noqa: BLE001  失敗したタスクは run_main が再実行する
            print(f"[failed] {task['target_dir']}\n{traceback.format_exc()}", flush=True)
            return False


def run_main(spec: dict, build_problem, prepare_cache, description: str) -> None:
    """
    概要: コマンドライン引数を解釈し，データのキャッシュを準備した後，全タスクを並列に実行する．
        未完了のタスクは最大 `MAX_ROUNDS` 回まで新しいプロセスで再実行する．
    引数:
        spec (dict)．実験の定義．
        build_problem (callable)．問題の定義を返すトップレベルの関数．
        prepare_cache (callable)．データのキャッシュを作成する関数．
        description (str)．ヘルプに表示する説明．
    戻り値: なし
    """
    args = parse_args(spec, description)
    prepare_cache()
    tasks = build_e2_tasks(spec, args.methods, args.seeds, args.epochs, os.path.abspath(args.output_root),
                           record_omega=not args.no_omega, inner_starts=tuple(args.inner_starts),
                           batch_sizes=args.batch_sizes)
    num_workers = max(1, min(args.workers, len(tasks)))
    print(f"タスク数: {len(tasks)}，並列プロセス数: {num_workers}，ω の記録: {not args.no_omega}", flush=True)
    worker = _Worker(build_problem)
    remaining = tasks
    for round_index in range(MAX_ROUNDS):
        # 1タスクごとに新しいプロセスで実行する（CUDAの初期化の失敗が他のタスクに影響しないようにする）．
        with multiprocessing.get_context("spawn").Pool(processes=min(num_workers, len(remaining)),
                                                       maxtasksperchild=1) as pool:
            pool.map(worker, remaining, chunksize=1)
        remaining = [t for t in remaining if not is_run_completed(t["target_dir"], t["run_config"]["epochs"])]
        if not remaining:
            break
        print(f"未完了のタスク {len(remaining)} 件を再実行します（{round_index + 2}回目）．", flush=True)
    if remaining:
        print(f"完了しなかったタスク: {[t['target_dir'] for t in remaining]}", flush=True)
    else:
        print("全ての学習が終了しました．", flush=True)
