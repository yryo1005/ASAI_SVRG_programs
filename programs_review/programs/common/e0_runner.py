"""
E0（内側ループの開始点の影響確認）と付録Bの再実施（バッチサイズ32・512）の画像分類実験に共通する，
タスクの構築とマルチプロセスでの並列実行．各実験の `train.py` は実験固有の定数と `build_problem` を
定義し，`run_main` を呼ぶ．

`spec` にはバッチサイズとエポック数を次のいずれかで与える．
- `batch_size` と `epochs`：1つのバッチサイズ（E0）．
- `epochs_by_batch_size`：{バッチサイズ: エポック数} の辞書（付録B）．
"""

import argparse
import itertools
import multiprocessing
import os
import time
import traceback

from vr_image_training import INNER_STARTS, is_run_completed, run_single_experiment

# WSL2 では，他のプロセスがGPUを高頻度に使用している間，新しいプロセスのCUDAの初期化が一時的に
# 失敗することがある（dxg の vmbus_sendpacket が EAGAIN を返し，`CUDA error: out of memory` となる）．
# 初期化を待ち時間を空けて再試行し，それでも失敗したタスクは新しいプロセスで再実行する．
CUDA_INIT_ATTEMPTS = 5
CUDA_INIT_WAIT_SECONDS = 20
MAX_ROUNDS = 5


def _init_cuda() -> None:
    """
    概要: CUDAを初期化する．失敗した場合は待ち時間を空けて再試行する．
    引数: なし
    戻り値: なし（`CUDA_INIT_ATTEMPTS` 回失敗した場合は最後の例外を送出する）
    """
    import torch

    if not torch.cuda.is_available():
        return
    for attempt in range(CUDA_INIT_ATTEMPTS):
        try:
            torch.zeros(1, device="cuda")
            return
        except RuntimeError:
            if attempt == CUDA_INIT_ATTEMPTS - 1:
                raise
            time.sleep(CUDA_INIT_WAIT_SECONDS)


def epochs_by_batch_size(spec: dict) -> dict:
    """
    概要: 実験の定義から {バッチサイズ: エポック数} の辞書を得る．
    引数: spec (dict)．実験の定義．
    戻り値: table (dict)．
    """
    if "epochs_by_batch_size" in spec:
        return dict(spec["epochs_by_batch_size"])
    return {spec["batch_size"]: spec["epochs"]}


def hp_name(spec: dict, inner_start: str, epochs: int, batch_size: int = None) -> str:
    """
    概要: 条件のディレクトリ名を構築する．
    引数:
        spec (dict)．実験の定義（`learning_rate`，`reg_lambda`，および `batch_size` を含む）．
        inner_start (str)．内側ループの開始点．
        epochs (int)．外側ループ数．
        batch_size (int) = None．ミニバッチサイズ．`None` の場合は `spec["batch_size"]`．
    戻り値: name (str)．
    """
    batch_size = spec["batch_size"] if batch_size is None else batch_size
    return (f"inner_{inner_start}_lr{spec['learning_rate']}_bs{batch_size}"
            f"_lambda{spec['reg_lambda']}_epochs{epochs}")


def build_tasks(spec: dict, methods, inner_starts, seeds, epochs, output_root: str, batch_sizes=None) -> list:
    """
    概要: 全ての (バッチサイズ, 手法, 内側ループの開始点, シード) の組のタスクを構築する．
    引数:
        spec (dict)．実験の定義（`experiment`，`learning_rate`，`reg_lambda`，バッチサイズとエポック数）．
        methods (list of str)．手法名．
        inner_starts (list of str)．内側ループの開始点．
        seeds (list of int)．シード．
        epochs (int or None)．外側ループ数．`None` の場合はバッチサイズごとの既定値．
        output_root (str)．結果の保存先のルート．
        batch_sizes (list of int) = None．ミニバッチサイズ．`None` の場合は `spec` の全て．
    戻り値: tasks (list of dict)．`vr_image_training.run_single_experiment` の引数 `task`．
        バッチサイズ（`spec` の順）を最も外側，次にシードのループとして並べる．シードごとに
        2つの `inner_start` の結果が揃う．
    """
    table = epochs_by_batch_size(spec)
    batch_sizes = list(table) if batch_sizes is None else list(batch_sizes)
    tasks = []
    for batch_size in batch_sizes:
        n_epochs = table[batch_size] if epochs is None else epochs
        run_config = {"learning_rate": spec["learning_rate"], "batch_size": batch_size, "epochs": n_epochs}
        for seed, inner_start, method in itertools.product(seeds, inner_starts, methods):
            name = hp_name(spec, inner_start, n_epochs, batch_size)
            tasks.append({
                "experiment": spec["experiment"],
                "method": method,
                "inner_start": inner_start,
                "seed": seed,
                "run_config": dict(run_config),
                "target_dir": os.path.join(output_root, method, name, str(seed)),
            })
    return tasks


def parse_args(spec: dict, description: str) -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．既定値は `spec` から取る．
    引数:
        spec (dict)．実験の定義（`methods`，`seeds`，`output_root`，`workers`，任意で `inner_starts`，
            `method_choices`（`--methods` で選べる手法．既定は `methods`））．
        description (str)．ヘルプに表示する説明．
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=spec["methods"],
                        choices=spec.get("method_choices", spec["methods"]))
    parser.add_argument("--inner-starts", nargs="+", default=list(spec.get("inner_starts", INNER_STARTS)),
                        choices=INNER_STARTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=spec["seeds"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None, choices=list(epochs_by_batch_size(spec)),
                        help="既定は全てのバッチサイズ")
    parser.add_argument("--epochs", type=int, default=None, help="既定はバッチサイズごとの既定値")
    parser.add_argument("--output-root", default=spec["output_root"])
    parser.add_argument("--workers", type=int, default=spec["workers"])
    return parser.parse_args()


class _Worker:
    """
    概要: 並列実行の各プロセスで，問題の定義を構築してから1つのタスクを実行する呼び出し可能
        オブジェクト．`spawn` で子プロセスへ渡すため，問題の定義そのものではなく構築関数を保持する．
    """

    def __init__(self, build_problem):
        """
        概要: 初期化する．
        引数: build_problem (callable)．引数なしで問題の定義（dict）を返す，モジュールの
            トップレベルの関数．
        戻り値: なし
        """
        self.build_problem = build_problem

    def __call__(self, task: dict) -> bool:
        """
        概要: 1つのタスクを実行する．例外は送出せずに表示し，他のタスクの実行を続ける．
        引数: task (dict)．`build_tasks` が返すタスク．
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
    引数:
        spec (dict)．実験の定義．
        build_problem (callable)．問題の定義を返すトップレベルの関数．
        prepare_cache (callable)．データのキャッシュを作成する関数．
        description (str)．ヘルプに表示する説明．
    戻り値: なし
    """
    args = parse_args(spec, description)
    prepare_cache()
    tasks = build_tasks(spec, args.methods, args.inner_starts, args.seeds, args.epochs,
                        os.path.abspath(args.output_root), args.batch_sizes)
    num_workers = max(1, min(args.workers, len(tasks)))
    print(f"タスク数: {len(tasks)}，並列プロセス数: {num_workers}", flush=True)
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
