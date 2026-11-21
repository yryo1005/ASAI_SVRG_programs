"""
E7（A）：ImageWoof・ResNet18（LayerNorm）における最終点のスナップショットの SVRG・NFG-SVRG の精度の実験．

`data.py`，`model.py` は `ex003_e0_imagewoof_resnet18_layernorm/` から変更せずにコピーした．
学習ループは `common/vr_image_training.py`（`run_single_experiment`）をそのまま用いる．

| 項目 | 設定 |
|---|---|
| データ・モデル | ImageWoof，ResNet18（LayerNorm）（`load_model(ResNet18LayerNorm, seed=seed)`） |
| 目的関数 | 交差エントロピー + 重みの L2 正則化（$ \\lambda = 5\\times10^{-4} $） |
| 学習率 | $ \\eta = 10^{-3} $ |
| 手法 | SVRG（最終点）（`SVRG_final`），NFG-SVRG（最終点）（`NFG_SVRG_final`） |
| 内側ループの開始点 | `snapshot`（最終点では `last_iterate` と同一の計算） |
| バッチサイズとエポック数 | 手法ごとに予算 $ \\#\\mathrm{grad}/N $（32：30，128：200，512：1000）に届く最小限（`EPOCHS`） |
| シード | 0〜4 |

学習損失が非有限となった場合はその外側ループで打ち切る（既定）．

実行例:
    python train.py --workers 10                         # 30試行（2手法 × 3バッチサイズ × 5シード）
    python train.py --methods SVRG --seeds 0 --epochs 2 --batch-sizes 32 128 512 --workers 1 \\
        --output-root ../../outputs/_verify/ex012        # 一致の確認（ex003・ex005 の SVRG と比べる）
"""

import argparse
import multiprocessing
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAMS_DIR = os.path.dirname(_THIS_DIR)
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))
sys.path.insert(0, _THIS_DIR)

from e0_runner import MAX_ROUNDS, _Worker, build_tasks  # noqa: E402

EXPERIMENT_NAME = "ex012_e7_imagewoof_final"
METHOD_CHOICES = ["SVRG_final", "NFG_SVRG_final", "SVRG"]
# (手法, バッチサイズ) ごとのエポック数．SVRG は初期 N・1エポック 3N，NFG-SVRG は1エポック 2N の勾配評価を数える．
EPOCHS = {
    ("SVRG_final", 32): 11, ("NFG_SVRG_final", 32): 16,
    ("SVRG_final", 128): 67, ("NFG_SVRG_final", 128): 100,
    ("SVRG_final", 512): 334, ("NFG_SVRG_final", 512): 500,
}
SPEC = {
    "experiment": EXPERIMENT_NAME,
    "methods": ["SVRG_final", "NFG_SVRG_final"],
    "seeds": [0, 1, 2, 3, 4],
    "learning_rate": 0.001,
    "batch_sizes": [32, 128, 512],
    # `common/e0_runner.build_tasks` が参照する表．エポック数を明示して呼ぶため，値は使われない．
    "epochs_by_batch_size": {32: 11, 128: 67, 512: 334},
    "reg_lambda": 5e-4,
    "output_root": os.path.join(_REVIEW_ROOT, "outputs", EXPERIMENT_NAME),
    "workers": 10,
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
    概要: 問題の定義（データ，モデル，損失，評価指標）を辞書にまとめる．並列実行の各プロセスで呼ぶ．
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


def epochs_for(method: str, batch_size: int, epochs_override: int = None) -> int:
    """
    概要: 手法とバッチサイズの組のエポック数を返す．
    引数: method (str)，batch_size (int)，epochs_override (int) = None．指定した場合はこの値を返す（確認用）．
    戻り値: epochs (int)．
    """
    if epochs_override is not None:
        return epochs_override
    if (method, batch_size) not in EPOCHS:
        raise ValueError(f"({method}, {batch_size}) のエポック数は定義されていない．--epochs で指定すること．")
    return EPOCHS[(method, batch_size)]


def build_e7_tasks(methods, batch_sizes, seeds, output_root: str, epochs_override: int = None) -> list:
    """
    概要: 手法とバッチサイズの組ごとに `common/e0_runner.build_tasks` でタスクを作り，連結する．
    引数: methods (list of str)，batch_sizes (list of int)，seeds (list of int)，output_root (str)，
        epochs_override (int) = None．指定した場合は全てのタスクのエポック数をこの値にする（確認用）．
    戻り値: tasks (list of dict)．バッチサイズ，手法の順に並べる．
    """
    tasks = []
    for batch_size in batch_sizes:
        for method in methods:
            epochs = epochs_for(method, batch_size, epochs_override)
            tasks += build_tasks(SPEC, [method], ["snapshot"], seeds, epochs, output_root, batch_sizes=[batch_size])
    return tasks


def order_longest_first(tasks: list) -> list:
    """
    概要: 並列実行の終盤に長い試行だけが残らないように，タスクをバッチサイズの大きい順（1試行の時間の長い順）に
        並べ替える．同じバッチサイズの中では元の順序を保つ．
    引数: tasks (list of dict)．
    戻り値: ordered (list of dict)．
    """
    return sorted(tasks, key=lambda task: -task["run_config"]["batch_size"])


def parse_args(argv=None) -> argparse.Namespace:
    """
    概要: コマンドライン引数を解釈する．
    引数: argv (list of str) = None．`None` の場合は `sys.argv` を用いる．
    戻り値: args (argparse.Namespace)．
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=SPEC["methods"], choices=METHOD_CHOICES)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=SPEC["batch_sizes"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SPEC["seeds"])
    parser.add_argument("--epochs", type=int, default=None, help="確認用：全てのタスクのエポック数を上書きする")
    parser.add_argument("--output-root", default=SPEC["output_root"])
    parser.add_argument("--workers", type=int, default=SPEC["workers"])
    return parser.parse_args(argv)


def run_main() -> None:
    """
    概要: `common/e0_runner.run_main` と同じ処理で，データのキャッシュを準備した後，全タスクを並列に実行する．
        タスクの作り方のみ `build_e7_tasks` に替えた．各タスクを新しいプロセスで実行し，未完了のタスクを
        最大 `MAX_ROUNDS` 回まで再実行する．
    引数: なし
    戻り値: なし
    """
    from data import prepare_cache
    from vr_image_training import is_run_completed

    args = parse_args()
    prepare_cache()
    tasks = order_longest_first(build_e7_tasks(args.methods, args.batch_sizes, args.seeds,
                                               os.path.abspath(args.output_root), args.epochs))
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


if __name__ == "__main__":
    run_main()
