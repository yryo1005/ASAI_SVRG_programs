"""
E6（最終点のスナップショットの SVRG・NFG-SVRG，CIFAR-10・AlexNet（LayerNorm），バッチサイズ32・512）の実験を実行するスクリプト．

`ex007_e2_cifar10_snapshot_gradient_factorial/train.py` のコピーであり，`SPEC` のみを変えた．
学習の実装（`e2_training.py`）は E2 と同一であり，バッチサイズ128の E2 の `SVRG_final`・`NFG_SVRG_final` と同じ計算を
バッチサイズ32・512で行う．

| 項目 | 設定 |
|---|---|
| モデル | `AlexNetCIFARNorm(norm_type="layernorm")`（`model.py` は ex007 のコピー） |
| 目的関数 | 交差エントロピー + 畳み込み層・全結合層の重みのL2正則化（$ \\lambda = 5\\times10^{-4} $） |
| 学習率 | $ \\eta = 10^{-3} $ |
| ミニバッチサイズとエポック数 | 32（$ K = 1563 $，12エポック），512（$ K = 98 $，192エポック）．付録B（`ex004`）と同じ |
| 手法 | `SVRG_final`（`SVRGFinalPoint`），`NFG_SVRG_final`（`NFGSVRGFinalPoint`） |
| 内側ループの開始点 | `inner_start="snapshot"`（最終点では `last_iterate` と同じ計算） |
| シード | 0〜4 |

実行例（`programs_review/programs` から）:
    python ex011_e6_cifar10_final_bs32_512/train.py --workers 4          # 全条件（2手法 × 2バッチサイズ × 5シード）
    python ex011_e6_cifar10_final_bs32_512/train.py --batch-sizes 32     # バッチサイズ32のみ
    python ex011_e6_cifar10_final_bs32_512/train.py --methods SVRG_final NFG_SVRG_final --seeds 0 --epochs 3 \\
        --batch-sizes 128 --workers 1 --output-root ../outputs/_verify/ex011_bs128   # E2 との一致の確認
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAMS_DIR = os.path.dirname(_THIS_DIR)
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))
sys.path.insert(0, _THIS_DIR)

from e2_runner import run_main  # noqa: E402

EXPERIMENT_NAME = "ex011_e6_cifar10_final_bs32_512"
NORM_TYPE = "layernorm"
SPEC = {
    "experiment": EXPERIMENT_NAME,
    "methods": ["SVRG_final", "NFG_SVRG_final"],
    "seeds": [0, 1, 2, 3, 4],
    "learning_rate": 0.001,
    "epochs_by_batch_size": {32: 12, 512: 192},
    "reg_lambda": 5e-4,
    "output_root": os.path.join(_REVIEW_ROOT, "outputs", EXPERIMENT_NAME),
    "workers": 4,
}


def _build_model(seed: int):
    """
    概要: 指定したシードでモデルを初期化する．
    引数: seed (int)．
    戻り値: model (torch.nn.Module)．
    """
    from model import AlexNetCIFARNorm, load_model

    return load_model(AlexNetCIFARNorm, seed=seed, norm_type=NORM_TYPE)


def build_problem() -> dict:
    """
    概要: 問題の定義（データ，モデル，損失，評価指標）を辞書にまとめる．並列実行の各プロセスで呼ぶ．
    引数: なし
    戻り値: problem (dict)．`e2_training.run_variance_reduced` の引数 `problem`．
    """
    from data import load_dataloader
    from model import loss_func, metrics_func

    return {
        "load_dataloader": load_dataloader,
        "build_model": _build_model,
        "loss_func": loss_func,
        "metrics_func": metrics_func,
        "reg_lambda": SPEC["reg_lambda"],
        "config_extra": {"dataset": "CIFAR-10", "model": "AlexNetCIFARNorm", "norm_type": NORM_TYPE},
    }


if __name__ == "__main__":
    from data import prepare_cache

    run_main(SPEC, build_problem, prepare_cache, __doc__)
