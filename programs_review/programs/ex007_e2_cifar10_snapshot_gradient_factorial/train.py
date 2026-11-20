"""
E2（スナップショットの選択方法と全勾配の推定方法の要因分離，CIFAR-10・AlexNet（LayerNorm））の実験を実行するスクリプト．

条件は E0 の CIFAR-10（`ex002_e0_cifar10_alexnet_layernorm`，論文4.2節）と同一である．

| 項目 | 設定 |
|---|---|
| モデル | `AlexNetCIFARNorm(norm_type="layernorm")`（`model.py` は ex002 のコピー） |
| 目的関数 | 交差エントロピー + 畳み込み層・全結合層の重みのL2正則化（$ \\lambda = 5\\times10^{-4} $） |
| 学習率 | $ \\eta = 10^{-3} $ |
| ミニバッチサイズ | 128（$ K = 391 $） |
| エポック数 | 48 |
| 内側ループの開始点 | `inner_start="snapshot"` |
| シード | 0〜4 |

2×3 の要因配置の6手法：

| | 最終点 | ランダム点 | 平均点 |
|---|---|---|---|
| g_s：全勾配 | `SVRG_final`（`SVRGFinalPoint`） | `SVRG`（`SVRG`） | `SVRG_average`（`SVRGAverage`） |
| g_s：平均勾配 | `NFG_SVRG_final`（`NFGSVRGFinalPoint`） | `NFG_SVRG`（`NFGSVRG`） | `ASAI_SVRG`（`ASAISVRG`） |

実行例:
    python train.py --workers 4                         # 全条件（6手法 × 5シード = 30試行）
    python train.py --methods SVRG NFG_SVRG ASAI_SVRG --seeds 0 --epochs 3 --output-root ../../outputs/_verify/ex007
    python train.py --methods ASAI_SVRG --seeds 0 --epochs 2 --no-omega --output-root ../../outputs/_verify/ex007_omega_off
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAMS_DIR = os.path.dirname(_THIS_DIR)
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))
sys.path.insert(0, _THIS_DIR)

from e2_runner import run_main  # noqa: E402
from e2_training import METHODS  # noqa: E402

EXPERIMENT_NAME = "ex007_e2_cifar10_snapshot_gradient_factorial"
NORM_TYPE = "layernorm"
SPEC = {
    "experiment": EXPERIMENT_NAME,
    "methods": list(METHODS),
    "seeds": [0, 1, 2, 3, 4],
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 48,
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
