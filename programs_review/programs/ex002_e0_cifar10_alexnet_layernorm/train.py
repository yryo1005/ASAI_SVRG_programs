"""
E0（内側ループの開始点の影響確認）のCIFAR-10・AlexNet（LayerNorm）の実験を実行するスクリプト．

条件は論文4.2節（既存の実験 `ex0024_cifar10_alexnet_layernorm_longrun` のバッチサイズ128）と同一である．

| 項目 | 設定 |
|---|---|
| モデル | `AlexNetCIFARNorm(norm_type="layernorm")` |
| 目的関数 | 交差エントロピー + 畳み込み層・全結合層の重みのL2正則化（$ \\lambda = 5\\times10^{-4} $） |
| 学習率 | $ \\eta = 10^{-3} $ |
| ミニバッチサイズ | 128（$ K = 391 $） |
| エポック数 | 48 |
| 手法 | SVRG（`SVRG`），NFG-SVRG（`NFGSVRG`），ASA-SVRG（`ASAISVRG`） |
| 要因 | `inner_start` ∈ {"last_iterate", "snapshot"} |
| シード | 0〜4 |

実行例:
    python train.py                                  # 全条件（3手法 × 2 × 5シード）
    python train.py --inner-starts last_iterate --seeds 0 --epochs 3 --output-root ../../outputs/_verify/ex002
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAMS_DIR = os.path.dirname(_THIS_DIR)
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))
sys.path.insert(0, _THIS_DIR)

from e0_runner import run_main  # noqa: E402

EXPERIMENT_NAME = "ex002_e0_cifar10_alexnet_layernorm"
NORM_TYPE = "layernorm"
SPEC = {
    "experiment": EXPERIMENT_NAME,
    "methods": ["SVRG", "NFG_SVRG", "ASAI_SVRG"],
    "seeds": [0, 1, 2, 3, 4],
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 48,
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
    from model import AlexNetCIFARNorm, load_model

    return load_model(AlexNetCIFARNorm, seed=seed, norm_type=NORM_TYPE)


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
        "config_extra": {"dataset": "CIFAR-10", "model": "AlexNetCIFARNorm", "norm_type": NORM_TYPE},
    }


if __name__ == "__main__":
    from data import prepare_cache

    run_main(SPEC, build_problem, prepare_cache, __doc__)
