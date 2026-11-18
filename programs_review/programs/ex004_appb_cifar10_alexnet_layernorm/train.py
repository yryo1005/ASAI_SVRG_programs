"""
付録B.2（CIFAR-10・AlexNet（LayerNorm），バッチサイズ32・512）を内側ループの開始点を論文のAlgorithm通り
（`inner_start="snapshot"`，$ {\\mathbf w}_{s}^{0} \\leftarrow {\\mathbf z}_{s} $）として再実施するスクリプト．

条件は既存の実験 `ex0024_cifar10_alexnet_layernorm_longrun` のバッチサイズ32・512と同一であり，開始点のみが異なる．
`last_iterate` の結果とSGDの結果は既存の実験のログを用いる（`analysis_appb.py`）．

| 項目 | 設定 |
|---|---|
| モデル | `AlexNetCIFARNorm(norm_type="layernorm")` |
| 目的関数 | 交差エントロピー + 畳み込み層・全結合層の重みのL2正則化（$ \\lambda = 5\\times10^{-4} $） |
| 学習率 | $ \\eta = 10^{-3} $ |
| ミニバッチサイズとエポック数 | 32（$ K = 1563 $，12エポック），512（$ K = 98 $，192エポック） |
| 予算（図の横軸） | バッチサイズ32：$ \\#\\mathrm{grad}/N \\approx 20 $，512：$ \\approx 350 $ |
| 手法 | SVRG（`SVRG`），NFG-SVRG（`NFGSVRG`），ASA-SVRG（`ASAISVRG`） |
| 内側ループの開始点 | `snapshot`（既定） |
| シード | 0〜4 |

学習損失が非有限となった場合はその外側ループで打ち切る（既存の実験と同じ基準）．

実行例:
    python train.py                                   # 全条件（2バッチサイズ × 3手法 × 5シード）
    python train.py --batch-sizes 32                  # バッチサイズ32のみ
    python train.py --inner-starts last_iterate --batch-sizes 32 512 --seeds 0 --epochs 2 \\
        --output-root ../../outputs/_verify/ex004     # 既存の実装との一致の確認
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAMS_DIR = os.path.dirname(_THIS_DIR)
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))
sys.path.insert(0, _THIS_DIR)

from e0_runner import run_main  # noqa: E402

EXPERIMENT_NAME = "ex004_appb_cifar10_alexnet_layernorm"
NORM_TYPE = "layernorm"
SPEC = {
    "experiment": EXPERIMENT_NAME,
    "methods": ["SVRG", "NFG_SVRG", "ASAI_SVRG"],
    "seeds": [0, 1, 2, 3, 4],
    "learning_rate": 0.001,
    "epochs_by_batch_size": {32: 12, 512: 192},
    "inner_starts": ["snapshot"],
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
