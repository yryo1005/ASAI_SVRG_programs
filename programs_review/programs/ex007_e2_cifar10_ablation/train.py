"""
E2（実験計画4章）：平均勾配を用いる手法のスナップショットと初期値のアブレーション（CIFAR-10・AlexNet（LayerNorm））
のうち，E0 に含まれない条件を実行するスクリプト．

E0（`ex002_e0_cifar10_alexnet_layernorm`）の6条件（SVRG・NFG-SVRG・ASA-SVRG × `last_iterate`・`snapshot`）と
合わせて，アブレーションの表を作る（`analysis_e2.py`）．条件は E0 と同一である．

| 項目 | 設定 |
|---|---|
| モデル | `AlexNetCIFARNorm(norm_type="layernorm")` |
| 目的関数 | 交差エントロピー + 畳み込み層・全結合層の重みのL2正則化（$ \\lambda = 5\\times10^{-4} $） |
| 学習率 | $ \\eta = 10^{-3} $ |
| ミニバッチサイズ | 128（$ K = 391 $） |
| エポック数 | 48 |
| シード | 0〜4 |

既定で実行する条件（実験計画4.2節の必須の2条件）：

| 手法名 | クラス | $ {\\mathbf g}_{s} $ | スナップショット | `inner_start` |
|---|---|---|---|---|
| `NFG_SVRG_final` | `NFGSVRGFinalPoint` | 平均勾配 | 最終点 | `last_iterate`（最終点では `snapshot` と同一の計算） |
| `SVRG_average` | `SVRGAverage` | 全勾配 | 平均点 | `last_iterate` |

任意の条件は `--methods` と `--inner-starts` で指定する（例：`--methods SVRG_average --inner-starts snapshot`，
`--methods SVRG_final`）．全ての条件で $ \\omega $（`omega_snapshot`，`omega_average`）を記録する．

実行例:
    python train.py --workers 10
    python train.py --methods SVRG ASAI_SVRG --seeds 0 --epochs 3 \\
        --output-root ../../outputs/_verify/ex007   # E0 のログとの一致の確認
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAMS_DIR = os.path.dirname(_THIS_DIR)
_REVIEW_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "common"))
sys.path.insert(0, _THIS_DIR)

from e0_runner import run_main  # noqa: E402

EXPERIMENT_NAME = "ex007_e2_cifar10_ablation"
NORM_TYPE = "layernorm"
SPEC = {
    "experiment": EXPERIMENT_NAME,
    "methods": ["NFG_SVRG_final", "SVRG_average"],
    "method_choices": ["NFG_SVRG_final", "SVRG_average", "SVRG_final", "SVRG", "NFG_SVRG", "ASAI_SVRG"],
    "seeds": [0, 1, 2, 3, 4],
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 48,
    "inner_starts": ["last_iterate"],
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
