"""
ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）のAlgorithm 1〜4，およびNFG SVRG原論文
（`references/No_Full_Grad_SVRG.pdf`）のAlgorithm 1に対応する最適化手法クラス群．

`.orders/order_020.md` の指示に基づき，従来1つのモジュール（`programs_old/optimizers/
optimizers.py`）にまとめられていた実装を，可読性のために手法ごとのモジュールへ分割した．
アルゴリズムの実装内容そのものは従来の実装をそのまま引き継いでいる．

| モジュール | クラス | 対応するアルゴリズム |
| :--- | :--- | :--- |
| `sgd.py` | `SGD` | ASAI SVRG論文 式(4)（オンライン学習のSGD） |
| `svrg.py` | `SVRG` | ASAI SVRG論文 Algorithm 1（スナップショットはランダム選択） |
| `nfg_svrg.py` | `NFGSVRG` | ASAI SVRG論文 Algorithm 2（スナップショットはランダム選択） |
| `asai_svrg.py` | `ASAISVRG` | ASAI SVRG論文 Algorithm 3（提案手法） |
| `svrg_final_point.py` | `SVRGFinalPoint` | NFG SVRG原論文が比較対象とする古典的SVRG（最終点採用） |
| `nfg_svrg_final_point.py` | `NFGSVRGFinalPoint` | NFG SVRG原論文 Algorithm 1（最終点採用） |

いずれのクラスも `torch.optim.Optimizer` のサブクラスであり，`torch.optim.SGD` 等のPyTorch公式
実装は使用せず更新則を陽に実装している．勾配は各パラメータの `.grad` 属性から読み取る．

SVRG系手法は外部ループ・内部ループ構造を持つため，通常の `step()` だけでは1エポック分の処理を
表現できない．そのため，各クラスは `step()` に加えて外部ループの境界で呼び出す
`begin_epoch()`／`end_epoch()`，およびスナップショットを取得する `get_snapshot_params()`／
`get_snapshot_gradient()` を提供する．
"""

from .asai_svrg import ASAISVRG
from .nfg_svrg import NFGSVRG
from .nfg_svrg_final_point import NFGSVRGFinalPoint
from .sgd import SGD
from .svrg import SVRG
from .svrg_final_point import SVRGFinalPoint

__all__ = ["SGD", "SVRG", "SVRGFinalPoint", "NFGSVRG", "NFGSVRGFinalPoint", "ASAISVRG"]
