"""
`programs/optimizers/` で手法ごとに分割定義された最適化手法クラス（SGD，SVRG，SVRGFinalPoint，
NFGSVRG，NFGSVRGFinalPoint，ASAISVRG）の単体テスト．いずれも `torch.optim.Optimizer` の
サブクラスとして陽実装されている．

分割前の実装（`programs_old/optimizers/optimizers.py`）と挙動が完全に一致することも
`test_split_modules_match_archived_implementation` で確認する．
"""

import importlib.util
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "programs"))

from optimizers import (  # noqa: E402
    ASAISVRG,
    NFGSVRG,
    NFGSVRGFinalPoint,
    SGD,
    SVRG,
    SVRGFinalPoint,
)


def test_optimizers_are_torch_optimizer_subclasses():
    """全クラスがtorch.optim.Optimizerのサブクラスであることを確認する．"""
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    assert isinstance(SGD([param], lr=0.1), torch.optim.Optimizer)
    assert isinstance(SVRG([param], lr=0.1, K=10), torch.optim.Optimizer)
    assert isinstance(SVRGFinalPoint([param], lr=0.1, K=10), torch.optim.Optimizer)
    assert isinstance(NFGSVRG([param], lr=0.1, K=10), torch.optim.Optimizer)
    assert isinstance(NFGSVRGFinalPoint([param], lr=0.1, K=10), torch.optim.Optimizer)
    assert isinstance(ASAISVRG([param], lr=0.1, K=10), torch.optim.Optimizer)


def test_each_optimizer_is_defined_in_its_own_module():
    """`.orders/order_020.md` の指示通り，各最適化手法が手法ごとの独立したモジュールに
    定義されていることを確認する．"""
    module_to_class = {
        "optimizers.sgd": SGD,
        "optimizers.svrg": SVRG,
        "optimizers.svrg_final_point": SVRGFinalPoint,
        "optimizers.nfg_svrg": NFGSVRG,
        "optimizers.nfg_svrg_final_point": NFGSVRGFinalPoint,
        "optimizers.asai_svrg": ASAISVRG,
    }
    for module_name, OptimizerClass in module_to_class.items():
        assert OptimizerClass.__module__ == module_name


def test_sgd_updates_params_correctly():
    """SGDが w_{k+1} = w_k - eta * grad を正しく計算することを確認する．"""
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    optimizer = SGD([param], lr=0.1)
    param.grad = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    optimizer.step()

    expected = torch.tensor([0.9, 1.9, 2.9], dtype=torch.float64)
    assert torch.allclose(param, expected)


def test_svrg_updates_params_correctly():
    """SVRGが v = grad_at_current - grad_at_snapshot + snapshot_gradient を用いて
    w_{k+1} = w_k - eta * v を正しく計算することを確認する．"""
    param = torch.nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float64))
    optimizer = SVRG([param], lr=0.5, K=1)
    optimizer.set_snapshot_gradient([torch.tensor([0.2, 0.2], dtype=torch.float64)])
    optimizer.begin_epoch(np.random.default_rng(0))

    param.grad = torch.tensor([1.0, 0.5], dtype=torch.float64)
    grad_at_snapshot = [torch.tensor([0.8, 0.3], dtype=torch.float64)]
    optimizer.step(grad_at_snapshot)

    # v = (1.0 - 0.8 + 0.2, 0.5 - 0.3 + 0.2) = (0.4, 0.4)
    expected = torch.tensor([1.0 - 0.5 * 0.4, 1.0 - 0.5 * 0.4], dtype=torch.float64)
    assert torch.allclose(param, expected)


def test_svrg_snapshot_selection_matches_target_k():
    """SVRGが，begin_epochで選ばれたステップ番号における更新前のパラメータ値を
    候補スナップショットとして正しく保持することを確認する．"""
    param = torch.nn.Parameter(torch.tensor([10.0], dtype=torch.float64))
    optimizer = SVRG([param], lr=0.0, K=3)  # lr=0としてパラメータ自体は変化させない
    optimizer.set_snapshot_gradient([torch.zeros(1, dtype=torch.float64)])

    class FixedRng:
        def integers(self, low, high):
            return 1  # 常にk=1を選ぶ

    optimizer.begin_epoch(FixedRng())

    for v in [10.0, 20.0, 30.0]:
        param.data.fill_(v)
        param.grad = torch.zeros(1, dtype=torch.float64)
        optimizer.step([torch.zeros(1, dtype=torch.float64)])

    snapshot = optimizer.get_snapshot_params()[0]
    assert torch.allclose(snapshot, torch.tensor([20.0], dtype=torch.float64))


def test_svrgfinalpoint_snapshot_is_the_last_trajectory_point():
    """SVRGFinalPointが，NFG SVRG原論文が比較対象とする古典的SVRGの実装通り，次エポックの
    スナップショットとして内部ループの最終パラメータを採用することを確認する（SVRGのような
    一様ランダム選択ではない）．"""
    param = torch.nn.Parameter(torch.tensor([10.0], dtype=torch.float64))
    optimizer = SVRGFinalPoint([param], lr=0.0, K=3)  # lr=0としてパラメータ自体は変化させない
    optimizer.set_snapshot_gradient([torch.zeros(1, dtype=torch.float64)])
    optimizer.begin_epoch()

    for v in [10.0, 20.0, 30.0]:
        param.data.fill_(v)
        param.grad = torch.zeros(1, dtype=torch.float64)
        optimizer.step([torch.zeros(1, dtype=torch.float64)])

    optimizer.end_epoch()
    snapshot = optimizer.get_snapshot_params()[0]
    assert torch.allclose(snapshot, torch.tensor([30.0], dtype=torch.float64))


def test_nfgsvrg_running_average_gradient_matches_numpy_mean():
    """NFGSVRGのend_epoch後のスナップショット勾配が，内部ループで観測した勾配の
    単純平均と一致することを確認する（ASAI SVRG論文の式(8)）．"""
    param = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
    optimizer = NFGSVRG([param], lr=0.0, K=4)
    optimizer.begin_epoch(np.random.default_rng(0))

    observed_grads = [
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.tensor([3.0, 4.0], dtype=torch.float64),
        torch.tensor([5.0, 6.0], dtype=torch.float64),
        torch.tensor([7.0, 8.0], dtype=torch.float64),
    ]
    for g in observed_grads:
        param.grad = g.clone()
        optimizer.step([torch.zeros(2, dtype=torch.float64)])

    optimizer.end_epoch()

    expected = torch.stack(observed_grads).mean(dim=0)
    assert torch.allclose(optimizer.get_snapshot_gradient()[0], expected)


def test_nfgsvrg_finalpoint_snapshot_is_the_last_trajectory_point():
    """NFGSVRGFinalPointが，NFG SVRG原論文Algorithm 1（11〜12行目，omega_{s+1}=x_s^n）通り，
    次エポックのスナップショットとして内部ループの最終パラメータを採用することを確認する
    （NFGSVRGのような一様ランダム選択ではない）．"""
    param = torch.nn.Parameter(torch.tensor([10.0], dtype=torch.float64))
    optimizer = NFGSVRGFinalPoint([param], lr=0.0, K=3)  # lr=0としてパラメータ自体は変化させない
    optimizer.begin_epoch()

    for v in [10.0, 20.0, 30.0]:
        param.data.fill_(v)
        param.grad = torch.zeros(1, dtype=torch.float64)
        optimizer.step([torch.zeros(1, dtype=torch.float64)])

    optimizer.end_epoch()
    snapshot = optimizer.get_snapshot_params()[0]
    assert torch.allclose(snapshot, torch.tensor([30.0], dtype=torch.float64))


def test_nfgsvrg_finalpoint_running_average_gradient_matches_numpy_mean():
    """NFGSVRGFinalPointの平均勾配（NFG SVRG原論文の式(6)）が，NFGSVRGと同様に単純平均と
    一致することを確認する．"""
    param = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
    optimizer = NFGSVRGFinalPoint([param], lr=0.0, K=4)
    optimizer.begin_epoch()

    observed_grads = [
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.tensor([3.0, 4.0], dtype=torch.float64),
        torch.tensor([5.0, 6.0], dtype=torch.float64),
        torch.tensor([7.0, 8.0], dtype=torch.float64),
    ]
    for g in observed_grads:
        param.grad = g.clone()
        optimizer.step([torch.zeros(2, dtype=torch.float64)])

    optimizer.end_epoch()

    expected = torch.stack(observed_grads).mean(dim=0)
    assert torch.allclose(optimizer.get_snapshot_gradient()[0], expected)


def test_asaisvrg_running_average_params_matches_numpy_mean():
    """ASAISVRGのend_epoch後のスナップショット点が，内部ループにおけるパラメータ列
    （更新前の値）の単純平均と一致することを確認する（式(13)）．"""
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    K = 3
    optimizer = ASAISVRG([param], lr=0.1, K=K)

    observed_params = []
    for _ in range(K):
        observed_params.append(param.detach().clone())
        param.grad = torch.zeros(1, dtype=torch.float64)
        optimizer.step([torch.zeros(1, dtype=torch.float64)])

    optimizer.end_epoch()

    expected = torch.stack(observed_params).mean(dim=0)
    assert torch.allclose(optimizer.get_snapshot_params()[0], expected)


def test_asaisvrg_first_epoch_does_not_move_parameters():
    """ASAISVRGの第1エポックでは z_0 = w_0 かつ g_0 = 0 であるため，補正勾配が常に0となり，
    パラメータが更新されないことを確認する（ASAI SVRG論文3.1節）．"""
    param = torch.nn.Parameter(torch.tensor([5.0, -3.0], dtype=torch.float64))
    optimizer = ASAISVRG([param], lr=0.1, K=5)
    optimizer.begin_epoch(np.random.default_rng(0))

    initial_value = param.detach().clone()
    for _ in range(5):
        grad = torch.tensor([1.23, -4.56], dtype=torch.float64)
        param.grad = grad.clone()
        # grad_at_snapshot は z_0 = w_0 における同一サンプルの勾配であるため grad と一致する
        optimizer.step([grad.clone()])

    assert torch.allclose(param, initial_value)


def test_svrg_reduces_to_gradient_descent_when_variance_is_zero():
    """スナップショットが現在点と一致する場合，SVRGの更新は通常の勾配降下法に一致することを
    2次関数 f(w) = 0.5 * ||w||^2 の最小化で確認する．"""
    param = torch.nn.Parameter(torch.tensor([10.0, -5.0], dtype=torch.float64))
    optimizer = SVRG([param], lr=0.1, K=1)

    for _ in range(200):
        optimizer.begin_epoch(np.random.default_rng(0))
        grad = param.detach().clone()
        optimizer.set_snapshot_gradient([grad])
        param.grad = grad.clone()
        optimizer.step([grad.clone()])

    assert torch.allclose(param, torch.zeros(2, dtype=torch.float64), atol=1e-6)


def test_split_modules_match_archived_implementation():
    """手法ごとに分割した実装が，分割前の `programs_old/optimizers/optimizers.py` の実装と
    同一の更新結果を与えることを確認する．"""
    # `programs/optimizers` パッケージと名前が衝突しないよう，ファイルパスを直接指定して読み込む．
    archived_path = os.path.join(_PROJECT_ROOT, "programs_old", "optimizers", "optimizers.py")
    spec = importlib.util.spec_from_file_location("archived_optimizers", archived_path)
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    ArchivedSVRGFinal = archived.SVRGFinalPoint
    ArchivedNFGFinal = archived.NFGSVRGFinalPoint
    ArchivedASAISVRG = archived.ASAISVRG

    K = 6
    grads = [torch.tensor([0.5 * k, -0.25 * k], dtype=torch.float64) for k in range(1, K + 1)]
    snapshot_grads = [torch.tensor([0.1, 0.2], dtype=torch.float64) for _ in range(K)]

    for NewClass, ArchivedClass in [
        (SVRGFinalPoint, ArchivedSVRGFinal),
        (NFGSVRGFinalPoint, ArchivedNFGFinal),
        (ASAISVRG, ArchivedASAISVRG),
    ]:
        results = []
        for OptimizerClass in (NewClass, ArchivedClass):
            param = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float64))
            optimizer = OptimizerClass([param], lr=0.3, K=K)
            optimizer.begin_epoch(np.random.default_rng(0))
            for g, g_snap in zip(grads, snapshot_grads):
                param.grad = g.clone()
                optimizer.step([g_snap.clone()])
            optimizer.end_epoch()
            results.append(
                (
                    param.detach().clone(),
                    optimizer.get_snapshot_params()[0],
                    optimizer.get_snapshot_gradient()[0],
                )
            )

        for new_tensor, archived_tensor in zip(*results):
            assert torch.allclose(new_tensor, archived_tensor)
