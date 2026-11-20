"""
E2（スナップショットと初期値のアブレーション）のプログラムのテスト．

- `SVRGAverage` のスナップショットが内側ループの更新前のパラメータの平均であり，
  $ {\\mathbf g}_{s} $ をそろえれば `ASAISVRG` と同じスナップショットになること．
- `OmegaAccumulator` の逐次計算が直接計算と一致し，平均点で最小となること．
- $ \\omega $ の記録の有無で学習の結果が変わらないこと．
- 勾配評価回数の増分が，全勾配の手法で初期 $ N $・1エポック $ 3N $，平均勾配の手法で $ 2N $ であること．
- 新しい手法でも `inner_start="snapshot"` の内側ループが $ {\\mathbf z}_{s} $ から始まること．
"""

import os
import sys

import pytest
import torch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REVIEW_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_REVIEW_ROOT, "programs", "common"))
sys.path.insert(0, _TESTS_DIR)

from machine_learning_utils import ResultLogger  # noqa: E402
from optimizers import ASAISVRG, SVRGAverage  # noqa: E402
import vr_image_training as vr  # noqa: E402
from test_e0_programs import _build_model, _make_problem, _run  # noqa: E402

NEW_METHODS = ("SVRG_average", "SVRG_final", "NFG_SVRG_final")


def _one_epoch(optimizer_class, seed=0, steps=5, lr=0.05):
    """
    概要: 小さなモデルで内側ループを1回実行し，更新前のパラメータの列とスナップショットを返す．
    引数: optimizer_class (type)．`SVRGAverage` または `ASAISVRG`．seed (int)．steps (int)．lr (float)．
    戻り値: (points, snapshot) の組．points は各ステップの更新前のパラメータのリスト，
        snapshot は `get_snapshot_params()` の値．
    """
    model = _build_model(seed)
    optimizer = optimizer_class(model.parameters(), lr=lr, K=steps)
    g = torch.Generator().manual_seed(seed + 1)
    optimizer.begin_epoch(None)
    points = []
    for _ in range(steps):
        points.append([p.detach().clone() for p in model.parameters()])
        for p in model.parameters():
            p.grad = torch.randn(p.shape, generator=g)
        grad_at_snapshot = [torch.randn(p.shape, generator=g) for p in model.parameters()]
        optimizer.step(grad_at_snapshot)
    optimizer.end_epoch()
    return points, optimizer.get_snapshot_params()


def test_svrg_average_snapshot_is_mean_of_iterates():
    """
    概要: スナップショットが更新前のパラメータの平均と一致し，$ {\\mathbf g}_{s} = {\\mathbf 0} $ の
        `ASAISVRG` のスナップショットとビット単位で一致することを確認する．
    引数: なし
    戻り値: なし
    """
    points, snapshot = _one_epoch(SVRGAverage)
    for i, z in enumerate(snapshot):
        mean = torch.stack([pt[i] for pt in points]).mean(0)
        assert torch.allclose(z, mean, atol=1e-6)
    _, snapshot_asa = _one_epoch(ASAISVRG)
    for a, b in zip(snapshot, snapshot_asa):
        assert torch.equal(a, b)


def test_omega_accumulator_matches_direct_computation():
    """
    概要: $ \\omega $ の逐次計算を直接計算と比べ，平均点で最小値となることを確認する．
    引数: なし
    戻り値: なし
    """
    g = torch.Generator().manual_seed(0)
    points = [[torch.randn(3, 2, generator=g) + 5.0, torch.randn(4, generator=g)] for _ in range(7)]
    acc = vr.OmegaAccumulator()
    for pt in points:
        acc.add(pt)
    flat = torch.stack([torch.cat([t.reshape(-1) for t in pt]).double() for pt in points])
    z_params = points[3]
    z = torch.cat([t.reshape(-1) for t in z_params]).double()
    direct_snapshot = float(((flat - z) ** 2).sum())
    direct_average = float(((flat - flat.mean(0)) ** 2).sum())
    omega_snapshot, omega_average = acc.omega(z_params)
    assert omega_snapshot == pytest.approx(direct_snapshot, rel=1e-10)
    assert omega_average == pytest.approx(direct_average, rel=1e-10)
    assert omega_average <= omega_snapshot
    mean_params = [torch.stack([pt[i] for pt in points]).mean(0) for i in range(2)]
    assert acc.omega(mean_params)[0] == pytest.approx(omega_average, rel=1e-6)


def _run_logger(method, inner_start, tmp_path, record_omega):
    """
    概要: 小さな問題で3エポック学習し，ロガーを返す．
    引数: method (str)，inner_start (str)，tmp_path，record_omega (bool)．
    戻り値: logger (ResultLogger)．
    """
    os.makedirs(tmp_path, exist_ok=True)
    logger = ResultLogger()
    logger.set_names(*vr.METRIC_NAMES)
    run_config = {"learning_rate": 0.1, "batch_size": 8, "epochs": 3, "record_omega": record_omega}
    vr.run_variance_reduced(method, inner_start, str(tmp_path), _make_problem([]), run_config,
                            torch.device("cpu"), 0, logger)
    return logger


@pytest.mark.parametrize("method", ["ASAI_SVRG", "SVRG", "SVRG_average", "NFG_SVRG_final"])
def test_omega_recording_does_not_change_training(method, tmp_path):
    """
    概要: $ \\omega $ の記録の有無で，$ \\omega $ 以外の記録がビット単位で一致することを確認する．
    引数: method (str)，tmp_path（pytestのフィクスチャ）．
    戻り値: なし
    """
    with_omega = _run_logger(method, "last_iterate", tmp_path / "a", True)
    without_omega = _run_logger(method, "last_iterate", tmp_path / "b", False)
    for key in ("oracle_calls", "train_loss", "test_accuracy", "approx_error", "full_grad_norm_sq"):
        assert with_omega[key] == without_omega[key]
    omega_snapshot = with_omega["omega_snapshot"]
    omega_average = with_omega["omega_average"]
    assert omega_snapshot[0] != omega_snapshot[0]  # 0エポック目は NaN
    for snap, avg in zip(omega_snapshot[1:], omega_average[1:]):
        assert avg <= snap * (1 + 1e-9) + 1e-12
    if method in ("ASAI_SVRG", "SVRG_average"):
        for snap, avg in zip(omega_snapshot[1:], omega_average[1:]):
            assert snap == pytest.approx(avg, rel=1e-5, abs=1e-9)


@pytest.mark.parametrize("method,per_epoch,initial", [
    ("SVRG_average", 3, 1), ("SVRG_final", 3, 1), ("NFG_SVRG_final", 2, 0)])
def test_oracle_calls_of_new_methods(method, per_epoch, initial, tmp_path):
    """
    概要: 勾配評価回数の増分を確認する（学習データ50件）．
    引数: method (str)，per_epoch (int)，initial (int)，tmp_path．
    戻り値: なし
    """
    logger = _run_logger(method, "last_iterate", tmp_path, True)
    n_train = 50
    expected = [0] + [(initial + per_epoch * s) * n_train for s in range(1, 4)]
    assert logger["oracle_calls"] == expected


@pytest.mark.parametrize("method", NEW_METHODS)
def test_new_methods_snapshot_inner_start(method, tmp_path, monkeypatch):
    """
    概要: `inner_start="snapshot"` で，各内側ループが直前のスナップショットから始まることを確認する．
    引数: method (str)，tmp_path，monkeypatch．
    戻り値: なし
    """
    start_points, snapshots, _ = _run(method, "snapshot", tmp_path, monkeypatch)
    for s in range(1, len(start_points)):
        for p, z in zip(start_points[s], snapshots[s]):
            assert torch.equal(p, z)


def test_final_point_methods_identical_for_both_inner_starts(tmp_path):
    """
    概要: 最終点をスナップショットとする `NFG_SVRG_final` で，2つの `inner_start` の結果が一致することを確認する．
    引数: tmp_path．
    戻り値: なし
    """
    a = _run_logger("NFG_SVRG_final", "last_iterate", tmp_path / "a", True)
    b = _run_logger("NFG_SVRG_final", "snapshot", tmp_path / "b", True)
    for key in ("train_loss", "test_accuracy", "approx_error"):
        assert a[key] == b[key]
