"""
E0のプログラムのテスト．

- `CachedTensorLoader` が，既存の実装の `DataLoader(dataset, shuffle=True, generator=...)` と
  同じ順序・同じ値のミニバッチを返し，乱数生成器を同じだけ消費すること．
- `inner_start="snapshot"` のとき，各外側ループの内側ループがスナップショット $ z_s $ から始まり，
  `"last_iterate"` のときは前の内側ループの最終点から始まること．
- 最終点をスナップショットとする手法では，2つの `inner_start` が同じ結果になること．
"""

import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

_REVIEW_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REVIEW_ROOT, "programs", "common"))

from cached_loader import CachedTensorLoader  # noqa: E402
from machine_learning_utils import ResultLogger  # noqa: E402
from optimizers import ASAISVRG, NFGSVRGFinalPoint  # noqa: E402
import vr_image_training as vr  # noqa: E402


def _make_data(n=50, d=4, seed=0):
    """
    概要: テスト用の小さな分類データを作る．
    引数: n (int)．データ数．d (int)．特徴量の次元．seed (int)．乱数シード．
    戻り値: (X, y) の組．X は形状 (n, d) のfloat32，y は形状 (n,) のint64．
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    y = (X[:, 0] > 0).long()
    return X, y


@pytest.mark.parametrize("as_numpy", [False, True])
def test_cached_loader_matches_dataloader(as_numpy):
    """
    概要: 3エポック分のミニバッチの値と順序，および乱数生成器の状態が既存の `DataLoader` と一致することを確認する．
    引数: as_numpy (bool)．入力を `numpy.ndarray` で渡すか．
    戻り値: なし
    """
    X, y = _make_data()
    reference = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True,
                           generator=torch.Generator().manual_seed(3))
    inputs = X.numpy() if as_numpy else X
    cached = CachedTensorLoader(inputs, y, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(3))
    assert len(cached) == len(reference)
    assert len(cached.dataset) == len(reference.dataset)
    for _ in range(3):
        for (x_ref, y_ref), (x_new, y_new) in zip(reference, cached):
            assert torch.equal(x_ref, x_new)
            assert torch.equal(y_ref, y_new)
    assert torch.equal(reference.generator.get_state(), cached.index_loader.generator.get_state())


class _TinyNet(torch.nn.Module):
    """
    概要: テスト用の小さな線形分類器．
    """

    def __init__(self):
        """
        概要: 初期化する．
        引数: なし
        戻り値: なし
        """
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        """
        概要: ロジットを返す．
        引数: x (torch.Tensor)，形状 (B, 4)．
        戻り値: logits (torch.Tensor)，形状 (B, 2)．
        """
        return self.linear(x)


def _loss_func(outputs, targets, model, reg_lambda):
    """
    概要: 交差エントロピーにL2正則化を加えた損失．
    引数: outputs (B, 2)，targets (B,)，model，reg_lambda (float)．
    戻り値: loss (torch.Tensor)．スカラー．
    """
    return torch.nn.functional.cross_entropy(outputs, targets) + 0.5 * reg_lambda * torch.sum(model.linear.weight ** 2)


def _metrics_func(outputs, targets):
    """
    概要: 分類精度を返す．
    引数: outputs (B, 2)，targets (B,)．
    戻り値: dict．{"accuracy": float}．
    """
    return {"accuracy": (outputs.argmax(-1) == targets).float().mean().item()}


def _build_model(seed):
    """
    概要: シードを固定して `_TinyNet` を初期化する．
    引数: seed (int)．
    戻り値: model (torch.nn.Module)．
    """
    torch.manual_seed(seed)
    return _TinyNet()


def _make_problem(start_points):
    """
    概要: `run_variance_reduced` に渡す問題の定義を作る．各外側ループの最初のステップで，
        `model` のパラメータを `start_points` に記録するように `iteration` を差し替える前提で用いる．
    引数: start_points (list)．記録先．
    戻り値: problem (dict)．
    """
    X, y = _make_data()

    def load_dataloader(seed, batch_size, device):
        g = torch.Generator().manual_seed(seed)
        return (CachedTensorLoader(X, y, batch_size, shuffle=True, generator=g),
                CachedTensorLoader(X, y, batch_size, shuffle=False))

    return {"load_dataloader": load_dataloader, "build_model": _build_model, "loss_func": _loss_func,
            "metrics_func": _metrics_func, "reg_lambda": 1e-3}


def _run(method, inner_start, tmp_path, monkeypatch):
    """
    概要: 小さな問題で3エポック学習し，各内側ループの開始点とスナップショットを記録する．
    引数: method (str)，inner_start (str)，tmp_path，monkeypatch（pytestのフィクスチャ）．
    戻り値: (start_points, snapshots, logger) の組．
    """
    start_points, snapshots = [], []
    original_train_epoch = vr.train_epoch

    def recording_train_epoch(model, dataloader, loss_func, reg_lambda, optimizer, snapshot_model, desc):
        start_points.append([p.detach().clone() for p in model.parameters()])
        snapshots.append([p.detach().clone() for p in snapshot_model.parameters()])
        original_train_epoch(model, dataloader, loss_func, reg_lambda, optimizer, snapshot_model, desc)

    monkeypatch.setattr(vr, "train_epoch", recording_train_epoch)
    logger = ResultLogger()
    logger.set_names(*vr.METRIC_NAMES)
    run_config = {"learning_rate": 0.1, "batch_size": 8, "epochs": 3}
    vr.run_variance_reduced(method, inner_start, str(tmp_path), _make_problem(start_points), run_config,
                            torch.device("cpu"), 0, logger)
    return start_points, snapshots, logger


@pytest.mark.parametrize("method", ["SVRG", "NFG_SVRG", "ASAI_SVRG"])
def test_inner_start_snapshot_starts_from_snapshot(method, tmp_path, monkeypatch):
    """
    概要: `"snapshot"` では全ての外側ループで w_s^0 = z_s となることを確認する．
    引数: method (str)，tmp_path，monkeypatch．
    戻り値: なし
    """
    start_points, snapshots, logger = _run(method, "snapshot", tmp_path, monkeypatch)
    for w0, z in zip(start_points, snapshots):
        assert all(torch.equal(a, b) for a, b in zip(w0, z))
    assert len(logger["epoch"]) == 4
    assert logger["terminated"] == [False] * 4


@pytest.mark.parametrize("method", ["SVRG", "NFG_SVRG", "ASAI_SVRG"])
def test_inner_start_last_iterate_differs_from_snapshot(method, tmp_path, monkeypatch):
    """
    概要: `"last_iterate"` では2回目以降の外側ループで w_s^0 ≠ z_s となる（平均点・ランダム点を
        スナップショットとする手法では一般に異なる）ことを確認する．
    引数: method (str)，tmp_path，monkeypatch．
    戻り値: なし
    """
    start_points, snapshots, _ = _run(method, "last_iterate", tmp_path, monkeypatch)
    differs = [not all(torch.equal(a, b) for a, b in zip(w0, z)) for w0, z in zip(start_points[1:], snapshots[1:])]
    assert any(differs)


def test_final_point_method_is_unaffected_by_inner_start():
    """
    概要: 最終点をスナップショットとする手法（`NFGSVRGFinalPoint`）では，`model` を z_s で上書きしても
        パラメータが変わらないこと（z_s = w_{s-1}^K）を確認する．
    引数: なし
    戻り値: なし
    """
    model = _build_model(0)
    optimizer = NFGSVRGFinalPoint(model.parameters(), lr=0.1, K=3)
    X, y = _make_data(n=3)
    optimizer.begin_epoch(np.random.default_rng(0))
    for i in range(3):
        optimizer.zero_grad()
        _loss_func(model(X[i:i + 1]), y[i:i + 1], model, 0.0).backward()
        optimizer.step([torch.zeros_like(p) for p in model.parameters()])
    optimizer.end_epoch()
    for p, z in zip(model.parameters(), optimizer.get_snapshot_params()):
        assert torch.equal(p.detach(), z)


def test_asai_snapshot_is_average_of_inner_iterates():
    """
    概要: ASA-SVRGのスナップショットが内側ループの更新前のパラメータ w^0, ..., w^{K-1} の平均であることを確認する．
    引数: なし
    戻り値: なし
    """
    model = _build_model(0).double()
    optimizer = ASAISVRG(model.parameters(), lr=0.1, K=3)
    X, y = _make_data(n=3)
    X = X.double()
    iterates = []
    optimizer.begin_epoch()
    for i in range(3):
        iterates.append([p.detach().clone() for p in model.parameters()])
        optimizer.zero_grad()
        _loss_func(model(X[i:i + 1]), y[i:i + 1], model, 0.0).backward()
        optimizer.step([torch.zeros_like(p) for p in model.parameters()])
    optimizer.end_epoch()
    for j, z in enumerate(optimizer.get_snapshot_params()):
        expected = sum(it[j] for it in iterates) / 3
        assert torch.allclose(z, expected, atol=1e-12)


def test_build_tasks_with_epochs_by_batch_size():
    """
    概要: `epochs_by_batch_size` を持つ実験の定義（付録B）で，バッチサイズごとのエポック数と保存先の
        ディレクトリ名が正しく構築され，`batch_size`・`epochs` だけを持つ定義（E0）の動作が変わらないことを確認する．
    引数: なし
    戻り値: なし
    """
    import e0_runner

    spec = {"experiment": "exp", "learning_rate": 0.001, "reg_lambda": 5e-4, "epochs_by_batch_size": {32: 12, 512: 192}}
    tasks = e0_runner.build_tasks(spec, ["ASAI_SVRG"], ["snapshot"], [0, 1], None, "/out")
    assert [(t["run_config"]["batch_size"], t["run_config"]["epochs"], t["seed"]) for t in tasks] == \
        [(32, 12, 0), (32, 12, 1), (512, 192, 0), (512, 192, 1)]
    assert tasks[0]["target_dir"] == "/out/ASAI_SVRG/inner_snapshot_lr0.001_bs32_lambda0.0005_epochs12/0"
    only_512 = e0_runner.build_tasks(spec, ["SVRG"], ["snapshot"], [0], 2, "/out", [512])
    assert only_512[0]["run_config"] == {"learning_rate": 0.001, "batch_size": 512, "epochs": 2}

    e0_spec = {"experiment": "exp", "learning_rate": 0.001, "reg_lambda": 5e-4, "batch_size": 128, "epochs": 48}
    e0_tasks = e0_runner.build_tasks(e0_spec, ["SVRG"], ["last_iterate", "snapshot"], [0], None, "/out")
    assert [t["target_dir"] for t in e0_tasks] == [
        "/out/SVRG/inner_last_iterate_lr0.001_bs128_lambda0.0005_epochs48/0",
        "/out/SVRG/inner_snapshot_lr0.001_bs128_lambda0.0005_epochs48/0"]
