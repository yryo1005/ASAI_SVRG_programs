"""
E2（`ex007_e2_cifar10_snapshot_gradient_factorial`）のプログラムのテスト．小さな線形モデルと乱数のデータを使い，CPU で数秒で終わる．

1. `SVRGAverage` のスナップショットが内側ループの更新前のパラメータの平均と一致し，$ {\\mathbf g}_{s} $ をそろえた
   `ASAISVRG` と1エポック後のスナップショットが一致すること．
2. $ \\omega $ の逐次計算が直接計算と一致し（相対 $ 10^{-10} $ 以下），`omega_average` $ \\le $ `omega_snapshot` であり，
   平均点の手法では両者が一致すること．
3. $ \\omega $ の記録の有無で，学習後のパラメータと記録がビット単位で一致すること．
4. `oracle_calls` の増分が，全勾配の手法で初期 $ N $・1エポック $ 3N $，平均勾配の手法で1エポック $ 2N $ であること．
5. `inner_start="snapshot"` で，6手法の各内側ループが $ {\\mathbf z}_{s} $ から始まること．
6. タスクの構築で，6手法 × 5シード = 30個の保存先が指示書2.3節の名前で作られること．
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

_REVIEW_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EX007_DIR = os.path.join(_REVIEW_ROOT, "programs", "ex007_e2_cifar10_snapshot_gradient_factorial")
sys.path.insert(0, os.path.join(_REVIEW_ROOT, "programs", "common"))
sys.path.insert(0, _EX007_DIR)

from cached_loader import CachedTensorLoader  # noqa: E402
from machine_learning_utils import ResultLogger  # noqa: E402
from optimizers import ASAISVRG  # noqa: E402
from svrg_average import SVRGAverage  # noqa: E402
import e2_training as e2  # noqa: E402

N_DATA = 40
BATCH_SIZE = 8
EPOCHS = 3


def _make_data(n=N_DATA, d=4, seed=0):
    """
    概要: テスト用の小さな分類データを作る．
    引数: n (int)．データ数．d (int)．特徴量の次元．seed (int)．乱数シード．
    戻り値: (X, y) の組．X は形状 (n, d) の float32，y は形状 (n,) の int64．
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).long()
    return X, y


class _TinyNet(torch.nn.Module):
    """
    概要: テスト用の小さな2層の分類器．
    """

    def __init__(self):
        """
        概要: 初期化する．
        引数: なし
        戻り値: なし
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 6)
        self.fc2 = torch.nn.Linear(6, 2)

    def forward(self, x):
        """
        概要: ロジットを返す．
        引数: x (torch.Tensor)，形状 (B, 4)．
        戻り値: logits (torch.Tensor)，形状 (B, 2)．
        """
        return self.fc2(torch.tanh(self.fc1(x)))


def _loss_func(outputs, targets, model, reg_lambda):
    """
    概要: 交差エントロピーに重みのL2正則化を加えた損失．
    引数: outputs (B, 2)，targets (B,)，model，reg_lambda (float)．
    戻り値: loss (torch.Tensor)．スカラー．
    """
    reg = torch.sum(model.fc1.weight ** 2) + torch.sum(model.fc2.weight ** 2)
    return torch.nn.functional.cross_entropy(outputs, targets) + 0.5 * reg_lambda * reg


def _metrics_func(outputs, targets):
    """
    概要: 分類精度を返す．
    引数: outputs (B, 2)，targets (B,)．
    戻り値: dict．{"accuracy": float}．
    """
    return {"accuracy": (outputs.argmax(-1) == targets).float().mean().item()}


def _make_problem(created_models=None):
    """
    概要: `run_variance_reduced` に渡す問題の定義を作る．
    引数: created_models (list) = None．指定した場合，`build_model` が作ったモデルを順に追加する
        （1つ目が `model`，2つ目が `snapshot_model`）．
    戻り値: problem (dict)．
    """
    X, y = _make_data()

    def load_dataloader(seed, batch_size, device):
        g = torch.Generator().manual_seed(seed)
        return (CachedTensorLoader(X, y, batch_size, shuffle=True, generator=g),
                CachedTensorLoader(X, y, batch_size, shuffle=False))

    def build_model(seed):
        torch.manual_seed(seed)
        model = _TinyNet()
        if created_models is not None:
            created_models.append(model)
        return model

    return {"load_dataloader": load_dataloader, "build_model": build_model, "loss_func": _loss_func,
            "metrics_func": _metrics_func, "reg_lambda": 1e-3}


def _run(method, tmp_path, record_omega=True, created_models=None, lr=0.1):
    """
    概要: 小さな問題で `EPOCHS` エポック学習し，ロガーを返す．
    引数: method (str)，tmp_path，record_omega (bool)，created_models (list)，lr (float)．
    戻り値: logger (ResultLogger)．
    """
    os.makedirs(tmp_path, exist_ok=True)
    logger = ResultLogger()
    logger.set_names(*e2.METRIC_NAMES)
    run_config = {"learning_rate": lr, "batch_size": BATCH_SIZE, "epochs": EPOCHS}
    e2.run_variance_reduced(method, "snapshot", str(tmp_path), _make_problem(created_models), run_config,
                            torch.device("cpu"), 0, logger, record_omega=record_omega)
    return logger


def _single_epoch(optimizer_class, model, X, y, record):
    """
    概要: 1エポック（1件ずつ $ K = $ len(X) ステップ）の内側ループを $ {\\mathbf g}_{s} = {\\mathbf 0} $，
        スナップショットにおける勾配 $ {\\mathbf 0} $ で実行する．
    引数: optimizer_class (type)，model (torch.nn.Module)，X (K, 4)，y (K,)，record (list)．更新前のパラメータの記録先．
    戻り値: optimizer．
    """
    optimizer = optimizer_class(model.parameters(), lr=0.1, K=len(X))
    optimizer.begin_epoch(np.random.default_rng(0))
    for i in range(len(X)):
        record.append([p.detach().clone() for p in model.parameters()])
        optimizer.zero_grad()
        _loss_func(model(X[i:i + 1]), y[i:i + 1], model, 0.0).backward()
        optimizer.step([torch.zeros_like(p) for p in model.parameters()])
    optimizer.end_epoch()
    return optimizer


# ---------------------------------------------------------------------------
# 1. SVRGAverage
# ---------------------------------------------------------------------------

def test_svrg_average_snapshot_is_average_of_inner_iterates():
    """
    概要: `SVRGAverage` のスナップショットが更新前のパラメータ w^0, ..., w^{K-1} の平均であることを確認する．
    引数: なし
    戻り値: なし
    """
    torch.manual_seed(0)
    model = _TinyNet().double()
    X, y = _make_data(n=5)
    iterates = []
    optimizer = _single_epoch(SVRGAverage, model, X.double(), y, iterates)
    for j, z in enumerate(optimizer.get_snapshot_params()):
        expected = sum(it[j] for it in iterates) / len(iterates)
        assert torch.allclose(z, expected, rtol=0.0, atol=1e-12)


def test_svrg_average_matches_asai_with_same_snapshot_gradient():
    """
    概要: $ {\\mathbf g}_{s} $ を両方とも $ {\\mathbf 0} $ にそろえると，同じ初期値・同じデータ順で `SVRGAverage` と `ASAISVRG` の
        1エポック後のスナップショットとパラメータがビット単位で一致することを確認する．
        また，`SVRGAverage` の `set_snapshot_gradient` / `get_snapshot_gradient` が値を保持することを確認する．
    引数: なし
    戻り値: なし
    """
    X, y = _make_data(n=6)
    torch.manual_seed(0)
    model_a = _TinyNet()
    torch.manual_seed(0)
    model_b = _TinyNet()
    opt_a = _single_epoch(SVRGAverage, model_a, X, y, [])
    opt_b = _single_epoch(ASAISVRG, model_b, X, y, [])
    for za, zb in zip(opt_a.get_snapshot_params(), opt_b.get_snapshot_params()):
        assert torch.equal(za, zb)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)

    grads = [torch.full_like(p, 0.5) for p in model_a.parameters()]
    opt_a.set_snapshot_gradient(grads)
    for g, h in zip(opt_a.get_snapshot_gradient(), grads):
        assert torch.equal(g, h)
    assert not hasattr(opt_a, "running_avg_grad")
    assert all("running_avg_grad" not in opt_a.state[p] for p in model_a.parameters())


# ---------------------------------------------------------------------------
# 2. ω
# ---------------------------------------------------------------------------

def test_omega_accumulator_matches_direct_computation():
    """
    概要: `OmegaAccumulator` の逐次計算が，全ての点を保持して直接計算した
        $ \\omega({\\mathbf z}) = \\sum_{k}\\|{\\mathbf w}^{k} - {\\mathbf z}\\|^{2} $ と相対 $ 10^{-10} $ 以下で一致し，
        平均点で最小になることを確認する．
    引数: なし
    戻り値: なし
    """
    g = torch.Generator().manual_seed(1)
    shapes = [(6, 4), (6,), (2, 6), (2,)]
    points = [[torch.randn(s, generator=g) + 3.0 for s in shapes] for _ in range(17)]
    acc = e2.OmegaAccumulator()
    for pt in points:
        acc.add(pt)
    z = points[5]
    direct = sum(sum(torch.sum((w.double() - zz.double()) ** 2).item() for w, zz in zip(pt, z)) for pt in points)
    assert abs(acc.omega(z) - direct) / direct < 1e-10

    mean = [sum(pt[j].double() for pt in points) / len(points) for j in range(len(shapes))]
    direct_min = sum(sum(torch.sum((w.double() - m) ** 2).item() for w, m in zip(pt, mean)) for pt in points)
    omega_snapshot, omega_average = acc.omega_pair(z)
    assert abs(omega_average - direct_min) / direct_min < 1e-10
    assert omega_average <= omega_snapshot
    acc.reset()
    assert math.isnan(acc.omega(z))


class _RecordingAccumulator(e2.OmegaAccumulator):
    """
    概要: 加算した点と，`omega_pair` に渡されたスナップショットを保持する `OmegaAccumulator`（テスト用）．
    """

    epochs = []

    def reset(self):
        """
        概要: 加算の状態を空に戻し，新しい外側ループの記録を始める．
        引数: なし
        戻り値: なし
        """
        super().reset()
        _RecordingAccumulator.epochs.append({"points": [], "z": None})

    def add(self, params):
        """
        概要: 点を加算し，そのコピーを保持する．
        引数: params (iterable of torch.Tensor)．
        戻り値: なし
        """
        params = list(params)
        _RecordingAccumulator.epochs[-1]["points"].append([p.detach().clone() for p in params])
        super().add(params)

    def omega_pair(self, z_params):
        """
        概要: スナップショットのコピーを保持してから $ \\omega $ を計算する．
        引数: z_params (list of torch.Tensor)．
        戻り値: (omega_snapshot, omega_average)．
        """
        _RecordingAccumulator.epochs[-1]["z"] = [z.detach().clone() for z in z_params]
        return super().omega_pair(z_params)


@pytest.mark.parametrize("method", e2.METHODS)
def test_logged_omega_matches_direct_computation(method, tmp_path, monkeypatch):
    """
    概要: 学習ループが記録した `omega_snapshot`・`omega_average` が，内側ループの更新前の全ての点と実際の
        スナップショット $ {\\mathbf z}_{s} $ から直接計算した値と相対 $ 10^{-10} $ 以下で一致すること，
        `omega_average` $ \\le $ `omega_snapshot` であること，平均点の手法では両者が（相対 $ 10^{-6} $ 以下で）一致すること，
        0エポック目が NaN であることを確認する．
    引数: method (str)，tmp_path，monkeypatch．
    戻り値: なし
    """
    _RecordingAccumulator.epochs = []
    monkeypatch.setattr(e2, "OmegaAccumulator", _RecordingAccumulator)
    logger = _run(method, tmp_path)
    assert math.isnan(logger["omega_snapshot"][0]) and math.isnan(logger["omega_average"][0])
    assert len(_RecordingAccumulator.epochs) == EPOCHS
    K = math.ceil(N_DATA / BATCH_SIZE)
    for s, record in enumerate(_RecordingAccumulator.epochs, start=1):
        points, z = record["points"], record["z"]
        assert len(points) == K
        direct = sum(sum(torch.sum((w.double() - zz.double()) ** 2).item() for w, zz in zip(pt, z)) for pt in points)
        mean = [sum(pt[j].double() for pt in points) / K for j in range(len(z))]
        direct_min = sum(sum(torch.sum((w.double() - m) ** 2).item() for w, m in zip(pt, mean)) for pt in points)
        omega_snapshot, omega_average = logger["omega_snapshot"][s], logger["omega_average"][s]
        if all(all(torch.equal(a, b) for a, b in zip(pt, points[0])) for pt in points):
            # 平均勾配の手法の1エポック目は g_0 = 0，w_0^0 = z_0 より補正勾配が常に 0 であり，内側ループが動かない．
            # このとき ω は 0 であり，記録される値は丸め誤差の大きさになる．
            assert s == 1 and method not in e2.FULL_GRADIENT_METHODS
            assert abs(omega_snapshot) < 1e-10 and abs(omega_average) < 1e-10
            continue
        assert abs(omega_snapshot - direct) <= 1e-10 * direct
        assert abs(omega_average - direct_min) <= 1e-10 * direct_min
        assert omega_average <= omega_snapshot * (1 + 1e-12)
        if method in ("SVRG_average", "ASAI_SVRG"):
            assert abs(omega_snapshot / omega_average - 1.0) < 1e-6
        else:
            assert omega_snapshot / omega_average > 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 3. ω の記録が学習に影響しないこと
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", e2.METHODS)
def test_omega_recording_does_not_change_training(method, tmp_path):
    """
    概要: $ \\omega $ の記録の有無で，学習後のパラメータ（`model`，`snapshot_model`）と，`omega_*`・`elapsed_time` 以外の
        全ての記録がビット単位で一致することを確認する．
    引数: method (str)，tmp_path．
    戻り値: なし
    """
    models_on, models_off = [], []
    log_on = _run(method, tmp_path / "on", record_omega=True, created_models=models_on)
    log_off = _run(method, tmp_path / "off", record_omega=False, created_models=models_off)
    for m_on, m_off in zip(models_on, models_off):
        for p_on, p_off in zip(m_on.parameters(), m_off.parameters()):
            assert torch.equal(p_on, p_off)
    for key in e2.METRIC_NAMES:
        if key in ("elapsed_time", "omega_snapshot", "omega_average"):
            continue
        assert np.array_equal(np.asarray(log_on[key], dtype=np.float64), np.asarray(log_off[key], dtype=np.float64),
                              equal_nan=True), key
    assert all(math.isnan(v) for v in log_off["omega_snapshot"])
    assert all(np.isfinite(log_on["omega_snapshot"][1:]))
    assert all(np.isfinite(log_on["omega_average"][1:]))


# ---------------------------------------------------------------------------
# 4. oracle_calls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", e2.METHODS)
def test_oracle_calls_increments(method, tmp_path):
    """
    概要: `oracle_calls` の増分が，全勾配の手法で初期 $ N $・1エポック $ 3N $，平均勾配の手法で1エポック $ 2N $ であり，
        全勾配の手法の `approx_error` が 0.0 であることを確認する．
    引数: method (str)，tmp_path．
    戻り値: なし
    """
    logger = _run(method, tmp_path)
    calls = np.asarray(logger["oracle_calls"])
    assert calls[0] == 0
    if method in e2.FULL_GRADIENT_METHODS:
        assert calls[1] == N_DATA + 3 * N_DATA
        assert np.all(np.diff(calls)[1:] == 3 * N_DATA)
        assert all(v == 0.0 for v in logger["approx_error"])
    else:
        assert np.all(np.diff(calls) == 2 * N_DATA)
        assert all(v > 0.0 for v in logger["approx_error"][1:])


def test_full_gradient_method_set():
    """
    概要: 全勾配を用いる手法の集合が指示書2.1節の表の通りであることを確認する．
    引数: なし
    戻り値: なし
    """
    assert e2.FULL_GRADIENT_METHODS == {"SVRG_final", "SVRG", "SVRG_average"}
    assert set(e2.METHODS) - e2.FULL_GRADIENT_METHODS == {"NFG_SVRG_final", "NFG_SVRG", "ASAI_SVRG"}


# ---------------------------------------------------------------------------
# 5. 内側ループの開始点
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", e2.METHODS)
def test_inner_loop_starts_from_snapshot(method, tmp_path, monkeypatch):
    """
    概要: `inner_start="snapshot"` で，全ての外側ループの内側ループが $ {\\mathbf w}_{s}^{0} = {\\mathbf z}_{s} $ から始まることを確認する．
        また，2回目以降のスナップショットが初期値から変化している（学習が進んでいる）ことを確認する．
    引数: method (str)，tmp_path，monkeypatch．
    戻り値: なし
    """
    start_points, snapshots = [], []
    original_train_epoch = e2.train_epoch

    def recording_train_epoch(model, dataloader, loss_func, reg_lambda, optimizer, snapshot_model, desc,
                              omega_acc=None):
        start_points.append([p.detach().clone() for p in model.parameters()])
        snapshots.append([p.detach().clone() for p in snapshot_model.parameters()])
        original_train_epoch(model, dataloader, loss_func, reg_lambda, optimizer, snapshot_model, desc, omega_acc)

    monkeypatch.setattr(e2, "train_epoch", recording_train_epoch)
    logger = _run(method, tmp_path)
    assert len(start_points) == EPOCHS
    for w0, z in zip(start_points, snapshots):
        assert all(torch.equal(a, b) for a, b in zip(w0, z))
    assert not all(torch.equal(a, b) for a, b in zip(snapshots[0], snapshots[-1]))
    assert logger["terminated"] == [False] * (EPOCHS + 1)


# ---------------------------------------------------------------------------
# 6. タスクの構築
# ---------------------------------------------------------------------------

def test_build_tasks_creates_thirty_target_dirs():
    """
    概要: 6手法 × 5シード = 30個のタスクの保存先が `{手法}/inner_snapshot_lr0.001_bs128_lambda0.0005_epochs48/{seed}` と
        なることを確認する．
    引数: なし
    戻り値: なし
    """
    import importlib.util

    import e2_runner

    # 他の実験の `train.py` と名前が衝突しないよう，ファイルのパスから一意の名前で読み込む．
    module_spec = importlib.util.spec_from_file_location("ex007_train", os.path.join(_EX007_DIR, "train.py"))
    e2_train = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(e2_train)

    spec = e2_train.SPEC
    tasks = e2_runner.build_e2_tasks(spec, spec["methods"], spec["seeds"], None, "/out")
    assert len(tasks) == 30
    expected = {f"/out/{m}/inner_snapshot_lr0.001_bs128_lambda0.0005_epochs48/{s}"
                for m in ["SVRG_final", "SVRG", "SVRG_average", "NFG_SVRG_final", "NFG_SVRG", "ASAI_SVRG"]
                for s in range(5)}
    assert {t["target_dir"] for t in tasks} == expected
    assert all(t["inner_start"] == "snapshot" and t["record_omega"] for t in tasks)
    assert all(t["run_config"] == {"learning_rate": 0.001, "batch_size": 128, "epochs": 48} for t in tasks)
    assert spec["output_root"].endswith(os.path.join("outputs", "ex007_e2_cifar10_snapshot_gradient_factorial"))

    args = e2_runner.parse_args(spec, "", ["--no-omega", "--epochs", "2"])
    short = e2_runner.build_e2_tasks(spec, ["ASAI_SVRG"], [0], args.epochs, "/v", record_omega=not args.no_omega)
    assert short[0]["target_dir"] == "/v/ASAI_SVRG/inner_snapshot_lr0.001_bs128_lambda0.0005_epochs2/0"
    assert short[0]["record_omega"] is False
