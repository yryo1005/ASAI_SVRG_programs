"""
programs_old/ex004_cifar10_resnet_minmax/ の単体テスト．
`.orders/order_009.md` の3点の修正（M_WORKERSワーカーによる分散環境模擬，フル勾配計算時の
Batch Normalization統計量の固定，sigmaの正則化勾配のスケール）が正しく機能していることを
重点的に確認する．
"""

import importlib.util
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROGRAMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "programs_old")
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "ex004_cifar10_resnet_minmax"))
sys.path.insert(0, _PROGRAMS_DIR)

# 他の実験（ex001, ex002, ex003等）のテストが同名モジュール（model, data, train）を
# sys.modulesにキャッシュしている場合，train.py内部の `from model import ...` 等の素朴な
# importが誤ったモジュールを解決してしまうため，明示的にキャッシュを破棄してから読み込む．
for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_MODEL_PATH = os.path.join(_PROGRAMS_DIR, "ex004_cifar10_resnet_minmax", "model.py")
_spec = importlib.util.spec_from_file_location("ex004_model", _MODEL_PATH)
ex004_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex004_model)

for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_TRAIN_PATH = os.path.join(_PROGRAMS_DIR, "ex004_cifar10_resnet_minmax", "train.py")
_spec2 = importlib.util.spec_from_file_location("ex004_train", _TRAIN_PATH)
ex004_train = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(ex004_train)

from machine_learning_utils import ResultLogger  # noqa: E402

MinMaxResNet18 = ex004_model.MinMaxResNet18


def test_distributed_backward_matches_single_pass_gradient_in_eval_mode():
    """M_WORKERS分割の勾配集約（データ項の重み付け平均＋正則化項を1回だけ加算）が，
    Batch Normalizationの挙動に依存しない場合（eval()モード）に，単一パスの
    backward_minmax_objectiveと数学的に同一の勾配を再現することを確認する．
    サブバッチ分割の集約数学（重み付け平均と正則化項の1回加算）そのものをBatch
    Normalizationの影響から切り離して検証するのが目的．"""
    torch.manual_seed(0)
    model_single = MinMaxResNet18(num_classes=10)
    model_dist = MinMaxResNet18(num_classes=10)
    model_dist.load_state_dict(model_single.state_dict())

    model_single.eval()
    model_dist.eval()

    # 5で割り切れないバッチサイズ（最後のサブバッチが1サンプルのみ）でも動作することを確認．
    x = torch.randn(17, 3, 32, 32)
    y = torch.randint(0, 10, (17,))

    ex004_train.backward_minmax_objective(model_single, x, y)
    ex004_train.backward_minmax_objective_distributed(model_dist, x, y, num_workers=5)

    for p_single, p_dist in zip(model_single.parameters(), model_dist.parameters()):
        assert torch.allclose(p_single.grad, p_dist.grad, atol=1e-5), (
            "サブバッチ分割による勾配集約が，単一パスの勾配と一致しない．"
        )


def test_full_gradient_computation_does_not_corrupt_bn_running_stats():
    """compute_full_gradient_and_metricsが，走査中にBatch Normalizationの
    running_mean/running_varを変更しないことを確認する（model.eval()化の修正）．"""
    torch.manual_seed(0)
    model = MinMaxResNet18(num_classes=10)
    model.train()  # 意図的にtrainモードにしてから呼び出し，関数内部でeval化されることを確認する

    running_stats_before = {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
        if "running_mean" in name or "running_var" in name
    }
    assert len(running_stats_before) > 0, "MinMaxResNet18にBatch Normalization層が見当たらない．"

    batches = [(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))) for _ in range(3)]
    device = torch.device("cpu")
    ex004_train.compute_full_gradient_and_metrics(model, batches, device)

    for name, buf in model.named_buffers():
        if "running_mean" in name or "running_var" in name:
            assert torch.allclose(buf, running_stats_before[name]), (
                f"{name} がフル勾配計算中に更新されている．model.eval()化が機能していない．"
            )


def test_regularization_gradient_added_exactly_once_regardless_of_num_workers():
    """正則化項 (lambda1/2)||w||^2 - (lambda2/2)||sigma||^2 の勾配が，num_workersの値に
    依らずグローバルミニバッチ全体に対して1回分のスケールで加わることを確認する．
    Batch Normalizationの影響を切り離すため，eval()モードで検証する．"""
    torch.manual_seed(0)
    base_model = MinMaxResNet18(num_classes=10)
    x = torch.randn(10, 3, 32, 32)
    y = torch.randint(0, 10, (10,))

    sigma_grad_reference = None
    for num_workers in (1, 2, 5):
        m = MinMaxResNet18(num_classes=10)
        m.load_state_dict(base_model.state_dict())
        m.eval()
        ex004_train.backward_minmax_objective_distributed(m, x, y, num_workers=num_workers)

        if sigma_grad_reference is None:
            sigma_grad_reference = m.sigma.grad.detach().clone()
        else:
            assert torch.allclose(m.sigma.grad, sigma_grad_reference, atol=1e-5), (
                f"num_workers={num_workers}のときsigmaの勾配がnum_workers=1と異なる．"
                "正則化項が重複して加算されている可能性がある．"
            )


def test_distributed_backward_runs_in_train_mode_with_uneven_batch():
    """train()モード（各サブバッチが自身の統計量でBatch Normalizationされる本来の用途）
    でも，割り切れないバッチサイズを含めエラーなく実行できることを確認する．"""
    torch.manual_seed(0)
    model = MinMaxResNet18(num_classes=10)
    model.train()
    x = torch.randn(13, 3, 32, 32)
    y = torch.randint(0, 10, (13,))

    outputs = ex004_train.backward_minmax_objective_distributed(model, x, y, num_workers=5)

    assert outputs.shape == (13, 10)
    assert model.sigma.grad is not None
    assert all(p.grad is not None for p in model.resnet.parameters())


def test_snapshot_model_bn_buffers_stay_synced_after_epoch(tmp_path):
    """`.orders/order_010.md` で報告されたバグの修正確認：train_variance_reducedを実行した
    場合，各エポック境界で snapshot_model のBatch Normalizationバッファ（running_mean，
    running_var）が，実際に学習した model の現在のバッファ値と一致する（同期されずに
    独立した学習履歴のまま古くなることがない）ことを一気通貫で確認する．"""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N_SAMPLES = 20
    X = torch.tensor(rng.standard_normal((N_SAMPLES, 3, 32, 32)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, 10, size=N_SAMPLES), dtype=torch.long)
    dataset = TensorDataset(X, y)

    def _make_dataloader(seed, batch_size):
        generator = torch.Generator().manual_seed(seed)
        train_dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=generator
        )
        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, test_dataloader

    # train_variance_reducedはload_modelを2回呼び出し，1回目が現在のパラメータを保持するmodel，
    # 2回目がスナップショットを保持するsnapshot_modelである．load_modelをスパイして両者への
    # 参照を取得する．
    created_models = []
    original_load_model = ex004_train.load_model

    def spy_load_model(ModelClass, weight_path=None, seed=0):
        model = original_load_model(ModelClass, weight_path=weight_path, seed=seed)
        created_models.append(model)
        return model

    ex004_train.load_model = spy_load_model
    try:
        logger = ResultLogger()
        logger.set_names(
            "epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error"
        )
        device = torch.device("cpu")
        ex004_train.train_variance_reduced(
            "ASAI_SVRG",
            str(tmp_path),
            MinMaxResNet18,
            _make_dataloader,
            epochs=1,
            batch_size=10,
            device=device,
            seed=0,
            logger=logger,
        )
    finally:
        ex004_train.load_model = original_load_model

    model, snapshot_model = created_models[0], created_models[1]
    trained_buffer_names = [
        name for name, _ in model.named_buffers()
        if "running_mean" in name or "running_var" in name
    ]
    assert len(trained_buffer_names) > 0, "MinMaxResNet18にBatch Normalization層が見当たらない．"

    for (name_m, b_model), (name_s, b_snapshot) in zip(
        model.named_buffers(), snapshot_model.named_buffers()
    ):
        assert name_m == name_s
        if "running_mean" in name_m or "running_var" in name_m:
            assert torch.allclose(b_model, b_snapshot), (
                f"{name_m}: エポック終了後にsnapshot_modelのBNバッファがmodelと同期されていない．"
            )
