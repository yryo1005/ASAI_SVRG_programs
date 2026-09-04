"""
programs/ex005_cifar10_resnet_classification/ の単体テスト．
`.orders/order_010.md` の指示に基づき，Ex004からmin-max構造（sigma）を取り除いた
純粋な多値分類問題として実装されていること，および `set_model_params` のBatch
Normalizationバッファ同期（`.orders/order_010.md` で報告されたバグの修正）が機能して
いることを重点的に確認する．
"""

import importlib.util
import os
import sys

import torch

_PROGRAMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "programs")
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "ex005_cifar10_resnet_classification"))
sys.path.insert(0, _PROGRAMS_DIR)

# 他の実験（ex001〜ex004等）のテストが同名モジュール（model, data, train）をsys.modulesに
# キャッシュしている場合，train.py内部の `from model import ...` 等の素朴なimportが誤った
# モジュールを解決してしまうため，明示的にキャッシュを破棄してから読み込む．
for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_MODEL_PATH = os.path.join(_PROGRAMS_DIR, "ex005_cifar10_resnet_classification", "model.py")
_spec = importlib.util.spec_from_file_location("ex005_model", _MODEL_PATH)
ex005_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex005_model)

for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_TRAIN_PATH = os.path.join(_PROGRAMS_DIR, "ex005_cifar10_resnet_classification", "train.py")
_spec2 = importlib.util.spec_from_file_location("ex005_train", _TRAIN_PATH)
ex005_train = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(ex005_train)

ResNet18 = ex005_model.ResNet18
load_model = ex005_model.load_model
set_model_params = ex005_model.set_model_params


def test_model_output_shape():
    """ResNet18の出力形状が (B, num_classes) であることを確認する．"""
    model = ResNet18(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_model_has_no_sigma_parameter():
    """Ex004のMinMaxResNet18と異なり，sigmaパラメータ（敵対的摂動）を保持しないことを確認する．"""
    model = ResNet18(num_classes=10)
    param_names = [name for name, _ in model.named_parameters()]
    assert not any("sigma" in name for name in param_names)


def test_backward_objective_descends_on_w():
    """backward_objectiveが，wに対して通常の勾配降下（Lを減少させる方向）に対応する
    勾配を設定することを確認する．"""
    torch.manual_seed(0)
    model = ResNet18(num_classes=10)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))

    ex005_train.backward_objective(model, x, y)
    grads = [p.grad.detach().clone() for p in model.parameters()]

    def compute_L(m):
        with torch.no_grad():
            outputs = m(x)
            ce = ex005_train.loss_func(outputs, y)
            w_sq = ex005_train.compute_squared_norm(m)
            return (ce + (ex005_train.LAMBDA1 / 2) * w_sq).item()

    L_before = compute_L(model)
    lr = 1e-4
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads):
            p.sub_(g, alpha=lr)
    L_after = compute_L(model)

    assert L_after < L_before


def test_distributed_backward_matches_single_pass_gradient_in_eval_mode():
    """M_WORKERS分割の勾配集約（データ項の重み付け平均＋正則化項を1回だけ加算）が，
    Batch Normalizationの挙動に依存しない場合（eval()モード）に，単一パスの
    backward_objectiveと数学的に同一の勾配を再現することを確認する．"""
    torch.manual_seed(0)
    model_single = ResNet18(num_classes=10)
    model_dist = ResNet18(num_classes=10)
    model_dist.load_state_dict(model_single.state_dict())

    model_single.eval()
    model_dist.eval()

    # 5で割り切れないバッチサイズ（最後のサブバッチが1サンプルのみ）でも動作することを確認．
    x = torch.randn(17, 3, 32, 32)
    y = torch.randint(0, 10, (17,))

    ex005_train.backward_objective(model_single, x, y)
    ex005_train.backward_objective_distributed(model_dist, x, y, num_workers=5)

    for p_single, p_dist in zip(model_single.parameters(), model_dist.parameters()):
        assert torch.allclose(p_single.grad, p_dist.grad, atol=1e-5), (
            "サブバッチ分割による勾配集約が，単一パスの勾配と一致しない．"
        )


def test_regularization_gradient_added_exactly_once_regardless_of_num_workers():
    """正則化項 (lambda1/2)||w||^2 の勾配が，num_workersの値に依らずグローバルミニバッチ
    全体に対して1回分のスケールで加わることを確認する．Batch Normalizationの影響を
    切り離すため，eval()モードで検証する．"""
    torch.manual_seed(0)
    base_model = ResNet18(num_classes=10)
    x = torch.randn(10, 3, 32, 32)
    y = torch.randint(0, 10, (10,))

    grad_reference = None
    for num_workers in (1, 2, 5):
        m = ResNet18(num_classes=10)
        m.load_state_dict(base_model.state_dict())
        m.eval()
        ex005_train.backward_objective_distributed(m, x, y, num_workers=num_workers)
        grad = next(m.parameters()).grad.detach().clone()

        if grad_reference is None:
            grad_reference = grad
        else:
            assert torch.allclose(grad, grad_reference, atol=1e-5), (
                f"num_workers={num_workers}のとき勾配がnum_workers=1と異なる．"
                "正則化項が重複して加算されている可能性がある．"
            )


def test_full_gradient_computation_does_not_corrupt_bn_running_stats():
    """compute_full_gradient_and_metricsが，走査中にBatch Normalizationの
    running_mean/running_varを変更しないことを確認する（model.eval()化）．"""
    torch.manual_seed(0)
    model = ResNet18(num_classes=10)
    model.train()  # 意図的にtrainモードにしてから呼び出し，関数内部でeval化されることを確認する

    running_stats_before = {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
        if "running_mean" in name or "running_var" in name
    }
    assert len(running_stats_before) > 0, "ResNet18にBatch Normalization層が見当たらない．"

    batches = [(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))) for _ in range(3)]
    device = torch.device("cpu")
    ex005_train.compute_full_gradient_and_metrics(model, batches, device)

    for name, buf in model.named_buffers():
        if "running_mean" in name or "running_var" in name:
            assert torch.allclose(buf, running_stats_before[name]), (
                f"{name} がフル勾配計算中に更新されている．model.eval()化が機能していない．"
            )


def test_set_model_params_syncs_bn_buffers_from_source_model():
    """`.orders/order_010.md` で報告されたバグの修正確認：set_model_paramsが，
    source_model指定時にBatch Normalizationのバッファ（running_mean，running_var等）も
    source_modelの現在値で同期することを確認する．"""
    trained_model = ResNet18(num_classes=10)
    snapshot_model = ResNet18(num_classes=10)

    trained_model.train()
    x = torch.randn(8, 3, 32, 32)
    trained_model(x)  # forwardによりBNのrunning_mean/running_varを更新する

    trained_buffers = [b.detach().clone() for b in trained_model.buffers()]
    snapshot_buffers_before = [b.detach().clone() for b in snapshot_model.buffers()]
    assert not all(
        torch.allclose(a, b) for a, b in zip(trained_buffers, snapshot_buffers_before)
    ), "事前条件: trained_modelとsnapshot_modelのBNバッファが既に一致してしまっている"

    param_values = [p.detach().clone() for p in trained_model.parameters()]
    set_model_params(snapshot_model, param_values, source_model=trained_model)

    for b_snapshot, b_trained in zip(snapshot_model.buffers(), trained_model.buffers()):
        assert torch.allclose(b_snapshot, b_trained), (
            "source_model指定時にBNバッファがsource_modelの値と同期されていない．"
        )
