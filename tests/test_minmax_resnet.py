"""
programs/ex003_cifar10_resnet_minmax/ の単体テスト．
特に，min-max定式化における sigma の符号反転（勾配上昇を勾配降下として扱う仕組み）が
正しく機能していることを重点的に確認する．
"""

import importlib.util
import os
import sys

import torch

_PROGRAMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "programs")
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "ex003_cifar10_resnet_minmax"))
sys.path.insert(0, _PROGRAMS_DIR)

# 他の実験（ex001, ex002等）のテストが同名モジュール（model, data, train）をsys.modulesに
# キャッシュしている場合，train.py内部の `from model import ...` 等の素朴なimportが誤った
# モジュールを解決してしまうため，明示的にキャッシュを破棄してから読み込む．
for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_MODEL_PATH = os.path.join(_PROGRAMS_DIR, "ex003_cifar10_resnet_minmax", "model.py")
_spec = importlib.util.spec_from_file_location("ex003_model", _MODEL_PATH)
ex003_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex003_model)

for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_TRAIN_PATH = os.path.join(_PROGRAMS_DIR, "ex003_cifar10_resnet_minmax", "train.py")
_spec2 = importlib.util.spec_from_file_location("ex003_train", _TRAIN_PATH)
ex003_train = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(ex003_train)

MinMaxResNet18 = ex003_model.MinMaxResNet18
load_model = ex003_model.load_model
set_model_params = ex003_model.set_model_params


def test_model_output_shape():
    """MinMaxResNet18の出力形状が (B, num_classes) であることを確認する．"""
    model = MinMaxResNet18(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_sigma_is_added_to_input():
    """sigmaが入力画像に加算されてからResNet-18へ渡されることを確認する．"""
    model = MinMaxResNet18(num_classes=10)
    x = torch.zeros(2, 3, 32, 32)

    with torch.no_grad():
        out_zero_sigma = model(x)
        model.sigma.fill_(1.0)
        out_nonzero_sigma = model(x)

    assert not torch.allclose(out_zero_sigma, out_nonzero_sigma)


def test_backward_minmax_objective_ascends_on_sigma():
    """backward_minmax_objectiveが，sigmaに対しては勾配上昇（Lを増大させる方向）の
    ステップに対応する符号の勾配を設定することを確認する．十分小さい学習率でsigma方向へ
    1歩降下（sigma -= lr * grad）した際に，min-max目的関数Lの値が増加することを検証する．"""
    torch.manual_seed(0)
    model = MinMaxResNet18(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    ex003_train.backward_minmax_objective(model, x, y)
    sigma_grad = model.sigma.grad.detach().clone()

    def compute_L(m):
        with torch.no_grad():
            outputs = m(x)
            ce = ex003_train.loss_func(outputs, y)
            w_sq, sigma_sq = ex003_train.compute_squared_norms(m)
            return (
                ce + (ex003_train.LAMBDA1 / 2) * w_sq - (ex003_train.LAMBDA2 / 2) * sigma_sq
            ).item()

    L_before = compute_L(model)

    lr = 1e-4
    with torch.no_grad():
        model.sigma.sub_(sigma_grad, alpha=lr)
    L_after = compute_L(model)

    assert L_after > L_before, (
        "sigmaについて勾配降下方向へ更新した結果Lが増加しなかった．"
        "F_sigma = -dL/dsigma の符号反転が正しく機能していない可能性がある．"
    )


def test_backward_minmax_objective_descends_on_w():
    """backward_minmax_objectiveが，wに対しては通常通りの勾配降下（Lを減少させる方向）に
    対応する勾配を設定することを確認する．"""
    torch.manual_seed(0)
    model = MinMaxResNet18(num_classes=10)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))

    ex003_train.backward_minmax_objective(model, x, y)
    w_grads = [p.grad.detach().clone() for p in model.resnet.parameters()]

    def compute_L(m):
        with torch.no_grad():
            outputs = m(x)
            ce = ex003_train.loss_func(outputs, y)
            w_sq, sigma_sq = ex003_train.compute_squared_norms(m)
            return (
                ce + (ex003_train.LAMBDA1 / 2) * w_sq - (ex003_train.LAMBDA2 / 2) * sigma_sq
            ).item()

    L_before = compute_L(model)

    lr = 1e-4
    with torch.no_grad():
        for p, g in zip(model.resnet.parameters(), w_grads):
            p.sub_(g, alpha=lr)
    L_after = compute_L(model)

    assert L_after < L_before, (
        "wについて勾配降下方向へ更新した結果Lが減少しなかった．"
    )


def test_compute_squared_norms_excludes_sigma_from_w_norm():
    """compute_squared_normsが，wのノルムにsigmaを含めないことを確認する．"""
    model = MinMaxResNet18(num_classes=10)
    with torch.no_grad():
        model.sigma.fill_(100.0)

    w_norm_sq, sigma_norm_sq = ex003_train.compute_squared_norms(model)
    expected_w_norm_sq = sum(p.pow(2).sum() for p in model.resnet.parameters())

    assert torch.allclose(w_norm_sq, expected_w_norm_sq)
    assert sigma_norm_sq.item() > 1e6  # 100^2 * (3*32*32) のオーダー


def test_set_model_params_overwrites_values():
    """set_model_paramsがモデルのパラメータ（sigmaを含む）を正しく上書きすることを確認する．"""
    source_model = MinMaxResNet18(num_classes=10)
    target_model = MinMaxResNet18(num_classes=10)

    source_values = [p.detach().clone() for p in source_model.parameters()]
    set_model_params(target_model, source_values)

    for p_target, p_source in zip(target_model.parameters(), source_values):
        assert torch.allclose(p_target, p_source)
