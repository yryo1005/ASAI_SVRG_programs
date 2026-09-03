"""
programs/ex001_mushroom_svrg/model.py の単体テスト．
自動微分（`loss.backward()`）により計算される勾配が，ロジスティック回帰の閉形式勾配と
一致することを確認する．
"""

import importlib.util
import os

import torch

# programs/ex001_mushroom_svrg/model.py と programs/ex002_cifar10_cnn/model.py は
# 同名（model.py）のため，sys.path経由の `import model` ではモジュール名が衝突する．
# importlib.util により一意なモジュール名で明示的に読み込む．
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "programs",
    "ex001_mushroom_svrg",
    "model.py",
)
_spec = importlib.util.spec_from_file_location("ex001_model", _MODEL_PATH)
_ex001_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ex001_model)

LogisticRegressionModel = _ex001_model.LogisticRegressionModel
compute_accuracy = _ex001_model.compute_accuracy
compute_gradient = _ex001_model.compute_gradient
compute_loss = _ex001_model.compute_loss
set_model_params = _ex001_model.set_model_params


def _closed_form_grad(weight, bias, X, y, reg_lambda):
    """テスト用に閉形式でロジスティック回帰の勾配を計算する（model.pyには依存しない）．"""
    logits = X @ weight.t() + bias
    prob = torch.sigmoid(logits.squeeze(-1))
    error = prob - y
    grad_w = (error.unsqueeze(0) @ X) / X.shape[0] + reg_lambda * weight
    grad_b = error.mean().reshape(1)
    return grad_w, grad_b


def test_compute_gradient_matches_closed_form():
    """自動微分による勾配が，ロジスティック回帰の閉形式勾配と一致することを確認する．"""
    torch.manual_seed(0)
    model = LogisticRegressionModel(input_dim=5)
    X = torch.randn(8, 5, dtype=torch.float64)
    y = (torch.rand(8, dtype=torch.float64) > 0.5).to(torch.float64)
    reg_lambda = 0.01

    grads = compute_gradient(model, X, y, reg_lambda)

    expected_w, expected_b = _closed_form_grad(
        model.linear.weight.detach(), model.linear.bias.detach(), X, y, reg_lambda
    )
    assert torch.allclose(grads[0], expected_w, atol=1e-10)
    assert torch.allclose(grads[1], expected_b, atol=1e-10)


def test_compute_gradient_populates_model_grad():
    """compute_gradientの呼び出し後，model.parameters()の.gradが直接更新されていることを
    確認する（optimizer.step()がこれを読み取れることを保証するため）．"""
    torch.manual_seed(0)
    model = LogisticRegressionModel(input_dim=3)
    X = torch.randn(4, 3, dtype=torch.float64)
    y = (torch.rand(4, dtype=torch.float64) > 0.5).to(torch.float64)

    compute_gradient(model, X, y, reg_lambda=0.0)

    for p in model.parameters():
        assert p.grad is not None


def test_compute_loss_and_accuracy_do_not_track_gradients():
    """compute_loss / compute_accuracyが勾配計算グラフを構築しないことを確認する．"""
    torch.manual_seed(0)
    model = LogisticRegressionModel(input_dim=3)
    X = torch.randn(4, 3, dtype=torch.float64)
    y = (torch.rand(4, dtype=torch.float64) > 0.5).to(torch.float64)

    loss = compute_loss(model, X, y, reg_lambda=0.01)
    accuracy = compute_accuracy(model, X, y)

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_set_model_params_overwrites_values():
    """set_model_paramsが指定した値でモデルのパラメータを正しく上書きすることを確認する．"""
    model = LogisticRegressionModel(input_dim=2)
    new_weight = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    new_bias = torch.tensor([3.0], dtype=torch.float64)

    set_model_params(model, [new_weight, new_bias])

    params = list(model.parameters())
    assert torch.allclose(params[0], new_weight)
    assert torch.allclose(params[1], new_bias)
