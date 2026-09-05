"""
programs/ex006_a9a_least_squares/ の単体テスト．
自動微分（`loss.backward()`）により計算される勾配が閉形式勾配と一致すること，平滑性定数の
計算が妥当な値を返すこと，4手法（SGD，SVRG，NFG，ASAI SVRG）が合成データ上でエラーなく
数エポック完走すること（スモークテスト）を確認する．
"""

import importlib.util
import os

import numpy as np
import torch

_EX006_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "programs",
    "ex006_a9a_least_squares",
)


def _load_module(name, filename):
    """テスト用に，`ex006_a9a_least_squares/` 内のモジュールを一意な名前で読み込む．"""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_EX006_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ex006_model = _load_module("ex006_model", "model.py")
_ex006_train = _load_module("ex006_train", "train.py")

LeastSquaresSigmoidModel = _ex006_model.LeastSquaresSigmoidModel
compute_accuracy = _ex006_model.compute_accuracy
compute_gradient = _ex006_model.compute_gradient
compute_loss = _ex006_model.compute_loss
set_model_params = _ex006_model.set_model_params

SGD = _ex006_train.SGD
_VARIANCE_REDUCED_OPTIMIZER_CLASSES = _ex006_train._VARIANCE_REDUCED_OPTIMIZER_CLASSES
compute_smoothness_constant = _ex006_train.compute_smoothness_constant
draw_batch = _ex006_train.draw_batch
run_sgd = _ex006_train.run_sgd
run_variance_reduced = _ex006_train.run_variance_reduced


def _closed_form_grad(weight, A, y):
    """テスト用に閉形式で非線形最小二乗損失（式(8)）の勾配を計算する（model.pyには依存しない）．"""
    z = A @ weight.t()
    s = torch.sigmoid(z).squeeze(-1)
    grad_w = (((2.0 * (s - y) * s * (1 - s)).unsqueeze(0)) @ A) / A.shape[0]
    return grad_w


def test_compute_gradient_matches_closed_form():
    """自動微分による勾配が，非線形最小二乗損失の閉形式勾配と一致することを確認する．"""
    torch.manual_seed(0)
    model = LeastSquaresSigmoidModel(input_dim=5)
    A = torch.randn(8, 5, dtype=torch.float64)
    y = (torch.rand(8, dtype=torch.float64) > 0.5).to(torch.float64)

    grads = compute_gradient(model, A, y)

    expected_w = _closed_form_grad(model.linear.weight.detach(), A, y)
    assert torch.allclose(grads[0], expected_w, atol=1e-10)


def test_model_has_no_bias():
    """式(8)（z_i = A_i・x）に忠実に，切片を持たないことを確認する．"""
    model = LeastSquaresSigmoidModel(input_dim=5)
    assert model.linear.bias is None


def test_compute_loss_and_accuracy_do_not_track_gradients():
    """compute_loss / compute_accuracyが勾配計算グラフを構築しないことを確認する．"""
    torch.manual_seed(0)
    model = LeastSquaresSigmoidModel(input_dim=3)
    A = torch.randn(4, 3, dtype=torch.float64)
    y = (torch.rand(4, dtype=torch.float64) > 0.5).to(torch.float64)

    loss = compute_loss(model, A, y)
    accuracy = compute_accuracy(model, A, y)

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
    assert loss >= 0.0


def test_set_model_params_overwrites_values():
    """set_model_paramsが指定した値でモデルのパラメータを正しく上書きすることを確認する．"""
    model = LeastSquaresSigmoidModel(input_dim=2)
    new_weight = torch.tensor([[1.0, 2.0]], dtype=torch.float64)

    set_model_params(model, [new_weight])

    params = list(model.parameters())
    assert torch.allclose(params[0], new_weight)


def test_compute_smoothness_constant_is_positive_and_finite():
    """平滑性定数 L が正の有限値であることを確認する．"""
    rng = np.random.default_rng(0)
    A = (rng.random((50, 10)) > 0.7).astype(np.float64)

    L = compute_smoothness_constant(A)

    assert L > 0.0
    assert np.isfinite(L)


def _make_synthetic_data(seed=0, N=40, d=123):
    """スモークテスト用に，ラベル付き合成データを生成する．"""
    rng = np.random.default_rng(seed)
    A = (rng.random((N, d)) > 0.6).astype(np.float64)
    y = (rng.random(N) > 0.5).astype(np.float64)
    return torch.tensor(A, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)


def test_sgd_smoke_runs_without_error():
    """SGDが合成データ上で数エポックエラーなく完走することを確認する（スモークテスト）．"""
    A_train, y_train = _make_synthetic_data(seed=0)
    A_test, y_test = _make_synthetic_data(seed=1)

    from machine_learning_utils import ResultLogger

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "objective_value", "grad_norm_sq", "accuracy", "approx_error")

    run_sgd(seed=0, A_train=A_train, y_train=y_train, A_test=A_test, y_test=y_test, eta=1e-2, epochs=2, logger=logger)

    assert len(logger["epoch"]) == 3  # 0エポック目 + 2エポック


def test_variance_reduced_methods_smoke_run_without_error():
    """SVRG，NFG，ASAI SVRGが合成データ上で数エポックエラーなく完走することを確認する
    （スモークテスト）．"""
    A_train, y_train = _make_synthetic_data(seed=0)
    A_test, y_test = _make_synthetic_data(seed=1)
    N_train = A_train.shape[0]

    from machine_learning_utils import ResultLogger

    for method in _VARIANCE_REDUCED_OPTIMIZER_CLASSES:
        logger = ResultLogger()
        logger.set_names("epoch", "oracle_calls", "elapsed_time", "objective_value", "grad_norm_sq", "accuracy", "approx_error")

        run_variance_reduced(
            method=method, seed=0, A_train=A_train, y_train=y_train, A_test=A_test, y_test=y_test,
            eta=1e-2, K=N_train, epochs=2, logger=logger,
        )

        assert len(logger["epoch"]) == 3, f"{method} が期待通りのエポック数を記録していない"
        assert all(np.isfinite(v) for v in logger["grad_norm_sq"]), f"{method} のgrad_norm_sqにNaN/Infが含まれる"
