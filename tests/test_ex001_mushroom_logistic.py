"""
`programs/ex001_mushroom_logistic/`（実験1：Mushroomデータセットを用いた二値分類問題）の
単体テスト．

自動微分による勾配が閉形式勾配（正則化項が重み・切片の両方に課されることを含む）と一致すること，
平滑性定数・収縮係数 $ \\rho $ の数値的性質，学習率の逆算（`solve_eta_for_target_rho`）の
妥当性，4手法が合成データ上でエラーなく完走すること，オラクル呼び出し回数の正しさを確認する．
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROGRAMS_DIR = os.path.join(_PROJECT_ROOT, "programs")
_EXPERIMENT_DIR = os.path.join(_PROGRAMS_DIR, "ex001_mushroom_logistic")
sys.path.insert(0, _EXPERIMENT_DIR)
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import ResultLogger  # noqa: E402

# 他の実験（ex000等）のテストが同名モジュール（model, data, train）をsys.modulesにキャッシュ
# している場合，train.py内部の `from model import ...` 等の素朴なimportが誤ったモジュールを
# 解決してしまうため，明示的にキャッシュを破棄してから読み込む．
for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_spec_model = importlib.util.spec_from_file_location(
    "ex001_model", os.path.join(_EXPERIMENT_DIR, "model.py")
)
ex001_model = importlib.util.module_from_spec(_spec_model)
_spec_model.loader.exec_module(ex001_model)

for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_spec_train = importlib.util.spec_from_file_location(
    "ex001_train", os.path.join(_EXPERIMENT_DIR, "train.py")
)
ex001_train = importlib.util.module_from_spec(_spec_train)
_spec_train.loader.exec_module(ex001_train)


def _make_synthetic_data(N: int = 50, d: int = 6, seed: int = 0):
    """
    概要: テスト用の小規模な合成データ（標準正規乱数の特徴量と{0,1}のラベル）を生成する．
    引数:
        N (int) = 50．サンプル数．
        d (int) = 6．特徴量の次元数．
        seed (int) = 0．乱数シード．
    戻り値: (X, y)．形状 (N, d) と (N,) の `torch.float64` テンソルの組．
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, d))
    y = rng.integers(0, 2, size=N).astype(np.float64)
    return torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)


def test_model_has_bias_term():
    """実験1のモデルは切片項を含むことを確認する（実験0とは異なる設計）．"""
    model = ex001_model.load_model(seed=0, input_dim=6)
    assert model.linear.bias is not None
    assert len(list(model.parameters())) == 2


def test_autograd_gradient_matches_closed_form_with_bias_regularization():
    """自動微分による勾配が，重み・切片の両方に正則化を課した閉形式勾配
    $ \\nabla f = \\frac{1}{N}X^\\top(\\sigma(Xw+b)-y) + \\lambda w $（重み），
    $ \\frac{1}{N}\\sum(\\sigma(Xw+b)-y) + \\lambda b $（切片）と一致することを確認する．"""
    X, y = _make_synthetic_data()
    model = ex001_model.load_model(seed=3, input_dim=X.shape[1])
    reg_lambda = 0.1

    grads = ex001_model.compute_gradient(model, X, y, reg_lambda)

    w = model.linear.weight.detach().numpy().reshape(-1)
    b = model.linear.bias.detach().numpy().reshape(-1)
    X_np, y_np = X.numpy(), y.numpy()
    p = 1.0 / (1.0 + np.exp(-(X_np @ w + b)))
    expected_grad_w = (X_np.T @ (p - y_np)) / X_np.shape[0] + reg_lambda * w
    expected_grad_b = np.mean(p - y_np) + reg_lambda * b

    assert np.allclose(grads[0].numpy().reshape(-1), expected_grad_w)
    assert np.allclose(grads[1].numpy().reshape(-1), expected_grad_b)


def test_smoothness_constant_matches_analytic_bound():
    """`compute_smoothness_constant` が $ L = \\frac{1}{4}\\max_i\\|[x_i,1]\\|^2 + \\lambda $
    を計算することを確認する．"""
    X, _ = _make_synthetic_data(N=40, d=5, seed=7)
    X_np = X.numpy()
    reg_lambda = 0.2

    L = ex001_train.compute_smoothness_constant(X_np, reg_lambda)

    X_augmented = np.hstack([X_np, np.ones((X_np.shape[0], 1))])
    expected_L = 0.25 * (X_augmented ** 2).sum(axis=1).max() + reg_lambda
    assert np.isclose(L, expected_L)


def test_contraction_rate_is_u_shaped_and_diverges_at_boundaries():
    """収縮係数 $ \\rho(\\eta) $（式(30)）が，$ \\eta \\to 0 $ と $ \\eta \\to 1/(3L) $ の
    両極限で増大するU字形であることを確認する．"""
    L, mu, K = 10.0, 0.5, 500
    eta_max = 1.0 / (3.0 * L)

    rho_small = ex001_train.compute_contraction_rate(eta_max * 0.001, L, mu, K)
    rho_mid = ex001_train.compute_contraction_rate(eta_max * 0.3, L, mu, K)
    rho_large = ex001_train.compute_contraction_rate(eta_max * 0.999, L, mu, K)

    assert rho_mid < rho_small
    assert rho_mid < rho_large


def test_solve_eta_for_target_rho_achieves_target():
    """`solve_eta_for_target_rho` が返す学習率で，実際に `compute_contraction_rate` が
    目標値と一致することを確認する．"""
    L, mu, K = 31.4, 0.9, 7311
    target_rho = 0.5

    eta = ex001_train.solve_eta_for_target_rho(L, mu, K, target_rho)
    rho = ex001_train.compute_contraction_rate(eta, L, mu, K)

    assert 0.0 < eta < 1.0 / (3.0 * L)
    assert np.isclose(rho, target_rho, rtol=1e-4)


def test_solve_eta_raises_when_target_rho_unreachable():
    """正則化が弱く $ \\rho $ の最小値が目標値を下回らない場合，
    `solve_eta_for_target_rho` が例外を送出することを確認する．"""
    L, mu, K = 31.4, 1e-6, 7311  # 極端に弱い正則化（muが小さい）ではrhoの最小値が大きくなる
    with pytest.raises(ValueError):
        ex001_train.solve_eta_for_target_rho(L, mu, K, target_rho=0.5)


@pytest.mark.parametrize("method", ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"])
def test_training_runs_without_error_on_synthetic_data(method):
    """4手法が合成データ上でエラーなく完走し，記録された評価指標の本数がエポック数+1と一致する
    ことを確認する（スモークテスト）．"""
    X_train, y_train = _make_synthetic_data(N=30, d=5, seed=11)
    X_test, y_test = _make_synthetic_data(N=10, d=5, seed=12)
    reg_lambda = 0.1
    epochs = 3

    _, f_star = ex001_train.compute_optimal_solution(X_train.numpy(), y_train.numpy(), reg_lambda)

    logger = ResultLogger()
    logger.set_names(*ex001_train.METRIC_NAMES)

    if method == "SGD":
        ex001_train.run_sgd(0, X_train, y_train, X_test, y_test, 0.1, reg_lambda, f_star, epochs, logger)
    else:
        ex001_train.run_variance_reduced(
            method, 0, X_train, y_train, X_test, y_test, 0.1, reg_lambda, f_star, epochs, logger
        )

    assert len(logger["epoch"]) == epochs + 1
    assert all(np.isfinite(v) for v in logger["objective_gap"])
    assert all(np.isfinite(v) for v in logger["grad_norm_sq"])
    # 強凸問題であり f_star は真の最適値であるため，目的関数の差分は常に非負のはずである．
    assert all(v >= -1e-9 for v in logger["objective_gap"])


def test_oracle_calls_accounting_per_epoch():
    """1エポックあたりのオラクル呼び出し回数が，SGDで N，NFG SVRG・ASAI SVRGで 2N，SVRGで 3N
    となることを確認する．"""
    X_train, y_train = _make_synthetic_data(N=24, d=4, seed=13)
    X_test, y_test = _make_synthetic_data(N=8, d=4, seed=14)
    reg_lambda = 0.1
    N = X_train.shape[0]
    epochs = 2

    _, f_star = ex001_train.compute_optimal_solution(X_train.numpy(), y_train.numpy(), reg_lambda)

    expected_per_epoch = {"SGD": N, "NFG_SVRG": 2 * N, "ASAI_SVRG": 2 * N, "SVRG": 3 * N}
    for method, per_epoch in expected_per_epoch.items():
        logger = ResultLogger()
        logger.set_names(*ex001_train.METRIC_NAMES)
        if method == "SGD":
            ex001_train.run_sgd(0, X_train, y_train, X_test, y_test, 0.05, reg_lambda, f_star, epochs, logger)
        else:
            ex001_train.run_variance_reduced(
                method, 0, X_train, y_train, X_test, y_test, 0.05, reg_lambda, f_star, epochs, logger
            )

        initial_full_grad = N if method == "SVRG" else 0
        assert logger["oracle_calls"][0] == 0
        assert logger["oracle_calls"][1] == initial_full_grad + per_epoch
        assert logger["oracle_calls"][2] - logger["oracle_calls"][1] == per_epoch
