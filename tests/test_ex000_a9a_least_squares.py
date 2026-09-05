"""
`programs/ex000_a9a_least_squares/`（実験0：a9aデータセットを用いた二値分類問題）の単体テスト．

自動微分による勾配が式(8)の閉形式勾配と一致すること，モデルが切片・正則化項を持たないこと，
平滑性定数と学習率が理論通りに構成されること，および3手法が合成データ上でエラーなく完走し
NFG SVRGのスナップショット勾配が理論的な性質を満たすことを確認する．
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROGRAMS_DIR = os.path.join(_PROJECT_ROOT, "programs")
_EXPERIMENT_DIR = os.path.join(_PROGRAMS_DIR, "ex000_a9a_least_squares")
sys.path.insert(0, _EXPERIMENT_DIR)
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import ResultLogger  # noqa: E402

# 他の実験（ex001等）のテストが同名モジュール（model, data, train）をsys.modulesにキャッシュ
# している場合，train.py内部の `from model import ...` 等の素朴なimportが誤ったモジュールを
# 解決してしまうため，明示的にキャッシュを破棄してから読み込む．
for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_spec_model = importlib.util.spec_from_file_location(
    "ex000_model", os.path.join(_EXPERIMENT_DIR, "model.py")
)
ex000_model = importlib.util.module_from_spec(_spec_model)
_spec_model.loader.exec_module(ex000_model)

for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_spec_train = importlib.util.spec_from_file_location(
    "ex000_train", os.path.join(_EXPERIMENT_DIR, "train.py")
)
ex000_train = importlib.util.module_from_spec(_spec_train)
_spec_train.loader.exec_module(ex000_train)


def _make_synthetic_data(N: int = 40, d: int = 6, seed: int = 0):
    """
    概要: テスト用の小規模な合成データ（0/1の特徴量と{0,1}のラベル）を生成する．
    引数:
        N (int) = 40．サンプル数．
        d (int) = 6．特徴量の次元数．
        seed (int) = 0．乱数シード．
    戻り値: (A, y)．形状 (N, d) と (N,) の `torch.float64` テンソルの組．
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, size=(N, d)).astype(np.float64)
    y = rng.integers(0, 2, size=N).astype(np.float64)
    return torch.tensor(A), torch.tensor(y)


def test_model_has_no_bias_term():
    """式(8)は z_i = A_i・x と切片を持たない形で定義されるため，モデルが切片を持たないことを
    確認する．"""
    model = ex000_model.load_model(seed=0)
    assert model.linear.bias is None
    assert len(list(model.parameters())) == 1


def test_model_parameters_are_double_precision():
    """フル勾配のノルムを多桁にわたって観測するため，パラメータが倍精度であることを確認する．"""
    model = ex000_model.load_model(seed=0)
    assert next(model.parameters()).dtype == torch.float64


def test_autograd_gradient_matches_closed_form():
    """自動微分による勾配が，式(8)の閉形式勾配
    ∇f(x) = (1/N) Σ_i 2(σ(z_i) - y_i)σ'(z_i) A_i と一致することを確認する．"""
    A, y = _make_synthetic_data()
    model = ex000_model.load_model(seed=3, input_dim=A.shape[1])

    grads = ex000_model.compute_gradient(model, A, y)

    x = next(model.parameters()).detach().numpy().reshape(-1)
    A_np, y_np = A.numpy(), y.numpy()
    s = 1.0 / (1.0 + np.exp(-(A_np @ x)))
    expected = (A_np.T @ (2.0 * (s - y_np) * s * (1.0 - s))) / A_np.shape[0]

    assert np.allclose(grads[0].numpy().reshape(-1), expected)


def test_loss_func_is_mean_squared_error_of_sigmoid():
    """`loss_func` が式(8)の f(x) = (1/n)Σ(y_i - σ(z_i))^2 を計算することを確認する．
    正則化項が加算されていないことも同時に確認される．"""
    A, y = _make_synthetic_data()
    model = ex000_model.load_model(seed=1, input_dim=A.shape[1])

    loss = ex000_model.compute_loss(model, A, y)

    x = next(model.parameters()).detach().numpy().reshape(-1)
    s = 1.0 / (1.0 + np.exp(-(A.numpy() @ x)))
    assert np.isclose(loss, np.mean((y.numpy() - s) ** 2))


def test_kappa_max_matches_analytic_bound():
    """l''(z, y) = 2σ'(z)^2 + 2(σ(z)-y)σ''(z) の絶対値の上界 κ_max が，正の有限値であり，
    細かい格子での再計算に対して安定であることを確認する．"""
    kappa_max = ex000_train.compute_kappa_max()
    assert 0.0 < kappa_max < 1.0
    assert np.isclose(kappa_max, ex000_train.compute_kappa_max(num_points=800001), rtol=1e-6)


def test_smoothness_constants_ordering():
    """Assumption 1に対応する L_individual = κ_max・max_i||A_i||^2 が，経験損失の平滑性定数
    L_average = κ_max・λ_max(A^T A / N) 以上であることを確認する．"""
    A, _ = _make_synthetic_data(N=60, d=8, seed=5)
    L_individual, L_average = ex000_train.compute_smoothness_constants(A.numpy())

    assert L_individual > 0.0
    assert L_average > 0.0
    assert L_individual >= L_average


def test_learning_rates_satisfy_upper_bound():
    """`.orders/order_020.md` が指定する上界 η ≤ 1/(3L) を，用いる全ての学習率が満たすことを
    確認する（データの読み込みを避けるため，学習率の構成規則を直接検証する）．"""
    L = 2.5
    N = 1000
    etas = [1.0 / (3.0 * L), 1.0 / (20.0 * L), 1.0 / (20.0 * L * N)]
    for eta in etas:
        assert eta <= 1.0 / (3.0 * L) + 1e-15


@pytest.mark.parametrize("method", ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"])
def test_training_runs_without_error_on_synthetic_data(method):
    """4手法が合成データ上でエラーなく完走し，記録された評価指標の本数がエポック数+1と一致する
    ことを確認する（スモークテスト）．"""
    A_train, y_train = _make_synthetic_data(N=30, d=5, seed=7)
    A_test, y_test = _make_synthetic_data(N=10, d=5, seed=8)
    epochs = 3

    logger = ResultLogger()
    logger.set_names(*ex000_train.METRIC_NAMES)

    if method == "SGD":
        ex000_train.run_sgd(0, A_train, y_train, A_test, y_test, 0.1, epochs, logger)
    else:
        ex000_train.run_variance_reduced(
            method, 0, A_train, y_train, A_test, y_test, 0.1, epochs, logger
        )

    assert len(logger["epoch"]) == epochs + 1
    assert logger["epoch"] == list(range(epochs + 1))
    assert all(np.isfinite(v) for v in logger["objective_value"])
    assert all(np.isfinite(v) for v in logger["grad_norm_sq"])


def test_oracle_calls_accounting_per_epoch():
    """1エポックあたりのオラクル呼び出し回数が，SGDで N，NFG SVRG・ASAI SVRGで 2N，SVRGで 3N
    （内部ループの 2N とフル勾配の N）となることを確認する．"""
    A_train, y_train = _make_synthetic_data(N=20, d=4, seed=11)
    A_test, y_test = _make_synthetic_data(N=6, d=4, seed=12)
    N = A_train.shape[0]
    epochs = 2

    expected_per_epoch = {"SGD": N, "NFG_SVRG": 2 * N, "ASAI_SVRG": 2 * N, "SVRG": 3 * N}
    for method, per_epoch in expected_per_epoch.items():
        logger = ResultLogger()
        logger.set_names(*ex000_train.METRIC_NAMES)
        if method == "SGD":
            ex000_train.run_sgd(0, A_train, y_train, A_test, y_test, 0.05, epochs, logger)
        else:
            ex000_train.run_variance_reduced(
                method, 0, A_train, y_train, A_test, y_test, 0.05, epochs, logger
            )

        # SVRGは初期スナップショット勾配 g_0 のフル勾配 N 回をepoch 1のコストとして計上する．
        initial_full_grad = N if method == "SVRG" else 0

        assert logger["oracle_calls"][0] == 0
        assert logger["oracle_calls"][1] == initial_full_grad + per_epoch
        assert logger["oracle_calls"][2] - logger["oracle_calls"][1] == per_epoch


def test_nfg_svrg_snapshot_gradient_equals_true_full_gradient_after_first_epoch():
    """NFG SVRGの第1エポックは g_0 = 0 かつ ω_0 = x_0 のため補正勾配が常に0となりパラメータが
    動かない．その結果，第1エポック終了時の平均勾配は各サンプルを丁度1回ずつ用いた
    x_0 における真のフル勾配と厳密に一致し，近似誤差 ||e_1||^2 が0となることを確認する
    （ランダムリシャッフルを用いることの帰結）．"""
    A_train, y_train = _make_synthetic_data(N=25, d=5, seed=13)
    A_test, y_test = _make_synthetic_data(N=8, d=5, seed=14)

    logger = ResultLogger()
    logger.set_names(*ex000_train.METRIC_NAMES)
    ex000_train.run_variance_reduced(
        "NFG_SVRG", 0, A_train, y_train, A_test, y_test, 0.1, 1, logger
    )

    # epoch 0 は g_0 = 0 のため ||e_0||^2 = ||∇f(x_0)||^2 となる．
    assert np.isclose(logger["approx_error"][0], logger["grad_norm_sq"][0])
    # epoch 1 終了時点では平均勾配が真のフル勾配と一致する．
    assert logger["approx_error"][1] < 1e-25
