"""
`programs/ex0001_a9a_logistic/`（実験0-1：a9aデータセットを用いたL2正則化付きロジスティック回帰）
の単体テスト．

モデルが切片項を含むこと，自動微分による勾配がL2正則化付きBCE損失の閉形式勾配と一致すること，
a9aに対して平滑性定数・正則化係数・学習率が解析的に期待される値となること，データの前処理が
実験0と同一（無加工の0/1特徴量）であること，4手法が合成データ上でエラーなく完走すること，
オラクル呼び出し回数の正しさを確認する．
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROGRAMS_DIR = os.path.join(_PROJECT_ROOT, "programs")
_EXPERIMENT_DIR = os.path.join(_PROGRAMS_DIR, "ex0001_a9a_logistic")
sys.path.insert(0, _EXPERIMENT_DIR)
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import ResultLogger  # noqa: E402


def _load_module(alias: str, filename: str):
    """
    概要: 実験ディレクトリ内のモジュールを，他の実験の同名モジュール（model，data，train）の
        キャッシュと衝突しないよう，キャッシュを破棄した上で別名で読み込む．
    引数:
        alias (str)．読み込んだモジュールに付与する名前．
        filename (str)．実験ディレクトリ内のファイル名．
    戻り値: module (module)．読み込んだモジュール．
    """
    for stale_name in ("model", "data", "train"):
        sys.modules.pop(stale_name, None)
    spec = importlib.util.spec_from_file_location(alias, os.path.join(_EXPERIMENT_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex0001_model = _load_module("ex0001_model", "model.py")
ex0001_data = _load_module("ex0001_data", "data.py")
ex0001_train = _load_module("ex0001_train", "train.py")


def _make_synthetic_data(N: int = 50, d: int = 6, seed: int = 0):
    """
    概要: テスト用の小規模な合成データ（0/1の二値特徴量と{0,1}のラベル）を生成する．
    引数:
        N (int) = 50．サンプル数．
        d (int) = 6．特徴量の次元数．
        seed (int) = 0．乱数シード．
    戻り値: (X, y)．形状 (N, d) と (N,) の `torch.float64` テンソルの組．
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(N, d)).astype(np.float64)
    y = rng.integers(0, 2, size=N).astype(np.float64)
    return torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)


def test_experiment_names_are_consistent():
    """data.py・train.pyの実験名が `ex0001_a9a_logistic` で一致することを確認する．"""
    assert ex0001_data.EXPERIMENT_NAME == "ex0001_a9a_logistic"
    assert ex0001_train.EXPERIMENT_NAME == "ex0001_a9a_logistic"
    assert ex0001_train.OUTPUT_ROOT.endswith(os.path.join("outputs", "ex0001_a9a_logistic"))


def test_model_has_bias_term_and_a9a_dimension():
    """実験0-1のモデルは切片項を含み（実験0とは異なる），既定の入力次元がa9aの123であることを
    確認する．"""
    model = ex0001_model.load_model(seed=0)
    assert model.linear.bias is not None
    assert model.linear.weight.shape == (1, 123)
    assert model.linear.weight.dtype == torch.float64


def test_loss_is_regularized_bce():
    """損失が `binary_cross_entropy(sigmoid(z), y) + λ/2(||w||^2 + b^2)` と一致することを
    確認する．"""
    X, y = _make_synthetic_data()
    model = ex0001_model.load_model(seed=1, input_dim=X.shape[1])
    reg_lambda = 0.3

    loss = ex0001_model.compute_loss(model, X, y, reg_lambda)

    with torch.no_grad():
        p = torch.sigmoid(model(X))
        expected = torch.nn.functional.binary_cross_entropy(p, y) + 0.5 * reg_lambda * (
            torch.sum(model.linear.weight ** 2) + torch.sum(model.linear.bias ** 2)
        )
    assert np.isclose(loss, expected.item())


def test_autograd_gradient_matches_closed_form_with_bias_regularization():
    """自動微分による勾配が，重み・切片の両方に正則化を課した閉形式勾配と一致することを確認する．"""
    X, y = _make_synthetic_data()
    model = ex0001_model.load_model(seed=3, input_dim=X.shape[1])
    reg_lambda = 0.1

    grads = ex0001_model.compute_gradient(model, X, y, reg_lambda)

    w = model.linear.weight.detach().numpy().reshape(-1)
    b = model.linear.bias.detach().numpy().reshape(-1)
    X_np, y_np = X.numpy(), y.numpy()
    p = 1.0 / (1.0 + np.exp(-(X_np @ w + b)))
    expected_grad_w = (X_np.T @ (p - y_np)) / X_np.shape[0] + reg_lambda * w
    expected_grad_b = np.mean(p - y_np) + reg_lambda * b

    assert np.allclose(grads[0].numpy().reshape(-1), expected_grad_w)
    assert np.allclose(grads[1].numpy().reshape(-1), expected_grad_b)


def test_a9a_features_are_raw_binary_like_ex000():
    """a9aの特徴量が実験0と同一の無加工の0/1値であり，分割後のデータ数が実験0と一致することを
    確認する（標準化を行っていないことの確認）．"""
    train_dataloader, test_dataloader = ex0001_data.load_dataloader(seed=0, batch_size=1)
    X_train, y_train = train_dataloader.dataset.tensors
    X_test, _ = test_dataloader.dataset.tensors

    assert X_train.shape == (29304, 123)
    assert X_test.shape == (3257, 123)
    assert set(torch.unique(X_train).tolist()) <= {0.0, 1.0}
    assert set(torch.unique(y_train).tolist()) == {0.0, 1.0}


def test_hyperparameters_on_a9a_match_analytic_values():
    """a9aに対して $ L_0 = 15/4 $，$ \\lambda = 0.03 L_0 $，$ L = L_0 + \\lambda $ となり，
    3つの学習率がいずれも上界 $ 1/(3L) $ 未満で，$ \\eta_b $ が $ \\rho = 0.5 $ を満たすことを
    確認する．"""
    hyperparameter_list = ex0001_train.build_hyperparameter_list()
    L0 = 15.0 / 4.0
    reg_lambda = ex0001_train.LAMBDA_RATIO * L0

    assert len(hyperparameter_list) == 3
    for hyperparams in hyperparameter_list:
        assert np.isclose(hyperparams["reg_lambda"], reg_lambda)
        assert np.isclose(hyperparams["L"], L0 + reg_lambda)
        assert 0.0 < hyperparams["eta"] < 1.0 / (3.0 * hyperparams["L"])

    eta_b = hyperparameter_list[1]
    assert eta_b["eta_rule"] == "eta_b_rho=0.5"
    assert np.isclose(eta_b["rho"], ex0001_train.TARGET_RHO, rtol=1e-4)


@pytest.mark.parametrize("method", ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"])
def test_training_runs_without_error_on_synthetic_data(method):
    """4手法が合成データ上でエラーなく完走し，記録された評価指標の本数がエポック数+1と一致する
    ことを確認する（スモークテスト）．"""
    X_train, y_train = _make_synthetic_data(N=30, d=5, seed=11)
    X_test, y_test = _make_synthetic_data(N=10, d=5, seed=12)
    reg_lambda = 0.1
    epochs = 3

    _, f_star = ex0001_train.compute_optimal_solution(X_train.numpy(), y_train.numpy(), reg_lambda)

    logger = ResultLogger()
    logger.set_names(*ex0001_train.METRIC_NAMES)

    if method == "SGD":
        ex0001_train.run_sgd(
            0, X_train, y_train, X_test, y_test, 0.1, reg_lambda, f_star, epochs, logger
        )
    else:
        ex0001_train.run_variance_reduced(
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

    _, f_star = ex0001_train.compute_optimal_solution(X_train.numpy(), y_train.numpy(), reg_lambda)

    expected_per_epoch = {"SGD": N, "NFG_SVRG": 2 * N, "ASAI_SVRG": 2 * N, "SVRG": 3 * N}
    for method, per_epoch in expected_per_epoch.items():
        logger = ResultLogger()
        logger.set_names(*ex0001_train.METRIC_NAMES)
        if method == "SGD":
            ex0001_train.run_sgd(
                0, X_train, y_train, X_test, y_test, 0.05, reg_lambda, f_star, epochs, logger
            )
        else:
            ex0001_train.run_variance_reduced(
                method, 0, X_train, y_train, X_test, y_test, 0.05, reg_lambda, f_star, epochs, logger
            )

        initial_full_grad = N if method == "SVRG" else 0
        assert logger["oracle_calls"][0] == 0
        assert logger["oracle_calls"][1] == initial_full_grad + per_epoch
        assert logger["oracle_calls"][2] - logger["oracle_calls"][1] == per_epoch
