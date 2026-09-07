"""
`programs/ex0021_cifar10_alexnet_norm/`（実験ex0021：CIFAR-10における正規化層・バッチサイズの
構造的変更による安定性の検証，および`.orders/order_024.md`によるスペクトル正規化の追加検証）の
単体テスト．

正規化層（LayerNorm2d，GroupNorm）がバッチ内の他サンプルに依存しない（サンプル単位で完結する）
こと，L2正則化が重みのみに課され正規化層のアフィンパラメータを除外すること，2手法（NFG SVRG，
ASAI SVRG）が合成データ上でエラーなく完走すること，オラクル呼び出し回数の正しさ，バッチサイズ
ごとのエポック数の対応が総イテレーション数をおおむね揃えることを確認する．

`.orders/order_024.md` 4節の指示により，SpectralConv2dのpower iterationバッファを凍結した後，
同一入力に対する勾配評価が完全に決定論的（再現可能）であることを検証する単体テスト
（`test_spectral_norm_snapshot_gradient_is_deterministic`）を必須項目として含む．
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROGRAMS_DIR = os.path.join(_PROJECT_ROOT, "programs")
_EXPERIMENT_DIR = os.path.join(_PROGRAMS_DIR, "ex0021_cifar10_alexnet_norm")
sys.path.insert(0, _EXPERIMENT_DIR)
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import ResultLogger  # noqa: E402

for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_spec_model = importlib.util.spec_from_file_location(
    "ex0021_model", os.path.join(_EXPERIMENT_DIR, "model.py")
)
ex0021_model = importlib.util.module_from_spec(_spec_model)
_spec_model.loader.exec_module(ex0021_model)

for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_spec_train = importlib.util.spec_from_file_location(
    "ex0021_train", os.path.join(_EXPERIMENT_DIR, "train.py")
)
ex0021_train = importlib.util.module_from_spec(_spec_train)
_spec_train.loader.exec_module(ex0021_train)


class _SyntheticDataset(torch.utils.data.Dataset):
    """テスト用の小規模な合成CIFAR-10形式データセット（ランダム画像・ランダムラベル）．"""

    def __init__(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        self.images = torch.tensor(rng.normal(size=(n, 3, 32, 32)), dtype=torch.float32)
        self.labels = torch.tensor(rng.integers(0, 10, size=n), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def _make_synthetic_dataloaders(n_train=24, n_test=8, batch_size=8, seed=0):
    train_ds = _SyntheticDataset(n_train, seed)
    test_ds = _SyntheticDataset(n_test, seed + 1)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl


@pytest.mark.parametrize("norm_type", ["none", "layernorm", "groupnorm", "spectralnorm"])
def test_model_has_no_batchnorm_or_dropout(norm_type):
    """BatchNormalization・Dropout層を含まないことを確認する．"""
    model = ex0021_model.AlexNetCIFARNorm(norm_type=norm_type)
    for module in model.modules():
        assert not isinstance(module, (torch.nn.BatchNorm2d, torch.nn.Dropout))


@pytest.mark.parametrize("norm_type", ["layernorm", "groupnorm"])
def test_normalization_is_sample_independent(norm_type):
    """LayerNorm2d・GroupNormが，バッチ内の他サンプルの値を変えても当該サンプルの出力に
    影響しない（サンプル単位で完結する）ことを確認する．SVRG系手法の理論的前提の検証．"""
    model = ex0021_model.AlexNetCIFARNorm(norm_type=norm_type)
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    x_perturbed = x.clone()
    x_perturbed[1] = torch.randn(3, 32, 32) * 10.0

    with torch.no_grad():
        y = model(x)[0]
        y_perturbed = model(x_perturbed)[0]

    assert torch.allclose(y, y_perturbed, atol=1e-5)


def test_layernorm2d_matches_manual_computation():
    """`LayerNorm2d` が (B,C,H,W) の各空間位置でチャネル方向に正規化することを，
    手動実装との比較で確認する．"""
    torch.manual_seed(0)
    num_channels = 8
    layer = ex0021_model.LayerNorm2d(num_channels)
    x = torch.randn(2, num_channels, 3, 3)

    y = layer(x)

    x_permuted = x.permute(0, 2, 3, 1)
    expected = torch.nn.functional.layer_norm(
        x_permuted, (num_channels,), layer.layer_norm.weight, layer.layer_norm.bias
    ).permute(0, 3, 1, 2)
    assert torch.allclose(y, expected, atol=1e-5)


def test_groupnorm_group_count_is_channels_divided_by_four():
    """GroupNormのグループ数が，チャネル数の4分の1（1グループあたり4チャンネル）で
    構築されることを確認する．"""
    model = ex0021_model.AlexNetCIFARNorm(norm_type="groupnorm")
    group_norms = [m for m in model.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert len(group_norms) == 5
    for gn in group_norms:
        assert gn.num_channels % gn.num_groups == 0
        assert gn.num_channels // gn.num_groups == 4


def test_l2_regularization_excludes_bias_and_norm_affine_params():
    """L2正則化項が畳み込み層・全結合層の重み（`weight`）のみに課され，切片および正規化層の
    アフィンパラメータを含まないことを確認する．"""
    model = ex0021_model.AlexNetCIFARNorm(norm_type="groupnorm")
    reg_lambda = 0.1

    expected = next(model.parameters()).new_zeros(())
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            expected = expected + torch.sum(module.weight ** 2)
    expected = 0.5 * reg_lambda * expected

    actual = ex0021_model.compute_l2_regularization(model, reg_lambda)
    assert torch.allclose(actual, expected)


def test_spectral_norm_l2_regularization_does_not_double_count():
    """`SpectralConv2d` は内部に生の `nn.Conv2d`（`self.conv`）を保持し，`model.modules()`
    の走査ではこの内部Conv2dも独立したConv2dとして訪問されるため，実装によっては正則化項が
    二重に計上されうる．`compute_l2_regularization` がこれを正しく回避し，全結合層の重みと
    各SpectralConv2dの生の重みをそれぞれ1回ずつ数え上げることを確認する．"""
    model = ex0021_model.AlexNetCIFARNorm(norm_type="spectralnorm")
    reg_lambda = 0.1

    expected = next(model.parameters()).new_zeros(())
    for module in model.modules():
        if isinstance(module, ex0021_model.SpectralConv2d):
            expected = expected + torch.sum(module.conv.weight ** 2)
        elif isinstance(module, torch.nn.Linear):
            expected = expected + torch.sum(module.weight ** 2)
    expected = 0.5 * reg_lambda * expected

    actual = ex0021_model.compute_l2_regularization(model, reg_lambda)
    assert torch.allclose(actual, expected)


def test_spectral_norm_total_parameter_count_matches_other_norm_types():
    """SpectralNormはpower iterationの補助ベクトル(u,v)をバッファ（学習対象パラメータでは
    ない）として保持するため，学習可能パラメータの総数は他の正規化パターンと一致する
    （`.orders/order_024.md` 3節が要求する層数・チャンネル数の公平性の一部）．"""
    n_none = sum(p.numel() for p in ex0021_model.AlexNetCIFARNorm(norm_type="none").parameters())
    n_spectral = sum(
        p.numel() for p in ex0021_model.AlexNetCIFARNorm(norm_type="spectralnorm").parameters()
    )
    assert n_none == n_spectral


def test_spectral_norm_unfrozen_buffers_update_every_forward_call():
    """`frozen=False`（既定，`model` 用）の間は，forwardのたびにpower iterationバッファが
    更新され，同一入力に対する出力が呼び出しごとに変化しうることを確認する（凍結時の決定論性
    との対比）．"""
    torch.manual_seed(0)
    model = ex0021_model.AlexNetCIFARNorm(norm_type="spectralnorm")
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        y1 = model(x).clone()
        y2 = model(x).clone()

    assert not torch.equal(y1, y2)


def test_spectral_norm_snapshot_gradient_is_deterministic():
    """`.orders/order_024.md` 4節が必須とする単体テスト．スナップショット
    （`refresh_and_freeze_spectral_norm` で凍結したモデル）に対し，同一インデックス（同一の
    入力バッチ）に対する勾配 $ \\nabla f_n(z_s) $ を複数回評価し，出力・勾配の双方が完全に
    同一の値になる（決定論的である）ことを確認する．"""
    torch.manual_seed(0)
    model = ex0021_model.AlexNetCIFARNorm(norm_type="spectralnorm")
    ex0021_model.refresh_and_freeze_spectral_norm(model)

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    def evaluate_gradient():
        model.zero_grad(set_to_none=True)
        outputs = model(x)
        loss = ex0021_model.loss_func(outputs, y, model, 0.01)
        loss.backward()
        return outputs.detach().clone(), [p.grad.clone() for p in model.parameters()]

    outputs_1, grads_1 = evaluate_gradient()
    outputs_2, grads_2 = evaluate_gradient()
    outputs_3, grads_3 = evaluate_gradient()

    assert torch.equal(outputs_1, outputs_2)
    assert torch.equal(outputs_1, outputs_3)
    for g1, g2 in zip(grads_1, grads_2):
        assert torch.equal(g1, g2)
    for g1, g3 in zip(grads_1, grads_3):
        assert torch.equal(g1, g3)


def test_spectral_norm_power_iteration_has_converged_after_burn_in():
    """`.orders/order_024.md` 4節が要求する収束確認．`refresh_and_freeze`（バーンイン，
    `_SNAPSHOT_POWER_ITERATIONS` 回のpower iteration）の後，さらに反復を重ねても最大特異値の
    推定値（`current_sigma`）がほとんど変化しない（連続する推定値の相対変化が閾値以下である）
    ことを，モデルの全SpectralConv2d層について確認する．"""
    torch.manual_seed(0)
    model = ex0021_model.AlexNetCIFARNorm(norm_type="spectralnorm")

    relative_change_threshold = 1e-2
    for module in model.modules():
        if not isinstance(module, ex0021_model.SpectralConv2d):
            continue

        module.refresh_and_freeze(power_iterations=ex0021_model._SNAPSHOT_POWER_ITERATIONS)
        sigma_after_burn_in = module.current_sigma().item()

        # バーンインの続きとしてさらに5回反復し，推定値がほとんど変化しないことを確認する．
        module.frozen = False
        with torch.no_grad():
            module._spectral_norm_weight(update_buffers=True, power_iterations=5)
        module.frozen = True
        sigma_after_more_iterations = module.current_sigma().item()

        relative_change = abs(sigma_after_more_iterations - sigma_after_burn_in) / abs(sigma_after_burn_in)
        assert relative_change < relative_change_threshold


@pytest.mark.parametrize("method", ["NFG_SVRG", "ASAI_SVRG"])
@pytest.mark.parametrize("norm_type", ["none", "layernorm", "groupnorm", "spectralnorm"])
def test_training_runs_without_error_on_synthetic_data(method, norm_type, tmp_path):
    """2手法 x 3正規化パターンが合成データ上でエラーなく完走することを確認する
    （スモークテスト）．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=16, n_test=8, batch_size=batch_size, seed=seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    ex0021_train.run_variance_reduced(
        method, str(tmp_path), load_dataloader_func, norm_type, batch_size,
        0.01, epochs, device, 0, logger,
    )

    assert len(logger["epoch"]) == epochs + 1
    assert all(np.isfinite(v) for v in logger["train_loss"])


def test_oracle_calls_accounting_per_epoch():
    """1エポックあたりのオラクル呼び出し回数が，NFG SVRG・ASAI SVRGいずれも 2N
    （内部ループの現在パラメータ・スナップショットの双方でのforward/backward）となることを
    確認する．"""
    device = torch.device("cpu")
    batch_size = 4
    epochs = 2
    n_train = 16

    def load_dataloader_func(seed=0, batch_size=batch_size):
        return _make_synthetic_dataloaders(n_train=n_train, n_test=8, batch_size=batch_size, seed=seed)

    for method in ["NFG_SVRG", "ASAI_SVRG"]:
        logger = ResultLogger()
        logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")
        ex0021_train.run_variance_reduced(
            method, "/tmp", load_dataloader_func, "none", batch_size, 0.01, epochs, device, 0, logger,
        )
        assert logger["oracle_calls"][0] == 0
        assert logger["oracle_calls"][1] == 2 * n_train
        assert logger["oracle_calls"][2] - logger["oracle_calls"][1] == 2 * n_train


def test_epochs_by_batch_size_keep_total_iterations_roughly_constant():
    """バッチサイズごとに設定したエポック数が，総イテレーション数（epoch×K）をおおむね
    一定に保つことを確認する（`.orders/order_023.md` 4.1節の要求）．"""
    n_train = 50000
    totals = []
    for batch_size, epochs in ex0021_train.EPOCHS_BY_BATCH_SIZE.items():
        k = -(-n_train // batch_size)  # ceil
        totals.append(epochs * k)

    baseline = totals[0]
    for total in totals:
        assert abs(total - baseline) / baseline < 0.05
