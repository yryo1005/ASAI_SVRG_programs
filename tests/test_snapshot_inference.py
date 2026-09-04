"""
`.orders/order_005.md` の指摘事項の確認：SVRG，NFG SVRG，ASAI SVRGが，検証用データに対する
推論（評価）を「スナップショット」のパラメータを用いて行っていることを，
`programs/ex002_cifar10_cnn/train.py` の実際の関数（`train_variance_reduced`）を用いて
統合テストとして検証する．特に，ASAI SVRGの推論時パラメータが，内部ループにおける
パラメータ列の平均（式(13)）と一致することを重点的に確認する．

`programs/optimizers/optimizers.py` 自体の単体テスト（`tests/test_optimizers.py`）は，
`ASAISVRG.get_snapshot_params()` が平均パラメータを返すことを既に確認済みだが，本テストは
その値が実際に `train.py` の評価コード（`epoch(snapshot_model, ...)`）へ正しく渡され，
推論に使用されていることまで一気通貫で確認する点で異なる．
"""

import importlib.util
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROGRAMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "programs")
sys.path.insert(0, os.path.join(_PROGRAMS_DIR, "ex002_cifar10_cnn"))
sys.path.insert(0, _PROGRAMS_DIR)

# 他の実験（ex001, ex003等）のテストが同名モジュール（model, data, train）をsys.modulesに
# キャッシュしている場合，train.py内部の `from model import ...` 等の素朴なimportが誤った
# モジュールを解決してしまうため，明示的にキャッシュを破棄してから読み込む．
for _stale_name in ("model", "data", "train"):
    sys.modules.pop(_stale_name, None)

_TRAIN_PATH = os.path.join(_PROGRAMS_DIR, "ex002_cifar10_cnn", "train.py")
_spec = importlib.util.spec_from_file_location("ex002_train_for_test", _TRAIN_PATH)
ex002_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex002_train)

from machine_learning_utils import ResultLogger  # noqa: E402


N_SAMPLES = 16
NUM_BATCHES_PER_EPOCH = 4  # batch_size = N_SAMPLES // NUM_BATCHES_PER_EPOCH = 4


def _make_synthetic_dataloader(seed: int, batch_size: int):
    """CIFAR-10の代わりに用いる小規模な合成データセットのDataLoaderを構築する．"""
    rng = np.random.default_rng(seed)
    X = torch.tensor(rng.standard_normal((N_SAMPLES, 3, 32, 32)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, 10, size=N_SAMPLES), dtype=torch.long)
    dataset = TensorDataset(X, y)

    generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def _run_with_instrumentation(method: str, epochs: int, tmp_path):
    """
    概要: `train_variance_reduced` を合成データで実行しつつ，(1) 各内部ループステップに
        おける現在のパラメータ w_s^k の軌跡，(2) `snapshot_model` へ実際に書き込まれた
        パラメータ値，(3) 検証用データの評価に用いられたモデルが `snapshot_model` と
        同一オブジェクトであったか，の3点を計装（instrumentation）により記録する．
    引数:
        method (str)．"SVRG"，"NFG_SVRG"，"ASAI_SVRG" のいずれか．
        epochs (int)．外部ループ数．
        tmp_path．pytestの一時ディレクトリ．
    戻り値:
        trajectories_per_epoch (list of list of torch.Tensor)．各エポックにおける
            w_s^k（重みパラメータのみ）の軌跡．
        snapshot_updates (list of list of torch.Tensor)．各エポック終了時に
            `snapshot_model` へ書き込まれたパラメータ値．
        eval_used_snapshot_model (list of bool)．各評価呼び出しが `snapshot_model` と
            同一オブジェクトを用いていたか．
    """
    trajectories_per_epoch = []
    current_epoch_trajectory = []
    snapshot_updates = []
    eval_used_snapshot_model = []

    # train_variance_reduced は load_model を2回呼び出し，1回目が現在のパラメータを保持する
    # model，2回目がスナップショットを保持する snapshot_model である．load_modelをスパイして
    # snapshot_modelの参照を，エポック0の評価が行われる前（set_model_paramsが1度も呼ばれない
    # 段階）から把握できるようにする．
    created_models = []
    original_load_model = ex002_train.load_model

    def spy_load_model(ModelClass, weight_path=None, seed=0):
        model = original_load_model(ModelClass, weight_path=weight_path, seed=seed)
        created_models.append(model)
        return model

    def snapshot_model_ref():
        return created_models[1] if len(created_models) > 1 else None

    original_set_model_params = ex002_train.set_model_params

    def spy_set_model_params(model, param_values, source_model=None):
        snapshot_updates.append([v.clone() for v in param_values])
        original_set_model_params(model, param_values, source_model=source_model)

    original_iteration = ex002_train.iteration

    def spy_iteration(model, inputs, teacher_signals, optimizer=None, snapshot_model=None):
        if optimizer is not None:
            # 更新前のパラメータ w_s^k を記録する（重み行列のみ，1つ目のConv層で代表させる）
            current_epoch_trajectory.append(next(model.parameters()).detach().clone())
        else:
            # 評価呼び出し．modelがsnapshot_modelと同一オブジェクトかどうかを記録する．
            eval_used_snapshot_model.append(model is snapshot_model_ref())
        return original_iteration(model, inputs, teacher_signals, optimizer, snapshot_model)

    original_epoch = ex002_train.epoch

    def spy_epoch(model, dataloader, device, optimizer=None, snapshot_model=None):
        nonlocal current_epoch_trajectory
        result = original_epoch(model, dataloader, device, optimizer, snapshot_model)
        if optimizer is not None:
            trajectories_per_epoch.append(current_epoch_trajectory)
            current_epoch_trajectory = []
        return result

    os.makedirs(str(tmp_path), exist_ok=True)

    # 内部ループの各ステップにおけるパラメータの移動を，軌跡上の点同士が明確に区別できる
    # 程度に大きくするため，学習率を一時的に引き上げる（この検証は学習の質ではなく，
    # スナップショットの構成方法という構造的な性質の確認が目的である）．
    original_learning_rate = ex002_train.LEARNING_RATE
    ex002_train.LEARNING_RATE = 1.0

    ex002_train.load_model = spy_load_model
    ex002_train.set_model_params = spy_set_model_params
    ex002_train.iteration = spy_iteration
    ex002_train.epoch = spy_epoch
    try:
        logger = ResultLogger()
        logger.set_names(
            "epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error"
        )
        device = torch.device("cpu")
        ex002_train.train_variance_reduced(
            method,
            str(tmp_path),
            ex002_train.CNNModel,
            _make_synthetic_dataloader,
            epochs,
            N_SAMPLES // NUM_BATCHES_PER_EPOCH,
            device,
            seed=0,
            logger=logger,
        )
    finally:
        ex002_train.LEARNING_RATE = original_learning_rate
        ex002_train.load_model = original_load_model
        ex002_train.set_model_params = original_set_model_params
        ex002_train.iteration = original_iteration
        ex002_train.epoch = original_epoch

    return trajectories_per_epoch, snapshot_updates, eval_used_snapshot_model


def test_evaluation_uses_snapshot_model_for_all_variance_reduced_methods(tmp_path):
    """SVRG，NFG SVRG，ASAI SVRGのいずれも，検証用データの評価に snapshot_model を
    用いていることを確認する（現在のパラメータを保持する model ではない）．"""
    for method in ["SVRG", "NFG_SVRG", "ASAI_SVRG"]:
        _, _, eval_used_snapshot_model = _run_with_instrumentation(
            method, epochs=2, tmp_path=tmp_path / method
        )
        assert len(eval_used_snapshot_model) > 0
        assert all(eval_used_snapshot_model), (
            f"{method}: 評価にsnapshot_model以外のモデルが使用された箇所がある"
        )


def test_asai_svrg_snapshot_equals_average_of_trajectory(tmp_path):
    """ASAI SVRGにおいて，第2エポック終了時にsnapshot_modelへ書き込まれるパラメータが，
    第2エポックの内部ループにおけるパラメータ列 w_s^k の単純平均と一致することを確認する．"""
    trajectories_per_epoch, snapshot_updates, _ = _run_with_instrumentation(
        "ASAI_SVRG", epochs=2, tmp_path=tmp_path
    )

    # epoch 0 (初期評価) はsnapshot_updatesに含まれないため，
    # snapshot_updates[0] が epoch 1終了時，snapshot_updates[1] が epoch 2終了時に対応する．
    second_epoch_trajectory = trajectories_per_epoch[1]
    expected_average = torch.stack(second_epoch_trajectory).mean(dim=0)

    actual_snapshot_weight = snapshot_updates[1][0]  # 1つ目のパラメータ（Conv層の重み）

    assert torch.allclose(actual_snapshot_weight, expected_average, atol=1e-5)


def test_svrg_and_nfg_svrg_snapshot_is_a_trajectory_point_not_the_average(tmp_path):
    """SVRG，NFG SVRGにおいて，snapshot_modelへ書き込まれるパラメータが，平均ではなく
    内部ループのパラメータ列のいずれか1点（一様ランダムに選ばれた点）と一致することを確認する．"""
    for method in ["SVRG", "NFG_SVRG"]:
        trajectories_per_epoch, snapshot_updates, _ = _run_with_instrumentation(
            method, epochs=2, tmp_path=tmp_path / method
        )
        second_epoch_trajectory = trajectories_per_epoch[1]
        actual_snapshot_weight = snapshot_updates[1][0]

        matches_some_point = any(
            torch.allclose(actual_snapshot_weight, w, atol=1e-6) for w in second_epoch_trajectory
        )
        average = torch.stack(second_epoch_trajectory).mean(dim=0)
        assert matches_some_point, f"{method}: スナップショットが軌跡上のいずれの点とも一致しない"
        assert not torch.allclose(actual_snapshot_weight, average, atol=1e-5), (
            f"{method}: スナップショットが平均と一致してしまっている（ASAI SVRGと区別できない）"
        )
