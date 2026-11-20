"""
E4（付録B.1，a9a の MSE・BCE の再実施）のプログラムのテスト．

`実験指示E4.md` 2.6節に対応する．
1. データの分割（学習・テストの添字と件数）が，既存の `programs/ex000_a9a_least_squares/data.py` と同一であること．
2. モデルの初期値が，既存のプログラム（`ex000`：MSE，`ex0001`：BCE）と同一であること．
3. $ L $，$ \\lambda $，$ \\eta $，BCE の $ f({\\mathbf w}^{*}) $ が実験指示E4.md 2.1節の表の値と一致すること．
4. 最終点をスナップショットとする手法（SVRG，NFG-SVRG）で，`inner_start` の2つの値の結果がビット単位で一致すること．
5. `inner_start="snapshot"` で，ASA-SVRG の各内側ループが $ {\\mathbf z}_{s} $ から始まること．
6. タスクの構築で，2実験 × 4手法 × 5シード = 40個の保存先が実験指示E4.md 2.4節の名前で作られること．

1・2の参照値は，既存のプログラム（`programs/ex000_a9a_least_squares/`，`programs/ex0001_a9a_logistic/`）の
`load_dataloader(seed)`・`load_model(seed=seed)` を，本リポジトリと同じ a9a のファイルに対して実行して求めた値である．
`programs_review/` の外を参照しないため，定数としてテストに埋め込む．
"""

import importlib
import os
import sys

import numpy as np
import pytest
import torch

_REVIEW_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EX008_DIR = os.path.join(_REVIEW_ROOT, "programs", "ex008_e4_a9a")
sys.path.insert(0, os.path.join(_REVIEW_ROOT, "programs", "common"))

from machine_learning_utils import ResultLogger  # noqa: E402
from optimizers import ASAISVRG  # noqa: E402


def _import_ex008_modules():
    """
    概要: `ex008_e4_a9a` の `data`，`model`，`train` を読み込む．他の実験のテスト（E1 等）が同じ名前の
        モジュールを `sys.modules` に登録しているため，一時的に退避してから読み込み，読み込んだ後に元に戻す．
    引数: なし
    戻り値: (data, model, train) のモジュールの組．
    """
    names = ("data", "model", "train")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    sys.path.insert(0, _EX008_DIR)
    try:
        modules = tuple(importlib.import_module(name) for name in names)
    finally:
        sys.path.remove(_EX008_DIR)
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return modules


e4_data, e4_model, e4_train = _import_ex008_modules()

_DATA_AVAILABLE = os.path.exists(e4_data.RAW_DATA_PATH)
requires_data = pytest.mark.skipif(not _DATA_AVAILABLE, reason="a9a のデータ（datasets/a9a/raw/a9a）が無い")

# 既存の data.py で求めた分割の参照値（ex000 と ex0001 で同一）．
# train_head10・test_head10 は全データ（32561件）に対する添字の先頭10件．checksum は
# (X @ [1, ..., d]) @ [1, ..., N] で，0/1 の特徴量に対して倍精度で厳密に計算できる整数値である．
EXPECTED_SPLIT = {
    0: {"train_head10": [31630, 30766, 14330, 388, 1196, 9125, 30171, 27147, 23015, 3],
        "test_head10": [18685, 31479, 5764, 21048, 18132, 16844, 20526, 18197, 7364, 21105],
        "X_train_checksum": 296862765700.0, "X_test_checksum": 3666012406.0},
    1: {"train_head10": [263, 28246, 23287, 22197, 19854, 8893, 17863, 867, 14738, 16017],
        "test_head10": [31930, 9615, 13110, 21713, 11228, 16481, 10518, 13131, 17394, 16065],
        "X_train_checksum": 296831999601.0, "X_test_checksum": 3674577409.0},
    2: {"train_head10": [31986, 31563, 3506, 19739, 26929, 28272, 25445, 17633, 31212, 1961],
        "test_head10": [4182, 23950, 12235, 14741, 21477, 9384, 9162, 16488, 2643, 25262],
        "X_train_checksum": 296872558448.0, "X_test_checksum": 3670564189.0},
    3: {"train_head10": [4218, 8024, 28264, 5003, 2932, 21098, 24817, 7626, 12350, 17746],
        "test_head10": [24415, 4998, 14152, 3220, 32533, 22397, 23316, 12173, 21458, 27431],
        "X_train_checksum": 296917355882.0, "X_test_checksum": 3667787510.0},
    4: {"train_head10": [5532, 2486, 4868, 6115, 30646, 20979, 17061, 11059, 24879, 15483],
        "test_head10": [18612, 28709, 23466, 20471, 20285, 19149, 3303, 23566, 6164, 16481],
        "X_train_checksum": 296888124836.0, "X_test_checksum": 3670563216.0},
}
EXPECTED_N_TRAIN, EXPECTED_N_TEST = 29304, 3257
EXPECTED_Y_TRAIN_SUM, EXPECTED_Y_TEST_SUM = 7057.0, 784.0

# 既存の model.py の load_model(seed=seed) の初期値の参照値．パラメータを weight（, bias）の順に平坦化した．
# 重みの先頭5成分は MSE と BCE で同一（同じ乱数の消費）．
_HEAD5 = {
    0: [0.0847665036826489, 0.03747697224199842, -0.007324633366307538, 0.07587508212588322, 0.026152769081811824],
    1: [-0.07915698668440453, -0.04967216748678745, -0.04792328773302904, -0.05822996432420278, 0.010110972740875402],
    2: [0.07530035456901046, -0.07268986263284158, 0.03764491944731631, 0.0072605143563318686, 0.07452439320348614],
    3: [-0.0840160986044692, -0.03845797237229172, 0.04921582362960744, -0.05862168534963838, 0.046051851774405336],
    4: [-0.0041279545229746546, 0.04177520649720089, -0.0797874769307213, 0.0799162437926171, 0.0809036650252475],
}
EXPECTED_INIT = {
    "mse": {0: (-0.22811952408519554, -0.07947418342353345), 1: (0.7281707949963047, 0.08506169992844052),
            2: (-0.4710655518240941, -0.014199778486703927), 3: (-0.9900164100622727, -0.0001038846136182539),
            4: (-0.04283320103820233, 0.049730654599943594)},
    "bce": {0: (-0.1574425713571967, 0.07067695272799883), 1: (0.6813328697639514, -0.04683792523235327),
            2: (-0.522931190433349, -0.05186563860925489), 3: (-0.9093069758211327, 0.08070943424113995),
            4: (-0.03960057890942967, 0.003232622128772664)},
}
EXPECTED_N_PARAMS = {"mse": 123, "bce": 124}

# 実験指示E4.md 2.1節の表．
EXPECTED_TABLE = {
    "mse": {"L": 2.1568199747494745, "eta": 1.5300303403316318e-1},
    "bce": {"L": 3.8625, "reg_lambda": 0.1125, "eta": 8.543689320388351e-2, "rho": 297.36,
            "f_star": 0.4740115090431359},
}
# MSE の既存の f_star_reference（非凸のため参照値．L-BFGS-B の停止位置が BLAS により僅かに変わり得る）．
EXPECTED_MSE_F_STAR_REFERENCE = 0.10329856539159467


def _rel(a: float, b: float) -> float:
    """
    概要: 相対誤差を計算する．
    引数: a, b (float)．
    戻り値: error (float)．
    """
    return abs(a - b) / abs(b)


def _checksum(X: np.ndarray) -> float:
    """
    概要: 特徴量行列の並びを含めたチェックサム $ (X [1, \\ldots, d]^{\\top})^{\\top} [1, \\ldots, N]^{\\top} $ を計算する．
    引数: X (numpy.ndarray)，形状 (N, d)．
    戻り値: value (float)．
    """
    return float((X @ np.arange(1, X.shape[1] + 1)) @ np.arange(1, X.shape[0] + 1))


# ---------------------------------------------------------------------------
# 1. データの分割
# ---------------------------------------------------------------------------

@requires_data
@pytest.mark.parametrize("seed", list(EXPECTED_SPLIT))
def test_data_split_matches_existing_program(seed):
    """
    概要: 学習・テストデータの件数，ラベルの合計，先頭10件の行，全体のチェックサムが既存の data.py と同一であることを確認する．
    引数: seed (int)．
    戻り値: なし
    """
    expected = EXPECTED_SPLIT[seed]
    A, _ = e4_data.build_features_and_labels()
    train_loader, test_loader = e4_data.load_dataloader(seed=seed)
    X_train, y_train = [t.numpy() for t in train_loader.dataset.tensors]
    X_test, y_test = [t.numpy() for t in test_loader.dataset.tensors]
    assert (len(y_train), len(y_test)) == (EXPECTED_N_TRAIN, EXPECTED_N_TEST)
    assert (float(y_train.sum()), float(y_test.sum())) == (EXPECTED_Y_TRAIN_SUM, EXPECTED_Y_TEST_SUM)
    assert np.array_equal(X_train[:10], A[expected["train_head10"]])
    assert np.array_equal(X_test[:10], A[expected["test_head10"]])
    assert _checksum(X_train) == expected["X_train_checksum"]
    assert _checksum(X_test) == expected["X_test_checksum"]


# ---------------------------------------------------------------------------
# 2. モデルの初期値
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss", ["mse", "bce"])
@pytest.mark.parametrize("seed", list(_HEAD5))
def test_initial_parameters_match_existing_program(loss, seed):
    """
    概要: `load_model(loss, seed=seed)` の初期値（先頭5成分，総和，最後の成分，パラメータ数）が既存のプログラムと
        ビット単位で同一であることを確認する．
    引数: loss (str)．seed (int)．
    戻り値: なし
    """
    model = e4_model.load_model(loss, seed=seed)
    flat = np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])
    expected_sum, expected_last = EXPECTED_INIT[loss][seed]
    assert flat.size == EXPECTED_N_PARAMS[loss]
    assert flat[:5].tolist() == _HEAD5[seed]
    assert float(flat.sum()) == expected_sum
    assert float(flat[-1]) == expected_last


# ---------------------------------------------------------------------------
# 3. 定数
# ---------------------------------------------------------------------------

@requires_data
def test_mse_constants_match_table():
    """
    概要: MSE の $ L $ と $ \\eta = 0.99/(3L) $ が表の値と相対 $ 10^{-12} $ 以下で一致することを確認する．
    引数: なし
    戻り値: なし
    """
    hp = e4_train.build_hyperparameters("mse")
    assert _rel(hp["L"], EXPECTED_TABLE["mse"]["L"]) <= 1e-12
    assert _rel(hp["L_individual"], EXPECTED_TABLE["mse"]["L"]) <= 1e-12
    assert _rel(hp["eta"], EXPECTED_TABLE["mse"]["eta"]) <= 1e-12
    assert hp["reg_lambda"] == 0.0
    assert hp["hp_name"] == "eta1.530030e-01_K29304_epochs100"


@requires_data
def test_bce_constants_match_table():
    """
    概要: BCE の $ L $，$ \\lambda $，$ \\eta $，$ \\rho $ が表の値と一致することを確認する
        （$ \\rho $ は表の有効数字（5桁）で比べる）．
    引数: なし
    戻り値: なし
    """
    hp = e4_train.build_hyperparameters("bce")
    table = EXPECTED_TABLE["bce"]
    assert _rel(hp["L"], table["L"]) <= 1e-12
    assert _rel(hp["reg_lambda"], table["reg_lambda"]) <= 1e-12
    assert hp["mu"] == hp["reg_lambda"]
    assert _rel(hp["eta"], table["eta"]) <= 1e-12
    assert abs(hp["rho"] - table["rho"]) < 5e-3
    assert hp["hp_name"] == "eta8.543689e-02_lambda1.125000e-01_K29304_epochs100"


@requires_data
def test_bce_f_star_matches_existing_value():
    """
    概要: シード0の BCE の最適値 $ f({\\mathbf w}^{*}) $（L-BFGS-B）が既存の値と相対 $ 10^{-12} $ 以下で一致し，
        MSE の参照値が既存の値と相対 $ 10^{-8} $ 以下で一致することを確認する．MSE は非凸であり，参照値としてのみ
        用いるため許容範囲を緩くする．
    引数: なし
    戻り値: なし
    """
    train_loader, _ = e4_data.load_dataloader(seed=0)
    X, y = [t.numpy() for t in train_loader.dataset.tensors]
    _, f_star = e4_train.compute_optimal_solution(X, y, "bce", e4_train.build_hyperparameters("bce")["reg_lambda"])
    assert _rel(f_star, EXPECTED_TABLE["bce"]["f_star"]) <= 1e-12
    _, f_ref = e4_train.compute_optimal_solution(X, y, "mse", 0.0)
    assert _rel(f_ref, EXPECTED_MSE_F_STAR_REFERENCE) <= 1e-8


def test_metric_names_start_with_existing_log_keys():
    """
    概要: `log.json` の先頭8項目が既存のログのキーと同じ名前・順序であることを確認する．
    引数: なし
    戻り値: なし
    """
    common = ["epoch", "oracle_calls", "full_grad_computations", "elapsed_time"]
    tail = ["grad_norm_sq", "accuracy", "approx_error"]
    assert list(e4_train.metric_names("mse")[:8]) == common + ["objective_value"] + tail
    assert list(e4_train.metric_names("bce")[:8]) == common + ["objective_gap"] + tail


# ---------------------------------------------------------------------------
# 4・5. 内側ループの開始点
# ---------------------------------------------------------------------------

def _make_small_data(n=12, d=3, seed=0):
    """
    概要: テスト用の小さな二値分類のデータを作る（0/1 の特徴量）．
    引数: n (int)．d (int)．seed (int)．
    戻り値: (X_train, y_train, X_test, y_test) の組．torch.float64 のテンソル．
    """
    g = torch.Generator().manual_seed(seed)
    X = (torch.rand(n, d, generator=g, dtype=torch.float64) > 0.5).double()
    y = (torch.rand(n, generator=g, dtype=torch.float64) > 0.5).double()
    return X, y, X[:4], y[:4]


def _run_vr(method, inner_start, loss, epochs=4):
    """
    概要: 小さなデータで SVRG 系の手法を実行し，ログを返す．
    引数: method (str)．inner_start (str)．loss (str)．epochs (int) = 4．
    戻り値: logger (ResultLogger)．
    """
    logger = ResultLogger()
    logger.set_names(*e4_train.metric_names(loss))
    reg_lambda = 1e-2 if loss == "bce" else 0.0
    e4_train.run_variance_reduced(method, inner_start, 0, _make_small_data(), loss, eta=5e-2, reg_lambda=reg_lambda,
                                  f_star=0.0, epochs=epochs, logger=logger)
    return logger


@pytest.mark.parametrize("loss", ["mse", "bce"])
@pytest.mark.parametrize("method", ["SVRG", "NFG_SVRG"])
def test_final_point_methods_are_bitwise_identical_for_both_inner_starts(method, loss):
    """
    概要: 最終点をスナップショットとする手法では $ {\\mathbf z}_{s} = {\\mathbf w}_{s-1}^{K} $ のため，
        `inner_start` の2つの値の結果（`elapsed_time` を除く全項目）がビット単位で一致することを確認する．
    引数: method (str)．loss (str)．
    戻り値: なし
    """
    a, b = _run_vr(method, "snapshot", loss), _run_vr(method, "last_iterate", loss)
    for key in e4_train.metric_names(loss):
        if key == "elapsed_time":
            continue
        assert np.array_equal(np.asarray(a[key], dtype=np.float64), np.asarray(b[key], dtype=np.float64),
                              equal_nan=True), key


def test_asa_svrg_inner_starts_differ():
    """
    概要: ASA-SVRG（平均点をスナップショットとする）では，2つの開始点の結果が異なることを確認する（切り替えが機能していること）．
    引数: なし
    戻り値: なし
    """
    a, b = _run_vr("ASAI_SVRG", "snapshot", "bce"), _run_vr("ASAI_SVRG", "last_iterate", "bce")
    assert a["objective_gap"][1] == b["objective_gap"][1]  # 1回目の内側ループは同じ点から始まる
    assert a["objective_gap"][2:] != b["objective_gap"][2:]


@pytest.mark.parametrize("loss", ["mse", "bce"])
def test_snapshot_inner_start_starts_each_inner_loop_from_z_s(loss):
    """
    概要: `inner_start="snapshot"` で，ASA-SVRG の各外側ループの内側ループが，直前の外側ループの終了時の
        スナップショット $ {\\mathbf z}_{s} $ から始まることを確認する．
    引数: loss (str)．
    戻り値: なし
    """
    data = _make_small_data()
    X_train, y_train, _, _ = data
    N_train, input_dim = X_train.shape
    reg_lambda = 1e-3 if loss == "bce" else 0.0

    recorded_start_points = []
    original_run_inner_loop = e4_train.run_inner_loop

    def recording_run_inner_loop(model, snapshot_model, optimizer, X, y, loss_, reg_lambda_, indices):
        recorded_start_points.append([p.detach().clone() for p in model.parameters()])
        original_run_inner_loop(model, snapshot_model, optimizer, X, y, loss_, reg_lambda_, indices)

    e4_train.run_inner_loop = recording_run_inner_loop
    try:
        logger = ResultLogger()
        logger.set_names(*e4_train.metric_names(loss))
        e4_train.run_variance_reduced("ASAI_SVRG", "snapshot", 0, data, loss, eta=1e-2, reg_lambda=reg_lambda,
                                      f_star=0.0, epochs=3, logger=logger)
    finally:
        e4_train.run_inner_loop = original_run_inner_loop

    # 独立に再現した z_s と，記録した開始点を比べる．
    model = e4_model.load_model(loss, seed=0, input_dim=input_dim)
    snapshot_model = e4_model.load_model(loss, seed=0, input_dim=input_dim)
    optimizer = ASAISVRG(model.parameters(), lr=1e-2, K=N_train)
    rng = np.random.default_rng(0)
    assert len(recorded_start_points) == 3
    for epoch in range(1, 4):
        indices = e4_train.get_epoch_indices(rng, N_train)
        current_params = [p.detach().clone() for p in model.parameters()]
        assert all(torch.equal(a, b) for a, b in zip(recorded_start_points[epoch - 1], current_params))
        optimizer.begin_epoch(rng)
        e4_train.run_inner_loop(model, snapshot_model, optimizer, X_train, y_train, loss, reg_lambda, indices)
        optimizer.end_epoch()
        z_s = optimizer.get_snapshot_params()
        e4_model.set_model_params(snapshot_model, z_s)
        e4_model.set_model_params(model, z_s)


def test_sgd_svrg_specific_fields_are_nan_and_oracle_calls():
    """
    概要: SGD のログで SVRG 系に固有の項目（`approx_error`，`approx_cos`）が NaN であり，1エポックの
        `oracle_calls` の増分が $ N_{\\mathrm{train}} $ であることを確認する．
    引数: なし
    戻り値: なし
    """
    data = _make_small_data(n=16)
    logger = ResultLogger()
    logger.set_names(*e4_train.metric_names("bce"))
    e4_train.run_sgd(0, data, "bce", eta=1e-2, reg_lambda=1e-3, f_star=0.0, epochs=3, logger=logger)
    assert all(np.isnan(v) for v in logger["approx_error"])
    assert all(np.isnan(v) for v in logger["approx_cos"])
    assert np.all(np.diff(logger["oracle_calls"]) == 16)


# ---------------------------------------------------------------------------
# 6. タスクの構築
# ---------------------------------------------------------------------------

@requires_data
def test_build_tasks_creates_40_directories_with_expected_names():
    """
    概要: 既定の条件（2実験 × 4手法 × 5シード，`inner_start="snapshot"`）で40個の保存先が作られ，
        名前が実験指示E4.md 2.4節の形式であることを確認する．
    引数: なし
    戻り値: なし
    """
    tasks = e4_train.build_tasks(["mse", "bce"], e4_train.ALL_METHODS, e4_train.DEFAULT_INNER_STARTS,
                                 e4_train.SEEDS, e4_train.EPOCHS, "/out")
    dirs = {t["target_dir"] for t in tasks}
    assert len(tasks) == 40 and len(dirs) == 40
    expected = set()
    for method in e4_train.ALL_METHODS:
        start = "none" if method == "SGD" else "snapshot"
        for seed in e4_train.SEEDS:
            expected.add(f"/out/mse/{method}/inner_{start}_eta1.530030e-01_K29304_epochs100/{seed}")
            expected.add(f"/out/bce/{method}/inner_{start}_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/{seed}")
    assert dirs == expected
    assert "/out/bce/ASAI_SVRG/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0" in dirs
