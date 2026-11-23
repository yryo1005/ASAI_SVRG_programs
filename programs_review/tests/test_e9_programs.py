"""
E9 の集計（`programs/analysis_e9.py`）のテスト．小さな合成のログで，手計算の値と一致することを確かめる．
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "programs"))

import analysis_e9 as e9  # noqa: E402


def _synthetic_log() -> dict:
    """
    概要: $ N = 1 $ として，評価点 $ \\#\\mathrm{grad}/N = 0, 400, 500, 700, 1000, 1200 $ の合成のログを作る．
    引数: なし
    戻り値: log (dict)．
    """
    return {
        "epoch": [0, 1, 2, 3, 4, 5],
        "oracle_calls": [0, 400, 500, 700, 1000, 1200],
        "train_loss": [4.0, 3.0, 2.5, 2.0, 1.5, 0.1],
        "test_accuracy": [0.10, 0.20, 0.30, 0.25, 0.35, 0.99],
        "approx_error": [None, 9.0, 5.0, 1.0, 3.0, 100.0],
        "terminated": [False] * 6,
    }


def test_value_at_budget_uses_last_point_within_budget():
    """予算での値は，予算以下の最後の評価点（1000）の値であり，予算を超える点（1200）は使わない．"""
    row = e9.summarize_run(_synthetic_log(), budget=1000.0, n_train=1)
    assert row["test_accuracy_pct"] == pytest.approx(35.0)
    assert row["train_loss"] == pytest.approx(1.5)
    assert row["recorded_epochs"] == 5
    assert row["max_grad_per_n"] == pytest.approx(1200.0)
    assert row["terminated"] is False


def test_value_at_budget_excludes_points_beyond_budget():
    """予算を下げると，その予算以下の最後の点の値になる．予算以下の点が無ければ NaN．"""
    x = np.array([0.0, 400.0, 500.0, 700.0])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert e9.value_at_budget(x, values, budget=600.0) == 3.0
    assert e9.value_at_budget(x, values, budget=500.0) == 3.0
    assert np.isnan(e9.value_at_budget(x + 10.0, values, budget=5.0))


def test_latter_half_fluctuation_matches_hand_calculation():
    """対象の組は (500, 700)，(700, 1000) のみ．|0.25-0.30| と |0.35-0.25| の平均 = 0.075 → 7.5 ポイント．"""
    row = e9.summarize_run(_synthetic_log(), budget=1000.0, n_train=1)
    assert row["fluctuation_pt"] == pytest.approx(7.5)


def test_latter_half_median_matches_hand_calculation():
    """500 < x ≤ 1000 の点は 700，1000 のみ（500 は含まない，1200 は予算外）．中央値は (1.0 + 3.0) / 2 = 2.0．"""
    row = e9.summarize_run(_synthetic_log(), budget=1000.0, n_train=1)
    assert row["approx_error_median"] == pytest.approx(2.0)


def test_terminated_and_mean_standard_error(tmp_path):
    """打ち切りの記録と，平均 ± 標準誤差（ddof=1），ファイルからの読み込みと表の作成を確かめる．"""
    log = _synthetic_log()
    log["terminated"][-1] = True
    assert e9.summarize_run(log, budget=1000.0, n_train=1)["terminated"] is True

    mean, standard_error = e9.mean_and_standard_error([1.0, 2.0, 3.0, float("nan")])
    assert mean == pytest.approx(2.0)
    assert standard_error == pytest.approx(1.0 / np.sqrt(3.0))
    assert np.isnan(e9.mean_and_standard_error([5.0])[1])

    for seed in (0, 2):
        os.makedirs(tmp_path / str(seed))
        with open(tmp_path / str(seed) / "log.json", "w", encoding="utf-8") as file:
            json.dump(_synthetic_log(), file)
    logs = e9.load_runs(str(tmp_path))
    assert sorted(logs) == [0, 2]
    output = tmp_path / "summary.md"
    e9.main(["--run-root", str(tmp_path), "--output", str(output)])
    text = output.read_text(encoding="utf-8")
    assert "## シードごとの値" in text and "## 平均 ± 標準誤差" in text
