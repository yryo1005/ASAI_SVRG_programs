# ex015_e10_a9a_bce_uniform：E10（付録B.1，a9a の BCE の独立一様サンプリング）

対応する実行指示：`実験指示E10.md`．対応する指示・報告：`.orders/order_082.md`，`.reports/report_082.md`．

## 1. 目的

論文の実験1（Mushroom）では，独立一様サンプリング（復元抽出）で NFG-SVRG と ASA-SVRG に誤差床が現れ，その大きさは
最適解での確率的勾配の分散から見積もった $ \sigma^{2}/K $ とほぼ一致した．本実験では，付録B.1 の a9a の BCE
（強凸のロジスティック回帰，$ \rho \approx 297.4 $）でも同じことが起きるかを確かめる．
条件は E4・E8 の a9a の BCE と同一とし，サンプリング方式だけを random reshuffling から独立一様サンプリングに変える．

| 項目 | 設定 |
|---|---|
| データ | a9a（学習用 $ N = 29304 $，テスト用 3257．シードごとに層化分割） |
| 損失 | BCE + $ \frac{\lambda}{2}(\|{\mathbf w}\|^{2} + b^{2}) $，$ \lambda = 0.1125 $ |
| 学習率 | $ \eta = 0.99/(3L) \approx 0.08544 $（$ L = 3.8625 $，$ \rho \approx 297.4 $） |
| 内側ループ | $ K = N $，ミニバッチサイズ1，外側ループ数100 |
| サンプリング | `uniform_with_replacement`：各外側ループの最初に `rng.integers(0, N, size=N)`（E1 と同一） |
| 手法 | `SGD`，`SVRG_random`，`NFG_SVRG_random`，`ASAI_SVRG`（SGD 以外は `inner_start="snapshot"`） |
| シード | 0〜4（4手法 × 5シード = 20試行） |

## 2. E8 との違い

`ex014_e8_a9a_supplement/` の `data.py`，`model.py`，`train.py` をコピーし，`train.py` だけを変えた．

1. `EXPERIMENT_NAME = "ex015_e10_a9a_bce_uniform"`．
2. サンプリング方式の選択を加えた（E1 の `ex006` と同一）．
   - `ALL_SAMPLINGS = ("random_reshuffle", "uniform_with_replacement")`．
   - `get_epoch_indices(rng, N_train, sampling)`：`random_reshuffle` は `rng.permutation(N_train)`，`uniform_with_replacement` は
     `rng.integers(0, N_train, size=N_train)` を返す．どちらも外側ループの最初（`begin_epoch` の直前）に1回だけ呼ぶ．
   - `run_variance_reduced`・`run_sgd` に引数 `sampling` を加えた．`config.json` の `sampling` にはタスクのサンプリング方式を記録する．
3. タスクに `sampling` を加え，保存先を `{output_root}/bce/{手法}/inner_{inner_start}_{sampling}_{hp_name}/{seed}/` とした（E1 と同じ命名）．
4. `--part` を削除した．既定値は `--loss bce`，`--methods SGD SVRG_random NFG_SVRG_random ASAI_SVRG`，`--inner-starts snapshot`，
   `--samplings uniform_with_replacement`，`--workers 20` である．`--samplings` は `ALL_SAMPLINGS` から選べる．

その他（$ \eta $ と定数の計算と検査，評価，`oracle_calls` の数え方，`approx_error` の記録，打ち切り，ランダム点の乱数
`snapshot_rng = default_rng([seed, 1])`，完了済みの試行のスキップ）は E8 のままである．

一致の確認のため `compare_e10.py` を新たに作った（3節）．

## 3. 実行方法（`programs_review/programs` から）

```bash
# 一致の確認 (a)：random reshuffling で E4・E8 を再現する
python ex015_e10_a9a_bce_uniform/train.py --samplings random_reshuffle --seeds 0 --epochs 3 \
    --output-root ../outputs/_verify/ex015_rr
# 一致の確認 (b)：一様サンプリングの動作
python ex015_e10_a9a_bce_uniform/train.py --seeds 0 --epochs 3 --output-root ../outputs/_verify/ex015_uniform
# 比較（E4 の手法は，本PCで E4 のプログラムを実行した結果でも比べる）
python ex015_e10_a9a_bce_uniform/compare_e10.py --rr-root ../outputs/_verify/ex015_rr \
    --uniform-root ../outputs/_verify/ex015_uniform --e4-this-pc-root ../outputs/_verify/ex014_e4_ref_ex008_this_pc \
    --output ../outputs/ex015_e10_a9a_bce_uniform/consistency.md
# 本実行（20試行）
python ex015_e10_a9a_bce_uniform/train.py --workers 20 > ../outputs/_run_logs/ex015.log 2>&1
# 集計
python analysis_e10.py
# テスト（programs_review/ で）
python -m pytest -q tests/test_e10_programs.py
```

## 4. 出力

```text
outputs/ex015_e10_a9a_bce_uniform/bce/{SGD,SVRG_random,NFG_SVRG_random,ASAI_SVRG}/inner_{snapshot|none}_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/{seed}/
outputs/ex015_e10_a9a_bce_uniform/consistency.md
outputs/e10_summary.md
outputs/_run_logs/ex015.log，ex015_verify_{rr,uniform}.log
```

各試行のディレクトリには `log.json` と `config.json` がある（項目は `ex008_e4_a9a/README.md` と同じ．`config.json` の `sampling` は
`uniform_with_replacement`，ランダム点の手法は `snapshot_rule`・`snapshot_rng_seed` も持つ）．

## 5. コピー元

| ファイル | コピー元 | 変更 |
|---|---|---|
| `data.py`，`model.py` | `ex014_e8_a9a_supplement/` | なし（`tests/test_e10_programs.py` で同一であることを確認する） |
| `train.py` | `ex014_e8_a9a_supplement/train.py` | 2節の通り |
| `compare_e10.py` | 新規 | — |
