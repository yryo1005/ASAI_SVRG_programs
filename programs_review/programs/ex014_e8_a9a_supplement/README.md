# ex014_e8_a9a_supplement：E8（付録B.1，a9a の追加実験）

対応する実行指示：`実験指示E8.md`．対応する指示・報告：`.orders/order_076.md`，`.reports/report_076.md`．

## 1. 目的

論文の付録B.1の a9a の2つの実験について，E4（`ex008_e4_a9a`）に無い試行を追加する．

| 部 | 実験 | 追加する手法 | 試行 | 役割 |
|---|---|---|---|---|
| A | BCE（強凸） | `SVRG_random`，`NFG_SVRG_random`（ランダム点，reset：`inner_start="snapshot"`） | 2手法 × 5シード = 10 | 実験1（Mushroom，`ex006`）と同じ構成（SGD，SVRG・NFG-SVRG のランダム点，ASA-SVRG（reset））にそろえる |
| B | MSE（非凸） | `ASAI_SVRG`（continue：`inner_start="last_iterate"`） | 1手法 × 5シード = 5 | 旧版の論文（v42）の設定を新しい実装で取り直す（参考） |

BCE の SGD・ASA-SVRG（reset），MSE の SGD・SVRG（最終点）・NFG-SVRG（最終点）・ASA-SVRG（reset）は E4 の結果を用いる．

## 2. E4 との違い

`ex008_e4_a9a/` の `data.py`，`model.py`，`train.py`，`compare_e4.py` をコピーし，`train.py` だけを変えた．

1. `EXPERIMENT_NAME = "ex014_e8_a9a_supplement"`．
2. 手法 `SVRG_random`（`common/optimizers/svrg.py` の `SVRG`），`NFG_SVRG_random`（`nfg_svrg.py` の `NFGSVRG`）を追加した．
   `SVRG`，`NFG_SVRG` は E4 と同じ最終点（`SVRGFinalPoint`，`NFGSVRGFinalPoint`）のままである．
   全勾配を用いる手法（`FULL_GRADIENT_METHODS = ("SVRG", "SVRG_random")`）は，`oracle_calls` を初期 $ N $・1外側ループ $ 3N $ と数える．
3. ランダム点の乱数は E1（`ex006`）と同じである．スナップショットの添字は `begin_epoch` で `rng.integers(0, K)` により1回選ぶ．
   `begin_epoch` には，添字の列の生成器 `rng = default_rng(seed)` とは別の生成器 `snapshot_rng = default_rng([seed, 1])` を全ての手法で渡す．
   最終点の手法と ASA-SVRG は `begin_epoch` で乱数を消費しないため，結果は E4 と変わらない（`outputs/ex014_e8_a9a_supplement/consistency.md`）．
   ランダム点の手法の `config.json` には `snapshot_rule: "uniform_random"`，`snapshot_rng_seed: [seed, 1]` を加える．
4. `--part`（`A`，`B`，`all`．既定 `all`）を加えた．`--loss`・`--methods`・`--inner-starts` のいずれかを明示した場合は `--part` を使わず，
   明示しなかった項目は E4 の既定値（MSE・BCE の両方，全ての手法，`snapshot`）とする．並列数の既定値は15とした．

その他（$ \eta = 0.99/(3L) $，$ K = N_{\mathrm{train}} = 29304 $，ミニバッチサイズ1，random reshuffling，外側ループ数100，シード0〜4，
評価点 $ {\mathbf z}_{s} $，ログの項目，完了済みの試行のスキップ）は E4 のままである．

## 3. 実行方法（`programs_review/programs` から）

```bash
# 3.1 E4 の再現（シード0，3外側ループ）
python ex014_e8_a9a_supplement/train.py --loss mse bce --methods SGD SVRG NFG_SVRG ASAI_SVRG \
    --inner-starts snapshot --seeds 0 --epochs 3 --output-root ../outputs/_verify/ex014_e4
# 3.2 ランダム点の手法の動作
python ex014_e8_a9a_supplement/train.py --part A --seeds 0 --epochs 3 --output-root ../outputs/_verify/ex014_random
# 3.3 MSE の ASA-SVRG（continue）と旧実装の一致
python ex014_e8_a9a_supplement/train.py --part B --seeds 0 --output-root ../outputs/_verify/ex014_continue
python ex014_e8_a9a_supplement/compare_e4.py --new-root ../outputs/_verify/ex014_continue \
    --existing mse=../../outputs/ex000_a9a_least_squares \
    --output ../outputs/ex014_e8_a9a_supplement/consistency_continue_with_ex000.md
# 本実行（15試行）
python ex014_e8_a9a_supplement/train.py --part all --workers 15 > ../outputs/_run_logs/ex014.log 2>&1
# 集計（E4 の結果のパスは --e4-root で変えられる）
python analysis_e8.py
# テスト（programs_review/ で）
python -m pytest -q tests/test_e8_programs.py
```

## 4. 出力

```text
outputs/ex014_e8_a9a_supplement/bce/{SVRG_random,NFG_SVRG_random}/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/{seed}/
outputs/ex014_e8_a9a_supplement/mse/ASAI_SVRG/inner_last_iterate_eta1.530030e-01_K29304_epochs100/{seed}/
outputs/ex014_e8_a9a_supplement/figures/e8_{bce,mse}_{A,B,C}_*.png
outputs/ex014_e8_a9a_supplement/consistency.md，consistency_continue_with_ex000.md
outputs/e8_summary.md
```

各試行のディレクトリには `log.json` と `config.json` がある（項目は `ex008_e4_a9a/README.md` 2・3節と同じ）．

## 5. コピー元

| ファイル | コピー元 | 変更 |
|---|---|---|
| `data.py`，`model.py`，`compare_e4.py` | `ex008_e4_a9a/` | なし（`tests/test_e8_programs.py` で同一であることを確認する） |
| `train.py` | `ex008_e4_a9a/train.py` | 2節の通り |
