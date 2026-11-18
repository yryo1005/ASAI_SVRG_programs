# ex006_e1_mushroom_rho_sampling：E1（Mushroomにおける $ \rho $ の異なる学習率での収束挙動とサンプリング方式の影響）

対応する実験計画：`実験計画.md` 3章（E1）．対応する実行指示：`実験指示E1.md`．
対応する指示・報告：`.orders/order_050.md`，`.reports/report_050.md`．

本ディレクトリは `ex001_e0_mushroom_logistic/`（E0のMushroom実験）を拡張したものであり，
以下の点のみが異なる．

1. **手法にSGDを追加**．既存の `programs/ex001_mushroom_logistic/train.py` の `run_sgd` を移植した．
2. **学習率を $ \rho $（定理1の縮小率の上界）で指定**．$ \rho \in \{0.5, 0.64, 0.8, 1.0, 2.0, 10, 298.4\} $ の
   7条件について，対応する $ \eta $ を求める．$ \rho = 0.5, 0.8, 1.0, 2.0, 10 $ は二分法（`solve_eta_for_target_rho`，
   大きい側の解）で求め，$ \rho = 0.64 $ と $ \rho = 298.4 $ は既存の実験（E0）と同一の値にするため，それぞれ
   $ \eta = 1/(20L) $，$ \eta = 0.99/(3L) $ を直接用いる（実験計画.md 3.3節の表の注記）．
3. **サンプリング方式を追加**．`random_reshuffle`（既存と同じ，`rng.permutation`）と
   `uniform_with_replacement`（`rng.integers(0, N_train, size=N_train)`，重複を許す独立一様サンプリング）．
   後者は $ \rho \in \{0.5, 1.0, 298.4\} $ のみ実施する．両方式とも，1エポック分の添字は外側ループの最初
   （`optimizer.begin_epoch(rng)` の直前）で `get_epoch_indices` により生成し，乱数の消費位置を揃える．
4. **`inner_start` は既定で `"snapshot"` のみ**．`"last_iterate"` は3節で述べる既存実装との一致確認にのみ用いる．

## 1. 実行方法

```bash
cd programs
# 本実行（既定の7×4×5 + 3×3×5 = 200試行）
python ex006_e1_mushroom_rho_sampling/train.py --workers 16
# 既存実装との一致の確認（last_iterate と snapshot，rho=298.4，SVRG系）
python ex006_e1_mushroom_rho_sampling/train.py --rhos 298.4 --samplings random_reshuffle \
    --methods SVRG NFG_SVRG ASAI_SVRG --inner-starts last_iterate snapshot --seeds 0 \
    --output-root ../outputs/_verify/ex006
python ex006_e1_mushroom_rho_sampling/train.py --rhos 298.4 --samplings random_reshuffle \
    --methods SGD --seeds 0 --output-root ../outputs/_verify/ex006
python compare_with_existing_logs.py --new-root ../outputs/_verify/ex006 \
    --existing-root ../outputs/ex001_e0_mushroom_logistic \
    --hp-map inner_last_iterate_random_reshuffle_rho298.4_eta1.049767e-02_lambda9.155984e-01_K7311_epochs100=inner_last_iterate_eta1.049767e-02_lambda9.155984e-01_K7311_epochs100 \
              inner_snapshot_random_reshuffle_rho298.4_eta1.049767e-02_lambda9.155984e-01_K7311_epochs100=inner_snapshot_eta1.049767e-02_lambda9.155984e-01_K7311_epochs100
python compare_with_existing_logs.py --new-root ../outputs/_verify/ex006 \
    --existing-root ../../outputs/ex001_mushroom_logistic \
    --hp-map inner_none_random_reshuffle_rho298.4_eta1.049767e-02_lambda9.155984e-01_K7311_epochs100=eta1.049767e-02_lambda9.155984e-01_K7311_epochs100
# 集計と図
python analysis_e1.py
# テスト
cd .. && python -m pytest -q tests/test_e1_programs.py
```

既定の `--rhos`，`--samplings`，`--methods` は `実験指示E1.md` 2.3節・2.4節の条件
（`random_reshuffle` が全7 $ \rho $，`uniform_with_replacement` が $ \rho \in \{0.5, 1.0, 298.4\} $）に対応する．
完了済みの試行（`log.json` と `config.json` があり，全エポックの記録があるか打ち切りで終了したもの）は
自動的にスキップされる．

## 2. 保存先のディレクトリ名

```text
outputs/ex006_e1_mushroom_rho_sampling/{手法}/inner_{inner_start}_{sampling}_rho{target_rho}_eta{eta:.6e}_lambda{λ:.6e}_K{K}_epochs{epochs}/{seed}/
```

SGDは `inner_start` を持たないため，`inner_none_...` とする．`config.json` には `eta`，
`rho`（$ \eta $ から計算し直した実際の縮小率），`target_rho`（上記の目標値），`L`，`mu` を15桁以上で保存する．

## 3. 既存実装との一致の確認

本実行の前に，`inner_start="last_iterate"` と `"snapshot"`，`random_reshuffle`，$ \rho = 298.4 $，
シード0の結果を，E0（`ex001_e0_mushroom_logistic`）および既存のSGD（リポジトリルートの
`outputs/ex001_mushroom_logistic/SGD/`）のログと比較した．結果は `outputs/ex006_e1_mushroom_rho_sampling/
consistency_with_e0.md`，`consistency_with_existing_sgd.md` に保存する．差は倍精度の丸め誤差の範囲内
（相対 $ 10^{-13} $ 程度以下）である．

## 4. 集計（`analysis_e1.py`）

- `outputs/e1_summary.md`：各 (sampling, $ \rho $, 手法) について，$ \eta $，$ \rho $，最終の目的関数誤差，
  実測の縮小率 $ \hat{r} $，最終のテスト精度，誤差床（目的関数誤差・近似誤差，最後の20外側ループの中央値），
  打ち切りの数を平均 ± 標準誤差（5シード）で示す．
- 図（`outputs/ex006_e1_mushroom_rho_sampling/figures/`）：
  - `e1_loss_{sampling}_{method}_{budget31,full}.png`：$ \rho $ の異なる曲線を重ねた目的関数誤差．
  - `e1_approx_error_{sampling}_{method}_{budget31,full}.png`：近似誤差（NFG-SVRG・ASA-SVRGのみ）．
  - `e1_sampling_overlay_{method}_{loss,approx_error}.png`：2つのサンプリング方式を重ねた図
    （$ \rho \in \{0.5, 1.0, 298.4\} $）．

## 5. `document.md` への転記内容（統合時）

親リポジトリの `programs_review/document.md` には，現時点でE0と付録Bのみが記載されている．統合時に
以下を追記する．

- 「現在実装している実験」の一覧にE1（本ディレクトリ）を追加する．
- ディレクトリ構成（1節）に `ex006_e1_mushroom_rho_sampling/`，`analysis_e1.py`，
  `tests/test_e1_programs.py`，`visualize_result_E1.ipynb` を追加する．
- 依存関係の図に，`ex006 train.py` → `common/optimizers`，`common/machine_learning_utils` の矢印，
  および `analysis_e1.py` → `outputs/`，`visualize_result_E1.ipynb` → `analysis_e1.py` の矢印を追加する．
- 2節末尾に，E1で新たに `config.json` に追加したキー（`rho`，`target_rho`，`sampling`）の説明を追加する．
