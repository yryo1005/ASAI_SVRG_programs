# ex008_e4_a9a：E4（付録B.1，a9a の MSE・BCE の再実施）

対応する実験計画：`実験計画.md` 6章（E4）．対応する実行指示：`実験指示E4.md`．
対応する指示・報告：`.orders/order_070.md`，`.reports/report_070.md`．

`実験計画.md` 6.3節は実験のディレクトリを `ex007_e4_a9a` としているが，`ex007` は E2 に割り当て済みのため `ex008_e4_a9a` とした．

本ディレクトリは E1（`ex006_e1_mushroom_rho_sampling/`）をコピーし，データとモデルを既存の a9a の実験
（`programs/ex000_a9a_least_squares/`：MSE，`programs/ex0001_a9a_logistic/`：BCE）と同一にしたものである．
E1 から次の仕組みをそのまま引き継いだ．`inner_start` の切り替え，乱数の消費位置（各外側ループの最初の `rng.permutation`），
`--workers` による並列実行（`multiprocessing.Pool`），完了済みの試行のスキップ，ログの形式，SGD の実装．
E1 との違いは次の通りである．

1. **`--loss mse|bce` で2つの実験を切り替える**．`model.py` は既存の2つのモデル（MSE：切片なし，BCE：切片付き）を持ち，
   目的関数の計算の順序も既存と同じである．
2. **学習率は $ \eta = 0.99/(3L) $ のみ**．$ L $（と BCE の $ \lambda $）は既存の `train.py` の関数
   （`compute_kappa_max`・`compute_smoothness_constants`（MSE），`compute_smoothness_constant`（BCE））をコピーし，
   シード0の学習データから計算する．`実験指示E4.md` 2.1節の表の値と相対 $ 10^{-12} $ 以下で一致しない場合は，警告を表示して表の値を用いる
   （本実行では一致した）．
3. **サンプリングは `random_reshuffle` のみ**．E1 の `--rhos`，`--samplings` は削除した．
4. **ログの目的関数の項目名は既存と同じ**．MSE は `objective_value`（$ f({\mathbf z}_{s}) $），BCE は `objective_gap`（$ f({\mathbf z}_{s}) - f({\mathbf w}^{*}) $）．

| ファイル | 内容 |
|---|---|
| `data.py` | a9a の読み込みと 9:1 の層化分割（既存の `data.py` と同一．保存先は `programs_review/datasets/a9a/raw/a9a`） |
| `model.py` | MSE と BCE のモデル，目的関数，勾配，精度 |
| `train.py` | 学習（SGD，SVRG，NFG-SVRG，ASA-SVRG），タスクの構築と並列実行 |
| `compare_e4.py` | 既存のログとの一致の確認（既存のログの全てのキー（`elapsed_time` を除く）と `config.json` の定数を比較） |

## 1. 実行方法

```bash
cd programs_review/programs
# 既存の実装との一致の確認（シード0）
python ex008_e4_a9a/train.py --loss mse bce --methods SGD SVRG NFG_SVRG --inner-starts snapshot --seeds 0 \
    --workers 6 --output-root ../outputs/_verify/ex008
python ex008_e4_a9a/train.py --loss mse bce --methods ASAI_SVRG --inner-starts last_iterate --seeds 0 \
    --workers 2 --output-root ../outputs/_verify/ex008
python ex008_e4_a9a/compare_e4.py --new-root ../outputs/_verify/ex008 \
    --existing mse=../../outputs/ex000_a9a_least_squares bce=../../outputs/ex0001_a9a_logistic \
    --output ../outputs/ex008_e4_a9a/consistency_with_existing.md
# 本実行（2実験 × 4手法 × 5シード = 40試行，inner_start="snapshot"）
python ex008_e4_a9a/train.py --loss mse bce --workers 20 > ../outputs/_run_logs/ex008.log 2>&1
# 集計と図（旧実装の ASA-SVRG のログのパスを引数で渡す）
python analysis_e4.py --existing mse=../../outputs/ex000_a9a_least_squares bce=../../outputs/ex0001_a9a_logistic
# ノートブック（programs_review/ で実行）
cd .. && jupyter nbconvert --to notebook --execute --inplace visualize_result_E4.ipynb
# テスト
python -m pytest -q tests
```

完了済みの試行（`log.json` と `config.json` があり，全エポックの記録があるか打ち切りで終了したもの）は自動的にスキップされる．
1試行は CPU の1プロセス（`torch.set_num_threads(1)`）で実行する．並列数1での最大常駐メモリは約0.85GB，
1外側ループ（SVRG）は約6.3秒であった（Intel Core Ultra 9 285K）．

## 2. 保存先のディレクトリ名

```text
outputs/ex008_e4_a9a/mse/{手法}/inner_{inner_start}_eta{eta:.6e}_K29304_epochs100/{seed}/
outputs/ex008_e4_a9a/bce/{手法}/inner_{inner_start}_eta{eta:.6e}_lambda{λ:.6e}_K29304_epochs100/{seed}/
```

SGD は `inner_start` を持たないため `inner_none_...` とする．`inner_` 以降は既存のログの条件名と同じである
（例：`bce/ASAI_SVRG/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/`）．
`config.json` には `loss`，`eta`，`L`，`reg_lambda`，`mu`（MSE は `null`），`inner_start`，`sampling`，`N_train`，`N_test`，`d` のほか，
MSE では `L_individual`，`L_average`，`f_star_reference`，BCE では `L0`，`lambda_ratio`，`rho`，`f_star` を保存する
（浮動小数点数は JSON の既定の17桁）．`f_star`・`f_star_reference` は各シードの学習データについて L-BFGS-B で求める．

## 3. ログの項目（`log.json`）

`epoch`，`oracle_calls`，`full_grad_computations`，`elapsed_time`，`objective_value`（MSE）または `objective_gap`（BCE），
`grad_norm_sq`，`accuracy`，`approx_error` は既存のログと同じ名前・順序・計算方法である．E1 と同じく
`test_correct`，`full_grad_norm_sq`（$ = $ `grad_norm_sq`），`approx_cos`，`terminated` を追加した．SGD では
`approx_error`，`approx_cos` を NaN とする．目的関数の値が非有限になった外側ループで打ち切る．

## 4. 既存の実装との一致の確認

`compare_e4.py` は既存のログの全てのキー（`elapsed_time` を除く）を比べる．`compare_with_existing_logs.py` の
`COMPARED_KEYS` には `objective_value` と `full_grad_computations` が含まれないため，本ディレクトリに別に置いた．
判定は，整数の量（`epoch`，`oracle_calls`，`full_grad_computations`，`accuracy`）は完全一致，実数の量は
既存の値の絶対値が $ 10^{-12} $ 以上の点で相対誤差 $ \le 10^{-12} $，それ未満の点（丸め誤差の水準まで小さくなった区間）で
絶対誤差 $ \le 10^{-15} $ とした．結果は `outputs/ex008_e4_a9a/consistency_with_existing.md` に保存した．

## 5. 集計（`analysis_e4.py`）

- `outputs/e4_summary.md`：各実験・手法について，$ \#\mathrm{grad}/N \le 31 $ の最後の評価点と100外側ループ終了時の
  主な評価量（MSE：$ \|\nabla f({\mathbf z}_{s})\|^{2} $，BCE：目的関数誤差），近似誤差，テスト精度（5シードの平均 ± 標準誤差），
  テスト精度の試行間の標準偏差，打ち切りの数，BCE の実測の縮小率 $ \hat{r} $（E1 と同じ定義）．
  旧実装の ASA-SVRG（既存のログ）を参考の行として示す．学習の前半の近似誤差（幾何平均）の表も示す．
- 図（`outputs/ex008_e4_a9a/figures/`）：`e4_{loss}_{A,B,C}_{primary,accuracy,approx_error}_{budget31,full}[_with_old].png`．
  (A) MSE は $ \|\nabla f({\mathbf z}_{s})\|^{2} $，BCE は目的関数誤差，(B) テスト精度，(C) 近似誤差（NFG-SVRG・ASA-SVRG）．
  `_with_old` は旧実装の ASA-SVRG を破線で重ねた確認用の図（論文には載せない）．
  補足として，MSE では (D) $ f({\mathbf z}_{s}) $ の図（`e4_mse_D_objective_value_*.png`）と，シードごとの最終の $ f({\mathbf z}_{s}) $ の表も作る．
  非凸の MSE では，シグモイド関数の飽和により $ f $ が大きいまま勾配が消える点（例：全てを負例と予測する点，$ f = $ 正例の割合 $ \approx 0.2408 $）
  があり，$ \|\nabla f({\mathbf z}_{s})\|^{2} $ だけでは収束の良し悪しを判断できないためである．
- 対数軸の量は E1 と同じく，下限で切り上げた値の常用対数の平均（幾何平均）± 標準誤差で描く．下限は目的関数誤差が
  $ 10^{-16} $，近似誤差と $ \|\nabla f({\mathbf z}_{s})\|^{2} $ が $ 10^{-300} $ である．

## 6. `document.md` への転記内容（統合時）

- 「現在実装している実験」の一覧に E4（本ディレクトリ，付録B.1 の a9a の MSE・BCE，`snapshot`，SGD を含む4手法）を追加する．
- ディレクトリ構成（1節）に `ex008_e4_a9a/`（`data.py`，`model.py`，`train.py`，`compare_e4.py`，`README.md`），
  `analysis_e4.py`，`tests/test_e4_programs.py`，`visualize_result_E4.ipynb`，`outputs/ex008_e4_a9a/`，`outputs/e4_summary.md`，
  `datasets/a9a/raw/a9a` を追加する．
- 依存関係の図に次の矢印を追加する．`ex008 train.py` → `common/optimizers`，`common/machine_learning_utils`，`ex008 data.py`，`ex008 model.py`．
  `ex008 compare_e4.py` → `outputs/`．`analysis_e4.py` → `outputs/`．`visualize_result_E4.ipynb` → `analysis_e4.py`．
- 1節末尾の条件のディレクトリ名の説明に，E4 は `{loss}/{手法}/inner_{inner_start}_eta{η}[_lambda{λ}]_K29304_epochs100` とすることと，
  `config.json` に `loss` を追加したことを追記する．
- 6節（実行方法）に本 README 1節のコマンド（本実行，集計）を追加する．
- 7節（ログの項目）に，a9a の MSE では目的関数の項目が `objective_value`（$ f({\mathbf z}_{s}) $ の値）であることを追記する．
