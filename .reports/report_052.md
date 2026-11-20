# report_052：E2（旧版）・E4 の成果物（`E2_results.zip`，`E4_results.zip`）の統合

## 0. 要点

- 別のPCで実施した2つの成果物を本リポジトリに取り込んだ．
  - `E4_results.zip`（E4，a9a，order_070／report_070）．
  - `E2_results.zip`（旧版の E2，6手法の要因配置，order_060／report_060）．
- E4 は既存のファイルとの衝突がなかった．
- 旧版の E2 は，本PCの E2（order_049）と5つのファイルの名前が衝突した．本PCのファイルは上書きせず，取り込むファイルの名前を `*_factorial` に変えた（1.2節）．
- 統合時に修正したのは1か所のみである．
  - 旧版の E2 の `e2_training.py` は，ω の2項目を `METRIC_NAMES` に加える．本PCの `common/vr_image_training.py` には既にこの2項目があり，重複してロガーが例外を出した．
  - 重複して加えないように1行を変えた（2節）．
- テストは全106件が通過した．内訳は，E0・付録B，E1，E2，旧版の E2，E3，E4 である．
- 本PCでの再実行による確認（3節）．
  - **旧版の E2：** 本PCで実行すると，本PCの E0 とビット単位で一致した．本PCの E2 のコードとも，学習の記録が全てビット単位で一致した．
    - zip のログとの差は GPU の違い（RTX 3090 と RTX PRO 6000）による丸め誤差である．
    - ω は2つの実装の計算式の違いにより，相対 $ 10^{-9} $ 程度異なる．
  - **E4：** 本PCで実行すると，SGD・SVRG・NFG-SVRG は MSE・BCE の両方で既存のログとビット単位で一致した．
    - report_070 の未解決の論点2（MSE の NFG-SVRG と既存のログとの相対 $ 10^{-6} $ の差）は，計算環境によるものと確定した．
- 統合後に，E2 のアブレーションの集計を完成させ，`.reports/report_049.md` にまとめた．最良の構成は ASA-SVRG（`last_iterate`）である．

## 1. 取り込んだファイル

### 1.1 E4（衝突なし）

| 種類 | パス |
|---|---|
| 指示・報告 | `.orders/order_070.md`，`.reports/report_070.md` |
| プログラム | `programs_review/programs/ex008_e4_a9a/`（`data.py`，`model.py`，`train.py`，`compare_e4.py`，`README.md`），`programs_review/programs/analysis_e4.py` |
| テスト・可視化 | `programs_review/tests/test_e4_programs.py`，`programs_review/visualize_result_E4.ipynb` |
| 結果 | `programs_review/outputs/ex008_e4_a9a/`（40試行，図28枚，`consistency_with_existing.md`），`outputs/e4_summary.md`，`outputs/_run_logs/ex008*.log` |

### 1.2 旧版の E2（5ファイルの名前を変更）

| zip 内の名前 | 取り込んだ名前 |
|---|---|
| `programs/analysis_e2.py` | `programs/analysis_e2_factorial.py`（出力先を `outputs/e2_factorial_summary.md` に変更） |
| `tests/test_e2_programs.py` | `tests/test_e2_factorial_programs.py` |
| `visualize_result_E2.ipynb` | `visualize_result_E2_factorial.ipynb`（`import analysis_e2_factorial` に変更） |
| `outputs/e2_summary.md` | `outputs/e2_factorial_summary.md` |
| `outputs/_run_logs/ex007.log` | `outputs/_run_logs/ex007_factorial.log` |

- 次のファイルは名前を変えずに取り込んだ．
  - `programs/ex007_e2_cifar10_snapshot_gradient_factorial/`．`README.md` の上記のファイル名は書き換えた．
  - `outputs/ex007_e2_cifar10_snapshot_gradient_factorial/`（30試行，図18枚，`consistency_*.md`）．
  - `outputs/_run_logs/ex007_{resume,verify_*}.log`，`.orders/order_060.md`，`.reports/report_060.md`．
- `report_060.md` と `order_060.md` は原文のまま残した．中のファイル名は変更前の名前である．
- 実験番号 `ex007` は，本PCの E2（`ex007_e2_cifar10_ablation`）と旧版の E2 が共に用いている．ディレクトリ名で区別できるため，番号は変えなかった．

### 1.3 統合の手順

- 作業用ディレクトリに展開した．
- 絶対パスや `..` を含むパスがないこと，名前の変更後に既存のファイルと衝突しないことを確認した．
- 上書きしない設定（`cp -rn`）で複写した．統合前に変更されていた5ファイル（`document.md` 等）以外に，既存のファイルの変更はない．
- 2つのプログラムが `programs_review/` の外を import・参照しないことを確認した．
  - 既存のログのパスは，コマンドライン引数で渡す．
- 旧版の E2 の `svrg_average.py` は実験のディレクトリ内にあり，`common/optimizers/svrg_average.py`（本PC）とは別のモジュールである．
- a9a のデータは本PCに無かったため，`data.py` の `download_raw_data()` で LIBSVM のサイトから取得した．
  - 保存先は `programs_review/datasets/a9a/raw/a9a`（Git管理外）である．
  - MD5 `fb3596510d780aaf76a06dc721d49662` は report_070 と一致する．

## 2. 統合時の修正

`ex007_e2_cifar10_snapshot_gradient_factorial/e2_training.py` の次の行を変えた．

```python
# 変更前
METRIC_NAMES = tuple(E0_METRIC_NAMES) + ("omega_snapshot", "omega_average")
# 変更後（本PCの common では既に2項目を含むため，重複して加えない）
METRIC_NAMES = tuple(E0_METRIC_NAMES) + tuple(
    name for name in ("omega_snapshot", "omega_average") if name not in E0_METRIC_NAMES)
```

- 修正前は，旧版の E2 のテスト29件中24件が `ResultLogger` の例外で失敗した．
- どちらの `common` でも，項目の並びは同じになる（E0 の項目，`omega_snapshot`，`omega_average`）．そのため，ログの形式は変わらない．

## 3. 本PCでの再実行による確認

### 3.1 旧版の E2（6手法，シード0，3エポック）

| 比較 | 結果 |
|---|---|
| 本PCで実行した旧版のコード ↔ 本PCの E0（SVRG，NFG-SVRG，ASA-SVRG，`snapshot`） | 学習の記録が全てビット単位で一致 |
| 本PCで実行した旧版のコード ↔ 本PCの E2 のコード（NFG-final，SVRG-final，SVRG-average，ASA-SVRG，`snapshot`） | 学習の記録（`train_loss`，`test_accuracy`，`approx_error`，`full_grad_norm_sq`，`approx_cos` 等）が全てビット単位で一致．ω は相対 $ 10^{-9} $ 程度の差 |
| 本PCで実行した旧版のコード ↔ zip のログ（RTX 3090） | `oracle_calls` は一致．学習損失は相対 $ 10^{-4} $ 以下，テスト精度は0.004以下の差（GPU の丸め誤差．report_060 1.2節と同じ現象） |

- ω の差は計算式の違いによる．旧版は $ \sum_k {\mathbf w}^k $ と $ \sum_k \|{\mathbf w}^k\|^2 $ をそのまま累積する．本PCは最初の点 $ {\mathbf w}^0 $ との差を累積する．
- 旧版は桁落ちのため，真の値が0の1エポック目で $ -9.3\times10^{-9} $ を返す．
- 結論に影響する差ではない．

48エポック目の PC 間の差は次の通りである．比較したのは，旧版の E2 と本PCの E0 で共通の3条件（`snapshot`）である．
- テスト精度：SVRG 0.002，ASA-SVRG 0.0002，NFG-SVRG 0.05．
- NFG-SVRG の差は，シード間の標準誤差（0.04）と同程度である．

このため，`analysis_e2.py` では次の方針とした（report_049 1節）．
- 本PC にある条件は本PC の結果を用いる．
- 本PC に無い最終点の条件と SVRG-average（`snapshot`）は，旧版の E2 の結果を用いる．

### 3.2 E4（8条件，シード0，100外側ループ）

| 比較 | 結果 |
|---|---|
| 本PCの再実行 ↔ zip のログ | 整数の量・精度は全て完全一致．実数の量の最大の差は，MSE の NFG-SVRG の $ f $ で相対 $ 7.4\times10^{-9} $．BCE は絶対 $ 2\times10^{-16} $ 以下 |
| 本PCの再実行 ↔ 既存のログ（`outputs/ex000_a9a_least_squares`，`ex0001_a9a_logistic`） | SGD・SVRG・NFG-SVRG は MSE・BCE の両方で全項目がビット単位で一致（42系列）．ASA-SVRG は既存のログが `last_iterate` のため一致しない（想定通り） |

- 結果は `outputs/ex008_e4_a9a/consistency_with_existing_this_pc.md` にある．
- MSE の NFG-SVRG の差（相対 $ 7.4\times10^{-9} $，$ \|\nabla f\|^2 $ で $ 1.7\times10^{-6} $）は，zip のログと既存のログの差（report_070 1.2節）と同じ大きさである．
- 本PCでは既存のログと完全に一致した．したがって，この差は別のPCの計算環境（CPU・BLAS）によるものであり，実装の差ではない．

### 3.3 集計の再現

- 2つの notebook を本PCで実行した（`visualize_result_E2_factorial.ipynb`，`visualize_result_E4.ipynb`）．
- 生成された `e2_factorial_summary.md` と `e4_summary.md` は，zip 内のものとバイト単位で同一であった．

## 4. その他の変更

- `programs_review/document.md`：
  - 旧版の E2 と E4 を，実験の一覧，ディレクトリ構成，依存関係の図，実行方法，ログの項目に追記した．
  - 内容は report_060 5.1節と `ex008_e4_a9a/README.md` 6節の転記内容に従う．
- `実験計画.md` 6.3〜6.5節：ディレクトリ名 `ex007_e4_a9a` を `ex008_e4_a9a` に改めた（report_070 5節の依頼）．
- `programs_review/programs/analysis_e2.py`：旧版の E2 の最終点の条件と SVRG-average（`snapshot`）を読み込むようにし，対応のある差を2つ加えた．
  - 追加した差：NFG-final − SVRG-final，および restart の ASA-SVRG − SVRG-average．
  - ω の図は2エポック目から描くようにした．
  - `visualize_result_E2.ipynb` は実行済みである．
- 確認の実行ログ：`outputs/_run_logs/verify_ex008_this_pc.log`，`verify_ex007_factorial_this_pc.log`，`verify_ex007_ablation_vs_factorial.log`．

## 5. 各実験の結果の要点

- **E2：** `.reports/report_049.md`（本PCの E2 と旧版の E2 を合わせたアブレーション）と `.reports/report_060.md`（旧版の E2 の要因配置）を参照．
- **E4：** `.reports/report_070.md` を参照．要点は次の通りである．
  - BCE（強凸）：ASA-SVRG が最も速く収束した（$ \hat{r} $ 0.0053，SVRG 0.0395，NFG-SVRG 0.1996）．付録B.1の記述は維持できる．
  - MSE（非凸）：旧実装から変わった．
    - 「学習の前半に ASA-SVRG の近似誤差が NFG-SVRG より数桁小さい」は成り立たない．$ \#\mathrm{grad}/N \gtrsim 12 $ から2〜3桁小さくなる．
    - 付録B.1（MSE）の本文と図B-1の修正が必要である．

## 6. 未解決の論点・次の作業の候補

1. 付録B.1（MSE）の本文の修正（report_070 7節）．
2. E2 の E0 の条件で ω を記録しての再実行（任意．report_049 5節）．
3. `common/` の共通化の提案（report_060 5.2節，report_070 5節）．今回は取り込んでいない．
4. 旧版の `実験指示E2.md` は，現行の E2 と内容が異なるが Git 管理下に残っている．
