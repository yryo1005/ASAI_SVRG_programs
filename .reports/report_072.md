# report_072：E3 の NFG-SVRG の restart での再計測（実験計画5.6節）

## 0. 要点

- E3（ImageWoof，実時間の比較）の NFG-SVRG を，restart（`inner_start="snapshot"`）で再計測した．
  - 条件は100エポック，シード0〜4である．
  - 5試行は1つずつ逐次に実行した．計測中は他の計算を実行していない．
  - 1試行は約31分であった．
- 実験計画5.6節の3つの確認は全て満たした（2節）．
  1. **E0 のログとの一致：** 5シードの全101評価点で，学習の記録が E0 の ImageWoof の NFG-SVRG（`snapshot`）とビット単位で一致した．
  2. **計算機の速度の変動：** time_optimizer／エポックは restart 10.573 ± 0.050 s，当初の continue 10.708 ± 0.086 s である．
     - 差は −1.26% で，基準の3%以内である．
     - したがって，SGD・SVRG・ASA-SVRG の計測は再実行せずにそのまま用いる．
  3. **計測環境：** 全ての試行で，機器とソフトウェアのバージョンが同一であった．
- 論文の E3 の表と図では，NFG-SVRG を restart の結果に置き換えた（`outputs/e3_summary.md`）．
  - 手法間の速度の比の結論は変わらない．
  - NFG-SVRG の最終のテスト精度は 0.266 から 0.304 に上がった．精度40%には依然として到達しない．

## 1. 変更したファイル

| ファイル | 変更 |
|---|---|
| `programs/ex009_e3_imagewoof_walltime/train.py` | `--inner-start {last_iterate,snapshot}` を追加した（既定は `last_iterate` で従来と同じ）．条件のディレクトリ名に `inner_start` を反映する． |
| `programs/analysis_e3.py` | 手法ごとに1つの条件を読む（`PAPER_CONDITIONS`．NFG-SVRG は `inner_snapshot…epochs100`）．continue の結果は確認用（`CONTINUE_CONDITION`）とした．5.6節の確認の節と確認用の図（`e3_check_*.png`）を追加した． |
| `tests/test_e3_programs.py` | 計時ありと計時なしの学習の一致を両方の `inner_start` で確認するテスト，`--inner-start` の既定値と `snapshot` の条件のテストを追加した． |
| `visualize_result_E3.ipynb` | 新規作成し，実行した． |
| `document.md` | E3 の説明（NFG-SVRG は restart），実行方法，ディレクトリ構成，依存関係の図を更新した． |

- 結果の保存先：`outputs/ex009_e3_imagewoof_walltime/NFG_SVRG/inner_snapshot_lr0.001_bs128_lambda0.0005_epochs100/{0..4}/`．
- 実行ログ：`outputs/_run_logs/ex009_nfg_restart.log`，事前確認 `outputs/_run_logs/verify_ex009_restart.log`．
- 当初の continue の結果（`inner_last_iterate…epochs100`）は削除せずに残した．

## 2. 実験計画5.6節の確認

### 2.1 確認1：E0 のログとの一致

- 比較対象：`outputs/ex003_e0_imagewoof_resnet18_layernorm/NFG_SVRG/inner_snapshot_lr0.001_bs128_lambda0.0005_epochs128/{seed}/log.json`．
- E0 は128エポックであるため，先頭の101評価点（0〜100エポック）を比べた．
- 比較した項目：epoch，oracle_calls，train_loss，test_accuracy，test_correct，approx_error．
- 結果：5シードの全項目・全評価点が完全に一致した．

計時の処理を入れても，学習の計算は変わらないことが確かめられた．

### 2.2 確認2：計算機の速度の変動

| 量 | restart（今回） | continue（当初） | restart／continue |
|---|---:|---:|---:|
| time_optimizer／エポック［s］ | 10.573 ± 0.050 | 10.708 ± 0.086 | 0.9874（−1.26%） |
| time_optimizer／全勾配の計算 | 1.4504 ± 0.0012 | 1.4478 ± 0.0011 | 1.0018 |

- 差は3%以内である．
- 全勾配の計算の時間で割った比は，0.2% しか違わない．restart と continue で1エポックの計算量は同じであるため，これは想定通りである．

### 2.3 確認3：計測環境

全ての試行（4手法×5シード，および continue の5試行）で，記録した計測環境の全項目が同一であった．
- 機器：Intel Xeon w7-3565X，NVIDIA RTX PRO 6000 Blackwell Max-Q．
- ソフトウェア：torch 2.11.0+cu128，CUDA 12.8，cuDNN 91900 等．

## 3. 結果（論文の E3 に用いる値）

| 手法 | エポック数 | time_optimizer／エポック［s］ | t／エポック［s］ | 最終の t［s］ | 精度40%到達の t［s］（到達数） | 最終の学習損失 | 最終のテスト精度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SGD | 200 | 6.020 ± 0.060 | 6.020 ± 0.060 | 1203.9 ± 12.0 | 510.6 ± 10.7（5/5） | 3.1407 ± 0.0252 | 0.4654 ± 0.0099 |
| SVRG | 67 | 10.583 ± 0.089 | 17.961 ± 0.153 | 1211.7 ± 10.3 | 1165.7 ± 18.8（3/5） | 3.6119 ± 0.0154 | 0.3885 ± 0.0064 |
| NFG-SVRG（restart） | 100 | 10.573 ± 0.050 | 10.573 ± 0.050 | 1057.3 ± 5.0 | —（0/5） | 3.8433 ± 0.0350 | 0.3041 ± 0.0120 |
| ASA-SVRG | 100 | 10.684 ± 0.124 | 10.684 ± 0.124 | 1068.4 ± 12.4 | 661.1 ± 9.6（5/5） | 3.3870 ± 0.0036 | 0.4570 ± 0.0020 |

### 3.1 計算機の速度の変化を除いた比（試行ごとの time_optimizer／全勾配の計算）

| 比 | 今回（restart） | 当初（continue） |
|---|---:|---:|
| NFG-SVRG／SVRG | 1.0111 | 1.0093 |
| ASA-SVRG／NFG-SVRG（平均パラメータの更新のオーバーヘッド） | 1.0103（1.03%） | 1.0121（1.21%） |
| ASA-SVRG／SVRG | 1.0215 | 1.0215（変更なし） |

- ASA-SVRG のオーバーヘッドは約1%である．この結論は変わらない．
- 生の時間の比 ASA-SVRG／NFG-SVRG は 1.011（オーバーヘッド 111.8 ms／エポック）である．

### 3.2 restart と continue の対応のある差（参考，同じシード，100エポック目）

| シード | 学習損失 restart | 学習損失 continue | 差 | テスト精度 restart | テスト精度 continue | 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.9744 | 4.0915 | −0.1171 | 0.2616 | 0.1922 | +0.0695 |
| 1 | 3.8140 | 4.0377 | −0.2237 | 0.3138 | 0.2275 | +0.0863 |
| 2 | 3.7741 | 3.8216 | −0.0474 | 0.3332 | 0.2960 | +0.0372 |
| 3 | 3.8036 | 3.7166 | +0.0870 | 0.3143 | 0.3426 | −0.0283 |
| 4 | 3.8504 | 3.9045 | −0.0540 | 0.2978 | 0.2726 | +0.0252 |
| 平均 ± 標準誤差 | | | −0.0710 ± 0.0506 | | | +0.0380 ± 0.0198 |

- 5シード中4シードで，restart の方が損失が低く精度が高い．
- ただし，差の平均は標準誤差の約1.4倍（損失）と約1.9倍（精度）にとどまる．
- どちらの開始点でも，NFG-SVRG は SVRG より損失が高く，精度40%に到達しない．NFG-SVRG と ASA-SVRG の順位は変わらない．

## 4. 図

`outputs/ex009_e3_imagewoof_walltime/figures/` にある．
- 論文用（NFG-SVRG は restart）：`e3_walltime_{train_loss,test_accuracy}.png`，`e3_grad_{train_loss,test_accuracy}.png`．
- 確認用（continue を淡い破線で重ねた図）：`e3_check_*.png`．

## 5. テスト

- `tests/` の全124件が通過した．
- E3 のテストは15件で，今回追加した分を含む．

## 6. 未解決の論点・次の作業の候補

1. 論文の4.3節（E3）の表と図を，restart の値（3節）に更新する．
   - 本文の「ASA-SVRG／NFG-SVRG のオーバーヘッド」は，正規化比で 1.21% から 1.03% に変わる．
   - 論文のリポジトリ（`ASAI_SVRG_Paper`）は今回変更していない．
2. E1 のランダム点の結果（report_071）の論文への反映．これも論文のリポジトリの作業である．
3. `実験計画 copy.md` は `実験計画.md` に反映済みのため，削除してよい．削除の判断はユーザーに委ねる．
