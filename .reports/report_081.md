# report_081：E9 ImageWoof・バッチサイズ512 の ASA-SVRG（reset）の残りのシード（1〜4）

指示書：`実験指示E9.md`．指示：`.orders/order_081.md`．番号は指示書0.3節のとおり（`order_081`，`report_081`）であり，既存の番号との衝突は無かった．

## 0. 要点

- ASA-SVRG（reset，`inner_start="snapshot"`）・ImageWoof・バッチサイズ512 のシード1〜4の4試行が全て完了した．既存のシード0と合わせて5シードがそろった．打ち切り・失敗は無い．
- 学習のプログラムは新たに作らず，既存の `ex005_appb_imagewoof_resnet18_layernorm/train.py` を `--batch-sizes 512 --methods ASAI_SVRG --inner-starts snapshot --seeds 1 2 3 4` で実行した．
- 一致の確認（2節）では，シード0の0〜3エポックの `oracle_calls`・`train_loss`・`test_accuracy`・`approx_error` が既存のシード0と**ビット単位で一致**した．実行環境（torch，CUDA，cuDNN，GPU）も同じである．
- 結果：予算（$ \#\mathrm{grad}/N \le 1000 $）でのテスト精度は 40.06 ± 0.16%（5シード），学習損失は 3.5933 ± 0.0051 である．シード0（39.83%，3.5808）とシード1〜4（39.83〜40.70%，3.5820〜3.6062）に大きな違いは無い．
- 本実行の実時間は9時間51分（4並列）であった．同じ時間帯に E10（CPU のみ）を並行して実行した．

## 1. 実行環境と並列数

| 項目 | 内容 |
|---|---|
| OS | Windows 上の WSL2（Linux 6.18.33.2-microsoft-standard-WSL2） |
| CPU | Intel Xeon w7-3565X（64論理コア），メモリ 125 GB |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition（97,887 MiB），ドライバ 582.08 |
| Python・主要パッケージ | Python 3.12.15（`.venv_pytorch`），torch 2.11.0+cu128，torchvision 0.26.0+cu128，CUDA 12.8，cuDNN 9.19.0，Pillow 13.0.0 |

`config.json` の `environment` は，シード1〜4の全てで既存のシード0と同じ値であった（GPU 名，torch・torchvision・CUDA・cuDNN・Pillow のバージョン）．

**並列数：4．** 根拠：2節の確認（並列数1，3エポック）で，`nvidia-smi -l 2` の `memory.used` は待機時 682 MiB，最大 23,183 MiB（1プロセスで約22.5 GB）であった．最大常駐メモリ（`/usr/bin/time -v`）は 9.4 GB であり，ImageWoof のキャッシュ（`numpy.memmap`）の共有ページを含む．
4並列では VRAM が約 90 GB（総量 97.9 GB）であり，タスクは4つなので全てを同時に実行できる．DRAM はシステム全体で約 7 GB であった（本実行中の `free`）．
なお，E7（`report_075`）と同じく，1プロセスでも GPU の使用率はほぼ100%であり，4並列の1エポックは約70秒（1プロセスでは約18秒）であった．したがって並列化による短縮はほぼ無いが，実時間は計測しないため結果には影響しない．

## 2. 本実行の前に残っていた試行（指示書0.5節）

本実行の前の `.../inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/` の状態は次のとおりであった．

| シード | 内容 | 記録エポック数 |
|---:|---|---|
| 0 | `log.json`，`config.json`，`best_model.pth`（2026-11-19 15:23） | 505（完了） |
| 1 | `best_model.pth` のみ（2026-11-19 20:19．以前の中止による） | `log.json` 無し（未完了） |
| 2 | 空のディレクトリ | `log.json` 無し（未完了） |
| 3 | 空のディレクトリ | `log.json` 無し（未完了） |
| 4 | 空のディレクトリ | `log.json` 無し（未完了） |

- シード1〜4はいずれも `is_run_completed` が False であり，`run_single_experiment` が最初からやり直した．シード1の古い `best_model.pth` は，学習中に上書きされた．
- 505 エポックの記録がそろったシード1〜4の結果は無かったため，飛ばされた試行は無い（ログに `[skip]` は無い）．
- 本実行の前に動いていた学習のプロセスは無かった．

## 3. 一致の確認（指示書2節）

詳細は `programs_review/outputs/_verify/e9/consistency.md` にある．

```bash
cd programs_review/programs/ex005_appb_imagewoof_resnet18_layernorm
../../../.venv_pytorch/bin/python train.py --batch-sizes 512 --methods ASAI_SVRG --inner-starts snapshot \
    --seeds 0 --epochs 3 --output-root ../../outputs/_verify/e9 --workers 1
```

| エポック | 指標 | 既存 | 今回 | 相対差 |
|---:|---|---:|---:|---:|
| 0 | `oracle_calls` | 0 | 0 | 0（一致） |
| 0 | `train_loss` | 4.4551721304473455 | 4.4551721304473455 | 0 |
| 0 | `test_accuracy` | 0.05701196233138203 | 0.05701196233138203 | 0 |
| 0 | `approx_error` | 116.58558907639235 | 116.58558907639235 | 0 |
| 1 | `oracle_calls` | 18050 | 18050 | 0（一致） |
| 1 | `train_loss` | 4.455173211454355 | 4.455173211454355 | 0 |
| 1 | `test_accuracy` | 0.05701196233138203 | 0.05701196233138203 | 0 |
| 1 | `approx_error` | 0.0013137801449869535 | 0.0013137801449869535 | 0 |
| 2 | `oracle_calls` | 36100 | 36100 | 0（一致） |
| 2 | `train_loss` | 4.236528958825193 | 4.236528958825193 | 0 |
| 2 | `test_accuracy` | 0.10435225477986859 | 0.10435225477986859 | 0 |
| 2 | `approx_error` | 0.1498300891601616 | 0.1498300891601616 | 0 |
| 3 | `oracle_calls` | 54150 | 54150 | 0（一致） |
| 3 | `train_loss` | 4.213856664853083 | 4.213856664853083 | 0 |
| 3 | `test_accuracy` | 0.11453296093577858 | 0.11453296093577858 | 0 |
| 3 | `approx_error` | 0.006116598743449941 | 0.006116598743449941 | 0 |

**判定：合格．** 全ての値がビット単位で一致した（基準は `oracle_calls` の一致と相対差 $ 10^{-3} $ 以下）．

## 4. 本実行

```bash
cd programs_review/programs/ex005_appb_imagewoof_resnet18_layernorm
../../../.venv_pytorch/bin/python train.py --batch-sizes 512 --methods ASAI_SVRG --inner-starts snapshot \
    --seeds 1 2 3 4 --workers 4 > ../../outputs/_run_logs/ex005_e9_bs512_asa_reset.log 2>&1
```

- 期間：2026-11-23 10:56:07〜20:47:00（実時間9時間51分）．4試行は 20:43〜20:47 に終了した．
- ログの末尾は「全ての学習が終了しました．」であり，`[done]` が4件，`[failed]` は0件であった．
- シード1〜4の `log.json` は全て506点（エポック0〜505）であり，打ち切りは無い（`config.json` の `recorded_epochs` は505，`terminated` は false）．
- 既存のシード0（`log.json` の更新日時は 2026-11-19 15:23 のまま）と，他の手法（`SVRG`，`NFG_SVRG`）・バッチサイズ32の結果は変更されていない（`git status` で `ex005` の下に変更されたファイルは無く，追加はシード1〜4のディレクトリのみ）．
- 本実行中に，12:34〜13:08 に E10（`ex015_e10_a9a_bce_uniform`，CPU のみ，20並列）を並行して実行した．E9 は実時間を計測しないため，結果には影響しない．

## 5. テスト

`programs_review/tests/test_e9_programs.py`（5件）．

- 小さな合成のログ（評価点 $ \#\mathrm{grad}/N = 0, 400, 500, 700, 1000, 1200 $）で，予算での値（1000 の点），揺らぎ（$ (500, 700) $ と $ (700, 1000) $ の組の平均 7.5 ポイント），近似誤差の中央値（700 と 1000 の点の中央値）が手計算と一致すること．
- 予算を超える評価点（1200）が予算での値に使われないこと．予算以下の点が無ければ NaN になること．
- 打ち切りの記録，平均 ± 標準誤差（`ddof=1`），ファイルからの読み込みと表の作成．

全てのテスト（`python -m pytest -q tests`）は214件が通った（2026-11-23．E10 のテスト13件を含む）．

## 6. 結果（`programs_review/outputs/e9_summary.md`）

横軸 $ \#\mathrm{grad}/N $ は `oracle_calls / 9025`，予算は $ \#\mathrm{grad}/N \le 1000 $ である（505 エポック目は 1010 であり，予算での値は 500 エポック目（1000）の値である）．揺らぎは，隣り合う評価点の両方が予算の後半（$ 500 \le \#\mathrm{grad}/N \le 1000 $）にある組の，テスト精度の変化の絶対値の平均である．近似誤差の中央値は $ 500 < \#\mathrm{grad}/N \le 1000 $ の評価点で求めた．

### 6.1 シードごとの値

| シード | 記録エポック数 | 打ち切り | 最後の #grad/N | 予算でのテスト精度［%］ | 予算での学習損失 | 予算の後半の揺らぎ［%pt］ | 予算の後半の近似誤差の中央値 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0（既存） | 505 | なし | 1010.0 | 39.83 | 3.5808 | 0.17 | 1.0256 |
| 1 | 505 | なし | 1010.0 | 39.83 | 3.6012 | 0.16 | 1.0666 |
| 2 | 505 | なし | 1010.0 | 40.01 | 3.5962 | 0.16 | 1.1556 |
| 3 | 505 | なし | 1010.0 | 40.70 | 3.5820 | 0.15 | 1.3984 |
| 4 | 505 | なし | 1010.0 | 39.93 | 3.6062 | 0.19 | 0.9992 |

### 6.2 平均 ± 標準誤差（`ddof=1`）

| 試行数 | 打ち切り | 予算でのテスト精度［%］ | 予算での学習損失 | 予算の後半の揺らぎ［%pt］ | 予算の後半の近似誤差の中央値 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0 | 40.06 ± 0.16 | 3.5933 ± 0.0051 | 0.16 ± 0.01 | 1.1291 ± 0.0724 |

## 7. 所見

### 7.1 事実

1. **シード0とシード1〜4に大きな違いは無い．** 予算でのテスト精度はシード0が 39.83%，シード1〜4が 39.83〜40.70% であり，シード0は5シードの範囲の中（最小と同じ値）にある．学習損失（3.5808 と 3.5820〜3.6062），揺らぎ（0.17 と 0.15〜0.19 ポイント），近似誤差の中央値（1.03 と 1.00〜1.40）も同様である．5シードのテスト精度の標準誤差は 0.16 ポイントと小さい．
2. **打ち切られた試行は無い．** 5シードとも 505 エポックの記録がある．
3. **参考（同じ PC，同じ予算の E7，`report_075` 6.1節）**：バッチサイズ512の予算でのテスト精度は，SVRG（最終点）41.14 ± 0.95%，NFG-SVRG（最終点）28.59 ± 3.49%，既存の SVRG（ランダム点）34.84%，NFG-SVRG（ランダム点）35.28%（いずれもシード0のみ）である．本実験の ASA-SVRG（reset）は 40.06 ± 0.16% である．予算の後半の揺らぎは，ASA-SVRG（reset）の 0.16 ポイントに対し，E7 の SVRG（最終点）は 1.40，NFG-SVRG（最終点）は 1.94 ポイントである．

### 7.2 推測

1. シード0の結果は特別な値ではなく，論文の図で5シードの平均 ± 標準誤差の破線として描いても，シード0だけの曲線と大きくは変わらないと考えられる．
2. ASA-SVRG（reset）の揺らぎ（0.16 ポイント）は，同じバッチサイズの SVRG・NFG-SVRG（最終点，ランダム点）より1桁小さい．スナップショットに平均点を用いることによる平滑化の効果と考えられる．ただし，比べた手法は実行時期と実験（E7，`ex005`）が異なり，対応のある比較ではない．
3. 予算でのテスト精度は，SVRG（最終点，41.14 ± 0.95%）と ASA-SVRG（reset，40.06 ± 0.16%）の差が標準誤差の和（約1.1 ポイント）と同程度であり，この差は区別できない．

## 8. 統合時の注意

- `programs_review/document.md` は変更してはいけないため，統合時に次を転記する．
  - 付録Bの `ex005` のバッチサイズ512について，「途中で中止した」という記述を，「ASA-SVRG（reset）は E9 で5シードをそろえた（SVRG・NFG-SVRG はシード0のみ）」に改める．
  - 実行方法に指示書3.2節のコマンド（`--batch-sizes 512 --methods ASAI_SVRG --inner-starts snapshot --seeds 1 2 3 4 --workers 4`）と `python programs/analysis_e9.py` を追加する．
  - ディレクトリ構成に `analysis_e9.py`，`tests/test_e9_programs.py`，`outputs/e9_summary.md` を追加する．
- zip に含めなかったもの：`*.pth`（シード0〜4の `best_model.pth`），`outputs/_verify/e9/` の学習結果（`ASAI_SVRG/...epochs3/`，`verify_run.log`，`gpu_mem.csv`，`vmstat.txt`．`consistency.md` のみ含める），`datasets/`，`.venv_*`，`tokens.json`．
- 結果のディレクトリにはシード0（既存）も含めた．別のPCで5シードをそろえて図を作るためである．
- シード1〜4の `log.json` には，シード0に無い項目 `omega_snapshot`・`omega_average`（ω の記録）がある．シード0は ω の記録が `common/vr_image_training.py` に加わる前（2026-11-19）に作られたためである．他の項目（`oracle_calls`，`train_loss`，`test_accuracy`，`approx_error` 等）は共通であり，図と集計には影響しない．ω を図に使う場合はシード1〜4だけになる．
- 実行ログ `outputs/_run_logs/ex005_e9_bs512_asa_reset.log` は，本実行の途中（12:02）でユーザーが commit した時点の内容が Git に記録されている．その後も追記されたため，作業ツリーでは変更ありと表示される．zip には最終の内容が含まれる．

## 9. 未解決の論点・次の作業の候補

1. 論文の実験3の図（ImageWoof，バッチサイズ512）に，ASA-SVRG（reset）の5シードの平均 ± 標準誤差を破線で描く．図は別のPCで作る．
2. 同じバッチサイズの SVRG・NFG-SVRG（ランダム点，`ex005`）はシード0の1試行のみである（`report_075` 9節の3）．図でこれらも5シードで示すなら，残りの8試行（4並列で約20時間）が必要である．
3. 7.2節2の揺らぎの比較を論文で述べる場合は，同じ実験の中での対応のある比較にするか，比較の条件の違いを明記する必要がある．

## 10. 成果物の zip

リポジトリのルートで指示書6.2節のコマンドを実行し，`E9_results.zip` を作った．`unzip -l E9_results.zip` の結果は次のとおりである．

```text
Archive:  E9_results.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
     1712  2026-11-23 10:54   .orders/order_081.md
    17339  2026-11-23 20:48   .reports/report_081.md
    11334  2026-11-23 10:56   programs_review/programs/analysis_e9.py
     3642  2026-11-23 10:56   programs_review/tests/test_e9_programs.py
        0  2026-11-18 22:05   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/
        0  2026-11-23 20:44   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/4/
   146350  2026-11-23 20:44   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/4/log.json
      689  2026-11-23 20:44   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/4/config.json
        0  2026-11-23 20:46   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/3/
   146316  2026-11-23 20:46   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/3/log.json
      689  2026-11-23 20:46   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/3/config.json
        0  2026-11-23 20:43   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/2/
   146419  2026-11-23 20:43   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/2/log.json
      689  2026-11-23 20:43   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/2/config.json
        0  2026-11-19 15:23   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/0/
   114861  2026-11-19 15:23   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/0/log.json
      689  2026-11-19 15:23   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/0/config.json
        0  2026-11-23 20:47   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/1/
   146299  2026-11-23 20:47   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/1/log.json
      689  2026-11-23 20:47   programs_review/outputs/ex005_appb_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/1/config.json
     2785  2026-11-23 10:56   programs_review/outputs/_verify/e9/consistency.md
     1624  2026-11-23 20:47   programs_review/outputs/e9_summary.md
  4277231  2026-11-23 20:47   programs_review/outputs/_run_logs/ex005_e9_bs512_asa_reset.log
---------                     -------
  5019357                     23 files
```
