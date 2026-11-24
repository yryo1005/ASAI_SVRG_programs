# report_083：ImageWoof・バッチサイズ128 の5手法の実時間の取り直し（ex016）

指示：`.orders/order_083.md`．実験のディレクトリ：`programs_review/programs/ex016_imagewoof_bs128_walltime/`，結果：`programs_review/outputs/ex016_imagewoof_bs128_walltime/`，集計：`programs_review/outputs/ex016_summary.md`．

## 0. 要点

- SGD，SVRG（最終点），NFG-SVRG（最終点），ASA-SVRG（continue），ASA-SVRG（reset）の5手法 × 5シードの25試行を，**1試行ずつ**（他の学習と同時に動かさず，毎回新しいプロセスで），シードごとに SGD → SVRG → NFG-SVRG → ASA-SVRG（continue）→ ASA-SVRG（reset）の順に5周実行した．全て完了し，失敗・打ち切りは無い．
- 学習の値（`oracle_calls`，`train_loss`，`test_accuracy`，`approx_error`）は，25試行の全エポックで既存の結果（ex009，ex003，ex012）と**ビット単位で一致**した．新たに得たのは，全エポックの実時間の記録である．
- 1エポックの時間（5周の平均）：SGD 6.01 s，SVRG（最終点）17.91 s（初期の全勾配 8.35 s は別），NFG-SVRG 10.67 s，ASA-SVRG（continue）10.82 s，ASA-SVRG（reset）10.85 s．
- 実行時刻に伴って，全ての手法の1エポックの時間が12時間で約3.6%ずつ伸びた．しかし同じ周の中の手法間の比は5周でほぼ変わらない（ASA ÷ SVRG は 0.603〜0.606，ASA ÷ NFG は 1.013〜1.015）．交互に実行したことで，時間の揺らぎは比からほぼ消えた．
- 予算（$ \#\mathrm{grad}/N \le 200 $）までの実時間は SGD 1203 s，SVRG 1190 s，NFG-SVRG 1067 s，ASA-SVRG（continue）1083 s，ASA-SVRG（reset）1085 s である．

## 1. 実行環境

| 項目 | 内容 |
|---|---|
| OS | Windows 上の WSL2（Linux 6.18.33.2-microsoft-standard-WSL2） |
| CPU | Intel Xeon w7-3565X（64論理コア），メモリ 125 GB |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition（97,887 MiB），ドライバ 582.08 |
| Python・主要パッケージ | Python 3.12.15（`.venv_pytorch`），torch 2.11.0+cu128，CUDA 12.8，cuDNN 9.19.0 |

並列数は1である．実時間を測るため，計測中は他の GPU のジョブや重い CPU の計算（テスト，集計を含む）を行わなかった．各試行の前に GPU のメモリ使用量が 2,000 MiB 未満になるまで待った．

## 2. プログラム

| ファイル | 内容 |
|---|---|
| `ex016_imagewoof_bs128_walltime/data.py`，`model.py`，`timed_training.py` | ex009 のものを変更せずにコピーした（テストで同一であることを確認）．E3（ex009）・E7 B（ex013）と同じ計測の仕組みである． |
| `ex016_imagewoof_bs128_walltime/train.py` | 新規．5条件 × シードのタスクを上の順に並べ，1試行ずつ新しいプロセスで実行する．`config.json` に `condition`，`started_at`，`finished_at` を加え，`best_model.pth` は消す． |
| `analysis_ex016.py` | 新規．1エポックの時間，予算までの実時間，手法間の比，実行の順序の表と図を作る． |
| `tests/test_ex016_programs.py` | 新規（5件）． |

条件：$ \eta = 10^{-3} $，$ \lambda = 5\times10^{-4} $，バッチサイズ128，エポック数は予算 $ \#\mathrm{grad}/N = 200 $ に合わせた（SGD 200，SVRG 67，NFG-SVRG・ASA-SVRG 100），`record_omega=False`．最終点の SVRG・NFG-SVRG は開始点によらず同じ計算になるため `inner_start="snapshot"` で実行した（E7 と同じ）．

実時間は `elapsed_time`（$ = $ `time_optimizer + time_svrg_full_grad` の累積）であり，評価と診断（`approx_error` など）の時間を含まない．

```bash
# programs_review/programs で実行
../../.venv_pytorch/bin/python ex016_imagewoof_bs128_walltime/train.py \
    --output-root ../outputs/ex016_imagewoof_bs128_walltime
../../.venv_pytorch/bin/python analysis_ex016.py
```

## 3. 一致の確認

詳細は `programs_review/outputs/ex016_imagewoof_bs128_walltime/consistency.md` にある．

- 本実行の前：シード0を2エポックだけ実行し（`_verify/ex016`），5条件とも既存の結果とビット単位で一致した．
- 本実行の後：25試行の全エポックを照合し，全てビット単位で一致した．

| 条件 | 比較先 | 照合したエポック |
|---|---|---|
| SGD | `ex009_e3_imagewoof_walltime/SGD/...epochs200` | 0〜200 |
| SVRG（最終点） | `ex012_e7_imagewoof_final/SVRG_final/...epochs67` | 0〜67 |
| NFG-SVRG（最終点） | `ex012_e7_imagewoof_final/NFG_SVRG_final/...epochs100` | 0〜100 |
| ASA-SVRG（continue） | `ex009_e3_imagewoof_walltime/ASAI_SVRG/inner_last_iterate_...epochs100` | 0〜100 |
| ASA-SVRG（reset） | `ex003_e0_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_...epochs128` | 0〜100 |

SGD の `approx_error` は，既存・今回ともに全て NaN である．

## 4. 本実行

- 期間：2026-11-24 18:34:06 〜 2026-11-25 07:14:33（12時間40分）．ログ：`programs_review/outputs/_run_logs/ex016.log`．
- `[done]` は25件，`[failed]` は0件であり，最後に「全ての学習が終了しました．」が出力された．
- 1周（5試行）は約2時間32分であった．同じ手法が連続した箇所は無い（`ex016_summary.md` 3節に全25試行の開始・終了時刻がある）．

## 5. テスト

```bash
cd programs_review && ../.venv_pytorch/bin/python -m pytest tests -q
```

`tests/test_ex016_programs.py` は5件が通過し，全体は219件が通過した．
テストを初めて実行したとき，`test_epoch_time_and_budget_values` の期待値に私の手計算の誤りがあった．増分が 12, 10, 11, 10 のとき，2エポック目以降の中央値は 10 であり，10.5 ではない．集計のプログラムが正しく，テストの期待値の方を直した．

## 6. 結果（`programs_review/outputs/ex016_summary.md`）

### 6.1 1エポックの時間と予算までの実時間（5シードの平均）

| 手法 | エポック数 | 1エポックの時間［s］（最小〜最大） | 初期の時間［s］ | 予算までの実時間［s］ | 予算でのテスト精度［%］ | 予算での学習損失 |
|---|---:|---:|---:|---:|---:|---:|
| SGD | 200 | 6.012（5.895〜6.107） | 0.00 | 1203.4 ± 7.7 | 46.54 ± 0.99 | 3.1407 ± 0.0252 |
| SVRG（最終点） | 67 | 17.905（17.562〜18.221） | 8.35 | 1190.0 ± 7.7 | 36.56 ± 1.11 | 3.6553 ± 0.0263 |
| NFG-SVRG（最終点） | 100 | 10.672（10.490〜10.834） | 0.00 | 1067.4 ± 6.4 | 27.16 ± 2.28 | 3.8912 ± 0.0485 |
| ASA-SVRG（continue） | 100 | 10.820（10.643〜10.989） | 0.00 | 1082.5 ± 6.5 | 45.70 ± 0.20 | 3.3870 ± 0.0036 |
| ASA-SVRG（reset） | 100 | 10.847（10.668〜11.038） | 0.00 | 1085.4 ± 6.5 | 37.61 ± 0.25 | 3.6699 ± 0.0040 |

1エポックの時間は，シードごとの2エポック目以降の増分の中央値である．± は標準誤差（`ddof=1`）である．

### 6.2 周ごとの1エポックの時間［s］と手法間の比

| 周（シード） | SGD | SVRG | NFG | ASA（cont.） | ASA（reset） | ASA（cont.）÷ SVRG | ASA（reset）÷ SVRG | ASA（cont.）÷ NFG | ASA（reset）÷ ASA（cont.） | NFG ÷ SVRG | ASA（cont.）÷ SGD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.895 | 17.562 | 10.490 | 10.643 | 10.668 | 0.606 | 0.607 | 1.015 | 1.002 | 0.597 | 1.805 |
| 1 | 5.970 | 17.738 | 10.588 | 10.731 | 10.761 | 0.605 | 0.607 | 1.014 | 1.003 | 0.597 | 1.797 |
| 2 | 6.021 | 17.921 | 10.675 | 10.813 | 10.841 | 0.603 | 0.605 | 1.013 | 1.003 | 0.596 | 1.796 |
| 3 | 6.068 | 18.083 | 10.773 | 10.927 | 10.928 | 0.604 | 0.604 | 1.014 | 1.000 | 0.596 | 1.801 |
| 4 | 6.107 | 18.221 | 10.834 | 10.989 | 11.038 | 0.603 | 0.606 | 1.014 | 1.004 | 0.595 | 1.799 |

### 6.3 同じ実時間（1060 s 以下の最後の評価点）での値

1060 s は全ての手法が到達する時刻である．

| 手法 | テスト精度［%］ | 学習損失 |
|---|---:|---:|
| SGD | 45.65 ± 1.45 | 3.2724 ± 0.0527 |
| SVRG（最終点） | 36.41 ± 1.16 | 3.6715 ± 0.0205 |
| NFG-SVRG（最終点） | 25.95 ± 1.71 | 3.9144 ± 0.0306 |
| ASA-SVRG（continue） | 45.48 ± 0.31 | 3.3986 ± 0.0058 |
| ASA-SVRG（reset） | 37.21 ± 0.30 | 3.6800 ± 0.0047 |

学習損失が閾値に初めて到達した実時間［s］（5シードの平均．括弧は到達したシード数）：

| 閾値 | SGD | SVRG | NFG-SVRG | ASA（cont.） | ASA（reset） |
|---:|---:|---:|---:|---:|---:|
| 3.9 | 144（5） | 388（5） | 422（5） | 236（5） | 447（5） |
| 3.7 | 345（5） | 907（5） | 877（2） | 513（5） | 1001（5） |
| 3.4 | 733（5） | 到達せず | 到達せず | 1059（5） | 到達せず |

6.3節の値は，報告用に一時的なスクリプトで計算したものであり，`analysis_ex016.py` には含まれていない．

### 6.4 図（`programs_review/outputs/ex016_imagewoof_bs128_walltime/figures/`）

横軸が実時間［s］の図と，$ \#\mathrm{grad}/N $ の図がある．縦軸は学習損失，テスト精度，近似誤差である．線は5シードの平均であり，帯は最小〜最大である．

- `ex016_train_loss_walltime.png`，`ex016_test_accuracy_walltime.png`，`ex016_approx_error_walltime.png`
- `ex016_train_loss_gradn.png`，`ex016_test_accuracy_gradn.png`，`ex016_approx_error_gradn.png`

## 7. 所見

### 7.1 事実

- 1エポックの時間は，5周を通して全ての手法で単調に伸びた（SGD 5.895 → 6.107 s，SVRG 17.562 → 18.221 s，ASA（continue）10.643 → 10.989 s．周0から周4で約3.6%）．1試行の中でも，周4では最初の10エポックより最後の10エポックの方がわずかに長い場合があった．
- 一方，同じ周の中の比の幅は小さい：ASA（continue）÷ SVRG は 0.603〜0.606，ASA（continue）÷ NFG は 1.013〜1.015，ASA（reset）÷ ASA（continue）は 1.000〜1.004．
- 前回の計測との比較：
  - E7 B（ex013，3周）の1エポックの時間は SGD 6.024 s，SVRG 17.908 s，NFG 10.635 s，ASA 10.752 s であり，今回の5周の平均との差は1%未満である．
  - ex009 の ASA（continue）・SGD のシードごとの値との比は 0.987〜1.011 であった．ただし，ex009 の ASA（continue）のシード4（10.173 s）に対しては 1.080 である．
- ASA-SVRG（continue）の1エポックは SVRG（最終点）の約0.60倍である．ただし1エポックの勾配の数が違う．ASA は $ \#\mathrm{grad}/N = 2 $，SVRG は 3 であり，SVRG の1エポック目は初期の全勾配を含めて 4 である．
  - SVRG の予算の点は66エポック目（$ \#\mathrm{grad}/N = 199 $）である．67エポック目は 202 であり，予算を超える．
  - $ \#\mathrm{grad}/N $ あたりの時間は ASA 5.41 s，SVRG 5.97 s である．このため，予算までの実時間の比 ASA（continue）÷ SVRG は 0.910 である．
- 実時間を横軸にしても，テスト精度・学習損失の順位は $ \#\mathrm{grad}/N $ の横軸と同じであった．SGD と ASA-SVRG（continue）が上位にあり，ASA-SVRG（reset）と SVRG（最終点）がほぼ同じで，NFG-SVRG（最終点）が最も低い．1060 s では SGD 45.65% と ASA（continue）45.48% の差は SGD の標準誤差（1.45）より小さい．学習損失は SGD の方が低い（3.27 対 3.40）．

### 7.2 推測

- 1エポックの時間が単調に伸びた原因は特定していない．夜間の時間帯の変化，GPU の温度や電力の制御，WSL2・Windows 側の常駐処理などが考えられる．全ての手法に同じ方向に同じ程度かかっているため，手法の性能ではなく計算機の状態によるものと考える．
- 手法を交互に並べたことで，この揺らぎは比からほぼ消えた．そのため，以前は揺らぎ（数%）に埋もれていた ASA ÷ NFG の差（約1.4%）も，今回は5周とも同じ値であり，実際の差と見てよいと考える．差の原因は，ASA の係数の計算などの追加の処理だと考えられるが，確認はしていない．
- ex009 の ASA（continue）のシード4が他より短かった（10.173 s）のも，同じ種類の時間の揺らぎによると考えられる．実時間の比を論文に載せる場合は，今回のように交互に計測した ex016 の値を使う方が妥当である．

## 8. 未解決の論点・次の作業の候補

- 時間が伸びた原因は調べていない．調べる場合は，計測中の GPU のクロック・温度・電力（`nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw`）を同時に記録する方法がある．
- 6.3節の「同じ実時間での値」と「閾値に到達する実時間」は `analysis_ex016.py` に入れていない．論文で使う場合は集計に加える．
- `programs_review/document.md` には ex016 をまだ反映していない．統合する際に，実験の一覧，ディレクトリ構成，実行コマンドに加える．
- 本PCでの作業であるため，結果の zip は作っていない．
