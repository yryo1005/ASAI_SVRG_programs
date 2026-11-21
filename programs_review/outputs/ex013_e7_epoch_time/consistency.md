# E7 B の動作の確認（実験指示3.4節）

実行：`python ex013_e7_epoch_time/train.py --datasets imagewoof --batch-sizes 128 --repeats 1 --output-root ../outputs/_verify/ex013`（ログ：`outputs/_run_logs/ex013_verify.log`，結果：`outputs/_verify/ex013/`）．

## 1. 実行の順序（ログの `[start]` 行）

- `[start] 2026-11-21T23:08:21 imagewoof bs128 rep0 ASAI_SVRG`
- `[start] 2026-11-21T23:09:51 imagewoof bs128 rep0 NFG_SVRG_final`
- `[start] 2026-11-21T23:11:21 imagewoof bs128 rep0 SVRG_final`
- `[start] 2026-11-21T23:12:50 imagewoof bs128 rep0 SGD`

ASA-SVRG → NFG-SVRG（最終点）→ SVRG（最終点）→ SGD の順であり，3.2節の順序と一致する．

## 2. `oracle_calls` の増分（$ N = 9025 $）

| 手法 | 記録（epoch 0〜4） | 規則 | 判定 |
|---|---|---|---|
| ASA-SVRG（continue） | [0, 18050, 36100, 54150, 72200] | 1エポック 2N | 一致 |
| NFG-SVRG（最終点） | [0, 18050, 36100, 54150, 72200] | 1エポック 2N | 一致 |
| SVRG（最終点） | [0, 36100, 63175, 90250, 117325] | 初期 N・1エポック 3N | 一致 |
| SGD | [0, 9025, 18050, 27075, 36100] | 1エポック N | 一致 |

## 3. 参考：1エポックの時間と ex009 の比較（判定には使わない）

1エポックの時間は 2〜4エポック目の `time_optimizer + time_svrg_full_grad` の平均である．ex009 は同じ定義で，5シードの各試行の2〜4エポック目の平均の，シード間の平均（最小〜最大）である．

| 手法 | 本確認［s］ | ex009 の手法 | ex009［s］（最小〜最大） | 差 |
|---|---:|---|---:|---:|
| ASA-SVRG（continue） | 10.910 | ASAI_SVRG（last） | 10.821（10.603〜11.087） | +0.82% |
| NFG-SVRG（最終点） | 10.837 | NFG_SVRG（snap） | 10.616（10.516〜10.951） | +2.09% |
| SVRG（最終点） | 18.344 | SVRG（last） | 17.906（17.460〜18.401） | +2.45% |
| SGD | 6.131 | SGD（none） | 6.006（5.889〜6.170） | +2.08% |

- SGD と ASA-SVRG は ex009 と同じ手法である．NFG-SVRG・SVRG は ex009 ではランダム点であり，1エポックの計算量は同じであるため参考として並べた．
