# ex016_imagewoof_bs128_walltime の動作の確認

実施日：2026-11-24．本PC：NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition，Intel Xeon w7-3565X，WSL2．
ログ：`outputs/_run_logs/ex016_verify.log`．結果：`outputs/_verify/ex016/`（Git 管理外）．

```bash
# programs_review/programs で実行
python ex016_imagewoof_bs128_walltime/train.py --seeds 0 --epochs 2 --output-root ../outputs/_verify/ex016
```

## 1. 実行の順序（ログの `[start]` 行）

- `[start] 2026-11-24T18:29:36 seed0 SGD`
- `[start] 2026-11-24T18:30:15 seed0 SVRG_final`
- `[start] 2026-11-24T18:31:06 seed0 NFG_SVRG_final`
- `[start] 2026-11-24T18:31:56 seed0 ASAI_SVRG_continue`
- `[start] 2026-11-24T18:32:48 seed0 ASAI_SVRG_reset`

## 2. 既存の結果との一致（シード0，エポック0〜2）

| 条件 | 比較先 | `oracle_calls` | `train_loss` | `test_accuracy` | `approx_error` |
|---|---|---|---|---|---|
| SGD | `ex009_e3_imagewoof_walltime/SGD/inner_none_..._epochs200/0` | 一致 | 一致 | 一致 | 一致（共に NaN） |
| SVRG（最終点） | `ex012_e7_imagewoof_final/SVRG_final/inner_snapshot_..._bs128_..._epochs67/0` | 一致 | 一致 | 一致 | 一致 |
| NFG-SVRG（最終点） | `ex012_e7_imagewoof_final/NFG_SVRG_final/inner_snapshot_..._bs128_..._epochs100/0` | 一致 | 一致 | 一致 | 一致 |
| ASA-SVRG（continue） | `ex009_e3_imagewoof_walltime/ASAI_SVRG/inner_last_iterate_..._epochs100/0` | 一致 | 一致 | 一致 | 一致 |
| ASA-SVRG（reset） | `ex003_e0_imagewoof_resnet18_layernorm/ASAI_SVRG/inner_snapshot_..._bs128_..._epochs128/0` | 一致 | 一致 | 一致 | 一致 |

（「一致」はビット単位の一致である．）

## 3. 参考：1エポックの時間（`time_optimizer + time_svrg_full_grad` の増分）

| 条件 | 1エポック目［s］ | 2エポック目［s］ | E7 B（`ex013`，3周の平均）［s］ |
|---|---:|---:|---:|
| SGD | 6.21 | 5.98 | 6.022 |
| SVRG（最終点） | 17.73 | 17.72 | 17.904（初期の全勾配 8.22 s は別） |
| NFG-SVRG（最終点） | 10.58 | 10.60 | 10.639 |
| ASA-SVRG（continue） | 10.70 | 10.72 | 10.758 |
| ASA-SVRG（reset） | 10.75 | 10.72 | 計測なし |

## 4. 本実行（全エポック）と既存の結果との一致

本実行（2026-11-24 18:34 〜 2026-11-25 07:14，ログ：`outputs/_run_logs/ex016.log`）の25試行の全エポックを，2節と同じ比較先と照合した．ASA-SVRG（reset）の比較先（`ex003`）は128エポックであるため，0〜100エポックを照合した．

| 条件 | シード | 照合したエポック | `oracle_calls` | `train_loss` | `test_accuracy` | `approx_error` |
|---|---|---|---|---|---|---|
| SGD | 0〜4 | 0〜200 | 一致 | 一致 | 一致 | 一致（共に全て NaN） |
| SVRG（最終点） | 0〜4 | 0〜67 | 一致 | 一致 | 一致 | 一致 |
| NFG-SVRG（最終点） | 0〜4 | 0〜100 | 一致 | 一致 | 一致 | 一致 |
| ASA-SVRG（continue） | 0〜4 | 0〜100 | 一致 | 一致 | 一致 | 一致 |
| ASA-SVRG（reset） | 0〜4 | 0〜100 | 一致 | 一致 | 一致 | 一致 |

（「一致」はビット単位の一致である．）したがって，ex016 の学習の値は既存の結果と同じであり，新たに得たのは実時間の記録だけである．
