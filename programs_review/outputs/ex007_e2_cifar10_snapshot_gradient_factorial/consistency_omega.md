# E2：$ \omega $ の記録が学習に影響しないことの確認

- ω の記録あり：`../outputs/_verify/ex007_omega_on`
- ω の記録なし（`--no-omega`）：`../outputs/_verify/ex007_omega_off`
- `elapsed_time`，`omega_snapshot`，`omega_average` 以外の全ての記録がビット単位で一致するかを確認する．

| 手法 | 条件 | シード | 指標 | 点数 | ビット単位で一致 | 最大絶対差 |
|---|---|---:|---|---:|---|---:|
| ASAI_SVRG | inner_snapshot_lr0.001_bs128_lambda0.0005_epochs2 | 0 | epoch | 3 | ○ | 0.000e+00 |
| ASAI_SVRG | inner_snapshot_lr0.001_bs128_lambda0.0005_epochs2 | 0 | oracle_calls | 3 | ○ | 0.000e+00 |
| ASAI_SVRG | inner_snapshot_lr0.001_bs128_lambda0.0005_epochs2 | 0 | train_loss | 3 | ○ | 0.000e+00 |
| ASAI_SVRG | inner_snapshot_lr0.001_bs128_lambda0.0005_epochs2 | 0 | test_accuracy | 3 | ○ | 0.000e+00 |
| ASAI_SVRG | inner_snapshot_lr0.001_bs128_lambda0.0005_epochs2 | 0 | approx_error | 3 | ○ | 0.000e+00 |
| ASAI_SVRG | inner_snapshot_lr0.001_bs128_lambda0.0005_epochs2 | 0 | test_correct | 3 | ○ | 0.000e+00 |
| ASAI_SVRG | inner_snapshot_lr0.001_bs128_lambda0.0005_epochs2 | 0 | full_grad_norm_sq | 3 | ○ | 0.000e+00 |
| ASAI_SVRG | inner_snapshot_lr0.001_bs128_lambda0.0005_epochs2 | 0 | approx_cos | 3 | ○ | 0.000e+00 |
| ASAI_SVRG | inner_snapshot_lr0.001_bs128_lambda0.0005_epochs2 | 0 | terminated | 3 | ○ | 0.000e+00 |
| ASAI_SVRG | inner_snapshot_lr0.001_bs128_lambda0.0005_epochs2 | 0 | omega_snapshot（記録あり） | 3 | nan, 4.94765e-08, 2.49039 | － |
| ASAI_SVRG | inner_snapshot_lr0.001_bs128_lambda0.0005_epochs2 | 0 | omega_snapshot（記録なし） | 3 | nan, nan, nan | － |

比較した系列数: 9，全てビット単位で一致: True
