# E4：既存のログとの一致の確認（シード0）

判定：整数の量（epoch, oracle_calls, full_grad_computations, accuracy）は完全一致．実数の量は，既存の値の絶対値が $ 10^{-12} $ 以上の点で相対誤差 $ \le 1e-12 $，それ未満の点（丸め誤差の区間）で絶対誤差 $ \le 1e-15 $．`elapsed_time` は比較しない．

## ログ

| 損失 | 手法 | 新しい条件 | シード | 指標 | 点数 | 完全一致 | 最大絶対誤差 | 最大相対誤差（有意な区間） | 丸め誤差の区間の点数 | 同区間の最大絶対誤差 | 判定 |
|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---|
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | objective_value | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | ○ | 0.000e+00 | 0.000e+00 | 98 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | approx_error | 101 | ○ | 0.000e+00 | 0.000e+00 | 98 | 0.000e+00 | 合格 |

## config.json の定数

| 損失 | 手法 | キー | 新しい値 | 既存の値 | 相対誤差 |
|---|---|---|---:|---:|---:|
| mse | ASAI_SVRG | eta | 0.15300303403316318 | 0.15300303403316318 | 0.000e+00 |
| mse | ASAI_SVRG | L_individual | 2.1568199747494745 | 2.1568199747494745 | 0.000e+00 |
| mse | ASAI_SVRG | L_average | 0.9686935764271645 | 0.9686935764271645 | 0.000e+00 |
| mse | ASAI_SVRG | f_star_reference | 0.10329856538975839 | 0.10329856539159467 | 1.778e-11 |
| mse | ASAI_SVRG | N_train | 29304 | 29304 | 0.000e+00 |
| mse | ASAI_SVRG | N_test | 3257 | 3257 | 0.000e+00 |
| mse | ASAI_SVRG | d | 123 | 123 | 0.000e+00 |
| mse | ASAI_SVRG | K | 29304 | 29304 | 0.000e+00 |
| mse | ASAI_SVRG | epochs | 100 | 100 | 0.000e+00 |

比較した試行数: 1，系列数: 7，完全一致: 7，合格: 7，新しいログに無いキー: なし
