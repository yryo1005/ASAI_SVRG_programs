# E4：既存のログとの一致の確認（シード0）

判定：整数の量（epoch, oracle_calls, full_grad_computations, accuracy）は完全一致．実数の量は，既存の値の絶対値が $ 10^{-12} $ 以上の点で相対誤差 $ \le 1e-12 $，それ未満の点（丸め誤差の区間）で絶対誤差 $ \le 1e-15 $．`elapsed_time` は比較しない．

## ログ

| 損失 | 手法 | 新しい条件 | シード | 指標 | 点数 | 完全一致 | 最大絶対誤差 | 最大相対誤差（有意な区間） | 丸め誤差の区間の点数 | 同区間の最大絶対誤差 | 判定 |
|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---|
| bce | ASAI_SVRG | `inner_last_iterate_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | ASAI_SVRG | `inner_last_iterate_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | ASAI_SVRG | `inner_last_iterate_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | ASAI_SVRG | `inner_last_iterate_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | objective_gap | 101 | × | 1.665e-16 | 2.082e-07 | 95 | 1.665e-16 | **不合格** |
| bce | ASAI_SVRG | `inner_last_iterate_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | × | 6.776e-19 | 2.060e-12 | 95 | 7.500e-24 | **不合格** |
| bce | ASAI_SVRG | `inner_last_iterate_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| bce | ASAI_SVRG | `inner_last_iterate_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | approx_error | 101 | × | 1.355e-19 | 7.617e-12 | 97 | 1.705e-24 | **不合格** |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | objective_gap | 101 | × | 2.220e-16 | 2.974e-05 | 83 | 1.110e-16 | **不合格** |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | × | 1.041e-17 | 8.652e-11 | 83 | 2.032e-23 | **不合格** |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | approx_error | 101 | × | 6.939e-18 | 5.610e-11 | 88 | 1.218e-23 | **不合格** |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | objective_gap | 101 | × | 2.220e-16 | 1.404e-14 | 0 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | × | 2.220e-16 | 1.190e-15 | 0 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | approx_error | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | objective_gap | 101 | × | 1.110e-16 | 3.688e-06 | 93 | 1.110e-16 | **不合格** |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | × | 3.469e-18 | 3.331e-11 | 93 | 1.615e-23 | **不合格** |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | approx_error | 101 | ○ | 0.000e+00 | 0.000e+00 | 101 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | objective_value | 101 | × | 5.551e-17 | 2.168e-16 | 0 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | × | 5.551e-17 | 1.778e-15 | 98 | 7.260e-30 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_last_iterate_eta1.530030e-01_K29304_epochs100` | 0 | approx_error | 101 | × | 7.047e-19 | 2.322e-15 | 98 | 2.474e-31 | 合格 |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | objective_value | 101 | × | 8.140e-10 | 7.383e-09 | 0 | 0.000e+00 | **不合格** |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | × | 1.966e-11 | 1.721e-06 | 1 | 1.464e-34 | **不合格** |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | approx_error | 101 | × | 1.423e-11 | 2.178e-06 | 1 | 1.033e-32 | **不合格** |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | objective_value | 101 | × | 5.551e-17 | 4.982e-16 | 0 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | × | 1.561e-17 | 1.765e-14 | 0 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | approx_error | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | objective_value | 101 | × | 8.327e-17 | 8.056e-16 | 0 | 0.000e+00 | 合格 |
| mse | SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | × | 1.563e-17 | 4.899e-09 | 0 | 0.000e+00 | **不合格** |
| mse | SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | approx_error | 101 | ○ | 0.000e+00 | 0.000e+00 | 101 | 0.000e+00 | 合格 |

## config.json の定数

| 損失 | 手法 | キー | 新しい値 | 既存の値 | 相対誤差 |
|---|---|---|---:|---:|---:|
| bce | ASAI_SVRG | eta | 0.08543689320388351 | 0.08543689320388351 | 0.000e+00 |
| bce | ASAI_SVRG | L | 3.8625 | 3.8625 | 0.000e+00 |
| bce | ASAI_SVRG | reg_lambda | 0.11249999999999999 | 0.11249999999999999 | 0.000e+00 |
| bce | ASAI_SVRG | mu | 0.11249999999999999 | 0.11249999999999999 | 0.000e+00 |
| bce | ASAI_SVRG | rho | 297.3550382338325 | 297.3550382338325 | 0.000e+00 |
| bce | ASAI_SVRG | f_star | 0.4740115090431358 | 0.4740115090431359 | 1.171e-16 |
| bce | ASAI_SVRG | N_train | 29304 | 29304 | 0.000e+00 |
| bce | ASAI_SVRG | N_test | 3257 | 3257 | 0.000e+00 |
| bce | ASAI_SVRG | d | 123 | 123 | 0.000e+00 |
| bce | ASAI_SVRG | K | 29304 | 29304 | 0.000e+00 |
| bce | ASAI_SVRG | epochs | 100 | 100 | 0.000e+00 |
| bce | NFG_SVRG | eta | 0.08543689320388351 | 0.08543689320388351 | 0.000e+00 |
| bce | NFG_SVRG | L | 3.8625 | 3.8625 | 0.000e+00 |
| bce | NFG_SVRG | reg_lambda | 0.11249999999999999 | 0.11249999999999999 | 0.000e+00 |
| bce | NFG_SVRG | mu | 0.11249999999999999 | 0.11249999999999999 | 0.000e+00 |
| bce | NFG_SVRG | rho | 297.3550382338325 | 297.3550382338325 | 0.000e+00 |
| bce | NFG_SVRG | f_star | 0.4740115090431358 | 0.4740115090431359 | 1.171e-16 |
| bce | NFG_SVRG | N_train | 29304 | 29304 | 0.000e+00 |
| bce | NFG_SVRG | N_test | 3257 | 3257 | 0.000e+00 |
| bce | NFG_SVRG | d | 123 | 123 | 0.000e+00 |
| bce | NFG_SVRG | K | 29304 | 29304 | 0.000e+00 |
| bce | NFG_SVRG | epochs | 100 | 100 | 0.000e+00 |
| bce | SGD | eta | 0.08543689320388351 | 0.08543689320388351 | 0.000e+00 |
| bce | SGD | L | 3.8625 | 3.8625 | 0.000e+00 |
| bce | SGD | reg_lambda | 0.11249999999999999 | 0.11249999999999999 | 0.000e+00 |
| bce | SGD | mu | 0.11249999999999999 | 0.11249999999999999 | 0.000e+00 |
| bce | SGD | rho | 297.3550382338325 | 297.3550382338325 | 0.000e+00 |
| bce | SGD | f_star | 0.4740115090431358 | 0.4740115090431359 | 1.171e-16 |
| bce | SGD | N_train | 29304 | 29304 | 0.000e+00 |
| bce | SGD | N_test | 3257 | 3257 | 0.000e+00 |
| bce | SGD | d | 123 | 123 | 0.000e+00 |
| bce | SGD | K | 29304 | 29304 | 0.000e+00 |
| bce | SGD | epochs | 100 | 100 | 0.000e+00 |
| bce | SVRG | eta | 0.08543689320388351 | 0.08543689320388351 | 0.000e+00 |
| bce | SVRG | L | 3.8625 | 3.8625 | 0.000e+00 |
| bce | SVRG | reg_lambda | 0.11249999999999999 | 0.11249999999999999 | 0.000e+00 |
| bce | SVRG | mu | 0.11249999999999999 | 0.11249999999999999 | 0.000e+00 |
| bce | SVRG | rho | 297.3550382338325 | 297.3550382338325 | 0.000e+00 |
| bce | SVRG | f_star | 0.4740115090431358 | 0.4740115090431359 | 1.171e-16 |
| bce | SVRG | N_train | 29304 | 29304 | 0.000e+00 |
| bce | SVRG | N_test | 3257 | 3257 | 0.000e+00 |
| bce | SVRG | d | 123 | 123 | 0.000e+00 |
| bce | SVRG | K | 29304 | 29304 | 0.000e+00 |
| bce | SVRG | epochs | 100 | 100 | 0.000e+00 |
| mse | ASAI_SVRG | eta | 0.15300303403316318 | 0.15300303403316318 | 0.000e+00 |
| mse | ASAI_SVRG | L_individual | 2.1568199747494745 | 2.1568199747494745 | 0.000e+00 |
| mse | ASAI_SVRG | L_average | 0.9686935764271647 | 0.9686935764271645 | 2.292e-16 |
| mse | ASAI_SVRG | f_star_reference | 0.10329856535152805 | 0.10329856539159467 | 3.879e-10 |
| mse | ASAI_SVRG | N_train | 29304 | 29304 | 0.000e+00 |
| mse | ASAI_SVRG | N_test | 3257 | 3257 | 0.000e+00 |
| mse | ASAI_SVRG | d | 123 | 123 | 0.000e+00 |
| mse | ASAI_SVRG | K | 29304 | 29304 | 0.000e+00 |
| mse | ASAI_SVRG | epochs | 100 | 100 | 0.000e+00 |
| mse | NFG_SVRG | eta | 0.15300303403316318 | 0.15300303403316318 | 0.000e+00 |
| mse | NFG_SVRG | L_individual | 2.1568199747494745 | 2.1568199747494745 | 0.000e+00 |
| mse | NFG_SVRG | L_average | 0.9686935764271647 | 0.9686935764271645 | 2.292e-16 |
| mse | NFG_SVRG | f_star_reference | 0.10329856535152805 | 0.10329856539159467 | 3.879e-10 |
| mse | NFG_SVRG | N_train | 29304 | 29304 | 0.000e+00 |
| mse | NFG_SVRG | N_test | 3257 | 3257 | 0.000e+00 |
| mse | NFG_SVRG | d | 123 | 123 | 0.000e+00 |
| mse | NFG_SVRG | K | 29304 | 29304 | 0.000e+00 |
| mse | NFG_SVRG | epochs | 100 | 100 | 0.000e+00 |
| mse | SGD | eta | 0.15300303403316318 | 0.15300303403316318 | 0.000e+00 |
| mse | SGD | L_individual | 2.1568199747494745 | 2.1568199747494745 | 0.000e+00 |
| mse | SGD | L_average | 0.9686935764271647 | 0.9686935764271645 | 2.292e-16 |
| mse | SGD | f_star_reference | 0.10329856535152805 | 0.10329856539159467 | 3.879e-10 |
| mse | SGD | N_train | 29304 | 29304 | 0.000e+00 |
| mse | SGD | N_test | 3257 | 3257 | 0.000e+00 |
| mse | SGD | d | 123 | 123 | 0.000e+00 |
| mse | SGD | K | 29304 | 29304 | 0.000e+00 |
| mse | SGD | epochs | 100 | 100 | 0.000e+00 |
| mse | SVRG | eta | 0.15300303403316318 | 0.15300303403316318 | 0.000e+00 |
| mse | SVRG | L_individual | 2.1568199747494745 | 2.1568199747494745 | 0.000e+00 |
| mse | SVRG | L_average | 0.9686935764271647 | 0.9686935764271645 | 2.292e-16 |
| mse | SVRG | f_star_reference | 0.10329856535152805 | 0.10329856539159467 | 3.879e-10 |
| mse | SVRG | N_train | 29304 | 29304 | 0.000e+00 |
| mse | SVRG | N_test | 3257 | 3257 | 0.000e+00 |
| mse | SVRG | d | 123 | 123 | 0.000e+00 |
| mse | SVRG | K | 29304 | 29304 | 0.000e+00 |
| mse | SVRG | epochs | 100 | 100 | 0.000e+00 |

比較した試行数: 8，系列数: 56，完全一致: 36，合格: 44，新しいログに無いキー: なし

## 判定のまとめと差の原因

**整数の量と精度．** 8試行の `epoch`，`oracle_calls`，`full_grad_computations`，`accuracy`（101点）は全て完全一致した．

**上の機械的な判定で「不合格」となった12系列．** 差の大きさは次の通りである．
- BCE（SVRG，NFG-SVRG，ASA-SVRG の `objective_gap`，`grad_norm_sq`，`approx_error`）と MSE の SVRG の `grad_norm_sq`：
  絶対誤差は `objective_gap` が $ \le 2.2\times10^{-16} $（$ f \approx 0.47 $ の1〜2 ulp），`grad_norm_sq`・`approx_error` が $ \le 1.6\times10^{-17} $ である．
  相対誤差が大きいのは，値が $ 10^{-12} $ 以上であっても $ f({\mathbf z}_{s}) - f({\mathbf w}^{*}) $ のように丸め誤差の水準の絶対誤差をもつ量の差であるか，
  学習が進んで値自体が丸め誤差の水準に近づいたためである．
- MSE の NFG-SVRG：`objective_value` の絶対誤差 $ 8.1\times10^{-10} $（相対 $ 7.4\times10^{-9} $），`grad_norm_sq` の相対誤差 $ 1.7\times10^{-6} $，
  `approx_error` の相対誤差 $ 2.2\times10^{-6} $．外側ループ1〜3では差は0または1 ulp で，外側ループごとに増大する（非凸の目的関数の軌道が
  丸め誤差を増幅する）．精度は101点全てで完全一致した．

**差の原因の確認（本PCでの旧コードとの比較）．** 既存のプログラム（`programs/ex000_a9a_least_squares/train.py`，`programs/ex0001_a9a_logistic/train.py`）の
`run_variance_reduced` を，ファイルを変更せずに本PCで12外側ループ実行し（データの読み込み先のみ `programs_review/datasets/a9a/raw/a9a` に差し替え，
BCE の $ f({\mathbf w}^{*}) $ は新しい試行の `config.json` の値を用いた），新しいプログラムの同じ試行の先頭13点と比べた．
MSE・BCE × SVRG（`snapshot`），NFG-SVRG（`snapshot`），ASA-SVRG（`last_iterate`）の6組全てで，`elapsed_time` を除く全てのキーが
**ビット単位で一致した**．MSE の NFG-SVRG では，既存のログとの相対誤差が外側ループ10で既に $ 5\times10^{-9} $ に達しているため，
この区間で新旧のコードが同一の計算をしていることが確認できる．したがって既存のログとの差は，実装の違いではなく，計算環境
（CPU，BLAS，torch のビルド）の違いによる丸め誤差と，その増幅によるものである．

**本PCの BCE の $ f({\mathbf w}^{*}) $．** 新しいプログラムの値 0.4740115090431358 と既存の値 0.4740115090431359 の差は1 ulp である．
MSE の `f_star_reference`（非凸，参照値のみ）は相対 $ 3.9\times10^{-10} $ 異なるが，ログの量の計算には用いない．
