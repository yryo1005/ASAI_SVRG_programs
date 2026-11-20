# E4：本PC（実験用リポジトリの既存のログを作成したPC）での再実行と既存のログとの一致（シード0，order_052）

本PCで `ex008_e4_a9a/train.py --inner-starts snapshot --seeds 0` を実行し，`compare_e4.py` で既存のログと比べた．
ASA-SVRG の既存のログは `last_iterate`（旧実装）であり，本実行は `snapshot` のため一致しないのは想定通りである（ASA-SVRG の `last_iterate` の一致は別のPCで `consistency_with_existing.md` に確認済み）．
SGD・SVRG・NFG-SVRG は MSE・BCE の両方で全ての項目がビット単位で一致した．


判定：整数の量（epoch, oracle_calls, full_grad_computations, accuracy）は完全一致．実数の量は，既存の値の絶対値が $ 10^{-12} $ 以上の点で相対誤差 $ \le 1e-12 $，それ未満の点（丸め誤差の区間）で絶対誤差 $ \le 1e-15 $．`elapsed_time` は比較しない．

## ログ

| 損失 | 手法 | 新しい条件 | シード | 指標 | 点数 | 完全一致 | 最大絶対誤差 | 最大相対誤差（有意な区間） | 丸め誤差の区間の点数 | 同区間の最大絶対誤差 | 判定 |
|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---|
| bce | ASAI_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | ASAI_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | ASAI_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | ASAI_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | objective_gap | 101 | × | 6.077e-07 | 7.447e-01 | 95 | 1.923e-13 | **不合格** |
| bce | ASAI_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | × | 8.570e-07 | 7.251e-01 | 95 | 6.398e-14 | **不合格** |
| bce | ASAI_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| bce | ASAI_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | approx_error | 101 | × | 5.799e-10 | 9.958e-01 | 97 | 6.739e-14 | **不合格** |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | objective_gap | 101 | ○ | 0.000e+00 | 0.000e+00 | 83 | 0.000e+00 | 合格 |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | ○ | 0.000e+00 | 0.000e+00 | 83 | 0.000e+00 | 合格 |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| bce | NFG_SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | approx_error | 101 | ○ | 0.000e+00 | 0.000e+00 | 88 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | objective_gap | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| bce | SGD | `inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | approx_error | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | objective_gap | 101 | ○ | 0.000e+00 | 0.000e+00 | 93 | 0.000e+00 | 合格 |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | ○ | 0.000e+00 | 0.000e+00 | 93 | 0.000e+00 | 合格 |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| bce | SVRG | `inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100` | 0 | approx_error | 101 | ○ | 0.000e+00 | 0.000e+00 | 101 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | ASAI_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | objective_value | 101 | × | 9.377e-02 | 3.894e-01 | 0 | 0.000e+00 | **不合格** |
| mse | ASAI_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | × | 1.698e-04 | 3.556e-15 | 98 | 1.698e-04 | **不合格** |
| mse | ASAI_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | accuracy | 101 | × | 8.044e-02 | 1.059e-01 | 0 | 0.000e+00 | **不合格** |
| mse | ASAI_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | approx_error | 101 | × | 2.512e-04 | 3.531e+00 | 98 | 2.512e-04 | **不合格** |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | objective_value | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | NFG_SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | approx_error | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | objective_value | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | accuracy | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | SGD | `inner_none_eta1.530030e-01_K29304_epochs100` | 0 | approx_error | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | epoch | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | oracle_calls | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | full_grad_computations | 101 | ○ | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 合格 |
| mse | SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | objective_value | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
| mse | SVRG | `inner_snapshot_eta1.530030e-01_K29304_epochs100` | 0 | grad_norm_sq | 101 | ○ | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 合格 |
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
| bce | ASAI_SVRG | f_star | 0.4740115090431359 | 0.4740115090431359 | 0.000e+00 |
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
| bce | NFG_SVRG | f_star | 0.4740115090431359 | 0.4740115090431359 | 0.000e+00 |
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
| bce | SGD | f_star | 0.4740115090431359 | 0.4740115090431359 | 0.000e+00 |
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
| bce | SVRG | f_star | 0.4740115090431359 | 0.4740115090431359 | 0.000e+00 |
| bce | SVRG | N_train | 29304 | 29304 | 0.000e+00 |
| bce | SVRG | N_test | 3257 | 3257 | 0.000e+00 |
| bce | SVRG | d | 123 | 123 | 0.000e+00 |
| bce | SVRG | K | 29304 | 29304 | 0.000e+00 |
| bce | SVRG | epochs | 100 | 100 | 0.000e+00 |
| mse | ASAI_SVRG | eta | 0.15300303403316318 | 0.15300303403316318 | 0.000e+00 |
| mse | ASAI_SVRG | L_individual | 2.1568199747494745 | 2.1568199747494745 | 0.000e+00 |
| mse | ASAI_SVRG | L_average | 0.9686935764271645 | 0.9686935764271645 | 0.000e+00 |
| mse | ASAI_SVRG | f_star_reference | 0.10329856538975839 | 0.10329856539159467 | 1.778e-11 |
| mse | ASAI_SVRG | N_train | 29304 | 29304 | 0.000e+00 |
| mse | ASAI_SVRG | N_test | 3257 | 3257 | 0.000e+00 |
| mse | ASAI_SVRG | d | 123 | 123 | 0.000e+00 |
| mse | ASAI_SVRG | K | 29304 | 29304 | 0.000e+00 |
| mse | ASAI_SVRG | epochs | 100 | 100 | 0.000e+00 |
| mse | NFG_SVRG | eta | 0.15300303403316318 | 0.15300303403316318 | 0.000e+00 |
| mse | NFG_SVRG | L_individual | 2.1568199747494745 | 2.1568199747494745 | 0.000e+00 |
| mse | NFG_SVRG | L_average | 0.9686935764271645 | 0.9686935764271645 | 0.000e+00 |
| mse | NFG_SVRG | f_star_reference | 0.10329856538975839 | 0.10329856539159467 | 1.778e-11 |
| mse | NFG_SVRG | N_train | 29304 | 29304 | 0.000e+00 |
| mse | NFG_SVRG | N_test | 3257 | 3257 | 0.000e+00 |
| mse | NFG_SVRG | d | 123 | 123 | 0.000e+00 |
| mse | NFG_SVRG | K | 29304 | 29304 | 0.000e+00 |
| mse | NFG_SVRG | epochs | 100 | 100 | 0.000e+00 |
| mse | SGD | eta | 0.15300303403316318 | 0.15300303403316318 | 0.000e+00 |
| mse | SGD | L_individual | 2.1568199747494745 | 2.1568199747494745 | 0.000e+00 |
| mse | SGD | L_average | 0.9686935764271645 | 0.9686935764271645 | 0.000e+00 |
| mse | SGD | f_star_reference | 0.10329856538975839 | 0.10329856539159467 | 1.778e-11 |
| mse | SGD | N_train | 29304 | 29304 | 0.000e+00 |
| mse | SGD | N_test | 3257 | 3257 | 0.000e+00 |
| mse | SGD | d | 123 | 123 | 0.000e+00 |
| mse | SGD | K | 29304 | 29304 | 0.000e+00 |
| mse | SGD | epochs | 100 | 100 | 0.000e+00 |
| mse | SVRG | eta | 0.15300303403316318 | 0.15300303403316318 | 0.000e+00 |
| mse | SVRG | L_individual | 2.1568199747494745 | 2.1568199747494745 | 0.000e+00 |
| mse | SVRG | L_average | 0.9686935764271645 | 0.9686935764271645 | 0.000e+00 |
| mse | SVRG | f_star_reference | 0.10329856538975839 | 0.10329856539159467 | 1.778e-11 |
| mse | SVRG | N_train | 29304 | 29304 | 0.000e+00 |
| mse | SVRG | N_test | 3257 | 3257 | 0.000e+00 |
| mse | SVRG | d | 123 | 123 | 0.000e+00 |
| mse | SVRG | K | 29304 | 29304 | 0.000e+00 |
| mse | SVRG | epochs | 100 | 100 | 0.000e+00 |

比較した試行数: 8，系列数: 56，完全一致: 49，合格: 49，新しいログに無いキー: なし
