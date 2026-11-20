# E3：ImageWoof における実時間の比較（集計結果）

値はシードの平均 ± 標準誤差である．実時間 $ t $ は time_optimizer + time_svrg_full_grad であり，近似誤差の診断（time_diagnostic）と評価（time_evaluation）の時間は含めない．全ての試行は逐次に1つずつ実行した．

## 時間の表

| 手法 | シード数 | エポック数 | 最終の $ \#\mathrm{grad}/N $ | time_optimizer／エポック［s］ | time_svrg_full_grad／エポック［s］ | t／エポック［s］ | time_diagnostic／エポック［s］（参考） | time_evaluation／エポック［s］（参考） | 最終の t［s］ | 精度40%到達の t［s］（到達数） | 最終の学習損失 | 最終のテスト精度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGD | 5 | 200 | 200.0 | 6.020 ± 0.060 | 0.000 ± 0.000 | 6.020 ± 0.060 | 0.000 ± 0.000 | 5.867 ± 0.060 | 1203.9 ± 12.0 | 510.6 ± 10.7（5/5） | 3.1407 ± 0.0252 | 0.4654 ± 0.0099 |
| SVRG | 5 | 67 | 202.0 | 10.583 ± 0.089 | 7.378 ± 0.064 | 17.961 ± 0.153 | 0.000 ± 0.000 | 1.779 ± 0.014 | 1211.7 ± 10.3 | 1165.7 ± 18.8（3/5） | 3.6119 ± 0.0154 | 0.3885 ± 0.0064 |
| NFG-SVRG | 5 | 100 | 200.0 | 10.708 ± 0.086 | 0.000 ± 0.000 | 10.708 ± 0.086 | 7.396 ± 0.065 | 1.780 ± 0.015 | 1070.8 ± 8.6 | —（0/5） | 3.9143 ± 0.0687 | 0.2662 ± 0.0262 |
| ASA-SVRG | 5 | 100 | 200.0 | 10.684 ± 0.124 | 0.000 ± 0.000 | 10.684 ± 0.124 | 7.292 ± 0.087 | 1.755 ± 0.020 | 1068.4 ± 12.4 | 661.1 ± 9.6（5/5） | 3.3870 ± 0.0036 | 0.4570 ± 0.0020 |

## 比（実験計画5.5節）

- SVRG の1エポックあたりの時間のうち，全勾配の計算を除いた部分（time_optimizer）：10.583 s．全勾配の計算：7.378 s（1エポックあたりの t の 41.1%）．
- NFG-SVRG の time_optimizer／SVRG の time_optimizer：1.012．
- ASA-SVRG の time_optimizer／SVRG の time_optimizer：1.010．
- ASA-SVRG の time_optimizer／NFG-SVRG の time_optimizer：0.998（平均パラメータの逐次更新のオーバーヘッド -23.6 ms／エポック）．
- SGD の time_optimizer／ASA-SVRG の time_optimizer：0.563．

## 計算機の速度の変化を除いた比

同じ計算でも，試行によって計算機の速度が数%異なる場合がある（下の試行ごとの表を参照）．そのため，各試行の time_optimizer を同じ試行の全勾配の計算の時間で割った比でも比較する．

- time_optimizer／全勾配の計算（試行ごとの比の平均）：SVRG 1.4345，NFG-SVRG 1.4478，ASA-SVRG 1.4653．
- NFG-SVRG／SVRG：1.0093．ASA-SVRG／SVRG：1.0215．
- ASA-SVRG／NFG-SVRG：1.0121（平均パラメータの逐次更新のオーバーヘッド 1.21%）．
- SGD／ASA-SVRG の time_optimizer（同じシードの比）：中央値 0.556（範囲 0.549〜0.602）．

### 試行ごとの1エポックあたりの時間

| シード | SVRG time_optimizer［s］ | SVRG 全勾配［s］ | SVRG 比 | NFG-SVRG time_optimizer［s］ | NFG-SVRG 全勾配［s］ | NFG-SVRG 比 | ASA-SVRG time_optimizer［s］ | ASA-SVRG 全勾配［s］ | ASA-SVRG 比 | SGD time_optimizer［s］ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.361 | 7.225 | 1.4340 | 10.523 | 7.257 | 1.4501 | 10.668 | 7.276 | 1.4663 | 5.858 |
| 1 | 10.491 | 7.303 | 1.4365 | 10.570 | 7.292 | 1.4494 | 10.662 | 7.257 | 1.4691 | 5.961 |
| 2 | 10.480 | 7.300 | 1.4357 | 10.622 | 7.329 | 1.4493 | 10.791 | 7.361 | 1.4661 | 5.963 |
| 3 | 10.770 | 7.503 | 1.4354 | 10.869 | 7.519 | 1.4456 | 11.032 | 7.551 | 1.4611 | 6.133 |
| 4 | 10.814 | 7.558 | 1.4308 | 10.956 | 7.583 | 1.4448 | 10.269 | 7.014 | 1.4641 | 6.181 |

## 計測環境

| 項目 | 値 |
|---|---|
| cpu | Intel(R) Xeon(R) w7-3565X |
| cuda | 12.8 |
| cudnn | 91900 |
| cudnn_benchmark | False |
| cudnn_deterministic | True |
| dataloader_num_workers | 0 |
| deterministic_algorithms | True |
| gpu | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| gpu_memory_gib | 95.6 |
| logical_cpus | 64 |
| memory_gib | 125.5 |
| pillow | 13.0.0 |
| platform | Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35 |
| python | 3.12.15 |
| torch | 2.11.0+cu128 |
| torch_num_threads | 1 |
| torchvision | 0.26.0+cu128 |

全ての試行で計測環境が同一：はい

## 図

- `ex009_e3_imagewoof_walltime/figures/e3_walltime_train_loss.png`
- `ex009_e3_imagewoof_walltime/figures/e3_walltime_test_accuracy.png`
- `ex009_e3_imagewoof_walltime/figures/e3_grad_train_loss.png`
- `ex009_e3_imagewoof_walltime/figures/e3_grad_test_accuracy.png`
