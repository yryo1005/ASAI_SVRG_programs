# E1：Mushroomにおける ρ の異なる学習率での収束挙動とサンプリング方式の影響（集計結果）

値は5シードの平均 ± 標準誤差である．最終値は100エポック終了時（打ち切りの場合は打ち切り直前）の評価点の値，誤差床は最後の20外側ループにおける中央値である．

## 表

| sampling | ρ（目標） | 手法 | η | ρ（実測） | 最終の目的関数誤差 | 実測の縮小率 r̂ | 最終のテスト精度 | 誤差床（目的関数誤差） | 誤差床（近似誤差） | 打ち切り数 |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| random reshuffling | 0.5 | SGD | 0.001105 ± 0.000000 | 0.5000 ± 0.0000 | 7.530e-04 ± 1.7e-04 | （適用外） | 0.8866 ± 0.0030 | 5.971e-04 ± 2.4e-05 | （適用外） | 0 |
| random reshuffling | 0.5 | SVRG | 0.001105 ± 0.000000 | 0.5000 ± 0.0000 | 4.441e-17 ± 2.7e-17 | 0.0009 ± 0.0003 | 0.8866 ± 0.0036 | 4.441e-17 ± 2.7e-17 | 0.000e+00 ± 0.0e+00 | 0 |
| random reshuffling | 0.5 | NFG-SVRG | 0.001105 ± 0.000000 | 0.5000 ± 0.0000 | 6.661e-17 ± 2.7e-17 | 0.1154 ± 0.0010 | 0.8866 ± 0.0036 | 6.661e-17 ± 2.7e-17 | 9.955e-31 ± 1.5e-31 | 0 |
| random reshuffling | 0.5 | ASA-SVRG | 0.001105 ± 0.000000 | 0.5000 ± 0.0000 | 6.661e-17 ± 2.7e-17 | 0.0135 ± 0.0002 | 0.8866 ± 0.0036 | 7.772e-17 ± 2.2e-17 | 4.326e-30 ± 2.9e-31 | 0 |
| random reshuffling | 0.64 | SGD | 0.001591 ± 0.000000 | 0.6399 ± 0.0000 | 1.166e-03 ± 1.8e-04 | （適用外） | 0.8856 ± 0.0030 | 9.265e-04 ± 4.9e-05 | （適用外） | 0 |
| random reshuffling | 0.64 | SVRG | 0.001591 ± 0.000000 | 0.6399 ± 0.0000 | 6.661e-17 ± 2.7e-17 | 0.0018 ± 0.0006 | 0.8866 ± 0.0036 | 6.661e-17 ± 2.7e-17 | 0.000e+00 ± 0.0e+00 | 0 |
| random reshuffling | 0.64 | NFG-SVRG | 0.001591 ± 0.000000 | 0.6399 ± 0.0000 | 1.110e-16 ± 3.5e-17 | 0.0772 ± 0.0008 | 0.8866 ± 0.0036 | 1.110e-16 ± 3.5e-17 | 9.948e-31 ± 2.0e-31 | 0 |
| random reshuffling | 0.64 | ASA-SVRG | 0.001591 ± 0.000000 | 0.6399 ± 0.0000 | 8.882e-17 ± 4.2e-17 | 0.0065 ± 0.0002 | 0.8866 ± 0.0036 | 6.661e-17 ± 2.7e-17 | 4.759e-30 ± 5.6e-31 | 0 |
| random reshuffling | 0.8 | SGD | 0.002027 ± 0.000000 | 0.8000 ± 0.0000 | 1.535e-03 ± 1.9e-04 | （適用外） | 0.8849 ± 0.0024 | 1.238e-03 ± 6.3e-05 | （適用外） | 0 |
| random reshuffling | 0.8 | SVRG | 0.002027 ± 0.000000 | 0.8000 ± 0.0000 | 2.220e-17 ± 2.2e-17 | 0.0024 ± 0.0011 | 0.8866 ± 0.0036 | 2.220e-17 ± 2.2e-17 | 0.000e+00 ± 0.0e+00 | 0 |
| random reshuffling | 0.8 | NFG-SVRG | 0.002027 ± 0.000000 | 0.8000 ± 0.0000 | 6.661e-17 ± 2.7e-17 | 0.0617 ± 0.0005 | 0.8866 ± 0.0036 | 4.441e-17 ± 2.7e-17 | 9.954e-31 ± 1.7e-31 | 0 |
| random reshuffling | 0.8 | ASA-SVRG | 0.002027 ± 0.000000 | 0.8000 ± 0.0000 | 6.661e-17 ± 2.7e-17 | 0.0041 ± 0.0001 | 0.8866 ± 0.0036 | 8.882e-17 ± 2.2e-17 | 5.044e-30 ± 9.3e-31 | 0 |
| random reshuffling | 1 | SGD | 0.002492 ± 0.000000 | 1.0000 ± 0.0000 | 1.915e-03 ± 1.9e-04 | （適用外） | 0.8851 ± 0.0018 | 1.578e-03 ± 9.1e-05 | （適用外） | 0 |
| random reshuffling | 1 | SVRG | 0.002492 ± 0.000000 | 1.0000 ± 0.0000 | 8.882e-17 ± 2.2e-17 | 0.0033 ± 0.0017 | 0.8866 ± 0.0036 | 8.882e-17 ± 2.2e-17 | 0.000e+00 ± 0.0e+00 | 0 |
| random reshuffling | 1 | NFG-SVRG | 0.002492 ± 0.000000 | 1.0000 ± 0.0000 | 8.882e-17 ± 2.2e-17 | 0.0546 ± 0.0022 | 0.8866 ± 0.0036 | 8.882e-17 ± 2.2e-17 | 9.548e-31 ± 2.1e-31 | 0 |
| random reshuffling | 1 | ASA-SVRG | 0.002492 ± 0.000000 | 1.0000 ± 0.0000 | 6.661e-17 ± 2.7e-17 | 0.0030 ± 0.0001 | 0.8866 ± 0.0036 | 8.882e-17 ± 2.2e-17 | 5.096e-30 ± 9.3e-31 | 0 |
| random reshuffling | 2 | SGD | 0.004165 ± 0.000000 | 2.0000 ± 0.0000 | 3.115e-03 ± 2.1e-04 | （適用外） | 0.8829 ± 0.0009 | 2.875e-03 ± 1.5e-04 | （適用外） | 0 |
| random reshuffling | 2 | SVRG | 0.004165 ± 0.000000 | 2.0000 ± 0.0000 | 6.661e-17 ± 2.7e-17 | 0.0043 ± 0.0020 | 0.8866 ± 0.0036 | 6.661e-17 ± 2.7e-17 | 0.000e+00 ± 0.0e+00 | 0 |
| random reshuffling | 2 | NFG-SVRG | 0.004165 ± 0.000000 | 2.0000 ± 0.0000 | 8.882e-17 ± 2.2e-17 | 0.0555 ± 0.0048 | 0.8866 ± 0.0036 | 8.882e-17 ± 2.2e-17 | 1.000e-30 ± 1.9e-31 | 0 |
| random reshuffling | 2 | ASA-SVRG | 0.004165 ± 0.000000 | 2.0000 ± 0.0000 | 2.220e-17 ± 2.2e-17 | 0.0018 ± 0.0000 | 0.8866 ± 0.0036 | 4.441e-17 ± 2.7e-17 | 4.732e-30 ± 4.0e-31 | 0 |
| random reshuffling | 10 | SGD | 0.008142 ± 0.000000 | 10.0000 ± 0.0000 | 5.308e-03 ± 4.7e-04 | （適用外） | 0.8800 ± 0.0015 | 5.902e-03 ± 2.4e-04 | （適用外） | 0 |
| random reshuffling | 10 | SVRG | 0.008142 ± 0.000000 | 10.0000 ± 0.0000 | 4.441e-17 ± 2.7e-17 | 0.0082 ± 0.0035 | 0.8866 ± 0.0036 | 4.441e-17 ± 2.7e-17 | 0.000e+00 ± 0.0e+00 | 0 |
| random reshuffling | 10 | NFG-SVRG | 0.008142 ± 0.000000 | 10.0000 ± 0.0000 | 8.882e-17 ± 4.2e-17 | 0.0761 ± 0.0072 | 0.8866 ± 0.0036 | 8.882e-17 ± 4.2e-17 | 1.027e-30 ± 2.1e-31 | 0 |
| random reshuffling | 10 | ASA-SVRG | 0.008142 ± 0.000000 | 10.0000 ± 0.0000 | 6.661e-17 ± 2.7e-17 | 0.0016 ± 0.0002 | 0.8866 ± 0.0036 | 4.441e-17 ± 2.7e-17 | 4.746e-30 ± 5.4e-31 | 0 |
| random reshuffling | 298.4 | SGD | 0.010498 ± 0.000000 | 298.4231 ± 0.0000 | 6.533e-03 ± 5.7e-04 | （適用外） | 0.8795 ± 0.0030 | 7.607e-03 ± 2.9e-04 | （適用外） | 0 |
| random reshuffling | 298.4 | SVRG | 0.010498 ± 0.000000 | 298.4231 ± 0.0000 | 8.882e-17 ± 2.2e-17 | 0.0091 ± 0.0025 | 0.8866 ± 0.0036 | 8.882e-17 ± 2.2e-17 | 0.000e+00 ± 0.0e+00 | 0 |
| random reshuffling | 298.4 | NFG-SVRG | 0.010498 ± 0.000000 | 298.4231 ± 0.0000 | 1.110e-16 ± 0.0e+00 | 0.0860 ± 0.0079 | 0.8866 ± 0.0036 | 1.110e-16 ± 0.0e+00 | 9.979e-31 ± 2.5e-31 | 0 |
| random reshuffling | 298.4 | ASA-SVRG | 0.010498 ± 0.000000 | 298.4231 ± 0.0000 | 6.661e-17 ± 2.7e-17 | 0.0016 ± 0.0003 | 0.8866 ± 0.0036 | 6.661e-17 ± 2.7e-17 | 4.383e-30 ± 5.0e-31 | 0 |
| uniform w/ replacement | 0.5 | SGD | 0.001105 ± 0.000000 | 0.5000 ± 0.0000 | 9.675e-04 ± 8.5e-05 | （適用外） | 0.8881 ± 0.0043 | 7.831e-04 ± 2.4e-05 | （適用外） | 0 |
| uniform w/ replacement | 0.5 | SVRG | 0.001105 ± 0.000000 | 0.5000 ± 0.0000 | 6.661e-17 ± 2.7e-17 | 0.0008 ± 0.0002 | 0.8866 ± 0.0036 | 6.661e-17 ± 2.7e-17 | 0.000e+00 ± 0.0e+00 | 0 |
| uniform w/ replacement | 0.5 | NFG-SVRG | 0.001105 ± 0.000000 | 0.5000 ± 0.0000 | 1.595e-04 ± 3.1e-05 | 0.9262 ± 0.0020 | 0.8888 ± 0.0035 | 1.726e-04 ± 3.8e-06 | 4.154e-04 ± 1.0e-05 | 0 |
| uniform w/ replacement | 0.5 | ASA-SVRG | 0.001105 ± 0.000000 | 0.5000 ± 0.0000 | 1.290e-04 ± 2.9e-05 | 0.9240 ± 0.0022 | 0.8883 ± 0.0034 | 1.377e-04 ± 2.4e-06 | 4.040e-04 ± 9.4e-06 | 0 |
| uniform w/ replacement | 1 | SGD | 0.002492 ± 0.000000 | 1.0000 ± 0.0000 | 1.898e-03 ± 2.1e-04 | （適用外） | 0.8868 ± 0.0040 | 1.870e-03 ± 9.4e-05 | （適用外） | 0 |
| uniform w/ replacement | 1 | SVRG | 0.002492 ± 0.000000 | 1.0000 ± 0.0000 | 6.661e-17 ± 2.7e-17 | 0.0019 ± 0.0004 | 0.8866 ± 0.0036 | 6.661e-17 ± 2.7e-17 | 0.000e+00 ± 0.0e+00 | 0 |
| uniform w/ replacement | 1 | NFG-SVRG | 0.002492 ± 0.000000 | 1.0000 ± 0.0000 | 1.666e-04 ± 3.5e-05 | 0.9266 ± 0.0021 | 0.8888 ± 0.0035 | 1.724e-04 ± 3.0e-06 | 4.121e-04 ± 9.1e-06 | 0 |
| uniform w/ replacement | 1 | ASA-SVRG | 0.002492 ± 0.000000 | 1.0000 ± 0.0000 | 1.490e-04 ± 3.3e-05 | 0.9254 ± 0.0022 | 0.8881 ± 0.0034 | 1.540e-04 ± 3.1e-06 | 4.043e-04 ± 9.3e-06 | 0 |
| uniform w/ replacement | 298.4 | SGD | 0.010498 ± 0.000000 | 298.4231 ± 0.0000 | 7.105e-03 ± 9.7e-04 | （適用外） | 0.8817 ± 0.0039 | 8.239e-03 ± 3.6e-04 | （適用外） | 0 |
| uniform w/ replacement | 298.4 | SVRG | 0.010498 ± 0.000000 | 298.4231 ± 0.0000 | 8.882e-17 ± 2.2e-17 | 0.0141 ± 0.0029 | 0.8866 ± 0.0036 | 8.882e-17 ± 2.2e-17 | 0.000e+00 ± 0.0e+00 | 0 |
| uniform w/ replacement | 298.4 | NFG-SVRG | 0.010498 ± 0.000000 | 298.4231 ± 0.0000 | 1.818e-04 ± 3.9e-05 | 0.9274 ± 0.0021 | 0.8883 ± 0.0034 | 1.746e-04 ± 6.1e-06 | 4.163e-04 ± 1.3e-05 | 0 |
| uniform w/ replacement | 298.4 | ASA-SVRG | 0.010498 ± 0.000000 | 298.4231 ± 0.0000 | 1.625e-04 ± 3.7e-05 | 0.9262 ± 0.0022 | 0.8881 ± 0.0034 | 1.637e-04 ± 3.1e-06 | 4.044e-04 ± 9.0e-06 | 0 |


## 図

- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_random_reshuffle_SGD_budget31.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_random_reshuffle_SGD_full.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_random_reshuffle_SVRG_budget31.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_random_reshuffle_SVRG_full.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_random_reshuffle_NFG_SVRG_budget31.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_random_reshuffle_NFG_SVRG_full.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_approx_error_random_reshuffle_NFG_SVRG_budget31.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_approx_error_random_reshuffle_NFG_SVRG_full.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_random_reshuffle_ASAI_SVRG_budget31.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_random_reshuffle_ASAI_SVRG_full.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_approx_error_random_reshuffle_ASAI_SVRG_budget31.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_approx_error_random_reshuffle_ASAI_SVRG_full.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_uniform_with_replacement_SGD_budget31.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_uniform_with_replacement_SGD_full.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_uniform_with_replacement_SVRG_budget31.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_uniform_with_replacement_SVRG_full.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_uniform_with_replacement_NFG_SVRG_budget31.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_uniform_with_replacement_NFG_SVRG_full.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_approx_error_uniform_with_replacement_NFG_SVRG_budget31.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_approx_error_uniform_with_replacement_NFG_SVRG_full.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_uniform_with_replacement_ASAI_SVRG_budget31.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_loss_uniform_with_replacement_ASAI_SVRG_full.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_approx_error_uniform_with_replacement_ASAI_SVRG_budget31.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_approx_error_uniform_with_replacement_ASAI_SVRG_full.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_sampling_overlay_SGD_loss.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_sampling_overlay_SVRG_loss.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_sampling_overlay_NFG_SVRG_loss.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_sampling_overlay_NFG_SVRG_approx_error.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_sampling_overlay_ASAI_SVRG_loss.png`
- `ex006_e1_mushroom_rho_sampling/figures/e1_sampling_overlay_ASAI_SVRG_approx_error.png`
