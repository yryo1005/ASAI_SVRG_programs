# E1 の最終点の試行の PC 間の差（本PCの再実行 ↔ 既存のログ）

本PCの再実行：`outputs/_verify/ex006_final_this_pc`，既存のログ：`outputs/ex006_e1_mushroom_rho_sampling`．

| 手法 | sampling | ρ | n | 整数の項目と精度が一致 | f* の差の最大 | 目的関数誤差の最大絶対差 | r̂ の差の最大絶対値 | log10 誤差床の比の最大絶対値 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NFG-SVRG | random reshuffling | 0.5 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 1.4e-07 | 0.045 |
| NFG-SVRG | random reshuffling | 0.64 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 4.7e-08 | 0.045 |
| NFG-SVRG | random reshuffling | 0.8 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 2.9e-07 | 0.045 |
| NFG-SVRG | random reshuffling | 1 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 1.1e-06 | 0.045 |
| NFG-SVRG | random reshuffling | 2 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 1.4e-07 | 0.045 |
| NFG-SVRG | random reshuffling | 10 | 5 | 5/5 | 2.2e-16 | 4.4e-16 | 9.9e-07 | 0.346 |
| NFG-SVRG | random reshuffling | 298.4 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 3.5e-07 | 0.045 |
| NFG-SVRG | uniform w/ replacement | 0.5 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 1.1e-14 | 0.000 |
| NFG-SVRG | uniform w/ replacement | 1 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 1.2e-14 | 0.000 |
| NFG-SVRG | uniform w/ replacement | 298.4 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 1.8e-14 | 0.000 |
| SVRG | random reshuffling | 0.5 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 2.4e-09 | 0.045 |
| SVRG | random reshuffling | 0.64 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 7.9e-10 | 0.045 |
| SVRG | random reshuffling | 0.8 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 5.2e-10 | 0.346 |
| SVRG | random reshuffling | 1 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 7.2e-08 | 0.045 |
| SVRG | random reshuffling | 2 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 1.4e-08 | 0.045 |
| SVRG | random reshuffling | 10 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 1.7e-07 | 0.045 |
| SVRG | random reshuffling | 298.4 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 9.8e-08 | 0.045 |
| SVRG | uniform w/ replacement | 0.5 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 2.0e-09 | 0.045 |
| SVRG | uniform w/ replacement | 1 | 5 | 5/5 | 2.2e-16 | 3.3e-16 | 2.4e-08 | 0.045 |
| SVRG | uniform w/ replacement | 298.4 | 5 | 5/5 | 2.2e-16 | 2.2e-16 | 3.9e-08 | 0.045 |
