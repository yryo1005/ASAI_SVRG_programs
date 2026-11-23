# E6 の一致の確認

本実行の前に，`ex011_e6_cifar10_final_bs32_512` のプログラムが既存の実験と同じ計算をすることを確かめた．
実行した PC の GPU は NVIDIA GeForce RTX 4090（24GB），torch 2.11.0+cu128，cuDNN 9.19.0 である．
比較には `programs/compare_with_existing_logs.py` を用いた（結果の全文は `consistency_bs128.md`，`consistency_bs32_512.md`）．
実行ログは `outputs/_run_logs/ex011_verify_bs128.log`，`ex011_verify_bs32_512.log` である．

## 1. 判定のまとめ

| 確認 | 比較先（GPU） | `oracle_calls` | 0エポック目 | 1エポック目以降 | 判定 |
|---|---|---|---|---|---|
| 3.1節：`SVRG_final`，`NFG_SVRG_final`，バッチサイズ128，3エポック | `ex007`（RTX 3090） | 完全一致 | `train_loss` の相対差 $ 5.9\times10^{-9} $，`test_accuracy` は一致 | `train_loss` の相対差は最大 $ 1.1\times10^{-4} $，`test_accuracy` の絶対差は最大 0.0016 | 合格（GPU の違いによる差） |
| 3.2節：`SVRG`，バッチサイズ32・512，2エポック | `ex004`（RTX PRO 6000 Blackwell Max-Q） | 完全一致 | `train_loss` の相対差は最大 $ 4.0\times10^{-8} $，`test_accuracy` は一致 | `train_loss` の相対差は最大 $ 1.5\times10^{-3} $（バッチサイズ32），`test_accuracy` の絶対差は最大 0.0035 | 合格（GPU の違いによる差） |

- `oracle_calls` は全ての系列で完全に一致した．そのため，勾配の計算回数の数え方とバッチサイズの扱いは既存と同じである．
- 0エポック目の差は相対 $ 10^{-5} $ を大きく下回り，丸め誤差の範囲である．
- 1エポック目以降の差は，比較先と GPU が異なる（本PC は RTX 4090）ことによる丸め誤差が，非凸の学習で拡大したものと考えられる．
  差はステップ数の多いバッチサイズ32（1エポックあたり $ K = 1563 $ ステップ）で最も大きく，ステップ数の少ないバッチサイズ512（$ K = 98 $）では相対 $ 5\times10^{-6} $ 以下である．この傾向は丸め誤差の蓄積と整合する（推測）．
- `NFG_SVRG_final` の `approx_error` の相対差（最大 0.14，3エポック目）は，近似誤差が2つの勾配の差の大きさであり，パラメータの小さな差に敏感な量であることによると考えられる．0エポック目の相対差は $ 3.2\times10^{-5} $ である．
- 以上より，3.3節の基準を満たすと判断し，本実行に進んだ．

## 2. エポックごとの値

### 3.1節（比較先：ex007）

#### SVRG_final inner_snapshot_lr0.001_bs128_lambda0.0005_epochs3  new_gpu=NVIDIA GeForce RTX 4090 old_gpu=NVIDIA GeForce RTX 3090
| epoch | train_loss(new) | train_loss(old) | rel diff | test_acc(new) | test_acc(old) | abs diff | approx_error rel diff |
| 0 | 2.52604303 | 2.52604305 | 5.89e-09 | 0.1042 | 0.1042 | 0.0000 | 0.00e+00 |
| 1 | 2.31432696 | 2.31423016 | 4.18e-05 | 0.2929 | 0.2931 | 0.0002 | 0.00e+00 |
| 2 | 2.11438955 | 2.11429635 | 4.41e-05 | 0.3644 | 0.3633 | 0.0011 | 0.00e+00 |
| 3 | 1.92903764 | 1.92925389 | 1.12e-04 | 0.4118 | 0.4134 | 0.0016 | 0.00e+00 |
elapsed_time(new, last)=29.3s
#### NFG_SVRG_final inner_snapshot_lr0.001_bs128_lambda0.0005_epochs3  new_gpu=NVIDIA GeForce RTX 4090 old_gpu=NVIDIA GeForce RTX 3090
| epoch | train_loss(new) | train_loss(old) | rel diff | test_acc(new) | test_acc(old) | abs diff | approx_error rel diff |
| 0 | 2.52604303 | 2.52604305 | 5.89e-09 | 0.1042 | 0.1042 | 0.0000 | 3.19e-05 |
| 1 | 2.52604304 | 2.52604305 | 2.78e-09 | 0.1042 | 0.1042 | 0.0000 | 3.16e-03 |
| 2 | 2.31408550 | 2.31390776 | 7.68e-05 | 0.3003 | 0.3005 | 0.0002 | 1.83e-02 |
| 3 | 2.19578284 | 2.19563092 | 6.92e-05 | 0.3339 | 0.3329 | 0.0010 | 1.38e-01 |
elapsed_time(new, last)=21.1s

### 3.2節（比較先：ex004）

#### SVRG inner_snapshot_lr0.001_bs32_lambda0.0005_epochs2  new_gpu=NVIDIA GeForce RTX 4090 old_gpu=NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
| epoch | train_loss(new) | train_loss(old) | rel diff | test_acc(new) | test_acc(old) | abs diff | approx_error rel diff |
| 0 | 2.52604307 | 2.52604314 | 2.86e-08 | 0.1042 | 0.1042 | 0.0000 | 0.00e+00 |
| 1 | 1.91442601 | 1.91396802 | 2.39e-04 | 0.4070 | 0.4076 | 0.0006 | 0.00e+00 |
| 2 | 1.65403883 | 1.65658209 | 1.54e-03 | 0.4887 | 0.4852 | 0.0035 | 0.00e+00 |
elapsed_time(new, last)=48.6s
#### SVRG inner_snapshot_lr0.001_bs512_lambda0.0005_epochs2  new_gpu=NVIDIA GeForce RTX 4090 old_gpu=NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
| epoch | train_loss(new) | train_loss(old) | rel diff | test_acc(new) | test_acc(old) | abs diff | approx_error rel diff |
| 0 | 2.52604305 | 2.52604315 | 3.96e-08 | 0.1042 | 0.1042 | 0.0000 | 0.00e+00 |
| 1 | 2.49477979 | 2.49478383 | 1.62e-06 | 0.2315 | 0.2318 | 0.0003 | 0.00e+00 |
| 2 | 2.46954599 | 2.46955824 | 4.96e-06 | 0.2624 | 0.2629 | 0.0005 | 0.00e+00 |
elapsed_time(new, last)=17.5s
