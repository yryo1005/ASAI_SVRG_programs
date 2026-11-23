# E6 の統合時の確認（本PCでの再実行）

実施日：2026-11-23．本PC：NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition（ドライバ 582.08），Intel Xeon w7-3565X，WSL2，Python 3.12.15，torch 2.11.0+cu128（CUDA 12.8，cuDNN 9.19.0）．
E6 の本実行は別のPC（NVIDIA GeForce RTX 4090，`.reports/report_080.md` 2節）で行われた．

```bash
# programs_review/programs で実行（ログ：outputs/_run_logs/ex011_verify_this_pc.log，結果：outputs/_verify/ex011_this_pc/．Git 管理外）
python ex011_e6_cifar10_final_bs32_512/train.py --methods SVRG_final NFG_SVRG_final --batch-sizes 32 512 \
    --inner-starts snapshot --seeds 0 --epochs 3 --workers 1 --output-root ../outputs/_verify/ex011_this_pc
```

## 1. zip のログ（RTX 4090）との比較（シード0，0〜3エポック）

| 手法 | バッチサイズ | `oracle_calls` | 0エポック目の `train_loss` の相対差 | 1〜3エポック目の `train_loss` の最大相対差 | `test_accuracy` の最大絶対差 |
|---|---:|---|---:|---:|---:|
| SVRG（最終点） | 32 | 完全一致 | $ 2.9\times10^{-8} $ | $ 1.4\times10^{-2} $ | 0.0077 |
| SVRG（最終点） | 512 | 完全一致 | $ 4.0\times10^{-8} $ | $ 3.0\times10^{-6} $ | 0.0001 |
| NFG-SVRG（最終点） | 32 | 完全一致 | $ 2.9\times10^{-8} $ | $ 2.3\times10^{-2} $ | 0.0525 |
| NFG-SVRG（最終点） | 512 | 完全一致 | $ 4.0\times10^{-8} $ | $ 7.1\times10^{-6} $ | 0.0002 |

0エポック目の `test_accuracy` は全て一致した．差は1エポックのステップ数が多いバッチサイズ32で大きく，512で小さい．`report_080.md` 4節の一致の確認（RTX 4090 と本PCの `ex004` の比較）と同じ大きさである．
NFG-SVRG（最終点）のバッチサイズ32は，本PCでも3エポック目に学習損失が上がり（1.8545 → 2.3346．zip は 1.8540 → 2.3891），テスト精度が下がった（0.4223 → 0.2798．zip は 0.4233 → 0.2273）．崩壊は GPU に依らず起きる．

## 2. 本PCの付録B（`ex004_appb_cifar10_alexnet_layernorm`，SVRG，シード0）との比較

本PCの再実行の SVRG（最終点）の0エポック目の `train_loss`・`test_accuracy`・`oracle_calls` は，`ex004` の SVRG（ランダム点）とバッチサイズ32・512の両方でビット単位で一致した（1エポック目以降はスナップショットの選び方が異なるため比べない）．
したがって，zip のログとの差は GPU の違いによる丸め誤差であり，プログラムの差ではない．

## 3. 集計の再現

本PCで `analysis_e6.py --output {作業用}/e6_summary.md --figure-dir {作業用}/figures` を実行した．表は zip 内の `outputs/e6_summary.md` と同一であった（違いは図の一覧の出力先のパスのみ）．図6枚はバイト単位で同一であった．
