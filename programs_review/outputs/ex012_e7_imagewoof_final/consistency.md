# ex012_e7_imagewoof_final の一致の確認

実施日：2026-11-22．ログ：`outputs/_run_logs/ex012_verify.log`．結果：`outputs/_verify/ex012/`（Git 管理外）．

## 1. 既存の SVRG（ランダム点）との一致

本実験の `train.py` で `--methods SVRG --seeds 0 --epochs 2 --batch-sizes 32 128 512 --workers 1` を実行し，既存の結果の先頭の3点（0〜2エポック）と比べた（`compare_with_existing_logs.py`）．

| バッチサイズ | 既存の結果 | oracle_calls | train_loss | test_accuracy | approx_error |
|---:|---|---|---|---|---|
| 32 | `ex005_appb_imagewoof_resnet18_layernorm/SVRG/inner_snapshot_lr0.001_bs32_lambda0.0005_epochs33/0` | 完全一致 | 完全一致 | 完全一致 | 完全一致 |
| 128 | `ex003_e0_imagewoof_resnet18_layernorm/SVRG/inner_snapshot_lr0.001_bs128_lambda0.0005_epochs128/0` | 完全一致 | 完全一致 | 完全一致 | 完全一致 |
| 512 | `ex005_appb_imagewoof_resnet18_layernorm/SVRG/inner_snapshot_lr0.001_bs512_lambda0.0005_epochs505/0` | 完全一致 | 完全一致 | 完全一致 | 完全一致 |

データ・モデル・学習のループ・シードの扱いが既存の実験と同じであることを確認した．

## 2. 最終点の2手法（バッチサイズ128，シード0，2エポック）

| 手法 | `config.json` の `method` | `inner_start` | oracle_calls | train_loss |
|---|---|---|---|---|
| `SVRG_final` | `SVRG_final` | `snapshot` | 0，36100，63175（$ = 0, 4N, 7N $，$ N = 9025 $） | 4.4552，4.1851，4.1418 |
| `NFG_SVRG_final` | `NFG_SVRG_final` | `snapshot` | 0，18050，36100（$ = 0, 2N, 4N $） | 4.4552，4.4552，4.1853 |

- oracle_calls は，SVRG（最終点）が初期 $ N $・1エポック $ 3N $（初期の全勾配の $ N $ は1エポック目の点に含めて記録），NFG-SVRG（最終点）が1エポック $ 2N $ であり，規則に一致した．
- NFG-SVRG（最終点）の1エポック目の学習損失は初期値とほぼ等しい（4.455171792 → 4.455171842）．これは $ {\mathbf g}_{0} = {\mathbf 0} $ による．1エポック目までの値は既存の NFG-SVRG（ランダム点，`ex003`）とビット単位で一致し，2エポック目（スナップショットの選び方が効く点）から異なる（4.1853 と 4.1986）．

## 3. 並列数

1プロセスの GPU のメモリ使用量（`memory.used` の差．WSL2 ではプロセスごとの値が表示されない）は，バッチサイズ512で約22.3 GB，128で約6.5 GB，32で約2.5 GB であった（待機時は 0.7 GB，総量 97.9 GB）．DRAM の使用量は全体で 5〜6 GB であった（総量 125 GB）．
タスクはバッチサイズの大きい順に実行するため，バッチサイズ512の4試行を同時に置ける4並列（約90 GB）とした（`report_073` の E5 のバッチサイズ512 と同じ）．
