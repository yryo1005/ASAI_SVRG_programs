# ex011_e6_cifar10_final_bs32_512：最終点のスナップショットの SVRG・NFG-SVRG（E6，CIFAR-10，バッチサイズ32・512）

## 1. 目的

論文の付録に，本文の CIFAR-10 の図（バッチサイズ128）と同じ構成の図を，バッチサイズ32・512について載せる．
図の5手法のうち，SVRG（最終点）と NFG-SVRG（最終点）にはバッチサイズ32・512の結果がないため，本実験で実行する．

| 図の手法 | バッチサイズ32・512のデータ |
|---|---|
| ASA-SVRG（平均点，continue） | 既存の実験 `ex0024_cifar10_alexnet_layernorm_longrun`（論文のリポジトリの環境） |
| ASA-SVRG（平均点，reset） | `outputs/ex004_appb_cifar10_alexnet_layernorm` |
| SGD | 既存の実験 `ex0024_cifar10_alexnet_layernorm_longrun` |
| SVRG（最終点） | 本実験 |
| NFG-SVRG（最終点） | 本実験 |

## 2. 手法

- SVRG（最終点）（`SVRG_final`，`SVRGFinalPoint`）：スナップショット $ {\mathbf z}_{s} $ を内側ループの最終点 $ {\mathbf w}_{s-1}^{K} $ とし，$ {\mathbf g}_{s} $ を $ {\mathbf z}_{s} $ での全勾配とする（SVRG 論文の Option I）．
- NFG-SVRG（最終点）（`NFG_SVRG_final`，`NFGSVRGFinalPoint`）：スナップショットは最終点とし，$ {\mathbf g}_{s} $ を直前の内側ループの確率的勾配の平均とする．
- 最終点では $ {\mathbf z}_{s} = {\mathbf w}_{s-1}^{K} $ であるため，内側ループの開始点を `snapshot` にしても `last_iterate` にしても同じ計算になる．E2 と同じく `inner_start="snapshot"` とした．

## 3. 条件

| 項目 | 設定 |
|---|---|
| データ・モデル | CIFAR-10，AlexNet（LayerNorm）．E2 と同じ |
| 目的関数 | 交差エントロピー + 重みの L2 正則化（$ \lambda = 5\times10^{-4} $） |
| 学習率 | $ \eta = 10^{-3} $ |
| バッチサイズとエポック数 | 32（$ K = 1563 $，12エポック），512（$ K = 98 $，192エポック）．付録B（`ex004`）と同じ |
| シード | 0〜4 |
| $ \omega $ の記録 | 有効（E2 と同じ） |
| 試行数 | 2手法 × 2バッチサイズ × 5シード $ = 20 $ |

- 保存先：`outputs/ex011_e6_cifar10_final_bs32_512/{手法}/inner_snapshot_lr0.001_bs{32,512}_lambda0.0005_epochs{12,192}/{seed}/`．
- `oracle_calls` の増分は，SVRG で初期 $ N $・1エポック $ 3N $，NFG-SVRG で1エポック $ 2N $ である（$ N = 50000 $）．
  - $ \#\mathrm{grad}/N $ の最大値は，バッチサイズ32で SVRG 37，NFG-SVRG 24 である．
  - バッチサイズ512では SVRG 577，NFG-SVRG 384 である．

## 4. ファイルとコピー元

| ファイル | 内容 |
|---|---|
| `data.py`，`model.py`，`e2_training.py`，`svrg_average.py` | `ex007_e2_cifar10_snapshot_gradient_factorial/` から変更せずにコピーした（テストで同一であることを確認する） |
| `e2_runner.py` | ex007 のコピー．変更は2点である．(1) `--batch-sizes` を加えて `common/e0_runner.build_tasks` に渡す．(2) `--methods` の選択肢を `e2_training.METHODS`（6手法）とする |
| `train.py` | ex007 のコピー．`SPEC`（手法，`epochs_by_batch_size`，保存先）のみを変えた |

- `--batch-sizes` の値は `epochs_by_batch_size` のキーに制限しない．`--epochs` を指定すると，エポック数の表を参照しない．
- そのため，`--batch-sizes 128 --epochs 3` で E2 との一致を確認できる．
- 集計は `programs/analysis_e6.py`（出力 `outputs/e6_summary.md`，図 `outputs/ex011_e6_cifar10_final_bs32_512/figures/`），テストは `tests/test_e6_programs.py` である．

## 5. 実行方法（`programs_review/programs` から）

```bash
# 一致の確認（E2，バッチサイズ128）
python ex011_e6_cifar10_final_bs32_512/train.py --methods SVRG_final NFG_SVRG_final --seeds 0 --epochs 3 \
    --batch-sizes 128 --workers 1 --output-root ../outputs/_verify/ex011_bs128
# 一致の確認（付録B，バッチサイズ32・512）
python ex011_e6_cifar10_final_bs32_512/train.py --methods SVRG --seeds 0 --epochs 2 --batch-sizes 32 512 \
    --workers 1 --output-root ../outputs/_verify/ex011_bs32_512
# 本実行（20試行）
python ex011_e6_cifar10_final_bs32_512/train.py --workers <並列数>
# 集計
python analysis_e6.py
```

## 6. 作成した PC での事前確認

プログラムを作成した PC（NVIDIA RTX PRO 6000 Blackwell Max-Q，付録B `ex004` と同じ GPU）で，5節の2つの一致の確認を行った．
結果は `outputs/_verify/`（Git 管理外）に保存した．実行ログは `outputs/_run_logs/verify_ex011_bs32_512.log`，`verify_ex011_bs128.log` である．

| 確認 | 比較先（GPU） | `oracle_calls` | 0エポック目 | 1エポック目以降 |
|---|---|---|---|---|
| SVRG，バッチサイズ32・512，2エポック | `ex004`（同じ GPU） | 完全一致 | 完全一致 | `train_loss`，`test_accuracy`，`approx_error` が完全一致 |
| `SVRG_final`，`NFG_SVRG_final`，バッチサイズ128，3エポック | `ex007`（RTX 3090） | 完全一致 | `train_loss` の相対差 $ 3.8\times10^{-8} $ | `train_loss` の相対差は最大 $ 2.3\times10^{-4} $（GPU の違いによる差） |
