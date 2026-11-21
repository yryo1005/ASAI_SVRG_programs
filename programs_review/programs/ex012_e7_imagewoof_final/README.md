# ex012_e7_imagewoof_final：最終点のスナップショットの SVRG・NFG-SVRG（E7 の A，ImageWoof，バッチサイズ32・128・512）

## 1. 目的

論文の実験3（ImageWoof）と表IV・V の比較手法である SVRG・NFG-SVRG を，SVRG 論文・NFG-SVRG 論文の実験と同じ最終点のスナップショットにそろえる．
既存の ImageWoof の SVRG・NFG-SVRG は全てランダム点であるため，最終点の2手法を本実験で実行する．

## 2. 手法

- SVRG（最終点）（`SVRG_final`，`SVRGFinalPoint`）：スナップショット $ {\mathbf z}_{s} $ を内側ループの最終点 $ {\mathbf w}_{s-1}^{K} $ とし，$ {\mathbf g}_{s} $ を $ {\mathbf z}_{s} $ での全勾配とする．
- NFG-SVRG（最終点）（`NFG_SVRG_final`，`NFGSVRGFinalPoint`）：スナップショットは最終点とし，$ {\mathbf g}_{s} $ を直前の内側ループの確率的勾配の平均とする．
- 最終点では，内側ループの開始点を `snapshot` にしても `last_iterate` にしても同じ計算になる．本実験では `inner_start="snapshot"` とした．
- 2手法は `common/vr_image_training.py` に登録済みであり，新しいオプティマイザや学習ループは実装していない．

## 3. 条件

| 項目 | 設定 |
|---|---|
| データ・モデル | ImageWoof，ResNet18（LayerNorm）（`ex003` と同じ） |
| 目的関数 | 交差エントロピー + 重みの L2 正則化（$ \lambda = 5\times10^{-4} $） |
| 学習率 | $ \eta = 10^{-3} $ |
| シード | 0〜4 |
| 評価点 | スナップショット $ {\mathbf z}_{s} $ |
| 打ち切り | 学習損失が非有限になった外側ループで打ち切る |

エポック数は，論文の予算 $ \#\mathrm{grad}/N $ に届く最小限とした（SVRG は初期 $ N $・1エポック $ 3N $，NFG-SVRG は1エポック $ 2N $）．

| バッチサイズ | 予算 | `SVRG_final` | `NFG_SVRG_final` |
|---:|---:|---:|---:|
| 32 | 30 | 11（34） | 16（32） |
| 128 | 200 | 67（202） | 100（200） |
| 512 | 1000 | 334（1003） | 500（1000） |

保存先：`outputs/ex012_e7_imagewoof_final/{手法}/inner_snapshot_lr0.001_bs{B}_lambda0.0005_epochs{E}/{seed}/{config.json,log.json,best_model.pth}`．

## 4. ファイルとコピー元

| ファイル | 内容 |
|---|---|
| `data.py`，`model.py` | `ex003_e0_imagewoof_resnet18_layernorm/` から変更せずにコピーした（テストで同一であることを確認する） |
| `train.py` | 新規．`EPOCHS`（手法とバッチサイズごとのエポック数）でタスクを作り，`common/e0_runner.build_tasks` で連結する．並列実行は `common/e0_runner.run_main` と同じ処理であり，タスクはバッチサイズの大きい順（1試行の時間の長い順）に実行する |

## 5. 実行方法（`programs_review/programs` から）

```bash
# 一致の確認（ex003・ex005 の SVRG と比べる）
python ex012_e7_imagewoof_final/train.py --methods SVRG --seeds 0 --epochs 2 --batch-sizes 32 128 512 \
    --workers 1 --output-root ../outputs/_verify/ex012
# 本実行（30試行）
python ex012_e7_imagewoof_final/train.py --workers <並列数>
# 集計
python analysis_e7.py
```

- `--methods` の選択肢は `SVRG_final`，`NFG_SVRG_final`，`SVRG` である．`SVRG` は一致の確認用であり，`--epochs` の指定が必要である．
- 一致の確認の結果は `outputs/ex012_e7_imagewoof_final/consistency.md` にある．
