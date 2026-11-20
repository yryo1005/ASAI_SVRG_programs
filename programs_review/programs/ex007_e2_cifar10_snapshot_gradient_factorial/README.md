# ex007：E2（スナップショットの選択方法と全勾配の推定方法の要因分離，CIFAR-10）

`実験計画.md` 4章（E2）の実験である．CIFAR-10・AlexNet（LayerNorm）で，スナップショット $ {\mathbf z}_{s} $ の選択方法（最終点，ランダム点，平均点）と，
スナップショット勾配 $ {\mathbf g}_{s} $ の推定方法（全勾配 $ \nabla f({\mathbf z}_{s}) $，平均勾配）の2×3の要因配置で6手法を比べる．
内側ループの開始点は論文のAlgorithm 通り `inner_start="snapshot"`（$ {\mathbf w}_{s}^{0} \leftarrow {\mathbf z}_{s} $）である．

| | 最終点 $ {\mathbf w}_{s-1}^{K} $ | ランダム点 | 平均点 $ \frac{1}{K}\sum_{k}{\mathbf w}_{s-1}^{k} $ |
|---|---|---|---|
| $ {\mathbf g}_{s} $：全勾配 | `SVRG_final`（`SVRGFinalPoint`） | `SVRG`（`SVRG`） | `SVRG_average`（`SVRGAverage`，新規） |
| $ {\mathbf g}_{s} $：平均勾配 | `NFG_SVRG_final`（`NFGSVRGFinalPoint`） | `NFG_SVRG`（`NFGSVRG`） | `ASAI_SVRG`（`ASAISVRG`） |

## 1. ファイル

| ファイル | 内容 |
|---|---|
| `data.py`，`model.py` | `ex002_e0_cifar10_alexnet_layernorm/` のコピー（変更なし）．キャッシュは `programs_review/datasets/cifar10/cache/` を共有する |
| `svrg_average.py` | `SVRGAverage`．スナップショットは `ASAISVRG` と同じ平均点（式(13)の逐次更新），$ {\mathbf g}_{s} $ は `SVRG` と同じ全勾配（`set_snapshot_gradient`）．平均勾配は持たない |
| `e2_training.py` | 学習ループ．`common/vr_image_training.py` の `iteration`，`train_epoch`，`_initial_snapshot_gradient`，`_end_outer_loop`，`run_variance_reduced`，`run_single_experiment` のコピーに，(1) 全勾配を用いる手法の集合 `FULL_GRADIENT_METHODS = {"SVRG_final", "SVRG", "SVRG_average"}` による分岐，(2) $ \omega $ の記録（`OmegaAccumulator`）を加えたもの．他の関数は `common/vr_image_training.py` から import する |
| `e2_runner.py` | タスクの構築と並列実行．`common/e0_runner.py` の `build_tasks`，`hp_name`，`_init_cuda` を使い，`_Worker`・`run_main` を E2 の学習ループを呼ぶようにコピーしたもの．`--no-omega` を追加 |
| `train.py` | 実行スクリプト（6手法 × 5シード = 30試行） |
| `check_consistency.py` | 本実行の前の一致の確認（E0 のログとの比較，$ \omega $ の記録の有無の比較）の結果を Markdown にまとめる |

集計は `programs_review/programs/analysis_e2_factorial.py`，可視化は `programs_review/visualize_result_E2_factorial.ipynb`，テストは `programs_review/tests/test_e2_factorial_programs.py` である．

## 2. $ \omega $ の記録

内側ループの各ステップで，`optimizer.step` の直前の（更新前の）パラメータ $ {\mathbf w}_{s-1}^{k} $ を
$ {\mathbf S}_{1} = \sum_{k}{\mathbf w}_{s-1}^{k} $（float64 のベクトル）と $ S_{2} = \sum_{k}\|{\mathbf w}_{s-1}^{k}\|^{2} $（float64 のスカラー）に加算する．
加算する点は `ASAISVRG` の `running_avg_params` に加える点と同じである．外側ループの終了時に

$$ \omega({\mathbf z}) = \sum_{k=0}^{K-1}\|{\mathbf w}_{s-1}^{k} - {\mathbf z}\|^{2} = S_{2} - 2\langle {\mathbf z}, {\mathbf S}_{1}\rangle + K\|{\mathbf z}\|^{2} $$

を float64 で計算し，`omega_snapshot`（$ {\mathbf z} = {\mathbf z}_{s} $，各手法が実際に選んだスナップショット）と `omega_average`
（$ {\mathbf z} = {\mathbf S}_{1}/K $，$ \omega $ の最小値）を記録する．0エポック目は NaN である．
加算はパラメータを読み出して float64 の新しいテンソルを作るだけであり，学習の計算を変えない（`tests/test_e2_factorial_programs.py` と
`outputs/ex007_.../consistency_omega.md` で確認）．

平均勾配の3手法の1エポック目は，$ {\mathbf g}_{0} = {\mathbf 0} $ かつ $ {\mathbf w}_{0}^{0} = {\mathbf z}_{0} $ より補正勾配
$ \nabla f_{n}({\mathbf w}) - \nabla f_{n}({\mathbf z}_{0}) + {\mathbf g}_{0} $ が常に $ {\mathbf 0} $ になり，内側ループが動かない．このため $ \omega = 0 $（記録される値は丸め誤差の大きさ）となる．
これは E0 の `NFG_SVRG`，`ASAI_SVRG` と同じ挙動である．

## 3. 実行方法

```bash
cd programs_review/programs
# 一致の確認（3.1：E0 のログとの比較）
python ex007_e2_cifar10_snapshot_gradient_factorial/train.py --methods SVRG NFG_SVRG ASAI_SVRG --seeds 0 --epochs 3 \
    --workers 1 --output-root ../outputs/_verify/ex007
python compare_with_existing_logs.py --new-root ../outputs/_verify/ex007 \
    --existing-root ../outputs/ex002_e0_cifar10_alexnet_layernorm \
    --hp-map inner_snapshot_lr0.001_bs128_lambda0.0005_epochs3=inner_snapshot_lr0.001_bs128_lambda0.0005_epochs48
python ex007_e2_cifar10_snapshot_gradient_factorial/check_consistency.py e0 --new-root ../outputs/_verify/ex007 \
    --existing-root ../outputs/ex002_e0_cifar10_alexnet_layernorm \
    --hp-map inner_snapshot_lr0.001_bs128_lambda0.0005_epochs3=inner_snapshot_lr0.001_bs128_lambda0.0005_epochs48 \
    --output ../outputs/ex007_e2_cifar10_snapshot_gradient_factorial/consistency_with_e0.md
# 一致の確認（3.2：ω の記録の有無）
python ex007_e2_cifar10_snapshot_gradient_factorial/train.py --methods ASAI_SVRG --seeds 0 --epochs 2 --workers 1 \
    --output-root ../outputs/_verify/ex007_omega_on
python ex007_e2_cifar10_snapshot_gradient_factorial/train.py --methods ASAI_SVRG --seeds 0 --epochs 2 --workers 1 \
    --no-omega --output-root ../outputs/_verify/ex007_omega_off
python ex007_e2_cifar10_snapshot_gradient_factorial/check_consistency.py omega --on-root ../outputs/_verify/ex007_omega_on \
    --off-root ../outputs/_verify/ex007_omega_off \
    --output ../outputs/ex007_e2_cifar10_snapshot_gradient_factorial/consistency_omega.md
# 本実行（30試行）
python ex007_e2_cifar10_snapshot_gradient_factorial/train.py --workers <並列数> > ../outputs/_run_logs/ex007_factorial.log 2>&1
# 集計と図
python analysis_e2_factorial.py
```

## 4. 出力

`programs_review/outputs/ex007_e2_cifar10_snapshot_gradient_factorial/{手法}/inner_snapshot_lr0.001_bs128_lambda0.0005_epochs48/{seed}/`
に `config.json`，`log.json`，`best_model.pth` を保存する．

- `log.json`：E0 の項目（`epoch`，`oracle_calls`，`elapsed_time`，`train_loss`，`test_accuracy`，`approx_error`，`test_correct`，
  `full_grad_norm_sq`，`approx_cos`，`terminated`）に `omega_snapshot`，`omega_average` を加えたもの．全勾配の手法の `approx_error` は 0.0．
- `config.json`：E0 の項目に，`snapshot_gradient`（`full_gradient` または `averaged_gradient`）と `record_omega` を加えたもの．
- `figures/`：`analysis_e2_factorial.py` の図．`consistency_with_e0.md`，`consistency_omega.md`：一致の確認の結果．
- `programs_review/outputs/e2_factorial_summary.md`：集計表．

勾配の評価回数 `oracle_calls` は，全勾配の3手法で初期の全勾配の $ N $ と1エポックあたり $ 3N $，平均勾配の3手法で1エポックあたり $ 2N $ である．
平均勾配の3手法の全勾配の計算は近似誤差の診断のためだけのものであり，`oracle_calls` に含めない．

## 5. `programs_review/document.md` への転記内容

統合時に `document.md` に次を追記する．

1. 冒頭の「現在実装している実験」に次を加える．
   - E2（スナップショットの選択方法と全勾配の推定方法の要因分離，CIFAR-10，`snapshot`，2×3の要因配置の6手法）．
2. 1節のディレクトリ構成に次を加える．

   ```text
   │   ├── ex007_e2_cifar10_snapshot_gradient_factorial/  # E2 CIFAR-10（data.py，model.py，svrg_average.py，e2_training.py，e2_runner.py，train.py，check_consistency.py，README.md）
   │   └── analysis_e2_factorial.py                   # E2の図と表
   ├── tests/test_e2_factorial_programs.py            # pytest（E2）
   ├── visualize_result_E2_factorial.ipynb            # E2の可視化（analysis_e2_factorial.py を呼ぶ）
   │   ├── ex007_e2_cifar10_snapshot_gradient_factorial/{手法}/{条件}/{seed}/...  # E2（figures/，consistency_*.md を含む）
   │   ├── e2_factorial_summary.md                    # 集計表（E2）
   ```

3. 2節の依存関係の図に次を加える．

   ```mermaid
   graph TD
       T7[ex007 train.py] --> R7[ex007 e2_runner.py]
       R7 --> R[common/e0_runner.py]
       R7 --> V7[ex007 e2_training.py]
       V7 --> V[common/vr_image_training.py]
       V7 --> O[common/optimizers/]
       V7 --> SA[ex007 svrg_average.py]
       T7 --> D7[ex007 data.py] --> C[common/cached_loader.py]
       T7 --> M7[ex007 model.py]
       CC[ex007 check_consistency.py] --> CMP[compare_with_existing_logs.py]
       E2A[analysis_e2_factorial.py] --> A[analysis_e0.py]
       E2A --> AS[analysis_e0_with_sgd.py]
       NB2[visualize_result_E2_factorial.ipynb] --> E2A
   ```

4. 6節の実行方法に，本 README の3節のコマンドを加える．
5. 7節のログの項目に，「E2 では `omega_snapshot`（$ \omega({\mathbf z}_{s}) $）と `omega_average`（$ \omega $ の最小値）を加えた．
   `config.json` には `snapshot_gradient` と `record_omega` を加えた」を追記する．
6. 8節（Git管理上の注意）は変更不要（`outputs/_verify/` と `*.pth` は既に管理外）．
