# ex010_e5_sgd_epoch_average：エポック内の平均点で評価した SGD（E5，実験計画8章）

## 1. 目的

論文4.4節の表IVでは，ASA-SVRG は内側ループの平均点 $ {\mathbf z}_{s} $ で評価され，SGD は最終点で評価される．
そのため，表IVの揺らぎの差が，手法の違いによるのか，平均点で評価したことによるのかを区別できない．
E5 では SGD を同じ1回の学習の中で最終点と平均点の両方で評価し，評価点の違いの影響を確かめる．

## 2. 条件

| 項目 | 設定 |
|---|---|
| 手法 | SGD のみ．$ \eta = 10^{-3} $，$ \lambda = 5\times10^{-4} $，random reshuffling |
| 評価点 | 最終点 $ {\mathbf w}^{K} $ と，エポック内の反復の平均点 $ \bar{\mathbf w} = \frac{1}{K}\sum_{k=0}^{K-1}{\mathbf w}^{k} $（更新前のパラメータの平均．ASA-SVRG の `running_avg_params` と同じ添字と式） |
| 設定（優先度の順） | ImageWoof・128・200エポック，CIFAR-10・128・90エポック，CIFAR-10・512・350エポック，ImageWoof・512・1000エポック |
| シード | 0〜4 |
| データ・モデル | ImageWoof：`ex009_e3_imagewoof_walltime`（ResNet18，LayerNorm），CIFAR-10：`ex002_e0_cifar10_alexnet_layernorm`（AlexNet，LayerNorm）．モジュールは複製せずに読み込む（`problems.py`） |

## 3. ファイル

| ファイル | 役割 |
|---|---|
| `problems.py` | 2つの既存の実験の `data.py`・`model.py` を別の名前で読み込み，問題の定義を返す |
| `e5_training.py` | 平均点でも評価する SGD の学習ループ（`EpochAverage`，`run_sgd_with_average`） |
| `train.py` | タスクの構築（優先度の順）と並列実行 |
| `compare_e5.py` | 最終点の記録と既存の SGD のログの比較（実験計画8.4節） |
| `import_reference_logs.py` | 表IVの元の既存ログ（ex0024，ex0051 の SGD と ASA-SVRG）を `outputs/ex010_e5_sgd_epoch_average/reference/` に複写する |

集計は `programs/analysis_e5.py`，可視化は `visualize_result_E5.ipynb`，テストは `tests/test_e5_programs.py` である．

## 4. 学習の軌跡を変えないための規則

- SGD の更新の演算は E3 の SGD（`timed_training.sgd_train_epoch`）と同一である．平均への加算は，逆伝播の後，更新の前に行い，パラメータと勾配を変えない．
- 最終点の評価は既存の SGD と同じく，シャッフルする学習用のローダーで行う．既存の実装ではこの評価がシャッフルの乱数を消費するため，同じ順序で消費する．
- 平均点の評価は，パラメータを複製した別のモデルと，シャッフルしない学習データのローダーで行う．評価の前後で CPU と CUDA の乱数の状態を保存・復元する．
- 評価用のモデルは `copy.deepcopy` で作り，モデルの初期化の乱数を消費しない．

## 5. 一致の確認（実験計画8.4節）

シード0の最初の5エポック（6評価点）の最終点の記録を比べた（`outputs/ex010_e5_sgd_epoch_average/consistency_check.md`）．

| 設定 | 比較対象 | 結果 |
|---|---|---|
| ImageWoof・128 | E3 の SGD | `oracle_calls`，`train_loss`，`test_accuracy`，`test_correct` が完全一致 |
| CIFAR-10・128 | 既存ログ ex0024 の SGD（`epochs96`） | `oracle_calls`，`train_loss`，`test_accuracy` が完全一致 |

本実行の全試行についても，同じ比較を `compare_e5.py` で行う（`consistency_full_runs.md`）．ImageWoof・512 の比較対象（ex0051）は前処理の実装が異なり，完全には一致しない（参考）．

## 6. 資源の目安（並列数1）

| 設定 | 1エポックの学習の時間 | 評価を含む1エポックの時間 | GPU メモリ |
|---|---:|---:|---:|
| ImageWoof・128 | 約6.3 s | 約20 s | 約6.4 GB |
| CIFAR-10・128 | 約4.3 s | 約8 s | 約2 GB |
| ImageWoof・512 | 約4.1 s | 約15 s（推定） | 約22 GB |
| CIFAR-10・512 | 約2.1 s | — | — |

ImageWoof のデータは `numpy.memmap` で読み出すため，複数のプロセスで OS のページキャッシュを共有する．
ImageWoof・512 は GPU メモリのため並列数を4とした．
