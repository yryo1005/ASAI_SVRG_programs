CIFAR-10データセットに対するResNet-18モデルの学習において、SVRGおよびASAI SVRGにおける補正勾配の発散・不安定化が「`snapshot_model`におけるBatch Normalization (BN) 統計量（`running_mean` / `running_var`）のドリフト」に起因するかを検証するため、以下の実験を実施し、スクリプト作成から実験実行、結果の保存までを行ってください。

### 1. 実験目的
`snapshot_model` の BN 統計量を固定（`eval()` モード指定）および再較正（Recalibration）した場合に、補正勾配の爆発や収束の不安定化が抑制されるかを検証し、分散低減手法（SVRG, ASAI SVRG）と BN 構造の相互作用を明らかにします。

### 2. ディレクトリ構成・実装・出力指定
* **プログラム配置先**: `programs/ex005_eval_snapshot_bn/`
* **出力結果保存先**: `outputs/ex005_eval_snapshot_bn/{method}/{condition}/{lr}/{seed}/`
  （規約に従い、`ResultLogger` を用いて JSON/CSV 等でメトリクスを保存すること）

### 3. 実験条件とハイパーパラメータ
* **データセット**: CIFAR-10
  * 通常学習時データ増強: Standard Normalize + RandomCrop + RandomHorizontalFlip
  * Recalibration時: Augmentation なし（標準 Normalize のみ、EvalTransform）
* **モデル構造**: `ResNet-18`
* **分散模擬環境**: $M = 5$ ワーカーの分散模擬環境
* **対象手法**:
  1. `SGD`（ベースライン）
  2. `SVRG`
  3. `ASAI SVRG`
* **比較条件（`snapshot_model` の動作制御）**:
  * **`Control`（従来設定）**: `snapshot_model` を `train()` モードで保持し、補正勾配計算時にも BN 統計量が更新される状態。
  * **`Treatment`（提案設定・再較正+固定）**: スナップショット作成時、訓練データから無作為抽出した10バッチ（Augmentationなし、`EvalTransform`）を `torch.no_grad()` かつ `snapshot_model.train()` の状態で順伝播させ、`running_mean` と `running_var` のみを再較正（Recalibration）した上で、`snapshot_model.eval()` に移行して統計量を完全固定する。
* **学習率**: メイン検証 $\gamma = 0.01$、補助検証 $\gamma = 0.001$
* **Seed数**: `seed = 42, 43, 44`（計3試行）
* **Epoch数**: 100 epoch
  * 安全策として Loss > 50 となった場合は発散と判定し、ログを記録して当該試行を早期終了してください。

### 4. 記録・出力メトリクス
各エポックで以下の数値を記録してください。
1. エポック数、累計総勾配計算回数（`grad_evals`）
2. 訓練損失 (Train Loss)、テスト精度 (Test Accuracy)
3. 補正勾配ノルム $\|\nabla f_i(w) - \nabla f_i(z)\|$
4. 補正勾配と真の全勾配とのコサイン類似度
5. スナップショット更新時の BN 統計量（`running_mean`）の L2 変化量

以上の実験を自動実行し、結果を正しく指定フォルダに保存してください。