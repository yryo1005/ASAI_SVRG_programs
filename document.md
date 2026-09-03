# document.md

本ドキュメントは，本リポジトリで作成されたプログラム・データセット・実験結果・文書の役割と
依存関係，実行方法をまとめたものである．

## 1. リポジトリの概要

本リポジトリは，論文『平均スナップショット近似に基づく確率的分散削減勾配法（Averaged Snapshot
Approximate Incremental SVRG，ASAI SVRG）』（`references/ASAI_SVRG_paper.pdf`）に掲載する数値実験を
再現・実施するためのものである．`.orders/order_001.md` の指示に基づき論文4.1節（マッシュルーム
データセットの二値分類問題，平滑かつ強凸な設定）を，`.orders/order_004.md`／`.orders/order_005.md`
の指示に基づき論文4.2節（CNNとCIFAR-10の多値分類問題，非凸な設定）の実験を実装した．

比較手法はSGD，SVRG，NFG SVRG，ASAI SVRG（提案手法）の4手法．最適化手法のクラスは
`programs/optimizers/` に実験に依存しない再利用可能な形で実装しており，Ex001・Ex002の両実験で
共通のクラスをそのまま再利用している．

`.orders/order_006.md` の指示に基づき，NFG SVRGの原論文（Medyakov et al., "Variance Reduction
Methods Do Not Need to Compute Full Gradients: Improved Efficiency through Shuffling"，
`references/No_Full_Grad_SVRG.pdf`）の実験結果とEx002の結果を比較する検証も行った（詳細は
`.reports/report_006.md`）．さらに`.orders/order_007.md` の指示に基づき，原論文の実験
（ResNet-18・CIFAR-10・min-max敵対的ロバスト性の定式化）そのものをEx003として再現する実験を
実施した（詳細は`.reports/report_007.md`）．`.orders/order_008.md` の指示に基づき，Ex003を
ミニバッチサイズ1（オンライン学習）に変更した追加検証も行った．詳細は`.reports/report_008.md`
を参照．

`.orders/order_009.md` の指示に基づき，Ex003の実装上の3つの問題点（M=5ワーカーによる分散環境の
簡略化，フル勾配計算時のBatch Normalization統計量の破損，sigmaの正則化勾配のスケール）を修正した
Ex004を実施した．詳細は`.reports/report_009.md`を参照．

## 2. ディレクトリ構成と各ファイルの役割

```text
./
├── .ai/                                # AI開発規約（サブモジュール）
├── references/
│   └── ASAI_SVRG_paper.pdf             # 参照論文
├── datasets/
│   ├── ex001_mushroom_svrg/
│   │   └── raw/                        # UCI Mushroomデータセットの生データ（自動ダウンロード）
│   ├── ex002_cifar10_cnn/
│   │   └── raw/                        # CIFAR-10データセットの生データ（自動ダウンロード）
│   ├── ex003_cifar10_resnet_minmax/
│   │   └── raw/                        # CIFAR-10データセットの生データ（ex002からコピー）
│   └── ex004_cifar10_resnet_minmax/
│       └── raw/                        # CIFAR-10データセットの生データ（ex003からコピー）
├── programs/
│   ├── optimizers/
│   │   ├── __init__.py
│   │   └── optimizers.py               # SGD, SVRG, SVRGFinalPoint, NFGSVRG, NFGSVRGFinalPoint,
│   │                                    # ASAISVRGの6クラス．いずれもtorch.optim.Optimizerの
│   │                                    # サブクラスとして独立に陽実装（.orders/order_002.md）．
│   │                                    # Ex001〜Ex003で共通
│   ├── ex001_mushroom_svrg/
│   │   ├── data.py                     # マッシュルームデータセットの取得・前処理・DataLoader構築
│   │   ├── model.py                    # ロジスティック回帰モデル．勾配は自動微分で計算
│   │   └── train.py                    # 4手法 x 5Seedの学習を実行するスクリプト（本体）
│   ├── ex002_cifar10_cnn/
│   │   ├── data.py                     # CIFAR-10データセットの取得・前処理・DataLoader構築
│   │   ├── model.py                    # CNNモデル（3畳み込み層+全結合層）
│   │   └── train.py                    # 4手法 x 5Seedの学習を実行するスクリプト（本体）
│   ├── ex003_cifar10_resnet_minmax/
│   │   ├── data.py                     # CIFAR-10データセットの取得・前処理・DataLoader構築
│   │   ├── model.py                    # ResNet-18（CIFAR向け）+ min-max敵対的摂動sigma
│   │   └── train.py                    # 4手法 x 5Seedの学習を実行するスクリプト（本体）
│   └── ex004_cifar10_resnet_minmax/
│       ├── data.py                     # CIFAR-10データセットの取得・前処理・DataLoader構築
│       │                                # （ex003と同一，パスのみex004向けに変更）
│       ├── model.py                    # ResNet-18 + min-max敵対的摂動sigma（ex003と同一）
│       └── train.py                    # Ex003の3点の修正（M_WORKERS分割によるBatch
│                                        # Normalization挙動の模擬，フル勾配計算のeval()化，
│                                        # sigma正則化勾配のスケール整合）を反映した学習
│                                        # スクリプト（order_009，本体）
├── outputs/
│   ├── ex001_mushroom_svrg/
│   │   └── {method}/{hyperparams}/{seed}/
│   │       ├── log.json                # ResultLoggerによる評価指標の履歴
│   │       ├── config.json             # ハイパーパラメータ等のメタデータ
│   │       └── best_model.pth          # 検証精度が最高となったエポックの重み
│   ├── ex002_cifar10_cnn/
│   │   └── {method}/{hyperparams}/{seed}/  # 同上の構成
│   ├── ex003_cifar10_resnet_minmax/
│   │   └── {method}/{hyperparams}/{seed}/  # 同上の構成
│   ├── ex004_cifar10_resnet_minmax/
│   │   └── {method}/{hyperparams}/{seed}/  # 同上の構成
│   └── order_006_archive/              # order_006検証（NFG_SVRG_FinalPoint）の実験結果．
│                                        # order_007によりEx002本体からは切り離して保存
├── tests/
│   ├── test_optimizers.py              # 最適化手法クラスの単体テスト（pytest）
│   ├── test_model.py                   # Ex001のモデル・勾配計算関数の単体テスト（pytest）
│   ├── test_cnn_model.py               # Ex002のCNNモデルの単体テスト（pytest）
│   ├── test_snapshot_inference.py      # SVRG系手法の推論がスナップショットを用いること，
│   │                                    # ASAI SVRGでは平均パラメータであることの統合テスト
│   │                                    # （order_005，pytest）
│   ├── test_minmax_resnet.py           # Ex003のResNet-18・min-max定式化（符号反転の正しさ）
│   │                                    # の単体テスト（order_007，pytest）
│   └── test_minmax_resnet_distributed.py  # Ex004のM_WORKERS分割による勾配集約，フル勾配計算の
│                                        # Batch Normalization統計量固定の単体テスト
│                                        # （order_009，pytest）
├── visualize_result.ipynb              # 実験結果の可視化ノートブック（ルート直下，Ex001〜Ex004
│                                        # 共通．横軸「勾配計算回数」は，各条件のconfig.jsonに
│                                        # 記録されたN_trainで除算し，#grad/N（SGDの1エポック
│                                        # 相当の計算量）として表示する，order_005での修正を反映）
├── .reports/
│   ├── report_001.md                   # Ex001の実験結果レポート
│   ├── report_002.md                   # 最適化手法クラスのtorch.optim.Optimizer化（order_002）
│   ├── report_003.md                   # 勾配計算の自動微分化（order_003）
│   ├── report_004.md                   # Ex002の実験結果レポート（order_004）
│   ├── report_005.md                   # Ex002の再実験（エポック数増加，#grad/N軸の修正，
│   │                                    # スナップショット推論の確認，order_005）
│   ├── report_006.md                   # NFG SVRG原論文との比較検証（order_006）
│   ├── report_007.md                   # Ex003：NFG SVRG原論文の実験の再現（order_007）
│   ├── report_008.md                   # Ex003のミニバッチサイズ1での再検証（order_008）
│   └── report_009.md                   # Ex004：M_WORKERS分割・BN固定・sigma勾配スケールの
│                                        # 3点修正による再検証（order_009）
├── requirements_pytorch.txt
├── .venv_pytorch/                      # Python仮想環境（Git管理対象外）
└── document.md                         # 本ファイル
```

## 3. プログラム間の依存関係

- `programs/ex001_mushroom_svrg/data.py`：`.ai/ai-dev-kit/src/machine_learning_utils.py` の
  `set_seed` を利用する．UCI Machine Learning Repositoryからマッシュルームデータセットを
  ダウンロードし（初回のみ），22種類のカテゴリ特徴量を順序符号化（`OrdinalEncoder`）した後，
  9:1に分割・標準化して `torch.utils.data.DataLoader` を構築する．
- `programs/ex001_mushroom_svrg/model.py`：`.ai/ai-dev-kit/src/machine_learning_utils.py` の
  `set_seed` を利用する．L2正則化付きロジスティック回帰モデル（`LogisticRegressionModel`）と，
  `torch.autograd`（`loss.backward()`）による勾配計算・損失・精度を計算する関数を定義する
  （`.orders/order_003.md` の指示）．
- `programs/optimizers/optimizers.py`：他モジュールに依存しない（`torch`，`numpy` のみに依存）．
  `SGD`，`SVRG`，`SVRGFinalPoint`，`NFGSVRG`，`NFGSVRGFinalPoint`，`ASAISVRG` の6クラスを提供
  する．`NFGSVRGFinalPoint`／`SVRGFinalPoint` は，NFG SVRG原論文
  （`references/No_Full_Grad_SVRG.pdf`）のAlgorithm 1，および原論文が比較対象とする古典的
  SVRGが実際に用いるスナップショット構成（内部ループの最終パラメータを採用）を忠実に再現した
  クラスであり，ASAI SVRG論文の理論解析上の都合による一様ランダム選択を用いる
  `NFGSVRG`／`SVRG` とは異なる．`.orders/order_006.md`／`.orders/order_007.md` の検証実験で
  導入し，Ex003における「原論文に忠実なSVRG・NFG」として利用する．Ex001・Ex002の正式な4手法
  比較には使用しない．
- `programs/ex001_mushroom_svrg/train.py`：上記3モジュールおよび `machine_learning_utils.py` の
  `ResultLogger`，`set_seed` を利用し，学習全体を統括する．

### 実装上の設計判断（`@.ai/ai-dev-kit/machine_learning.md` との差異）

`machine_learning.md` は，`iteration`/`epoch`/`train` 関数によるミニバッチ epoch 学習ループの
テンプレートを定めているが，SVRG系手法（Algorithm 1〜4）は外部ループ・内部ループから成る特有の
反復構造を持ち，このテンプレートとは根本的に構造が異なる（内部ループでは同一サンプルに対して
現在パラメータとスナップショットの双方で勾配を評価する必要があり，外部ループはスナップショットの
再構成を伴う）．そのため `train.py` では，論文のAlgorithm 1〜4に忠実な専用の学習ループ
（`run_sgd`，`run_variance_reduced`）を実装し，`iteration`/`epoch`/`train` 関数の形式は用いていない．
`set_seed`，`ResultLogger`，出力ディレクトリ規則，`load_model`/`load_dataloader` のインターフェース
等，テンプレートと両立する部分はそのまま踏襲している．

最適化手法クラス（`programs/optimizers/optimizers.py`）は，`.orders/order_002.md` の指示に
基づき，SGD，SVRG，NFG SVRG，ASAI SVRGの4クラスをすべて `torch.optim.Optimizer` の
サブクラスとして独立に陽実装している（`torch.optim.SGD` 等，PyTorch公式の最適化手法実装は
使用しない）．SVRG，NFG SVRG，ASAI SVRGの内部ループの更新則には共通する部分があるが，
指示に従い共通の基底クラスへ抽出せず，各クラスが自身のアルゴリズム（論文Algorithm 1〜3）を
完結して実装している．勾配は各パラメータの `.grad` 属性から読み取る通常の
`torch.optim.Optimizer` の流儀に従う．`.orders/order_003.md` の指示に基づき，`.grad` は
`model.py` の `compute_gradient` が `loss.backward()`（PyTorchの標準的な自動微分）を実行する
ことで設定される．

SVRG系手法は，現在のパラメータ w_s^k とスナップショット z_s の双方における勾配を必要とする．
自動微分は実際にその値でforwardした計算グラフに対してのみ機能するため，`train.py` では
現在のパラメータを保持する `model` と，スナップショットを保持する別インスタンス
`snapshot_model`（`model.py` の `set_model_params` でパラメータを更新する）の2つの
モデルインスタンスを用意し，同一サンプルに対してそれぞれforward／backwardを実行することで
2種類の勾配を得ている．

SVRG系3手法は，`step()` に加えて外部ループの境界で呼び出す `begin_epoch()`／`end_epoch()`，
および次エポックのスナップショットを取得する `get_snapshot_params()`／`get_snapshot_gradient()`
を提供する．3クラスとも同じインターフェースを持つため，`train.py` の
`run_variance_reduced()` は，このインターフェースを介して1つの関数で3手法を扱う
（Optimizerクラス自体はそれぞれ独立に陽実装されている）．

- `programs/ex002_cifar10_cnn/data.py`：`.ai/ai-dev-kit/src/machine_learning_utils.py` の
  `set_seed` を利用する．`torchvision.datasets.CIFAR10` によりCIFAR-10を取得し（初回のみ
  ダウンロード），チャネルごとの平均・標準偏差による標準化を行う．CIFAR-10は学習用
  （50000枚）・検証用（10000枚）の公式な分割が定義されているため，独自の9:1分割は行わない．
- `programs/ex002_cifar10_cnn/model.py`：3つの畳み込み層と1つの全結合層から成る
  `CNNModel`，および `load_model`／`set_model_params` を定義する．
- `programs/ex002_cifar10_cnn/train.py`：`.orders/order_004.md` の指示に基づき，
  `loss_func`／`metrics_func`／`iteration`／`epoch`／`train` の関数構成を用いる．
  ただし，SVRG系手法は同一ミニバッチに対して現在のパラメータ `model` とスナップショット
  `snapshot_model` の双方における勾配を必要とするため，`iteration` 関数は
  `snapshot_model` を任意引数として受け取れるよう拡張している．この点のみが標準テンプレート
  からの変更点である．外部ループ（`begin_epoch`/`end_epoch`/スナップショット更新）は
  `train_variance_reduced()` が担い，1エポック分の内部ループは `epoch` 関数（ミニバッチの
  DataLoaderを1周）がそのまま担う．すなわち，Ex001（オンライン学習，内部ループ長
  K=N_train個の単一サンプル）とは異なり，Ex002では「1エポック＝1回のDataLoader走査」が
  そのままSVRGの内部ループ（K=ミニバッチ数）に対応する，実装上より自然な構成となっている．

  非凸設定のため，目的関数の真の最適値 `f(w*)` は求まらない．そのため，論文4.1節で用いる
  目的関数の「誤差」 `f(z_s) - f(w*)` の代わりに，目的関数の値 `f(z_s)` 自体（学習損失）を
  記録する．また，学習率はリプシッツ定数から導出せず，`.orders/order_004.md` の指示に従い
  適当な値（`LEARNING_RATE = 0.01`）を定めている．

- `programs/ex003_cifar10_resnet_minmax/model.py`：CIFAR向けResNet-18（`ResNet18`）と，
  min-max敵対的ロバスト性の定式化のための敵対的摂動 `sigma`（画像1枚分の形状）を保持する
  `MinMaxResNet18`，および `load_model`／`set_model_params` を定義する．
- `programs/ex003_cifar10_resnet_minmax/data.py`：Ex002と同様にCIFAR-10を取得・前処理する．
- `programs/ex003_cifar10_resnet_minmax/train.py`：`.orders/order_007.md` の指示に基づき，
  NFG SVRG原論文7節の実験（ResNet-18・min-max敵対的ロバスト性の定式化）を再現する．
  min-maxの $ \sigma $ に対する勾配上昇は，目的関数 $ L $ の $ \sigma $ に関する自然な勾配を
  反転させた $ F_\sigma = -\partial L/\partial \sigma $ を「勾配」として扱うことで実現し
  （`backward_minmax_objective`），既存の最適化手法クラスを一切変更せずに再利用している．
  比較手法は，原論文と同一のスナップショット構成（最終点採用）を用いる `SVRGFinalPoint`，
  `NFGSVRGFinalPoint`，および提案手法自身のスナップショット構成（平均パラメータ）を用いる
  `ASAISVRG` の3手法に，`SGD` を加えた4手法．ResNet-18はBatch Normalization層を含むため，
  `epoch` 関数内で `optimizer` の有無に応じ `model.train()`／`model.eval()` を明示的に
  切り替えている．

  **重要な発見**：ASAI SVRGは，5Seedすべてで学習の途中から指数的に発散する現象が観測された．
  当初はBatch Normalizationの移動平均統計量とスナップショット（平均パラメータ）との不整合を
  仮説として立てたが，統計量の再較正を試みても発散は解消せず，この仮説は反証された．真の原因は
  `.reports/report_007.md` の時点では特定できていない．また，`.orders/order_008.md` の指示に
  基づく追加検証（ミニバッチサイズ1，`.reports/report_008.md`）では，ミニバッチサイズ1に変更
  すると，Ex003で安定していたSVRG（真のフル勾配を用いる古典的手法）までもが1エポック目から
  発散することが判明し，発散の原因が近似誤差ではなく，SVRG系手法の補正勾配自体が単一サンプル・
  高次元非凸ネットワークにおいて高い分散を持つことに起因する可能性が示唆された．

- `programs/ex004_cifar10_resnet_minmax/`：`.orders/order_009.md` の指示に基づき，Ex003の
  実装上の3つの問題点を修正した検証実験．`model.py`・`data.py` はEx003と同一（パス以外の変更
  なし）．`train.py` は次の3点を修正する．
  1. **M=5ワーカーによる分散環境の模倣**：`backward_minmax_objective_distributed` 関数を
     新設し，グローバルミニバッチ（`BATCH_SIZE = 128`）を `M_WORKERS = 5` 個のサブバッチ
     （`torch.chunk`で分割）に分割して各サブバッチを独立にforward・backwardすることで，
     各ワーカーが自身のローカルミニバッチの統計量でBatch Normalizationを適用する挙動を模倣
     する．交差エントロピー損失の勾配は各サブバッチのサンプル数で重み付け平均し（全体1回の
     forwardと数学的に同一の平均勾配を再構成），正則化項の勾配はサブバッチ数に依らず
     グローバルミニバッチ全体に対して1回だけ加える．学習時（`iteration`関数）はこの関数を
     用い，`epoch`関数の`train()`/`eval()`切り替えロジックはEx003から変更していない．
  2. **フル勾配計算時のBatch Normalization統計量の固定**：`compute_full_gradient_and_metrics`
     の`model.train()`を`model.eval()`に変更した．全データセットを走査する間，移動平均統計量
     （running_mean/running_var）が更新され続けるとスナップショット勾配の数学的整合性が
     崩れるため，評価モードに固定した上で（autogradの勾配計算グラフの構築は妨げられない）
     パラメータの`.grad`を計算する．
  3. **sigmaの正則化勾配のスケール**：上記1の設計（正則化項をグローバルミニバッチ全体に対して
     1回だけ加える）により構造的に解決される．

  Ex003との比較のため，比較手法・Seed数・ハイパーパラメータ（学習率，lambda1，lambda2，
  ミニバッチサイズ128）はEx003と同一に保っている．

## 4. 外部モジュールとの依存関係

- PyTorch（`torch`）：パラメータの保持・演算，および `torch.autograd` による勾配計算．
- torchvision：CIFAR-10データセットの取得・前処理（Ex002）．
- scikit-learn：データ分割（`train_test_split`），前処理（`OrdinalEncoder`，`StandardScaler`，Ex001）．
- scipy：最適解 `w*` および最適値 `f(w*)` を求めるための `scipy.optimize.minimize`（L-BFGS-B，Ex001）．
- pandas, requests：マッシュルームデータセットの取得・読み込み（Ex001）．

## 5. Python環境の構築方法

```bash
uv venv .venv_pytorch --python 3.11
uv pip install --python .venv_pytorch/bin/python -r requirements_pytorch.txt
```

VS CodeからJupyterカーネルとして利用する場合は，カーネル名 `asai_svrg_pytorch` を選択する
（`.venv_pytorch/bin/python -m ipykernel install --user --name=asai_svrg_pytorch` で登録済み）．

## 6. プログラムの実行方法

```bash
# Ex001の学習実行（4手法 x 5Seed = 20条件をマルチプロセスで並列実行．既に完了した条件はスキップ）
.venv_pytorch/bin/python programs/ex001_mushroom_svrg/train.py

# Ex002の学習実行（4手法 x 5Seed = 20条件を4プロセス並列で実行．GPU使用，既に完了した条件はスキップ）
.venv_pytorch/bin/python programs/ex002_cifar10_cnn/train.py

# Ex003の学習実行（4手法 x 5Seed = 20条件を4プロセス並列で実行．GPU使用，既に完了した条件はスキップ）
.venv_pytorch/bin/python programs/ex003_cifar10_resnet_minmax/train.py

# 単体テスト
.venv_pytorch/bin/python -m pytest tests/ -v

# 結果の可視化（Jupyter上で実行，またはnbconvertで一括実行．先頭セルのEXPERIMENT変数で
# "ex001_mushroom_svrg"／"ex002_cifar10_cnn"／"ex003_cifar10_resnet_minmax" を切り替える）
.venv_pytorch/bin/jupyter nbconvert --to notebook --execute --inplace visualize_result.ipynb
```

## 7. 実験結果・文書の保存場所

- 学習結果（各Seedのログ・メタデータ）：
  `outputs/{ex001_mushroom_svrg,ex002_cifar10_cnn,ex003_cifar10_resnet_minmax}/{method}/{hyperparams}/{seed}/`
- 可視化結果（グラフ画像）：`outputs/{ex001_mushroom_svrg,ex002_cifar10_cnn,ex003_cifar10_resnet_minmax}/` 直下
- レポート：`.reports/report_001.md`（Ex001の実験結果），`.reports/report_002.md`（最適化手法
  クラスの設計），`.reports/report_003.md`（勾配計算方式の変更），`.reports/report_004.md`
  （Ex002の実験結果，初回），`.reports/report_005.md`（Ex002の再実験），`.reports/report_006.md`
  （NFG SVRG原論文との比較検証），`.reports/report_007.md`（Ex003：原論文実験の再現），
  `.reports/report_008.md`（Ex003：ミニバッチサイズ1での再検証）

## 8. 必要なAPIキーや設定ファイル

Ex001（UCI Machine Learning Repository）・Ex002／Ex003（CIFAR-10，torchvision経由）とも公開
データセットのみを用いるため，APIキーは不要である．`tokens.json`（Gemini，Hugging Face Hub用）
はプロジェクトルートに存在し，`.gitignore` で管理対象外としている．

## 9. Git管理上の注意事項

- `datasets/`，`outputs/`，`.venv_pytorch/`，`tokens.json` はいずれも `.gitignore` によりGit管理対象外．
- `.vscode/` は `settings.json` のみGit管理対象とする．
