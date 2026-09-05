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

`.orders/order_010.md` の指示に基づき，(1) `model.py` の `set_model_params()` が
Batch Normalizationのバッファ（`running_mean`，`running_var`）を同期していなかったバグを
Ex001〜Ex004すべてに対して修正し，(2) min-max構造（敵対的摂動sigma）を取り除いた純粋な
CIFAR-10多値分類問題をEx005として実装した．詳細は`.reports/report_010.md`を参照．

`.orders/order_011.md` の指示に基づき，NFG SVRG原論文（`references/No_Full_Grad_SVRG.pdf`）の
7節（ResNet-18・CIFAR-10）の再現は断念し，同論文の付録A.1（LEAST SQUARES REGRESSION，式(8)の
非線形最小二乗回帰問題）をa9aデータセットで再現するEx006を実装した．詳細は`.reports/report_011.md`
を参照．

### 1.1 `.orders/order_020.md` によるディレクトリ構成の変更（重要）

`.orders/order_020.md` により，Ex001〜Ex006は「論文に掲載する実験ではなく，性能の簡易評価の
ための事前実験」と位置付けられ，以降に実施する論文掲載用の実験と混同しないよう，次の3つの
ディレクトリを改名した．改名に伴い，各モジュール内のパスの参照も併せて修正しており，改名前と
同様に実行できる状態を維持している．

| 改名前 | 改名後 | 内容 |
| :--- | :--- | :--- |
| `programs/` | `programs_old/` | Ex001〜Ex006の事前実験のプログラム |
| `outputs/` | `outputs_old/` | Ex001〜Ex006の事前実験の実験結果 |
| `tests/` | `tests_old/` | Ex001〜Ex006の事前実験の単体テスト |

改名後の `programs/`，`outputs/`，`tests/` には，`.orders/order_020.md` が定める論文掲載用の
実験（実験0〜実験3）のみを配置する．本セッションでは，まず**実験0（a9aデータセットを用いた
二値分類問題，`ex000_a9a_least_squares`）**を実装・実施した．実験0は，NFG SVRG原論文の非凸実験
設定（付録A.1）を再現し，本リポジトリの実装が先行研究の挙動を正しく再現できることを確認する
検証実験である．さらにユーザーからのチャットでの追加指示に基づき，論文全体を通じて提案手法
ASAI SVRGが一貫して比較対象に含まれるよう，本来`.orders/order_020.md`ではASAI SVRGを対象外と
してよいとされていた実験0にもASAI SVRGを追加し，最終的に4手法（SGD，SVRG，NFG SVRG，
ASAI SVRG）で実施した．詳細は`.reports/report_020.md`を参照．

なお，`programs/optimizers/` は，`.orders/order_020.md` の指示に基づき，従来1つのモジュール
（`programs_old/optimizers/optimizers.py`）にまとめられていた6つの最適化手法クラスを，可読性の
ために手法ごとのモジュールへ分割したものである．アルゴリズムの実装内容そのものは従来の実装を
そのまま引き継いでおり，両者が同一の更新結果を与えることは
`tests/test_optimizers.py::test_split_modules_match_archived_implementation` で確認している．

### 1.2 `.orders/order_021.md` による実験1（Mushroom，強凸設定）の実装

`.orders/order_021.md` は，実験0（`report_020.md`）で観察された (a) 学習率が理論上界
$ \eta=1/(3L) $ に近い場合のNFG SVRG・ASAI SVRGの振動，(b) ASAI SVRGの近似誤差
$ \|e_s\|^2 $ がNFG SVRGより一貫して小さい現象を，ASAI SVRG論文のAssumption 1〜4
（$ L $-平滑性，凸性，$ \mu $-強凸性，分散の一様有界性）を厳密に満たす設定（Mushroom
データセットを用いたL2正則化付きロジスティック回帰）で定量的に検証するよう指示している．
具体的には，定理1（収束特性，式(30)(31)）・定理2（誤差床の上界比較，式(37)）の理論的主張が
実測でも成立するかを，SGD，SVRG，NFG SVRG，ASAI SVRGの4手法（実験0と異なりASAI SVRGは
理論的検証の主対象であるため必須）で検証する実験1として実装した．詳細は`.reports/
report_021.md`を参照．

## 2. ディレクトリ構成と各ファイルの役割

```text
./
├── .ai/                                # AI開発規約（サブモジュール）
├── references/
│   └── ASAI_SVRG_paper.pdf             # 参照論文
├── datasets/
│   ├── ex000_a9a_least_squares/
│   │   └── raw/                        # a9aデータセットの生データ（LIBSVM Data，自動ダウンロード）
│   │                                    # ★ order_020以降の論文掲載用の実験0
│   ├── ex001_mushroom_logistic/
│   │   └── raw/                        # UCI Mushroomデータセットの生データ（自動ダウンロード）
│   │                                    # ★ order_021の論文掲載用の実験1
│   ├── ex001_mushroom_svrg/
│   │   └── raw/                        # UCI Mushroomデータセットの生データ（自動ダウンロード）
│   ├── ex002_cifar10_cnn/
│   │   └── raw/                        # CIFAR-10データセットの生データ（自動ダウンロード）
│   ├── ex003_cifar10_resnet_minmax/
│   │   └── raw/                        # CIFAR-10データセットの生データ（ex002からコピー）
│   ├── ex004_cifar10_resnet_minmax/
│   │   └── raw/                        # CIFAR-10データセットの生データ（ex003からコピー）
│   ├── ex005_cifar10_resnet_classification/
│   │   └── raw/                        # CIFAR-10データセットの生データ（ex004からコピー）
│   └── ex006_a9a_least_squares/
│       └── raw/                        # a9aデータセットの生データ（LIBSVM Data，自動ダウンロード）
├── programs/                           # ★ order_020以降の論文掲載用の実験
│   ├── optimizers/                     # 最適化手法を手法ごとのモジュールに分割（order_020）
│   │   ├── __init__.py                 # 6クラスをまとめて再エクスポート
│   │   ├── sgd.py                      # SGD（ASAI SVRG論文 式(4)）
│   │   ├── svrg.py                     # SVRG（ASAI SVRG論文 Algorithm 1，ランダム選択）
│   │   ├── nfg_svrg.py                 # NFG SVRG（ASAI SVRG論文 Algorithm 2，ランダム選択）
│   │   ├── asai_svrg.py                # ASAI SVRG（ASAI SVRG論文 Algorithm 3，提案手法）
│   │   ├── svrg_final_point.py         # SVRG（NFG SVRG原論文の比較対象，最終点採用）
│   │   └── nfg_svrg_final_point.py     # NFG SVRG（NFG SVRG原論文 Algorithm 1，最終点採用）
│   ├── ex000_a9a_least_squares/        # 実験0：a9aの二値分類（非線形最小二乗誤差，非凸）
│   │   ├── data.py                     # a9aデータセット（LIBSVM Data）の取得・前処理・
│   │   │                                # DataLoader構築．標準化等は行わず0/1の特徴量をそのまま使用
│   │   ├── model.py                    # 非線形最小二乗回帰モデル（切片なし線形結合+シグモイド，
│   │   │                                # 倍精度）．勾配は自動微分で計算
│   │   └── train.py                    # 4手法 x 3学習率 x 5Seed = 60条件の学習を実行する
│   │                                    # スクリプト（本体，order_020／チャットでの追加指示で
│   │                                    # ASAI SVRG追加）
│   └── ex001_mushroom_logistic/        # 実験1：Mushroomの二値分類（BCE+L2正則化，強凸）
│       ├── data.py                     # マッシュルームデータセット（順序符号化+標準化，
│       │                                # N=8124, d=22）の取得・前処理・DataLoader構築
│       ├── model.py                    # L2正則化付きロジスティック回帰（切片含む）．正則化は
│       │                                # 重み・切片の両方に課し，全パラメータの mu-強凸性を
│       │                                # 厳密に満たす（事前実験 ex001_mushroom_svrg との差異）
│       └── train.py                    # 4手法 x 3学習率 x 5Seed = 60条件の学習を実行する
│                                        # スクリプト（本体，order_021）．L・lambda・収縮係数
│                                        # rhoの数値的な逆算（solve_eta_for_target_rho）を含む
├── programs_old/                       # order_020以前の事前実験（Ex001〜Ex006）
│   ├── optimizers/
│   │   ├── __init__.py
│   │   └── optimizers.py               # SGD, SVRG, SVRGFinalPoint, NFGSVRG, NFGSVRGFinalPoint,
│   │                                    # ASAISVRGの6クラス．いずれもtorch.optim.Optimizerの
│   │                                    # サブクラスとして独立に陽実装（.orders/order_002.md）．
│   │                                    # programs/optimizers/ の分割前の実装
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
│   ├── ex004_cifar10_resnet_minmax/
│   │   ├── data.py                     # CIFAR-10データセットの取得・前処理・DataLoader構築
│   │   │                                # （ex003と同一，パスのみex004向けに変更）
│   │   ├── model.py                    # ResNet-18 + min-max敵対的摂動sigma（ex003と同一）．
│   │   │                                # set_model_paramsのBNバッファ同期修正済み（order_010）
│   │   └── train.py                    # Ex003の3点の修正（M_WORKERS分割によるBatch
│   │                                    # Normalization挙動の模擬，フル勾配計算のeval()化，
│   │                                    # sigma正則化勾配のスケール整合）を反映した学習
│   │                                    # スクリプト（order_009，本体）
│   ├── ex005_cifar10_resnet_classification/
│   │   ├── data.py                     # CIFAR-10データセットの取得・前処理・DataLoader構築
│   │   │                                # （ex004と同一，パスのみex005向けに変更）
│   │   ├── model.py                    # ResNet-18のみ（sigma・min-max構造を除去）．
│   │   │                                # set_model_paramsのBNバッファ同期を最初から実装
│   │   └── train.py                    # Ex004からsigma・min-max構造を取り除いた，純粋な
│   │                                    # 多値分類問題の学習スクリプト（order_010，本体）
│   └── ex006_a9a_least_squares/
│       ├── data.py                     # a9aデータセット（LIBSVM Data）の取得・前処理・
│       │                                # DataLoader構築．標準化等は行わず0/1の特徴量をそのまま使用
│       ├── model.py                    # 非線形最小二乗回帰モデル（切片なし線形結合+シグモイド）．
│       │                                # 勾配は自動微分で計算
│       └── train.py                    # 4手法 x 5Seedの学習を実行するスクリプト（本体，order_011）
├── outputs/                            # ★ order_020以降の論文掲載用の実験結果
│   ├── ex000_a9a_least_squares/
│   │   ├── {method}/{hyperparams}/{seed}/
│   │   │   ├── log.json                # ResultLoggerによる評価指標の履歴
│   │   │   └── config.json             # 学習率・平滑性定数等のメタデータ
│   │   ├── grad_norm_sq_vs_full_grad_computations.png  # 原論文Figure 3に対応する図
│   │   ├── comparison_all_axes_{eta}.png               # 3横軸 × 3評価指標の比較
│   │   ├── accuracy_vs_epoch.png                       # 分類精度の推移
│   │   └── final_epoch_summary.md                      # 最終エポックの評価指標の表
│   └── ex001_mushroom_logistic/
│       ├── {method}/{hyperparams}/{seed}/
│       │   ├── log.json                # ResultLoggerによる評価指標の履歴（objective_gap等）
│       │   └── config.json             # L・lambda・rho等の理論的な根拠を含むメタデータ
│       ├── comparison_all_axes_{eta}.png               # 3横軸 × 3評価指標の比較
│       ├── accuracy_vs_epoch.png                       # 分類精度の推移
│       └── final_epoch_summary.md                      # 最終エポックの評価指標の表
├── outputs_old/                        # order_020以前の事前実験の結果
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
│   ├── ex005_cifar10_resnet_classification/
│   │   └── {method}/{hyperparams}/{seed}/  # 同上の構成
│   ├── ex006_a9a_least_squares/
│   │   └── {method}/{hyperparams}/{seed}/  # 同上の構成（best_model.pthは保存しない）
│   └── order_006_archive/              # order_006検証（NFG_SVRG_FinalPoint）の実験結果．
│                                        # order_007によりEx002本体からは切り離して保存
├── tests/                              # ★ order_020以降の論文掲載用の実験の単体テスト
│   ├── test_optimizers.py              # 手法ごとに分割した最適化手法クラスの単体テスト．
│   │                                    # 分割前の実装と同一結果になることも確認する
│   ├── test_ex000_a9a_least_squares.py # 実験0のモデル・勾配・平滑性定数・学習率の上界・
│   │                                    # オラクル呼び出し回数・3手法のスモークテスト
│   └── test_ex001_mushroom_logistic.py # 実験1のモデル（重み・切片両方への正則化）・勾配・
│                                        # 平滑性定数・収縮係数rhoのU字形・rho逆算の妥当性・
│                                        # 4手法のスモークテスト・オラクル呼び出し回数．
│                                        # 実験0のテストとのモジュール名（model/data/train）
│                                        # 衝突をimportlibによる明示的な読み込みで回避
├── tests_old/                          # order_020以前の事前実験の単体テスト
│   ├── test_optimizers.py              # 最適化手法クラスの単体テスト（pytest）
│   ├── test_model.py                   # Ex001のモデル・勾配計算関数の単体テスト（pytest）
│   ├── test_cnn_model.py               # Ex002のCNNモデルの単体テスト（pytest）
│   ├── test_snapshot_inference.py      # SVRG系手法の推論がスナップショットを用いること，
│   │                                    # ASAI SVRGでは平均パラメータであることの統合テスト
│   │                                    # （order_005，pytest）
│   ├── test_minmax_resnet.py           # Ex003のResNet-18・min-max定式化（符号反転の正しさ）
│   │                                    # の単体テスト（order_007，pytest）
│   ├── test_minmax_resnet_distributed.py  # Ex004のM_WORKERS分割による勾配集約，フル勾配計算の
│   │                                    # Batch Normalization統計量固定，snapshot_modelのBN
│   │                                    # バッファ同期（order_009／order_010）の単体テスト
│   ├── test_resnet_classification.py   # Ex005のResNet18（sigmaなし），M_WORKERS分割勾配，
│   │                                    # BNバッファ同期の単体テスト（order_010，pytest）
│   └── test_least_squares_regression.py  # Ex006の非線形最小二乗回帰モデル・平滑性定数計算・
│                                        # 4手法のスモークテスト（order_011，pytest）
├── visualize_result.ipynb              # 実験結果の可視化ノートブック（ルート直下）．
│                                        # OUTPUTS_ROOT変数で outputs/（論文掲載用）と
│                                        # outputs_old/（事前実験）を切り替えられる．
│                                        # 横軸「フル勾配の計算回数」は，オラクル呼び出し回数を
│                                        # N_trainで除した値（train.pyが
│                                        # full_grad_computationsとして記録）を用いる
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
│   ├── report_009.md                   # Ex004：M_WORKERS分割・BN固定・sigma勾配スケールの
│   │                                    # 3点修正による再検証（order_009）
│   ├── report_010.md                   # set_model_paramsのBNバッファ同期バグ修正，および
│   │                                    # Ex005：min-max構造を除いた純粋な多値分類問題（order_010）
│   ├── report_011.md                   # Ex006：NFG SVRG原論文 付録A.1（LEAST SQUARES
│   │                                    # REGRESSION）の再現実験（order_011）
│   ├── report_020.md                   # ディレクトリの改名・Optimizerの手法別分割，および
│   │                                    # 実験0（a9aの二値分類，ASAI SVRG追加を含む）の実施
│   │                                    # （order_020／チャットでの追加指示）
│   └── report_021.md                   # 実験1：Mushroomの二値分類（強凸設定）による
│                                        # 定理1・定理2の定量的検証（order_021）
├── requirements_pytorch.txt
├── .venv_pytorch/                      # Python仮想環境（Git管理対象外）
└── document.md                         # 本ファイル
```

## 3. プログラム間の依存関係

### 3.1 論文掲載用の実験（`programs/`，`.orders/order_020.md` 以降）

- `programs/optimizers/`：他モジュールに依存しない（`torch`，`numpy` のみに依存）．
  `sgd.py`，`svrg.py`，`nfg_svrg.py`，`asai_svrg.py`，`svrg_final_point.py`，
  `nfg_svrg_final_point.py` の6モジュールに1クラスずつ定義し，`__init__.py` がこれらを
  まとめて再エクスポートするため，利用側は `from optimizers import SGD, SVRGFinalPoint` の
  ように従来と同じ形で読み込める．
- `programs/ex000_a9a_least_squares/data.py`：`machine_learning_utils.py` の `set_seed` を
  利用する．LIBSVM Dataからa9aを取得し（初回のみ），ラベルを $ \{-1, +1\} $ から
  $ \{0, 1\} $ へ写像した上で9:1に分割する．原論文の式(8)が標準化に言及していないため，
  特徴量は0/1のまま無加工で用いる．
- `programs/ex000_a9a_least_squares/model.py`：`machine_learning_utils.py` の `set_seed` を
  利用する．切片・正則化項を持たない非線形最小二乗回帰モデルと，自動微分による勾配計算・
  損失・精度を計算する関数を定義する．フル勾配のノルムを多桁にわたって観測するため，
  パラメータは倍精度（`torch.float64`）で保持する．
- `programs/ex000_a9a_least_squares/train.py`：上記の3つに加え `machine_learning_utils.py` の
  `ResultLogger`，`set_seed`，および `scipy.optimize`（参考値 $ f(x^*) $ の算出）を利用し，
  学習全体を統括する．60条件をマルチプロセスで並列実行するため，NumPy・PyTorchのimportより
  前に `OMP_NUM_THREADS` 等を1に設定し，BLASによるスレッドの過剰起動を防いでいる．
- `programs/ex001_mushroom_logistic/data.py`：`machine_learning_utils.py` の `set_seed` を
  利用する．UCI Machine Learning Repositoryからマッシュルームデータセットを取得し（初回の
  み），22種類のカテゴリ特徴量を順序符号化（`OrdinalEncoder`）した後，9:1に分割・標準化して
  `DataLoader` を構築する．この前処理はASAI SVRG論文4.1節の記載（$ N=8124 $，$ d=22 $）と
  一致する．
- `programs/ex001_mushroom_logistic/model.py`：`machine_learning_utils.py` の `set_seed` を
  利用する．切片項を含むL2正則化付きロジスティック回帰モデルを定義する．正則化項は重み
  $ w $ と切片 $ b $ の両方に課し（事前実験 `programs_old/ex001_mushroom_svrg/model.py` は
  $ w $ のみ），全パラメータについて $ \mu $-強凸性（Assumption 2(a)）を厳密に満たす．
- `programs/ex001_mushroom_logistic/train.py`：上記の2つに加え `machine_learning_utils.py` の
  `ResultLogger`，`set_seed`，`optimizers`（`SGD`，`SVRGFinalPoint`，`NFGSVRGFinalPoint`，
  `ASAISVRG`），および `scipy.optimize`（最適値 $ f(w^*) $ の算出，収縮係数 $ \rho $ を目標値
  に一致させる学習率の数値的な逆算）を利用し，学習全体を統括する．`.reports/report_020.md`
  5.6節で判明したCPU競合（物理32コアに対し論理64スレッドが存在する2-way SMT環境で，論理
  スレッド数まで並列化すると物理コア数を超過する）を踏まえ，並列数を物理コア数相当（論理
  スレッド数の半分）に制限し，60条件全てを同一の競合条件下で実行することで，実行時間の
  比較可能性を確保している．

### 3.2 事前実験（`programs_old/`，`.orders/order_011.md` まで）

- `programs_old/ex001_mushroom_svrg/data.py`：`.ai/ai-dev-kit/src/machine_learning_utils.py` の
  `set_seed` を利用する．UCI Machine Learning Repositoryからマッシュルームデータセットを
  ダウンロードし（初回のみ），22種類のカテゴリ特徴量を順序符号化（`OrdinalEncoder`）した後，
  9:1に分割・標準化して `torch.utils.data.DataLoader` を構築する．
- `programs_old/ex001_mushroom_svrg/model.py`：`.ai/ai-dev-kit/src/machine_learning_utils.py` の
  `set_seed` を利用する．L2正則化付きロジスティック回帰モデル（`LogisticRegressionModel`）と，
  `torch.autograd`（`loss.backward()`）による勾配計算・損失・精度を計算する関数を定義する
  （`.orders/order_003.md` の指示）．
- `programs_old/optimizers/optimizers.py`：他モジュールに依存しない（`torch`，`numpy` のみに依存）．
  `SGD`，`SVRG`，`SVRGFinalPoint`，`NFGSVRG`，`NFGSVRGFinalPoint`，`ASAISVRG` の6クラスを提供
  する．`NFGSVRGFinalPoint`／`SVRGFinalPoint` は，NFG SVRG原論文
  （`references/No_Full_Grad_SVRG.pdf`）のAlgorithm 1，および原論文が比較対象とする古典的
  SVRGが実際に用いるスナップショット構成（内部ループの最終パラメータを採用）を忠実に再現した
  クラスであり，ASAI SVRG論文の理論解析上の都合による一様ランダム選択を用いる
  `NFGSVRG`／`SVRG` とは異なる．`.orders/order_006.md`／`.orders/order_007.md` の検証実験で
  導入し，Ex003における「原論文に忠実なSVRG・NFG」として利用する．Ex001・Ex002の正式な4手法
  比較には使用しない．
- `programs_old/ex001_mushroom_svrg/train.py`：上記3モジュールおよび `machine_learning_utils.py` の
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

最適化手法クラス（`programs_old/optimizers/optimizers.py`，分割後は `programs/optimizers/`）は，`.orders/order_002.md` の指示に
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

- `programs_old/ex002_cifar10_cnn/data.py`：`.ai/ai-dev-kit/src/machine_learning_utils.py` の
  `set_seed` を利用する．`torchvision.datasets.CIFAR10` によりCIFAR-10を取得し（初回のみ
  ダウンロード），チャネルごとの平均・標準偏差による標準化を行う．CIFAR-10は学習用
  （50000枚）・検証用（10000枚）の公式な分割が定義されているため，独自の9:1分割は行わない．
- `programs_old/ex002_cifar10_cnn/model.py`：3つの畳み込み層と1つの全結合層から成る
  `CNNModel`，および `load_model`／`set_model_params` を定義する．
- `programs_old/ex002_cifar10_cnn/train.py`：`.orders/order_004.md` の指示に基づき，
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

- `programs_old/ex003_cifar10_resnet_minmax/model.py`：CIFAR向けResNet-18（`ResNet18`）と，
  min-max敵対的ロバスト性の定式化のための敵対的摂動 `sigma`（画像1枚分の形状）を保持する
  `MinMaxResNet18`，および `load_model`／`set_model_params` を定義する．
- `programs_old/ex003_cifar10_resnet_minmax/data.py`：Ex002と同様にCIFAR-10を取得・前処理する．
- `programs_old/ex003_cifar10_resnet_minmax/train.py`：`.orders/order_007.md` の指示に基づき，
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

- `programs_old/ex004_cifar10_resnet_minmax/`：`.orders/order_009.md` の指示に基づき，Ex003の
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

- **`set_model_params()` のBatch Normalizationバッファ同期バグの修正（`.orders/order_010.md`）**：
  `set_model_params()` は，従来 `model.parameters()`（学習可能パラメータ）のみを上書きし，
  Batch Normalizationの移動平均統計量（`model.buffers()`）には触れていなかった．そのため，
  `train.py` がエポック境界で `snapshot_model` のパラメータを更新しても，そのBN統計量は
  それ以前の独立した学習履歴のまま古くなっていた．`set_model_params(model, param_values,
  source_model=None)` に `source_model` 引数を追加し，指定時は `model.buffers()` を
  `source_model.buffers()`（実際に学習した `model` の現在値）で同期するよう修正した．
  Ex001〜Ex004の全 `model.py`／`train.py`（呼び出し箇所を `source_model=model` に変更）に
  同一の修正を適用し，Ex005は最初からこの修正済みの実装で構築した．ただし，`outputs/`
  以下の既存のEx001〜Ex004の実験結果は，この修正が適用される**前**に生成されたものである点に
  注意（詳細は`.reports/report_010.md` 3.3節）．

- **`programs_old/ex005_cifar10_resnet_classification/`（`.orders/order_010.md`）**：Ex004から
  min-max構造（敵対的摂動 `sigma`，および正則化項 $ -(\lambda_2/2)\|\sigma\|^2 $）のみを
  取り除いた，純粋な多値分類問題の実験．`model.py` は Ex003/Ex004の `ResNet18` クラスをそのまま
  流用し（`MinMaxResNet18` は定義しない），`train.py` は `backward_minmax_objective(_distributed)`
  から `backward_objective(_distributed)` に改名した上でsigma関連の処理を除去している．
  M_WORKERS分割によるBatch Normalization挙動の模擬，フル勾配計算時の `model.eval()` 化は
  min-max構造に依存しないためEx004からそのまま引き継いでいる．データセット・モデル構造・
  学習率・`lambda1`・ミニバッチサイズ・エポック数・Seed数はEx004と同一に揃え，min-max構造の
  有無のみを比較可能にしている．

- **`programs_old/ex006_a9a_least_squares/`（`.orders/order_011.md`）**：NFG SVRG原論文
  （`references/No_Full_Grad_SVRG.pdf`）付録A.1（LEAST SQUARES REGRESSION）の再現実験．
  `data.py` はLIBSVM Dataのa9aデータセット（32561サンプル，特徴量次元数123，値は0/1の
  One-Hotベクトル）を取得し，9:1に分割する．原論文の式(8) $ f(x) = (1/n)\sum_i(y_i-h_i)^2 $，
  $ h_i = 1/(1+\exp(-A_i\cdot x)) $ は標準化に言及していないため，Ex001とは異なり
  `StandardScaler` 等の前処理は行わず，特徴量を無加工のまま用いる．`model.py` の
  `LeastSquaresSigmoidModel` は，式(8)通り切片なし（`bias=False`）の線形結合にシグモイド
  関数を適用し，二乗誤差を損失とする．正則化項も式(8)には現れないため付加していない．

  比較手法は`SGD`，原論文Algorithm 1に忠実な`SVRGFinalPoint`，`NFGSVRGFinalPoint`，
  ASAI SVRG論文自身のスナップショット構成を用いる`ASAISVRG`の4手法（Ex003〜Ex005と同じ選択）．
  学習率は，原論文Theorem 1（非凸設定，Algorithm 1）の学習率上界 $ \gamma \le 1/(20Ln) $ の
  半分の値を4手法共通に用いる．平滑性定数 $ L $ は，1サンプル分の損失
  $ l_i(x) = (y_i - \sigma(A_i\cdot x))^2 $ のヘッセ行列が
  $ \kappa_{\max}\cdot A_i A_i^\top $（$ \kappa_{\max} $はzに関する2階微分の絶対値の上界，
  数値的な格子探索で算出）で上から抑えられることを用い，
  $ L = \kappa_{\max}\cdot\lambda_{\max}(A^\top A/N) $ として計算する（`compute_smoothness_constant`）．
  原論文はAppendix A.1で「理論的なステップ幅では，チューニング済み（tuned）のステップ幅と比べて
  収束が劣ることが予想される」と述べており，`.orders/order_011.md` の指示によりtuned版は実装
  していない．評価指標は，原論文Figure 3・4と同じ「真のフル勾配のノルムの2乗
  $ \|\nabla f(z_s)\|^2 $」（`grad_norm_sq`）を主軸とし，目的関数の値・分類精度（補助指標）・
  NFG/ASAI SVRGのフル勾配の近似誤差 $ \|e_s\|^2 $ もあわせて記録する．目的関数 $ f(x) $ は
  非凸であるため，`scipy.optimize.minimize`（L-BFGS-B）で得られる$ f(x^*) $ は大域的最適値の
  保証がなく，参考値として`config.json`にのみ記録する（グラフの縦軸には用いない）．

## 4. 外部モジュールとの依存関係

- PyTorch（`torch`）：パラメータの保持・演算，および `torch.autograd` による勾配計算．
- torchvision：CIFAR-10データセットの取得・前処理（Ex002）．
- scikit-learn：データ分割（`train_test_split`），前処理（`OrdinalEncoder`，`StandardScaler`，
  Ex001，実験1）．
- scipy：最適解 `w*` および最適値 `f(w*)` を求めるための `scipy.optimize.minimize`
  （L-BFGS-B．Ex001，Ex006，実験0，実験1），収縮係数 $ \rho $ を目標値に一致させる学習率の
  数値的な逆算（`scipy.optimize.minimize_scalar`，`scipy.optimize.brentq`．実験1）．
- pandas, requests：マッシュルームデータセットの取得・読み込み（Ex001，実験1）．requestsは
  a9aデータセットの取得（Ex006，実験0）にも用いる．
- scikit-learn：LIBSVM形式のデータの読み込み（`load_svmlight_file`．Ex006，実験0）．
- matplotlib：`visualize_result.ipynb` によるグラフの描画．
- tqdm：学習ループの進捗の表示．

## 5. Python環境の構築方法

```bash
uv venv .venv_pytorch --python 3.11
uv pip install --python .venv_pytorch/bin/python -r requirements_pytorch.txt
```

VS CodeからJupyterカーネルとして利用する場合は，カーネル名 `asai_svrg_pytorch` を選択する
（`.venv_pytorch/bin/python -m ipykernel install --user --name=asai_svrg_pytorch` で登録済み）．

## 6. プログラムの実行方法

```bash
# --- 論文掲載用の実験（.orders/order_020.md 以降）---

# 実験0の学習実行（4手法 x 3学習率 x 5Seed = 60条件をCPUでマルチプロセス並列実行．
# 既に完了した条件はスキップ）
.venv_pytorch/bin/python programs/ex000_a9a_least_squares/train.py

# 実験1の学習実行（4手法 x 3学習率 x 5Seed = 60条件を物理コア数相当で並列実行．
# 既に完了した条件はスキップ）
.venv_pytorch/bin/python programs/ex001_mushroom_logistic/train.py

# --- 事前実験（.orders/order_011.md まで）---

# Ex001の学習実行（4手法 x 5Seed = 20条件をマルチプロセスで並列実行．既に完了した条件はスキップ）
.venv_pytorch/bin/python programs_old/ex001_mushroom_svrg/train.py

# Ex002の学習実行（4手法 x 5Seed = 20条件を4プロセス並列で実行．GPU使用，既に完了した条件はスキップ）
.venv_pytorch/bin/python programs_old/ex002_cifar10_cnn/train.py

# Ex003の学習実行（4手法 x 5Seed = 20条件を4プロセス並列で実行．GPU使用，既に完了した条件はスキップ）
.venv_pytorch/bin/python programs_old/ex003_cifar10_resnet_minmax/train.py

# Ex004の学習実行（4手法 x 5Seed = 20条件を4プロセス並列で実行．GPU使用，既に完了した条件はスキップ）
.venv_pytorch/bin/python programs_old/ex004_cifar10_resnet_minmax/train.py

# Ex005の学習実行（4手法 x 5Seed = 20条件を4プロセス並列で実行．GPU使用，既に完了した条件はスキップ）
.venv_pytorch/bin/python programs_old/ex005_cifar10_resnet_classification/train.py

# Ex006の学習実行（4手法 x 5Seed = 20条件をCPUでマルチプロセス並列実行．既に完了した条件はスキップ）
.venv_pytorch/bin/python programs_old/ex006_a9a_least_squares/train.py

# --- 単体テスト ---
# tests/ と tests_old/ はいずれも `optimizers` という名前のパッケージ（分割後・分割前）を
# 読み込むため，1回のpytestの実行では名前が衝突する．次のように別々に実行する．
.venv_pytorch/bin/python -m pytest tests/ -v          # 論文掲載用の実験
.venv_pytorch/bin/python -m pytest tests_old/ -v      # 事前実験

# --- 結果の可視化 ---
# Jupyter上で実行，またはnbconvertで一括実行する．「可視化する条件の指定」セルの
# EXPERIMENT・METHODS・HYPERPARAMS_LIST で対象を切り替える．事前実験の結果を見る場合は，
# 冒頭のセルの OUTPUTS_ROOT を Path("outputs_old") に変更する．
.venv_pytorch/bin/jupyter nbconvert --to notebook --execute --inplace visualize_result.ipynb
```

## 7. 実験結果・文書の保存場所

- 学習結果（各Seedのログ・メタデータ）：
  `outputs/{ex000_a9a_least_squares,ex001_mushroom_logistic}/{method}/{hyperparams}/{seed}/`
  （論文掲載用），
  `outputs_old/{ex001_mushroom_svrg,...,ex006_a9a_least_squares}/{method}/{hyperparams}/{seed}/`（事前実験）
- 可視化結果（グラフ画像）：上記各実験ディレクトリ直下
- レポート：`.reports/report_001.md`（Ex001の実験結果），`.reports/report_002.md`（最適化手法
  クラスの設計），`.reports/report_003.md`（勾配計算方式の変更），`.reports/report_004.md`
  （Ex002の実験結果，初回），`.reports/report_005.md`（Ex002の再実験），`.reports/report_006.md`
  （NFG SVRG原論文との比較検証），`.reports/report_007.md`（Ex003：原論文実験の再現），
  `.reports/report_008.md`（Ex003：ミニバッチサイズ1での再検証），`.reports/report_009.md`
  （Ex004：M_WORKERS分割・BN固定・sigma勾配スケールの3点修正），`.reports/report_010.md`
  （`set_model_params`のBNバッファ同期バグ修正，Ex005：min-max構造を除いた純粋な多値分類問題），
  `.reports/report_011.md`（Ex006：NFG SVRG原論文 付録A.1，LEAST SQUARES REGRESSIONの再現実験），
  `.reports/report_020.md`（実験0：a9aの二値分類，ASAI SVRG追加を含む），`.reports/report_021.md`
  （実験1：Mushroomの二値分類，強凸設定での定理1・定理2の検証）

## 8. 必要なAPIキーや設定ファイル

Ex001（UCI Machine Learning Repository）・Ex002／Ex003（CIFAR-10，torchvision経由）・
Ex006（a9a，LIBSVM Data経由）とも公開データセットのみを用いるため，APIキーは不要である．
`tokens.json`（Gemini，Hugging Face Hub用）は`.gitignore`で管理対象外としているが，本セッションの
作業ディレクトリには暗号化済みバックアップ（`tokens.json.enc`）のみが存在し，復号済みの
`tokens.json`自体は存在しない．本リポジトリの実験はいずれも外部APIを使用しないため実行には
影響しないが，`tokens.json`を必要とする作業を行う場合は復号が必要である旨をユーザーに警告する．

## 9. Git管理上の注意事項

- `datasets/`，`.venv_pytorch/`，`tokens.json`，`*.pth` は `.gitignore` によりGit管理対象外．
- `outputs/`（および改名後の `outputs_old/`）は，`.gitignore` の該当行がコメントアウトされて
  いるためGit管理対象に含まれる．実験結果のログ（`log.json`，`config.json`）と可視化した図が
  リポジトリに含まれる点に注意する．
- `.vscode/` は `settings.json` のみGit管理対象とする．
- `.orders/order_020.md` による改名は `git mv` で実施しているため，`programs_old/`・
  `outputs_old/`・`tests_old/` の履歴は改名前から継続して追跡できる．
