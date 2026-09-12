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

### 1.3 `.orders/order_022.md` による実験2（CIFAR-10・AlexNet，非凸設定）の実装

`.orders/order_022.md` は，実験0・実験1で得られた知見（ASAI SVRGの近似誤差がNFG SVRGより
一貫して小さいこと，内部ループ長 $ K $ と誤差床の恒久性の関係，学習率依存の不安定化）を，
理論解析の仮定を満たさない非凸かつ実践的な設定（CIFAR-10画像分類，AlexNetをCIFAR向けに
縮小したCNN）で検証するよう指示している．SVRG系手法との理論的整合性（補正勾配の再現性）を
保つため，BatchNormalization・Dropout・データ拡張は用いない．バッチサイズ（8, 32, 128）×
学習率（0.01, 0.001, 0.0001）×4手法×5Seed = 180条件のグリッドを実施した．GPUが必要なため
`.venv_pytorch_gpu`（1.3.1節参照）を新設した．結果として，NFG SVRG・ASAI SVRGは9条件中7条件
（学習率0.01の全3条件，学習率0.001のバッチサイズ8）で大半または全てのSeedが学習の途中で発散
（NaN）し，発散が生じなかった2条件（バッチサイズ32・128，学習率0.001）でのみASAI SVRGの
近似誤差がNFG SVRGより一貫して小さいことを確認できた．また，実験1で提示した「$ K $ が大きい
ほど誤差が自己減衰する」という仮説は非凸・ミニバッチ設定では逆転し，$ K $ が最も大きい
（バッチサイズが最も小さい）条件が最も不安定であった．詳細は`.reports/report_022.md`を参照．

#### 1.3.1 GPU環境の構築（`.venv_pytorch_gpu`）

本マシンのGPU（NVIDIA RTX PRO 6000 Blackwell，CUDA capability sm_120）は，実験0・1が用いる
`.venv_pytorch`のPyTorch 2.5.1+cu124（対応compute capability sm_50〜sm_90）では認識されるが
実行時カーネルが存在せず使用できなかった（`CUDA error: no kernel image is available for
execution on the device`）．Blackwell世代に対応するcu128ビルド（torch 2.11.0）を，既存の
`.venv_pytorch`を変更せず新規環境`.venv_pytorch_gpu`として追加することで解決した．詳細は
5.1節を参照．

### 1.4 `.orders/order_023.md` による実験ex0021（正規化層・バッチサイズの構造的変更）の実装

`.orders/order_023.md` は，実験2でNFG SVRG・ASAI SVRGが大半の条件で発散した結果を受け，
学習率の再探索ではなく，(1) サンプル単位で完結する正規化層（LayerNorm・GroupNorm）の追加，
(2) バッチサイズの拡大（128, 256, 512）という構造的変更により発散を緩和できるかを検証する
よう指示している．学習率は実験2で唯一発散が生じなかった $ \eta=0.001 $ に固定し，比較手法は
実験2で安定性が確認済みのSGD・SVRGを除きNFG SVRG・ASAI SVRGの2手法に絞る．バッチサイズ
間で総イテレーション数（epoch×K）を揃えるため，バッチサイズごとにエポック数を調整する
（128：12エポック，256：24エポック，512：48エポック）．正規化層3種×バッチサイズ3種×
手法2種×5Seed = 90条件のグリッドを実施した．結果は顕著であった：**90条件全てで発散
（NaN）が完全に解消**され，正規化層なしでは両手法ともチャンスレベル（約10%）にとどまる
一方，LayerNorm・GroupNormの追加によりASAI SVRGは全バッチサイズで60〜63%まで学習が進んだ
（NFG SVRGは15〜34%にとどまる）．正規化層の有無を要因分離すると，発散の解消と学習の進行は
主として正規化層の追加によるものであり，バッチサイズの拡大単独では寄与しないことが判明した．
また，NFG SVRGは学習後半で近似誤差が急増し分類精度が崩壊する（発散はしない）現象が全ての
バッチサイズで一貫して観察され，ASAI SVRGの誤差床がNFG SVRG以下であるという定理2の主張が
90条件全てで再確認された．詳細は`.reports/report_023.md`を参照．

### 1.5 `.orders/order_024.md` によるスペクトル正規化の追加検証

`.reports/report_023.md` 7.2節は，LayerNorm・GroupNormが発散を防いだ理由を「活性化のスケール
安定化」と推測していたが未検証であった．`.orders/order_024.md` は，メカニズムが異なる
スペクトル正規化（Miyato et al., 2018，重み行列の最大特異値を制約し層のリプシッツ定数を
直接抑える）を正規化層の第4パターンとして追加することで，「活性化スケールの安定化」と
「リプシッツ定数の抑制」のどちらが発散抑制の本質かを切り分けるよう指示している．
`AlexNetCIFARNorm` に `SpectralConv2d` を追加し，既存90条件に2手法×3バッチサイズ×5Seed=
30条件を追加した（合計120条件）．

power iterationの補助バッファ（$ u, v $）がforwardのたびに更新される標準実装は，SVRG系
手法の理論的前提（スナップショット $ z_s $ に対する勾配が同一の $ n $ に対して常に同じ値を
返すこと）を破るため，`SpectralConv2d` に `frozen` フラグを実装し，スナップショット専用
モデルの重み更新直後に20回のpower iteration（バーンイン，連続する反復間の相対変化が
$ 10^{-3} $ 程度に収束することを確認済み）でバッファを確定・凍結する設計とした．この
決定論性は単体テストで直接検証している．

結果は「発散の防止」と「高精度への到達」で異なる結論を示した．**SpectralNorm条件も30条件
全てで発散は完全に解消**され，リプシッツ定数の抑制のみで発散防止には十分であることが確認
された．一方，**分類精度はLayerNorm・GroupNorm（ASAI SVRGで60〜63%）に対しSpectralNormでは
18〜20%にとどまり大幅に劣る**．さらに特筆すべき点として，LayerNorm・GroupNormで顕著だった
ASAI SVRGのNFG SVRGに対する優位性（29〜45ポイント差）が，SpectralNorm条件ではほぼ消失した
（差0.1〜0.2ポイント）．近似誤差の時系列を調べると，NFG SVRGの近似誤差が学習後半に増大する
現象自体はSpectralNorm条件でも再現するが，到達するスケールが$10^{-3}$程度にとどまり
（LayerNorm・GroupNormでは$10^0\sim10^2$），精度崩壊を引き起こすには至らない．これは，
ASAI SVRGの優位性が「近似誤差が大きく爆発しうる状況でその爆発を食い止める」という点に由来
しており，SpectralNormのように爆発の余地自体を構造的に排除する設定では両手法を区別する
材料がなくなるためと解釈できる．この結果を踏まえ，`.reports/report_023.md` 7.6節で提案した
後続実験（4手法比較）の正規化層はGroupNormのまま変更しないことを結論した．詳細は
`.reports/report_024.md`（`report_023.md`の実験を拡張する追加検証として位置づけ）を参照．

### 1.6 `.orders/order_025.md` による実験ex0022（GroupNorm下での4手法比較）

実験2・ex0021・ex0022はいずれもNFG SVRG・ASAI SVRGの2手法比較にとどまり，実験2の終了基準4
「分散削減の利点（同一オラクル呼び出し回数でのより高い精度）」の検証（＝SGD・SVRGとの比較）
は未実施のまま残っていた．`.orders/order_025.md` は，GroupNorm固定のもとSGD・SVRG・
NFG SVRG・ASAI SVRGの4手法をバッチサイズ（512, 128, 32）・学習率（0.01, 0.001）のグリッドで
比較するよう指示している．`programs/ex0022_cifar10_alexnet_groupnorm/` に，
`programs/ex0021_cifar10_alexnet_norm/` を複製する形で実装し，学習率0.001・GroupNorm・
バッチサイズ512, 128のNFG SVRG・ASAI SVRG，計20条件はex0021の結果をディレクトリコピーで
再利用した（初期値の再現性は単体テストで確認済み）．

実装着手前に，複製元の実験2・ex0021が`.orders/order_022.md`・`.orders/order_023.md`の指示
（`SVRGFinalPoint`／`NFGSVRGFinalPoint`の使用，NFG SVRG・ASAI SVRGの診断専用フル勾配計算を
`elapsed_time`から除外）と実際の実装（`SVRG`／`NFGSVRG`を使用，診断専用フル勾配計算時間を
`elapsed_time`に含めていた）の間に食い違いがあることを発見し，ユーザーに確認の上，既存2実験
との一貫性を優先する方針（前者は`SVRG`／`NFGSVRG`に統一，後者は本実験から修正）で対応した．

結果は，SGD・SVRGが全120条件で発散なく高精度（56〜77%）に到達する一方，**NFG SVRGはほぼ
全条件で近似誤差の急増による精度崩壊を示し**，**ASAI SVRGは条件依存で挙動が大きく異なる**
ことを明らかにした．バッチサイズ512（両学習率）・バッチサイズ128（学習率0.001）では，
ASAI SVRGは同一オラクル呼び出し回数のもとで一貫してSVRGを上回り（分散削減の効率性の
実証），バッチサイズ512・学習率0.01ではSGDにほぼ匹敵する精度（75.9%）に達した．しかし
**バッチサイズ128・学習率0.01では5Seed中4Seedが発散**し，**バッチサイズ32では両学習率で
SVRGに対し劣位**となった．NFG SVRGで確認されてきた「近似誤差爆発による精度崩壊」パターンは
ASAI SVRGにも条件次第で生じることが本実験で初めて確認された．詳細は`.reports/report_025.md`
を参照．

### 1.7 `.orders/order_026.md` による実験ex0023（GroupNorm・学習率0.001固定での長期エポック学習）

ex0022は，学習率0.001では全条件で発散が生じないことを示したが，バッチサイズ32では
ASAI SVRGがSVRGに劣位となることも報告していた．`.orders/order_026.md` は，学習率0.01を
対象から外し学習率0.001に絞った上で，エポック数をex0022比で約4倍（バッチサイズ512, 128, 32
についてそれぞれ192, 48, 12エポック）に延長し，近似誤差・分類精度がプラトーに達するまで
観察するよう指示している．`programs/ex0023_cifar10_alexnet_groupnorm_longrun/` に
`programs/ex0022_cifar10_alexnet_groupnorm/` を複製する形で実装した．ex0022の内部
Optimizer状態（running average等）はディスクに保存されていないため学習を再開できず，
全60条件（4手法×3バッチサイズ×5Seed）をゼロから再学習した．

結果は，**バッチサイズ512, 128ではASAI SVRGの精度が明確にプラトーに達しSGD・SVRGとほぼ
同水準（74%前後）まで収束**し，実験1（`.reports/report_021.md`）で確認された「誤差床は
transientである」という強凸設定での知見が非凸CNN設定でも成立することを示した．一方，
**NFG SVRGは192エポックに至っても一度もプラトーに達せず，32〜47%の範囲で慢性的に振動**し
続け，恒久的な不安定性を示した．さらに重要な新規知見として，**バッチサイズ32ではNFG SVRG・
ASAI SVRGの全5Seedが学習開始から早い段階（エポック2〜3）で近似誤差爆発を起こし，分類精度が
チャンスレベルに固定される恒久的な崩壊状態に陥った**．訓練損失は非有限値化しなかったため
NaN検知による早期打ち切りは作動せず，「損失は有限のままモデルが機能不全に陥る」という
見えにくい崩壊モードを新たに確認した．ex0022でバッチサイズ32・学習率0.001が示した
「劣位」（NFG 43.2%，ASAI 50.8%，3エポックで打ち切り）は，実際にはこの崩壊が始まる直前で
たまたま観測を止めていた結果であったことが判明した．詳細は`.reports/report_026.md`を参照．

### 1.8 `.orders/order_027.md` による実験3（Tiny Shakespeare・Transformer）Stage A

`.orders/order_020.md` が定める実験3（Tiny Shakespeare，Transformer）に着手する．
SVRG系分散削減手法が系列データ（自然言語）にも適用可能かを検証するとともに，実験0〜
ex0023の知見がTransformerアーキテクチャでも成立するかを確認する．実験2が正規化層なしの
モデルでいきなり本比較を実施し9条件中7条件で発散したという教訓（`.reports/report_022.md`）
を踏まえ，`.orders/order_027.md` は最初から段階的に実施するよう指示している．Stage A
（本レポート対応）はNFG SVRG・ASAI SVRGの2手法に絞った安定性探索，Stage B（別途オーダー
作成予定）はSGD・SVRGを含めた4手法比較・長期学習である．

`programs/ex003_tinyshakespeare_transformer/` に，文字レベル言語モデリング用の
Decoder-only Transformer（4層，隠れ次元128，パラメータ数826,433）を実装した．Dropout・
BatchNormalizationは使用せず，LayerNorm（Pre-LN構成，各トークンごとに正規化されるため
SVRG系手法の理論的前提と両立する）を用いる．データセットはTiny Shakespeare（1,115,394
文字，語彙サイズ65）を，固定長128トークンの非重複チャンク（学習用7,781チャンク，検証用
864チャンク）に分割する．

Stage A（NFG SVRG・ASAI SVRG×バッチサイズ{512,128,32}×学習率{0.01,0.001}×3Seed，計36
条件，12エポック）の結果，**36条件全てで発散（NaN）が皆無**であった．これは
正規化層なしで実施した実験2の初回グリッド（9条件中7条件が発散）と著しく対照的であり，
LayerNormを最初から採用したことが主因と考えられる．近似誤差の爆発パターン（実験2以降
繰り返し観察）や，ex0023で確認された「損失は有限のままモデルが機能不全に陥る」見えない
崩壊も，本Stage Aでは一切観察されなかった．この結果を踏まえ，Stage Bの条件案
（バッチサイズ128・32を中心に，学習率0.01を中心としつつより大きい学習率での安定性限界の
探索も検討，エポック数をex0023同等の総イテレーション数まで延長）を提案した．詳細は
`.reports/report_027.md`を参照．

### 1.9 `.orders/order_028.md` による実験3 Stage B（SGD・SVRGを含めた4手法比較）

Stage Aと完全に同一のモデル・データセット・グリッドのまま，SGD・SVRGを比較対象に加えた
4手法比較を行う．NFG SVRG・ASAI SVRGの36条件はStage Aの結果をそのまま再利用し（`is_run_
completed` による自動スキップ），`programs/ex003_tinyshakespeare_transformer/train.py` に
SGD・SVRGの学習ループ（`run_sgd`，`run_variance_reduced`のSVRG分岐）を追加して新規36条件
のみを学習した（計72条件）．

結果は，**SGD・SVRGを含めた全72条件で発散が皆無**であり，Stage Aの安定性がそのまま
維持されることを確認した．オラクル呼び出し回数を横軸とした効率性比較では，**探索した
全6条件（バッチサイズ3種×学習率2種）でASAI SVRGが一貫してSVRGを上回り**（ex0022，
`.reports/report_025.md`で確認されたGroupNorm下のCIFAR-10での知見がTransformer・系列
データでも成立することを実証），特にバッチサイズ128・学習率0.001では約1.49倍の精度差と
なった．一方，12エポックでは全条件で精度がまだ収束しておらず，長期学習（Stage C相当）の
実施を推奨する結論に至った．詳細は`.reports/report_028.md`を参照．

### 1.10 `.orders/order_029.md` による実験3 Stage C（長期学習によるASAI SVRGの効率性優位性の持続性検証）

Stage Bで観察されたASAI SVRGの効率性優位性（最大約1.49倍）が学習途中の値に基づくもので
あったこと，およびex0023で確認された遅延崩壊がTransformer設定でも起こりうることを踏まえ，
学習率を0.01に絞った上でエポック数をStage Bの4倍（48エポック）に延長した長期学習を実施
した．`.orders/order_029.md` 末尾の指示により，エポック数を区別しやすくするため実験
ディレクトリ名を`ex0031_tinyshakespeare_transformer_longrun`とした（Stage A/Bの
`ex003_tinyshakespeare_transformer`とは別ディレクトリ）．モデル・データセットはStage A/Bと
完全に同一，全36条件（バッチサイズ3種×手法4種×Seed3）を新規に学習した（Optimizer内部
状態非保存のため継続学習は不可，ex0023と同様）．

結果は，**36条件全てで発散・遅延崩壊が皆無**であり，特にバッチサイズ32のNFG SVRG・
ASAI SVRGの精度軌跡は48エポックを通じて滑らかに単調増加し，ex0023（CIFAR-10）で観察
された「損失は有限のまま精度がチャンスレベルに固定される」崩壊は一切再現しなかった．
オラクル呼び出し回数を揃えたASAI SVRG対SVRGの精度比は，**Stage B終了時点（12エポック
相当）で最大約1.25倍（バッチサイズ512）に達した後，学習を継続すると急速に縮小し，以降は
バッチサイズにより約1〜9%の小さいが正の水準で安定する**という推移を示した．すなわち，
Stage Bで観測された大きな優位性は学習曲線が急峻な初期を切り取ったことによる過大評価
だった可能性が高いが，優位性自体は消失せず，長期学習でも持続することが確認された．
エポック数を揃えた（コストを揃えない）比較では48エポック時点の4手法の最終精度はいずれの
バッチサイズでも0.05ポイント未満の差に収束しており，オラクル呼び出し回数という「コスト」
を揃えて初めて優位性が可視化される点はStage Bと同様であった．プラトー判定（末尾6エポック
の相対変化）により，48エポック時点でおおむねプラトーに到達していることを確認した．学習率
0.001での追加の長期学習は，主要な結論を左右しないと判断し見送ることを提案した．詳細は
`.reports/report_029.md`を参照．

### 1.11 `.orders/order_030.md` による実験4（WikiText-2・単語レベル言語モデリング）Stage A

実験3（Tiny Shakespeare，文字レベル）の長期学習（`report_029.md`）で，全手法の最終精度が
ほぼ同水準に収束するという結果が判明したことを受け，文字レベル・小規模コーパスというタスク
難度の低さが分散削減の恩恵を限定的にしていた可能性を検証するため，単語レベル言語モデリング
（WikiText-2，語彙サイズ約33,000）へ切り替えた．実験2・実験3と同様の教訓を踏まえ，Stage A
（安定性探索，NFG SVRG・ASAI SVRGの2手法のみ，12エポック）から段階的に実施した．

`programs/ex004_wikitext2_transformer/`に，PyTorch公式word language modelサンプルが配布
する前処理済みWikiText-2（学習用語彙サイズ33,277，$N_{\text{train}}=31{,}567$）を用いる
実装を追加した．モデルは実験3と同一のTransformer本体（4層，隠れ次元128）を維持しつつ，
系列長を$T=64$に短縮し，パラメータ数は約935万（Token Embedding・出力射影層で全体の91.4%）
となった．

実装直後の学習実行で，バッチサイズ512の条件（出力層のロジットテンソルが1プロセスあたり
約23.5GB要求）を8プロセス並列実行した際にGPU（WSL2のGPU仮想化層）が応答不能状態に陥る
事象が発生した．原因は，VRAM実測を最小バッチサイズ（32）でのみ行っており，最悪ケース
（バッチサイズ512）での実測を怠っていたことであった．並列数を2に制限することで解決し，
以降は正常に学習が完了した（詳細・教訓は`.reports/report_030.md` 5.3節を参照）．

結果は，**36条件全てで発散（NaN）は皆無**であった．一方，バッチサイズ512・学習率0.001の
条件（両手法とも全3Seed）は，チャンスレベル張り付き判定に該当したが，精度が単調に増加して
おり崩壊ではなく学習不足と判断した．学習実行中に**損失値が交差エントロピー損失ではなく
L2正則化項（Token Embeddingの初期スケールに起因）に支配されている**ことが判明し，
正則化係数$\lambda=5\times10^{-4}$（実験2・実験3から不変）が本実験のEmbedding層規模には
過大であることが分かった．近似誤差の絶対水準は実験3と比べて多くの条件で同等かむしろ小さく，
「タスクが簡単すぎた」という仮説を直接支持する結果は得られなかった．Stage Bの実施前に
正則化係数の見直しが必須であるとの結論に至った．詳細は`.reports/report_030.md`を参照．

### 1.12 `.orders/order_031.md` による実験4b（ex0041，正則化修正後のStage A再実行）

実験4 Stage A（`report_030.md` 8.3節）で判明した「訓練損失がL2正則化項に支配され交差
エントロピー損失の寄与が視認できない」問題を修正した上で，実験4 Stage Aと同一のグリッド
（NFG SVRG・ASAI SVRG，バッチサイズ3種×学習率2種×3Seed，12エポック，36条件）を
`programs/ex0041_wikitext2_transformer_fixedreg/`として再実行した．

正則化係数$\lambda=5\times10^{-4}$自体は実験2〜4から変更せず，代わりにToken Embedding層・
出力射影層の初期化を$\mathcal N(0,\,1/\sqrt d)$（$d=128$）へ変更した．修正後のモデルの
初期状態でのL2正則化項は約19.0であり，チャンスレベル交差エントロピー理論値（約10.4）の
1.83倍に収まることを学習開始前に検証した（`verify_regularization_is_not_dominant`関数）．
`report_030.md` 5.3節の教訓（VRAM実測は最悪ケースのバッチサイズで行うこと）を踏まえ，
最初から最大バッチサイズ（512）でVRAM使用量を実測し（約23.5GB，実験4と同一），
`num_workers=2`を採用した結果，実験4で発生したGPU応答不能事象は再発しなかった．

結果は，**36条件全てで発散・チャンスレベル張り付きが皆無**であった．訓練損失の絶対値は
修正前（969〜1080）から修正後（約24.5〜29.2）へ大幅に縮小し，交差エントロピー損失を反映
する妥当な値になった．**修正の効果が最も劇的に現れたのはバッチサイズ512・学習率0.001
条件**で，修正前は「学習不足」（チャンスレベル付近に張り付き）だった最終精度が，修正後は
約67〜71倍に改善し（0.0005→0.0357/0.0334），エポック6付近から明確な加速（S字型学習曲線）
が観察された．また，バッチサイズ128・学習率0.001条件で報告されていた近似誤差の異常な増大
（実験3比10〜50倍）も，修正後は他条件と比べて突出しない水準に解消した．一方，学習率0.01
条件では，修正後の最終精度が修正前よりもやや低下するという副次的な現象が観察され，これは
修正前の過大な初期化がたまたま短期的な学習に有利な多様な初期表現を与えていた可能性が
考えられるとして考察した．詳細は`.reports/report_031.md`を参照．

### 1.13 `.orders/order_033.md` による実験5（Imagewoofを用いたResNet18画像分類）Stage A

より本物のImageNet画像に近い，高解像度（$224\times224$）かつ意味的に難しい分類タスク
での検証として，Imagewoof（fast.aiが配布する，ImageNet-1kから抽出した犬種10クラスの
サブセット，$N_{\text{train}}=9025$，$N_{\text{test}}=3929$）・ResNet18を用いた
`programs/ex005_imagewoof_resnet18/`を実装した．正規化層は，本論文の2つの非凸実践設定
（CIFAR-10とImagewoof）全体でLayerNormを標準として報告するため，CIFAR-10系列（GroupNorm）
とは異なり**LayerNormに統一**した．`torchvision.models.resnet18(weights=None,
num_classes=10)`の全20層のBatchNorm2dをLayerNorm2d（ex0021由来の実装）に置換する方式を
採用し，パラメータ数（11,181,642）が標準BatchNorm版ResNet18と完全一致することを確認した．

実行前の計算コスト見積もり（`report_030.md` 5.3節の教訓に従い，VRAM実測は最大バッチサイズ
128で実施）では約1時間程度と概算したが，8プロセス並列実行時のGPU資源競合により実測は
約2.5時間を要した．NFG SVRG・ASAI SVRGの2手法，バッチサイズ128/64/32・学習率0.01/0.001・
3Seed，12エポックの36条件（Stage A）を実行した結果，**学習率0.001の18条件は全て非崩壊**
であった一方，**学習率0.01ではNFG SVRGが全バッチサイズで崩壊し，ASAI SVRGはバッチサイズ
128でのみ3Seed中2Seedが崩壊を免れる**という，学習率を主要因としつつバッチサイズ・手法の
組み合わせに依存する崩壊境界が観察された．また，学習率0.001の条件では，実験3
（Tiny Shakespeare）で見られた「団子」現象とは異なり，バッチサイズが小さいほどNFG SVRGと
ASAI SVRGの精度差が拡大する傾向（バッチサイズ32で+19.4ポイント）が確認された．詳細は
`.reports/report_033.md`を参照．

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
│   ├── ex002_cifar10_alexnet/
│   │   └── raw/                        # CIFAR-10データセットの生データ（自動ダウンロード）
│   │                                    # ★ order_022の論文掲載用の実験2
│   ├── ex0021_cifar10_alexnet_norm/，ex0022_cifar10_alexnet_groupnorm/，
│   │   ex0023_cifar10_alexnet_groupnorm_longrun/
│   │   └── raw/                        # CIFAR-10データセットの生データ（自動ダウンロード，
│   │                                    # または既存実験からコピー）
│   ├── ex003_tinyshakespeare_transformer/，ex0031_tinyshakespeare_transformer_longrun/
│   │   └── raw/                        # Tiny Shakespeareの生データ（input.txt，自動ダウン
│   │                                    # ロード，またはex003からコピー）．
│   │                                    # ★ order_027/028/029の論文掲載用の実験3
│   ├── ex004_wikitext2_transformer/，ex0041_wikitext2_transformer_fixedreg/
│   │   └── raw/                        # WikiText-2の生データ（train.txt/valid.txt/test.txt，
│   │                                    # PyTorch公式word_language_modelサンプルより自動
│   │                                    # ダウンロード，またはex004からコピー）．
│   │                                    # ★ order_030/031の論文掲載用の実験4
│   ├── ex005_imagewoof_resnet18/
│   │   └── raw/                        # Imagewoof2-320の生データ（imagewoof2-320.tgz，
│   │                                    # fast.aiより自動ダウンロード）とchannel_stats.json
│   │                                    # （チャネル統計量のキャッシュ）．★ order_033の
│   │                                    # 論文掲載用の実験5
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
│   ├── ex001_mushroom_logistic/        # 実験1：Mushroomの二値分類（BCE+L2正則化，強凸）
│   │   ├── data.py                     # マッシュルームデータセット（順序符号化+標準化，
│   │   │                                # N=8124, d=22）の取得・前処理・DataLoader構築
│   │   ├── model.py                    # L2正則化付きロジスティック回帰（切片含む）．正則化は
│   │   │                                # 重み・切片の両方に課し，全パラメータの mu-強凸性を
│   │   │                                # 厳密に満たす（事前実験 ex001_mushroom_svrg との差異）
│   │   └── train.py                    # 4手法 x 3学習率 x 5Seed = 60条件の学習を実行する
│   │                                    # スクリプト（本体，order_021）．L・lambda・収縮係数
│   │                                    # rhoの数値的な逆算（solve_eta_for_target_rho）を含む
│   ├── ex002_cifar10_alexnet/          # 実験2：CIFAR-10の多値分類（AlexNet縮小版，非凸）
│   │   ├── data.py                     # CIFAR-10（公式の学習・検証分割，データ拡張なし）の
│   │   │                                # 取得・前処理・DataLoader構築
│   │   ├── model.py                    # AlexNetCIFAR（5畳み込み層+3全結合層，BN・Dropoutなし，
│   │   │                                # 約717万パラメータ）．L2正則化は重みのみに課す
│   │   └── train.py                    # 4手法 x 3バッチサイズ x 3学習率 x 5Seed = 180条件の
│   │                                    # 学習を実行するスクリプト（本体，order_022）．
│   │                                    # `machine_learning.md` のiteration/epoch関数構成に
│   │                                    # 従い，snapshot_model引数で拡張．GPU使用，8プロセス
│   │                                    # 並列（chunksize=1で動的負荷分散）
│   └── ex0021_cifar10_alexnet_norm/    # 実験ex0021：正規化層・バッチサイズの構造的変更
│       ├── data.py                     # CIFAR-10（ex002と同一の前処理）の取得・DataLoader構築
│       ├── model.py                    # AlexNetCIFARNorm．正規化層（なし/LayerNorm2d/
│       │                                # GroupNorm）を選択可能．いずれもサンプル単位で完結
│       └── train.py                    # 2手法（NFG SVRG，ASAI SVRG）x 3バッチサイズ
│                                        # (128,256,512) x 3正規化層 x 5Seed = 90条件の学習を
│                                        # 実行するスクリプト（本体，order_023）．バッチサイズ
│                                        # ごとに総イテレーション数が揃うようエポック数を調整
│                                        # （128:12, 256:24, 512:48）．正規化層の第4パターン
│                                        # SpectralConv2d（order_024）も含む
│   ├── ex0022_cifar10_alexnet_groupnorm/  # 実験ex0022：GroupNorm下での4手法比較
│   │   ├── data.py                     # ex0021と同一（EXPERIMENT_NAMEのみ変更）
│   │   ├── model.py                    # ex0021と同一（GroupNorm固定で使用）
│   │   └── train.py                    # 4手法（SGD, SVRG, NFG SVRG, ASAI SVRG）x 3バッチ
│   │                                    # サイズ(512,128,32) x 2学習率(0.01,0.001) x 5Seed =
│   │                                    # 120条件の学習を実行するスクリプト（本体，order_025）．
│   │                                    # うち学習率0.001・バッチサイズ512,128のNFG SVRG・
│   │                                    # ASAI SVRG計20条件はex0021の結果をコピーして再利用
│   │                                    # （reuse_ex0021_results関数）
│   ├── ex0023_cifar10_alexnet_groupnorm_longrun/  # 実験ex0023：学習率0.001固定・長期学習
│   │   ├── data.py                     # ex0022と同一（EXPERIMENT_NAMEのみ変更）
│   │   ├── model.py                    # ex0022と同一
│   │   └── train.py                    # 4手法 x 3バッチサイズ(512,128,32) x 5Seed = 60条件
│   │                                    # （学習率0.001固定，本体，order_026）．エポック数は
│   │                                    # ex0022比で約4倍（512:192, 128:48, 32:12）．
│   │                                    # `compute_trailing_relative_change`関数でプラトー
│   │                                    # 判定を行う．訓練損失が非有限値化した場合は学習を
│   │                                    # 打ち切る（is_run_completedもこれを完了済みとして扱う）
│   ├── ex003_tinyshakespeare_transformer/  # 実験3 Stage A・Stage B：Tiny Shakespeare・
│       │                                # Transformer
│       ├── data.py                     # Tiny Shakespeare（自動ダウンロード）を文字レベルで
│       │                                # トークナイズし，非重複チャンク（T=128）へ分割
│       ├── model.py                    # DecoderOnlyTransformer（4層，隠れ次元128，
│       │                                # 826,433パラメータ）．Dropout・BatchNorm不使用，
│       │                                # LayerNorm（Pre-LN）・Causalマスク・学習可能な
│       │                                # 位置埋め込みを使用
│       └── train.py                    # 4手法（SGD, SVRG, NFG SVRG, ASAI SVRG）x 3バッチ
│                                        # サイズ(512,128,32) x 2学習率(0.01,0.001) x 3Seed =
│                                        # 72条件を実行するスクリプト（Stage A：NFG SVRG・
│                                        # ASAI SVRG 36条件，order_027／Stage B：SGD・SVRG
│                                        # 36条件を追加，order_028）．Stage Aの36条件は
│                                        # `is_run_completed` により自動的に再利用（スキップ）
│                                        # される．`is_stuck_near_chance`関数で見えない崩壊を検出
│   ├── ex0031_tinyshakespeare_transformer_longrun/  # 実験3 Stage C：学習率0.01固定・
│       │                                # 48エポックへの長期学習（エポック数を区別しやすい
│       │                                # よう order_029末尾の指示でex0031と命名）
│       ├── data.py                     # ex003と同一（EXPERIMENT_NAMEのみ変更）
│       ├── model.py                    # ex003と同一
│       └── train.py                    # 4手法 x 3バッチサイズ(512,128,32) x 3Seed = 36条件
│                                        # （学習率0.01固定・48エポック一律，本体，order_029）．
│                                        # ex003 Stage Bの4手法学習ロジックと，ex0023由来の
│                                        # `compute_trailing_relative_change`（プラトー判定）・
│                                        # 崩壊時の学習打ち切りロジックを組み合わせる．
│                                        # 全36条件を新規学習（継続学習は不可のため）
│   ├── ex004_wikitext2_transformer/    # 実験4 Stage A：WikiText-2・単語レベル言語モデリング
│       ├── data.py                     # WikiText-2（PyTorch公式word_language_modelサンプル
│       │                                # より自動ダウンロード）を単語レベルでトークナイズし，
│       │                                # 非重複チャンク（T=64）へ分割．公式train/valid分割
│       │                                # をそのまま使用
│       ├── model.py                    # ex003と同一のTransformer本体（4層，隠れ次元128）．
│       │                                # vocab_size=33277によりパラメータ数は約935万
│       │                                # （Token Embedding・出力射影層が全体の91.4%）
│       └── train.py                    # 2手法（NFG SVRG, ASAI SVRG）x 3バッチサイズ
│                                        # (512,128,32) x 2学習率(0.01,0.001) x 3Seed = 36条件
│                                        # を実行するスクリプト（Stage A，order_030）．
│                                        # `is_stuck_near_chance`の許容幅を本実験の語彙サイズ
│                                        # 向けに再調整（0.001）．`num_workers=2`（バッチ
│                                        # サイズ512のVRAM実測に基づく，report_030.md 5.3節）
│   └── ex0041_wikitext2_transformer_fixedreg/  # 実験4b：正則化修正後のStage A再実行
│       ├── data.py                     # ex004と同一（EXPERIMENT_NAMEのみ変更）
│       ├── model.py                    # ex004と同一のTransformer本体だが，Token Embedding・
│       │                                # 出力射影層の初期化を$\mathcal N(0,1/\sqrt d)$
│       │                                # （$d=128$）へ修正（order_031 3.2節）
│       └── train.py                    # ex004と同一グリッド（2手法 x 3バッチサイズ x
│                                        # 2学習率 x 3Seed = 36条件，Stage A再実行，
│                                        # order_031）．学習開始前に
│                                        # `verify_regularization_is_not_dominant`関数で
│                                        # 正則化項がチャンスレベル交差エントロピーの10倍を
│                                        # 超えないことを検証．最大バッチサイズ（512）で
│                                        # VRAM実測を行った上で`num_workers=2`を採用
│   └── ex005_imagewoof_resnet18/       # 実験5 Stage A：Imagewoof・ResNet18画像分類
│       ├── data.py                     # Imagewoof2-320（fast.aiより自動ダウンロード，
│       │                                # $N_{\text{train}}=9025$，$N_{\text{test}}=3929$）．
│       │                                # Resize(256)+CenterCrop(224)のみの決定論的前処理，
│       │                                # チャネル統計量は学習データから算出しキャッシュ
│       ├── model.py                    # ResNet18LayerNorm．torchvision公式ResNet18の全20層
│       │                                # のBatchNorm2dをLayerNorm2d（ex0021由来）に置換．
│       │                                # パラメータ数11,181,642（標準BatchNorm版と完全一致）
│       └── train.py                    # 2手法（NFG SVRG, ASAI SVRG）x 3バッチサイズ
│                                        # (128,64,32) x 2学習率(0.01,0.001) x 3Seed = 36条件
│                                        # を実行するスクリプト（Stage A，order_033）．
│                                        # `is_stuck_near_chance`の許容幅は10クラス設定向けに
│                                        # 0.03．最大バッチサイズ（128）でVRAM実測（約6.1GB）
│                                        # を行った上で`num_workers=8`を採用
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
│   ├── ex001_mushroom_logistic/
│   │   ├── {method}/{hyperparams}/{seed}/
│   │   │   ├── log.json                # ResultLoggerによる評価指標の履歴（objective_gap等）
│   │   │   └── config.json             # L・lambda・rho等の理論的な根拠を含むメタデータ
│   │   ├── comparison_all_axes_{eta}.png               # 3横軸 × 3評価指標の比較
│   │   ├── accuracy_vs_epoch.png                       # 分類精度の推移
│   │   └── final_epoch_summary.md                      # 最終エポックの評価指標の表
│   ├── ex002_cifar10_alexnet/
│   │   ├── {method}/{lr,bs,lambda,epochs}/{seed}/
│   │   │   ├── log.json                # ResultLoggerによる評価指標の履歴（train_loss，
│   │   │   │                            # test_accuracy，approx_error等．発散した場合はNaN）
│   │   │   ├── config.json             # バッチサイズ・学習率・K等のメタデータ
│   │   │   └── best_model.pth          # 検証精度が最高となったエポックの重み
│   │   ├── accuracy_grid.png                     # 分類精度の推移（縦：バッチサイズ，横：学習率）
│   │   ├── approx_error_grid.png                 # 近似誤差の推移（同上，NFG・ASAIのみ）
│   │   ├── approx_error_stable_conditions.png    # 発散が生じなかった2条件での近似誤差比較
│   │   └── grid_summary.md                       # 全180条件の最終・最高精度，発散Seed数の表
│   ├── ex0021_cifar10_alexnet_norm/
│   │   └── {method}/{lr,bs,norm,lambda,epochs}/{seed}/
│   │       ├── log.json                # ResultLoggerによる評価指標の履歴
│   │       ├── config.json             # バッチサイズ・正規化層・K・総イテレーション数等
│   │       └── best_model.pth          # 検証精度が最高となったエポックの重み
│   ├── ex0022_cifar10_alexnet_groupnorm/
│   │   └── {method}/{lr,bs,norm,lambda,epochs}/{seed}/
│   │       ├── log.json                # ResultLoggerによる評価指標の履歴（コピー再利用分は
│   │       │                            # ex0021からそのままコピーされたもの）
│   │       ├── config.json             # 再利用分は experiment/reused_from フィールドを追記
│   │       └── best_model.pth          # 検証精度が最高となったエポックの重み
│   ├── ex0023_cifar10_alexnet_groupnorm_longrun/
│   │   └── {method}/{lr,bs,norm,lambda,epochs}/{seed}/
│   │       ├── log.json                # ResultLoggerによる評価指標の履歴．崩壊により打ち
│   │       │                            # 切られた場合は記録数がepochs+1未満
│   │       ├── config.json             # collapsedフィールドで崩壊の有無を明示
│   │       └── best_model.pth          # 検証精度が最高となったエポックの重み
│   ├── ex003_tinyshakespeare_transformer/
│   │   └── {method}/{lr,bs,lambda,epochs}/{seed}/
│   │       ├── log.json                # ResultLoggerによる評価指標の履歴（次文字予測精度，
│   │       │                            # 近似誤差等）
│   │       ├── config.json             # collapsedフィールド，vocab_size等のメタデータ
│   │       └── best_model.pth          # 検証精度が最高となったエポックの重み
│   ├── ex0031_tinyshakespeare_transformer_longrun/
│   │   └── {method}/{lr,bs,lambda,epochs}/{seed}/
│   │       ├── log.json                # ResultLoggerによる評価指標の履歴（49エントリ，
│   │       │                            # epoch0〜48）
│   │       ├── config.json             # collapsedフィールド（全36条件でfalse）
│   │       └── best_model.pth          # 検証精度が最高となったエポックの重み
│   ├── ex004_wikitext2_transformer/
│   │   └── {method}/{lr,bs,lambda,epochs}/{seed}/
│   │       ├── log.json                # ResultLoggerによる評価指標の履歴（次単語予測精度，
│   │       │                            # 近似誤差等，13エントリ，epoch0〜12）
│   │       ├── config.json             # collapsedフィールド（全36条件でfalse），
│   │       │                            # sequence_length等のメタデータ
│   │       └── best_model.pth          # 検証精度が最高となったエポックの重み
│   ├── ex0041_wikitext2_transformer_fixedreg/
│   │   └── {method}/{lr,bs,lambda,epochs}/{seed}/
│   │       ├── log.json                # ResultLoggerによる評価指標の履歴（正則化修正後，
│   │       │                            # 13エントリ，epoch0〜12）
│   │       ├── config.json             # collapsedフィールド（全36条件でfalse）
│   │       └── best_model.pth          # 検証精度が最高となったエポックの重み
│   └── ex005_imagewoof_resnet18/
│       └── {method}/{lr,bs,lambda,epochs}/{seed}/
│           ├── log.json                # ResultLoggerによる評価指標の履歴（分類精度，
│           │                            # 近似誤差等，13エントリ，epoch0〜12）
│           ├── config.json             # collapsedフィールド（NaN発散のみ判定，全36条件で
│           │                            # false．チャンスレベル張り付きは別途手動判定，
│           │                            # report_033.md 8.1節参照），K・N_train等のメタデータ
│           └── best_model.pth          # 検証精度が最高となったエポックの重み
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
│   ├── test_ex001_mushroom_logistic.py # 実験1のモデル（重み・切片両方への正則化）・勾配・
│   │                                    # 平滑性定数・収縮係数rhoのU字形・rho逆算の妥当性・
│   │                                    # 4手法のスモークテスト・オラクル呼び出し回数．
│   │                                    # 実験0のテストとのモジュール名（model/data/train）
│   │                                    # 衝突をimportlibによる明示的な読み込みで回避
│   ├── test_ex002_cifar10_alexnet.py   # 実験2のモデル（BN・Dropoutの不在，L2正則化が
│   │                                    # 切片を除外すること）・4手法のスモークテスト・
│   │                                    # オラクル呼び出し回数．合成画像テンソルを用い
│   │                                    # CIFAR-10の実データダウンロードを伴わない
│   └── test_ex0021_cifar10_alexnet_norm.py  # 実験ex0021の正規化層（LayerNorm2d・GroupNorm）
│                                        # がサンプル単位で完結すること（他サンプルの値を
│                                        # 変えても出力が変化しない）の検証，L2正則化が
│                                        # 正規化層のアフィンパラメータを除外すること，
│                                        # 2手法のスモークテスト，バッチサイズごとの
│                                        # エポック数が総イテレーション数を揃えることの検証．
│                                        # SpectralConv2d（order_024）については，凍結後の
│                                        # 勾配評価が完全に決定論的であること（必須テスト），
│                                        # power iterationのバーンイン後の収束，L2正則化の
│                                        # 二重計上防止，パラメータ数の公平性を検証
│   ├── test_ex0022_cifar10_alexnet_groupnorm.py  # 実験ex0022．ex0021とのモデル初期値・
│   │                                    # データローダー初期順序の一致（コピー再利用の前提，
│   │                                    # order_025必須項目），4手法のスモークテスト，
│   │                                    # オラクル呼び出し回数（SGD:N, NFG/ASAI:2N, SVRG:3N），
│   │                                    # elapsed_timeからNFG/ASAIの診断専用フル勾配計算時間が
│   │                                    # 除外されること（人為的遅延によるモック検証）を確認
│   ├── test_ex0023_cifar10_alexnet_groupnorm_longrun.py  # 実験ex0023．4手法のスモーク
│   │                                    # テスト，オラクル呼び出し回数，elapsed_timeの
│   │                                    # 診断専用フル勾配計算除外（ex0022からの踏襲確認），
│   │                                    # compute_trailing_relative_change（プラトー判定）の
│   │                                    # 正しさ，学習率を極端に大きくして人為的に崩壊させた
│   │                                    # 場合に学習が打ち切られること，打ち切られたログを
│   │                                    # is_run_completedが完了済みとして扱うことを検証
│   ├── test_ex003_tinyshakespeare_transformer.py  # 実験3 Stage A・Stage B．Dropout・
│   │                                    # BatchNorm不使用の確認，モデルの決定論性，Causal
│   │                                    # マスクが未来のトークンに依存しないこと，チャンク
│   │                                    # 分割の非重複性，L2正則化がLayerNormを除外すること，
│   │                                    # 4手法のスモークテスト，オラクル呼び出し回数
│   │                                    # （SGD:N, NFG/ASAI:2N, SVRG:3N），elapsed_timeの
│   │                                    # 診断専用フル勾配計算除外，is_stuck_near_chance
│   │                                    # （見えない崩壊検出）の正しさ，Stage BのグリッドがStage
│   │                                    # Aと一致すること（既存結果再利用の前提）を検証
│   ├── test_ex0031_tinyshakespeare_transformer_longrun.py  # 実験3 Stage C．Transformer構造
│   │                                    # の性質（ex003と同様）に加え，compute_trailing_
│   │                                    # relative_change（プラトー判定，ex0023から踏襲）の
│   │                                    # 境界動作，崩壊時の学習早期打ち切り，打ち切られた
│   │                                    # ログをis_run_completedが完了済みとして扱うこと，
│   │                                    # EPOCHS==48・LEARNING_RATE==0.01等の実験条件定数を検証
│   ├── test_ex004_wikitext2_transformer.py  # 実験4 Stage A．Transformer構造の性質（ex003と
│   │                                    # 同様）に加え，語彙構築の決定論性，検証用テキストが
│   │                                    # 学習用語彙に対しOOVを持たないこと，パラメータ数が
│   │                                    # 見積もり（約935万）と概ね一致すること，2手法
│   │                                    # （NFG SVRG，ASAI SVRG）のスモークテスト，オラクル
│   │                                    # 呼び出し回数（2N），elapsed_timeの診断専用フル勾配
│   │                                    # 計算除外，is_stuck_near_chance（本実験用の許容幅
│   │                                    # 0.001）が学習進行中の精度・最頻出単語縮退の双方を
│   │                                    # 誤検知しないことを検証
│   ├── test_ex0041_wikitext2_transformer_fixedreg.py  # 実験4b．実験4と共通の検証項目に加え，
│   │                                    # Token Embedding・出力射影層が$\mathcal N(0,1/\sqrt d)$
│   │                                    # スケールで初期化されていること，初期状態でのL2正則化
│   │                                    # 項がチャンスレベル交差エントロピーの10倍を超えない
│   │                                    # こと，`verify_regularization_is_not_dominant`関数が
│   │                                    # 妥当な設定で合格し極端な語彙サイズでは正しく
│   │                                    # AssertionErrorを送出することを検証
│   └── test_ex005_imagewoof_resnet18.py  # 実験5 Stage A．全BatchNorm2d層がLayerNorm2d
│                                        # （20層）に置換され，Dropoutを含まないこと，出力の
│                                        # 決定論性，パラメータ数が標準ResNet18（BatchNorm版）
│                                        # と完全一致すること，L2正則化がConv2d・Linearの重み
│                                        # のみに課されること，2手法（NFG SVRG，ASAI SVRG）
│                                        # のスモークテスト，オラクル呼び出し回数（2N），
│                                        # elapsed_timeの診断専用フル勾配計算除外，
│                                        # is_stuck_near_chance（10クラス設定用の許容幅0.03）
│                                        # の正しさ，Stage Aのグリッドが実験条件と一致すること
│                                        # を検証
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
│                                        # full_grad_computationsとして記録）を用いる．
│                                        # 実験2（バッチサイズ×学習率の2軸グリッド）は単一軸を
│                                        # 前提とした既存セルでは可視化できないため，末尾に
│                                        # 専用セル（ex002_load_all等）を追加している
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
│   ├── report_021.md                   # 実験1：Mushroomの二値分類（強凸設定）による
│   │                                    # 定理1・定理2の定量的検証（order_021）
│   ├── report_022.md                   # 実験2：CIFAR-10・AlexNet（非凸設定）による
│   │                                    # ASAI SVRGの実用性検証，Kと誤差床の恒久性の
│   │                                    # 関係の検証（order_022）
│   ├── report_023.md                   # 実験ex0021：正規化層・バッチサイズの構造的変更に
│   │                                    # よるNFG SVRG・ASAI SVRGの安定性の検証（order_023）
│   ├── report_024.md                   # 実験ex0021 追加検証：スペクトル正規化によるメカニズム
│   │                                    # 解明（活性化正規化 vs リプシッツ定数抑制，order_024）
│   ├── report_025.md                   # 実験ex0022：GroupNorm下での4手法比較，分散削減の
│   │                                    # 効率性検証（order_025）．実験2・ex0021の実装上の
│   │                                    # 食い違い（SVRG系Optimizerクラスの選択，elapsed_time
│   │                                    # の計上方法）の発見・対応も記載
│   ├── report_026.md                   # 実験ex0023：GroupNorm・学習率0.001固定での長期
│   │                                    # エポック学習，誤差床の収束観察（order_026）．
│   │                                    # transient vs 恒久的誤差床の判定，バッチサイズ32での
│   │                                    # 新規崩壊の発見を記載
│   ├── report_027.md                   # 実験3 Stage A：Tiny Shakespeare・Transformerでの
│   │                                    # 安定性探索（order_027）．36条件全てで発散が皆無
│   │                                    # であった結果と考察，Stage Bの条件案を記載
│   ├── report_028.md                   # 実験3 Stage B：SGD・SVRGを含めた4手法比較
│   │                                    # （order_028）．全72条件で発散が皆無，ASAI SVRGが
│   │                                    # 全6条件でオラクル呼び出し回数あたりSVRGを上回る
│   │                                    # 結果と考察，Stage C（長期学習）実施の推奨を記載
│   ├── report_029.md                   # 実験3 Stage C：学習率0.01固定・48エポックへの長期
│   │                                    # 学習（order_029）．全36条件で発散・遅延崩壊が皆無，
│   │                                    # ASAI SVRGの効率性優位性はStage Bの最大約1.25倍
│   │                                    # （バッチサイズ512）から学習継続で急速に縮小した後，
│   │                                    # 小さいが正の水準（約1〜9%）で安定するという結論，
│   │                                    # Stage B/C同一エポック時点での完全一致による整合性
│   │                                    # 確認，学習率0.001長期学習の見送り提案を記載
│   ├── report_030.md                   # 実験4 Stage A：WikiText-2・単語レベル言語モデリング
│   │                                    # （order_030）．全36条件で発散が皆無，バッチサイズ
│   │                                    # 512×学習率0.001のみチャンス張付き判定に該当したが
│   │                                    # 学習不足であり崩壊ではないと判断．8プロセス並列時に
│   │                                    # VRAM総量超過でGPUが応答不能になった事象とnum_workers
│   │                                    # =2への対応，損失値がL2正則化項（Token Embeddingの
│   │                                    # 初期スケール由来）に支配されている問題の発見，
│   │                                    # 近似誤差の絶対水準は実験3と同等かむしろ小さいという
│   │                                    # 結果，正則化係数見直しを含むStage B提案を記載
│   └── report_031.md                   # 実験4b（ex0041）：正則化修正後のStage A再実行
│                                        # （order_031）．Token Embedding・出力射影層の初期化を
│                                        # $\mathcal N(0,1/\sqrt d)$へ修正し，初期L2正則化項が
│                                        # チャンスレベル交差エントロピーの1.83倍に収まることを
│                                        # 検証．全36条件で発散・チャンス張付きが皆無．訓練損失は
│                                        # 969〜1080から24.5〜29.2へ縮小し交差エントロピーを
│                                        # 反映する値になった．バッチサイズ512・学習率0.001は
│                                        # 最終精度が約67〜71倍に改善（学習不足の大幅解消），
│                                        # バッチサイズ128・学習率0.001の近似誤差異常値も解消．
│                                        # 一方学習率0.01条件は精度がやや低下するという副次的な
│                                        # 現象を考察，Stage Bへの提案を記載
│   └── report_033.md                   # 実験5 Stage A：Imagewoof・ResNet18画像分類での
│                                        # 安定性探索（order_033）．全BatchNorm2d層をLayerNorm2d
│                                        # に置換したResNet18の実装，実行前見積もり（約1時間）と
│                                        # 実測（約2.5時間，8プロセス並列時のGPU資源競合が主因）
│                                        # の乖離，学習率0.001の18条件は全て非崩壊，学習率0.01は
│                                        # NFG SVRGが全バッチサイズで崩壊しASAI SVRGはバッチ
│                                        # サイズ128でのみ部分的に崩壊回避という結果と考察，
│                                        # 実験3の「団子」現象とは異なりバッチサイズが小さいほど
│                                        # 手法間の精度差が拡大する傾向を記載，Stage Bへの提案を
│                                        # 記載
├── requirements_pytorch.txt
├── requirements_pytorch_gpu.txt        # 実験2用GPU環境の依存ライブラリ（torch 2.11.0+cu128等）
├── .venv_pytorch/                      # Python仮想環境（Git管理対象外）
├── .venv_pytorch_gpu/                  # GPU用Python仮想環境（Git管理対象外，実験2）
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
- `programs/ex002_cifar10_alexnet/data.py`：`machine_learning_utils.py` の `set_seed` を
  利用する．torchvision経由でCIFAR-10を取得し，公式の学習・検証分割（50000枚／10000枚）を
  そのまま用いる．チャネルごとの標準化のみを行い，データ拡張（ランダムクロップ・反転等）は
  用いない（`.orders/order_022.md` 2節の制約，SVRG系手法との理論的整合性のため）．
- `programs/ex002_cifar10_alexnet/model.py`：`machine_learning_utils.py` の `set_seed` を
  利用する．AlexNetをCIFAR-10向けに縮小したCNN（`AlexNetCIFAR`，BatchNormalization・
  Dropoutを含まない）と，L2正則化項（畳み込み層・全結合層の重みのみに課す）を加えた損失関数
  を定義する．
- `programs/ex002_cifar10_alexnet/train.py`：上記の2つに加え `machine_learning_utils.py` の
  `ResultLogger`，`set_seed`，`optimizers`（`SGD`，`SVRG`，`NFGSVRG`，`ASAISVRG`．ミニバッチ
  平均勾配をそのまま渡すことで内部ループ長 $ K=\lceil N_{\text{train}}/\text{batch
  size}\rceil $ のミニバッチ学習に対応でき，Optimizerクラス自体の変更は不要であることを
  確認した）を利用し，学習全体を統括する．`@.ai/ai-dev-kit/machine_learning.md` が定める
  `iteration`／`epoch` 関数構成に従い，`iteration` 関数がSVRG系手法用に `snapshot_model`
  引数を任意に受け取れるよう拡張している（`programs_old/ex002_cifar10_cnn/train.py` で
  確立した拡張と同一）．GPU（`.venv_pytorch_gpu`）を用い，8プロセスを `chunksize=1` で
  並列実行する（タスクごとの実行時間がバッチサイズにより6〜8倍異なるため，静的な均等
  チャンク分割では負荷分散が偏ることを避けるため）．
- `programs/ex0021_cifar10_alexnet_norm/data.py`：`programs/ex002_cifar10_alexnet/data.py`
  と同一の前処理（`.orders/order_023.md` の指示）．
- `programs/ex0021_cifar10_alexnet_norm/model.py`：`machine_learning_utils.py` の `set_seed`
  を利用する．`AlexNetCIFAR`（実験2）をベースに，各畳み込み層に適用する正規化を4パターン
  （なし／`LayerNorm2d`／`GroupNorm`／`SpectralConv2d`，`.orders/order_024.md` でSpectralNormを
  追加）から選択できるよう拡張した `AlexNetCIFARNorm` を定義する．`LayerNorm2d` はConvNeXt
  （Liu et al., 2022）のブロック設計に倣い，`(B,C,H,W)` を `(B,H,W,C)` に並べ替えてチャネル
  方向にLayerNormを適用する自前実装．GroupNormは1グループあたり4チャンネルとなるよう
  グループ数を決定する．`SpectralConv2d` は，畳み込み層自体の重みを最大特異値で除する
  自前実装で，power iterationの補助バッファ（$ u, v $）を `frozen` フラグにより凍結できる
  （スナップショットモデルの決定論性を保証するため）．いずれもバッチ内の他サンプルに
  依存しないことを単体テストで確認済み．
- `programs/ex0021_cifar10_alexnet_norm/train.py`：上記の2つに加え `machine_learning_utils.py`
  の `ResultLogger`，`set_seed`，`optimizers`（`NFGSVRG`，`ASAISVRG`．`.orders/order_023.md`
  2.5節の指示によりSGD・SVRGは対象外）を利用し，学習全体を統括する．バッチサイズ間で総
  イテレーション数（epoch×K）を揃えるため，バッチサイズごとに異なるエポック数
  （128:12, 256:24, 512:48）を用いる．スナップショットモデルの重みを更新するたびに
  `model.refresh_and_freeze_spectral_norm` を呼び出し，power iterationバッファを凍結する
  （`SpectralConv2d` を含まないモデルに対しては何もしない）．GPU（`.venv_pytorch_gpu`）を
  用い，実験2と同様に8プロセスを `chunksize=1` で並列実行する．
- `programs/ex0022_cifar10_alexnet_groupnorm/data.py`／`model.py`：`programs/
  ex0021_cifar10_alexnet_norm/` の同名ファイルと同一（`EXPERIMENT_NAME` のみ変更）．
  GroupNorm以外の正規化パターンのコードは，比較のため削らずそのまま保持している．
- `programs/ex0022_cifar10_alexnet_groupnorm/train.py`：`machine_learning_utils.py` の
  `ResultLogger`，`set_seed`，`optimizers`（`SGD`，`SVRG`，`NFGSVRG`，`ASAISVRG`．
  `.orders/order_025.md` の指示は`SVRGFinalPoint`／`NFGSVRGFinalPoint`を挙げていたが，
  実験2・ex0021が実際に使用してきた`SVRG`／`NFGSVRG`との一貫性を優先する方針をユーザーから
  確認した）を利用する．学習率0.001・GroupNorm・バッチサイズ512, 128のNFG SVRG・ASAI SVRG
  計20条件は，`reuse_ex0021_results` 関数により `outputs/ex0021_cifar10_alexnet_norm/` から
  ディレクトリごとコピーして再利用する（`hp_name` 関数がex0021と文字列レベルで一致する
  命名規則を持つよう設計しているため，リネーム不要）．また，実験2・ex0021の実装では
  NFG SVRG・ASAI SVRGの近似誤差算出専用のフル勾配計算時間が `elapsed_time` に誤って
  計上されていたバグを修正し，SVRGの（アルゴリズムに必要な）フル勾配計算時間とを区別する．
  GPU（`.venv_pytorch_gpu`）を用い，8プロセスを `chunksize=1` で並列実行する．
- `programs/ex0023_cifar10_alexnet_groupnorm_longrun/data.py`／`model.py`：`programs/
  ex0022_cifar10_alexnet_groupnorm/` の同名ファイルと同一（`EXPERIMENT_NAME` のみ変更）．
- `programs/ex0023_cifar10_alexnet_groupnorm_longrun/train.py`：ex0022と同一の
  Optimizerクラス・`elapsed_time`計上方法を踏襲する．学習率0.001に固定し，バッチサイズ
  512, 128, 32のエポック数をex0022比で約4倍（192, 48, 12）に延長する．ex0022の内部
  Optimizer状態は保存されていないため学習を再開できず，全60条件をゼロから再学習する
  （既存結果のコピー再利用は行わない）．`compute_trailing_relative_change` 関数で，
  末尾複数エポックの相対変化の最大値によりプラトー到達を数値的に判定する．訓練損失が
  非有限値化した場合（崩壊）は学習を打ち切り，`is_run_completed` はこれを完了済みとして
  扱う．GPU（`.venv_pytorch_gpu`）を用い，8プロセスを `chunksize=1` で並列実行する．
- `programs/ex003_tinyshakespeare_transformer/data.py`：`urllib.request` でTiny
  Shakespeareを自動ダウンロードし，文字レベルで語彙を構築する．コーパスの前方90%を
  学習用，後方10%を検証用とし，それぞれ独立に固定長 $ T+1=129 $ の非重複チャンクへ
  分割する（`_ChunkedTextDataset`）．チャンクは学習の全過程を通じて固定．
- `programs/ex003_tinyshakespeare_transformer/model.py`：`machine_learning_utils.py` の
  `set_seed` を利用する．Decoder-only Transformer（`DecoderOnlyTransformer`，4層，
  隠れ次元128，826,433パラメータ）を定義する．Pre-LN構成のTransformerブロック
  （`CausalSelfAttention` + Feed Forward）を用い，Dropout・BatchNormalizationは使用
  しない．Causalマスクは `torch.tril` による決定論的なバッファ．位置エンコーディングは
  学習可能な `nn.Embedding`．L2正則化は `nn.Linear`・`nn.Embedding` の重みのみに課し，
  LayerNormのアフィンパラメータは対象外とする．
- `programs/ex003_tinyshakespeare_transformer/train.py`：ex0022・ex0023と同一の
  Optimizerクラス（`SGD`，`SVRG`，`NFGSVRG`，`ASAISVRG`）・`elapsed_time`計上方法を用いる．
  Stage A（`.orders/order_027.md` 6節）として，NFG SVRG・ASAI SVRGの2手法×バッチサイズ
  (512,128,32)×学習率(0.01,0.001)×3Seed＝36条件を12エポックで学習し，Stage B（`.orders/
  order_028.md`）でSGD・SVRGを追加した4手法×同一グリッド＝72条件に拡張した（Stage Aの
  36条件は`is_run_completed`により自動的に再利用）．`is_stuck_near_chance` 関数で，次文字
  予測精度がチャンスレベル付近に張り付く「見えない崩壊」（`.reports/report_026.md` 5.1節）
  の兆候を検出する．GPU（`.venv_pytorch_gpu`）を用い，8プロセスを `chunksize=1` で並列
  実行する．
- `programs/ex0031_tinyshakespeare_transformer_longrun/data.py`／`model.py`：
  `programs/ex003_tinyshakespeare_transformer/` の同名ファイルと完全に同一
  （`EXPERIMENT_NAME` のみ変更）．
- `programs/ex0031_tinyshakespeare_transformer_longrun/train.py`：Stage C
  （`.orders/order_029.md`）として，ex003 Stage Bの4手法学習ロジックを踏襲し，学習率を
  0.01のみに固定，全バッチサイズ一律48エポック（Stage Bの4倍）で36条件（4手法×3バッチ
  サイズ×3Seed）を学習する．ex0023由来の `compute_trailing_relative_change`（プラトー
  判定）・崩壊時の学習打ち切りロジックを流用する．Optimizer内部状態が保存されないため，
  Stage Bの12エポック分の結果からの継続学習は行わず，全36条件をゼロから再学習する
  （既存結果のコピー再利用は行わない）．GPU（`.venv_pytorch_gpu`）を用い，8プロセスを
  `chunksize=1` で並列実行する．
- `programs/ex004_wikitext2_transformer/data.py`：PyTorch公式word_language_modelサンプル
  が配布する前処理済みWikiText-2（単語分割済み，`<unk>`によるレア語処理済み）を
  `urllib.request` で自動ダウンロードする．学習用テキストのみから語彙を構築し
  （`sorted(set(...))` による決定論的な構築），公式train/valid分割をそのまま用いる．
  チャンク分割ロジック（`_ChunkedTextDataset`）はex003と共通の設計．
- `programs/ex004_wikitext2_transformer/model.py`：ex003の`model.py`と完全に同一の
  Transformer本体（`DecoderOnlyTransformer`）を用いる．`vocab_size`（33,277）・
  `max_seq_len`（64）のみが実際の使用時に異なる．
- `programs/ex004_wikitext2_transformer/train.py`：Stage A（`.orders/order_030.md`）として，
  NFG SVRG・ASAI SVRGの2手法×バッチサイズ(512,128,32)×学習率(0.01,0.001)×3Seed＝36条件を
  12エポックで学習する．`is_stuck_near_chance`の許容幅を本実験の語彙サイズ向けに0.001へ
  再調整している．GPU（`.venv_pytorch_gpu`）を用いるが，出力層のロジットテンソルが
  バッチサイズ512で1プロセスあたり約23.5GBのVRAMを要求するため，`num_workers=2`に制限
  している（8プロセス並列で実行した場合，最初にバッチサイズ512の条件が複数同時に実行され
  VRAM総量を超過しGPUが応答不能になった事象を踏まえた対応，`.reports/report_030.md` 5.3節
  参照）．
- `programs/ex0041_wikitext2_transformer_fixedreg/data.py`：`programs/
  ex004_wikitext2_transformer/` の同名ファイルと完全に同一（`EXPERIMENT_NAME` のみ変更）．
- `programs/ex0041_wikitext2_transformer_fixedreg/model.py`：ex004の`model.py`とほぼ同一
  だが，`__init__`内でToken Embedding層・出力射影層（`head`）の重みを明示的に
  `nn.init.normal_(..., std=1/sqrt(d_model))`で再初期化する（`.orders/order_031.md` 3.2節，
  `.reports/report_031.md` 3.1節）．PyTorchの`nn.Embedding`デフォルト初期化
  （標準偏差1）に起因する，初期状態でのL2正則化項の過大な支配（`.reports/report_030.md`
  8.3節）を修正するための変更である．
- `programs/ex0041_wikitext2_transformer_fixedreg/train.py`：ex004 Stage Aと同一グリッド
  （2手法×バッチサイズ3種×学習率2種×3Seed＝36条件，12エポック）を再実行する
  （`.orders/order_031.md`）．`main`関数の冒頭で`verify_regularization_is_not_dominant`
  関数を実行し，初期状態でのL2正則化項がチャンスレベル交差エントロピー理論値
  （$\ln(\text{vocab\_size})$）の10倍を超えないことを学習開始前に検証する．最大バッチ
  サイズ（512）でVRAM使用量を実測した上で`num_workers=2`を採用しており（`.reports/
  report_030.md` 5.3節の教訓の反映），GPU応答不能事象は再発しなかった．
- `programs/ex005_imagewoof_resnet18/data.py`：fast.aiが配布するImagewoof2-320
  （`https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz`）を`urllib.request`で
  自動ダウンロードし，`tarfile`で展開する．`torchvision.datasets.ImageFolder`で公式
  train/val分割をそのまま読み込む．前処理は`Resize(256)→CenterCrop(224)→ToTensor→
  Normalize`のみの決定論的な処理．チャネル統計量（mean/std）は学習用画像から算出し，
  `channel_stats.json`にキャッシュして全プロセスで再利用する。
- `programs/ex005_imagewoof_resnet18/model.py`：`ResNet18LayerNorm`．
  `torchvision.models.resnet18(weights=None, num_classes=10)`をインスタンス化した後，
  モジュール木を再帰的に走査し全20層の`nn.BatchNorm2d`を`LayerNorm2d`（ex0021由来の実装，
  `_replace_batchnorm_with_layernorm`関数）に置換する。パラメータ数（11,181,642）は
  標準BatchNorm版ResNet18と完全一致する。
- `programs/ex005_imagewoof_resnet18/train.py`：Stage A（`.orders/order_033.md`）として，
  NFG SVRG・ASAI SVRGの2手法×バッチサイズ(128,64,32)×学習率(0.01,0.001)×3Seed＝36条件を
  12エポックで学習する。`is_stuck_near_chance`の許容幅を10クラス設定向けに0.03へ設定
  している。実行前に最大バッチサイズ（128）でVRAM使用量を実測（約6.1GB）し
  （`.reports/report_030.md` 5.3節の教訓の反映），`num_workers=8`を採用した。

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
  実験2は `.venv_pytorch_gpu` のtorch 2.11.0+cu128（本マシンのBlackwell世代GPUに対応）を
  用いる．
- torchvision：CIFAR-10データセットの取得・前処理（Ex002，実験2）．
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

### 5.1 GPU用環境（`.venv_pytorch_gpu`，実験2以降）

本マシンのGPU（NVIDIA RTX PRO 6000 Blackwell，CUDA capability sm_120）は，`.venv_pytorch`が
使用するPyTorch 2.5.1+cu124（対応compute capability sm_50〜sm_90）では認識されるものの
カーネルが存在せず（`CUDA error: no kernel image is available for execution on the device`），
実際には計算に使用できない．実験2（CIFAR-10・AlexNet）はGPUなしでは180条件の実行が非現実的な
時間を要するため，Blackwell世代のGPUに対応した新しいPyTorchビルド（cu128，torch 2.11.0）を
別の仮想環境 `.venv_pytorch_gpu` にインストールした．既存の `.venv_pytorch`（実験0・1，Ex001〜
Ex006が依存）への影響を避けるため，同一環境の更新ではなく新規環境として追加している．

```bash
uv venv .venv_pytorch_gpu --python 3.11
uv pip install --python .venv_pytorch_gpu/bin/python -r requirements_pytorch_gpu.txt
```

VS CodeからJupyterカーネルとして利用する場合は，カーネル名 `asai_svrg_pytorch_gpu` を選択する
（`.venv_pytorch_gpu/bin/python -m ipykernel install --user --name=asai_svrg_pytorch_gpu` で
登録済み）．実験2の学習・単体テストは `.venv_pytorch_gpu/bin/python` で実行する．

## 6. プログラムの実行方法

```bash
# --- 論文掲載用の実験（.orders/order_020.md 以降）---

# 実験0の学習実行（4手法 x 3学習率 x 5Seed = 60条件をCPUでマルチプロセス並列実行．
# 既に完了した条件はスキップ）
.venv_pytorch/bin/python programs/ex000_a9a_least_squares/train.py

# 実験1の学習実行（4手法 x 3学習率 x 5Seed = 60条件を物理コア数相当で並列実行．
# 既に完了した条件はスキップ）
.venv_pytorch/bin/python programs/ex001_mushroom_logistic/train.py

# 実験2の学習実行（4手法 x 3バッチサイズ x 3学習率 x 5Seed = 180条件をGPUで8プロセス並列
# 実行．既に完了した条件はスキップ．GPU対応の .venv_pytorch_gpu を用いる）
.venv_pytorch_gpu/bin/python programs/ex002_cifar10_alexnet/train.py

# 実験ex0021の学習実行（2手法 x 3バッチサイズ x 4正規化層 x 5Seed = 120条件をGPUで8プロセス
# 並列実行．既に完了した条件はスキップ）
.venv_pytorch_gpu/bin/python programs/ex0021_cifar10_alexnet_norm/train.py

# 実験ex0022の学習実行（4手法 x 3バッチサイズ x 2学習率 x 5Seed = 120条件のうち，ex0021から
# コピーで再利用可能な20条件をコピーし，残り100条件をGPUで8プロセス並列実行．既に完了した
# 条件はスキップ）
.venv_pytorch_gpu/bin/python programs/ex0022_cifar10_alexnet_groupnorm/train.py

# 実験ex0023の学習実行（学習率0.001固定，4手法 x 3バッチサイズ x 5Seed = 60条件を全て新規に
# GPUで8プロセス並列実行．既に完了した条件（崩壊による打ち切りを含む）はスキップ）
.venv_pytorch_gpu/bin/python programs/ex0023_cifar10_alexnet_groupnorm_longrun/train.py

# 実験3 Stage A・Stage Bの学習実行（4手法 x 3バッチサイズ x 2学習率 x 3Seed = 72条件のうち，
# Stage Aで完了済みの36条件はスキップし，Stage B新規36条件（SGD・SVRG）をGPUで8プロセス
# 並列実行）
.venv_pytorch_gpu/bin/python programs/ex003_tinyshakespeare_transformer/train.py

# 実験3 Stage Cの学習実行（学習率0.01固定・48エポック一律，4手法 x 3バッチサイズ x 3Seed =
# 36条件を全て新規にGPUで8プロセス並列実行．既に完了した条件はスキップ）
.venv_pytorch_gpu/bin/python programs/ex0031_tinyshakespeare_transformer_longrun/train.py

# 実験4 Stage Aの学習実行（2手法 x 3バッチサイズ x 2学習率 x 3Seed = 36条件をGPUで2プロセス
# 並列実行．バッチサイズ512のVRAM使用量が大きいため並列数を2に制限，既に完了した条件はスキップ）
.venv_pytorch_gpu/bin/python programs/ex004_wikitext2_transformer/train.py

# 実験4b（ex0041，正則化修正後のStage A再実行）の学習実行（同一グリッド36条件をGPUで
# 2プロセス並列実行．既に完了した条件はスキップ）
.venv_pytorch_gpu/bin/python programs/ex0041_wikitext2_transformer_fixedreg/train.py

# 実験5 Stage Aの学習実行（2手法 x 3バッチサイズ x 2学習率 x 3Seed = 36条件をGPUで8プロセス
# 並列実行．既に完了した条件はスキップ）
.venv_pytorch_gpu/bin/python programs/ex005_imagewoof_resnet18/train.py

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
.venv_pytorch/bin/python -m pytest tests/ -v          # 論文掲載用の実験（実験2の単体テストは
                                                       # GPU不要でCPU venvでも実行可能）
.venv_pytorch_gpu/bin/python -m pytest tests/ -v      # 同上（GPU venvから実行する場合）
.venv_pytorch/bin/python -m pytest tests_old/ -v      # 事前実験

# --- 結果の可視化 ---
# Jupyter上で実行，またはnbconvertで一括実行する．「可視化する条件の指定」セルの
# EXPERIMENT・METHODS・HYPERPARAMS_LIST で対象を切り替える．事前実験の結果を見る場合は，
# 冒頭のセルの OUTPUTS_ROOT を Path("outputs_old") に変更する．
.venv_pytorch/bin/jupyter nbconvert --to notebook --execute --inplace visualize_result.ipynb
```

## 7. 実験結果・文書の保存場所

- 学習結果（各Seedのログ・メタデータ）：
  `outputs/{ex000_a9a_least_squares,ex001_mushroom_logistic,ex002_cifar10_alexnet,ex0021_cifar10_alexnet_norm,ex0022_cifar10_alexnet_groupnorm,ex0023_cifar10_alexnet_groupnorm_longrun,ex003_tinyshakespeare_transformer,ex0031_tinyshakespeare_transformer_longrun,ex004_wikitext2_transformer,ex0041_wikitext2_transformer_fixedreg,ex005_imagewoof_resnet18}/{method}/{hyperparams}/{seed}/`
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
  （実験1：Mushroomの二値分類，強凸設定での定理1・定理2の検証），`.reports/report_022.md`
  （実験2：CIFAR-10・AlexNetの多値分類，非凸設定でのASAI SVRGの実用性検証），
  `.reports/report_023.md`（実験ex0021：正規化層・バッチサイズの構造的変更による安定性の検証）

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
