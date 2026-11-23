# report_077：E6 の成果物（`E6_results.zip`）の統合

指示：`.orders/order_077.md`．E6 の実施の記録は `.orders/order_080.md`，`.reports/report_080.md`（別のPCで作成，原文のまま取り込んだ）である．

## 0. 要点

- 別のPC（NVIDIA GeForce RTX 4090）で実施した E6（`ex011_e6_cifar10_final_bs32_512`：CIFAR-10 のバッチサイズ32・512における SVRG（最終点）・NFG-SVRG（最終点），20試行）の成果物を本リポジトリに取り込んだ．
- zip 内の64ファイルのうち，プログラムの9ファイルは本PCで作成済み（order_074）のものとバイト単位で同一であった．残りの55ファイルは新規であり，既存のファイルとの衝突は無かった．統合時に修正したファイルは無い．
- テストは全196件が通った．
- 本PCでの確認（3節）：
  - 本PCで短く再実行すると，zip のログと `oracle_calls` が完全に一致し，0エポック目の学習損失の差は相対 $ 4\times10^{-8} $ 以下であった．本PCの再実行の0エポック目は，本PCの付録B（`ex004`）とビット単位で一致した．したがって zip との差は GPU の違いによる丸め誤差である．
  - NFG-SVRG（最終点）のバッチサイズ32が3エポック目に崩壊する現象は，本PCでも再現した．
  - `analysis_e6.py` を本PCで実行すると，集計表は zip 内の `e6_summary.md` と同一，図6枚はバイト単位で同一であった．
- `programs_review/document.md` に E6 を追記した．同じ機会に，E7（report_075）と E8（report_076）の「統合時の注意」の内容も追記した（4節）．

## 1. 取り込んだファイル

| 種類 | パス | 状態 |
|---|---|---|
| 指示・報告 | `.orders/order_080.md`，`.reports/report_080.md` | 新規 |
| プログラム | `programs_review/programs/ex011_e6_cifar10_final_bs32_512/`（`data.py`，`model.py`，`train.py`，`e2_training.py`，`e2_runner.py`，`svrg_average.py`，`README.md`），`programs_review/programs/analysis_e6.py` | 本PCのものと同一 |
| テスト | `programs_review/tests/test_e6_programs.py` | 本PCのものと同一 |
| 結果 | `programs_review/outputs/ex011_e6_cifar10_final_bs32_512/`（20試行の `log.json`・`config.json`，`figures/`（6枚），`consistency.md`，`consistency_bs128.md`，`consistency_bs32_512.md`），`programs_review/outputs/e6_summary.md` | 新規 |
| 実行ログ | `programs_review/outputs/_run_logs/ex011.log`（約28MB．tqdm の出力を含む），`ex011_verify_bs128.log`，`ex011_verify_bs32_512.log` | 新規 |

統合の手順は，これまでの統合（report_052）と同じである．

- 作業用ディレクトリ（スクラッチ）に展開した．絶対パスや `..` を含むパスは無かった．
- 展開した各ファイルをリポジトリの同じパスと比べ，同一（9），新規（55），内容が異なる（0）に分けた．
- 上書きしない設定（`cp -rn`）で複写し，複写後の全ファイルが zip の内容と同一であることを確かめた．
- `best_model.pth`（20個）は zip に含まれておらず，本PCにも無い．集計には使わない．

`report_080.md` の末尾の `unzip -l` の一覧は zip を作る前の値であり，`report_080.md` 自身のサイズ（19246）は実物（32351）と異なる．これは `report_080.md` 12節の注記のとおりである．

## 2. 統合時の修正

なし．取り込んだファイルは全て zip のとおりである．

## 3. 本PCでの確認

詳細は `programs_review/outputs/ex011_e6_cifar10_final_bs32_512/consistency_this_pc.md` にある．再実行の結果は `outputs/_verify/ex011_this_pc/`（Git 管理外），ログは `outputs/_run_logs/ex011_verify_this_pc.log` である．

### 3.1 zip のログ（RTX 4090）との比較（シード0，0〜3エポック）

| 手法 | バッチサイズ | `oracle_calls` | 0エポック目の `train_loss` の相対差 | 1〜3エポック目の `train_loss` の最大相対差 | `test_accuracy` の最大絶対差 |
|---|---:|---|---:|---:|---:|
| SVRG（最終点） | 32 | 完全一致 | $ 2.9\times10^{-8} $ | $ 1.4\times10^{-2} $ | 0.0077 |
| SVRG（最終点） | 512 | 完全一致 | $ 4.0\times10^{-8} $ | $ 3.0\times10^{-6} $ | 0.0001 |
| NFG-SVRG（最終点） | 32 | 完全一致 | $ 2.9\times10^{-8} $ | $ 2.3\times10^{-2} $ | 0.0525 |
| NFG-SVRG（最終点） | 512 | 完全一致 | $ 4.0\times10^{-8} $ | $ 7.1\times10^{-6} $ | 0.0002 |

- 差は1エポックのステップ数が多いバッチサイズ32（$ K = 1563 $）で大きく，512（$ K = 98 $）で小さい．report_080 4節の一致の確認（RTX 4090 と本PCの `ex004` の比較）と同じ大きさである．
- NFG-SVRG（最終点）のバッチサイズ32は，本PCでも3エポック目に学習損失が上がり（1.8545 → 2.3346．zip は 1.8540 → 2.3891），テスト精度が下がった（0.4223 → 0.2798．zip は 0.4233 → 0.2273）．崩壊は GPU に依らず起きる．

### 3.2 本PCの付録B（`ex004`）との比較

本PCの再実行の SVRG（最終点）の0エポック目の `train_loss`・`test_accuracy`・`oracle_calls` は，`ex004` の SVRG（シード0）とバッチサイズ32・512の両方でビット単位で一致した．1エポック目以降は，スナップショットの選び方（最終点とランダム点）が異なるため比べない．

### 3.3 集計の再現

本PCで `analysis_e6.py` を出力先だけ作業用ディレクトリに変えて実行した．表は zip 内の `e6_summary.md` と同一であり（違いは図の一覧の出力先のパスのみ），図6枚はバイト単位で同一であった．したがって，取り込んだ `e6_summary.md` と図はそのまま用いる．

### 3.4 テスト

`python -m pytest -q tests`：196件が通った（E0・付録B，E1，E2，旧版のE2，E3，E4，E5，E6，E7，E8）．report_080 で skip された E4 の9件は，本PCには a9a のデータがあるため実行された．

## 4. その他の変更

`programs_review/document.md` に次を追記した．

- 実験の一覧に E6，E7，E8 を加えた（実験の数を7から10に改めた）．最終点の手法（E6，E7）では内側ループの開始点が結果に影響しないことを加えた．
- ディレクトリ構成に `ex011`〜`ex014` のプログラム，`analysis_e6.py`〜`analysis_e8.py`，`tests/test_e6_programs.py`〜`test_e8_programs.py`，結果のディレクトリと `e6_summary.md`〜`e8_summary.md` を加えた．
- 依存関係の図に `ex011`〜`ex014` の `train.py` と `analysis_e6.py`〜`analysis_e8.py` を加えた．
- 実行方法に E6，E7（B → A の順），E8 のコマンドを加えた．
- ログの項目に E7 B の1エポックの時間の定義を加えた．
- 環境構築に，E4・E6・旧版のE2 を実行した PC と，GPU が異なると1エポック目以降の値が丸め誤差の水準で異なることを加えた．

## 5. E6 の結果の要点（report_080 と本PCの E7 との対応）

E6 の詳細は `.reports/report_080.md` 6・7節にある．値は5シードの平均 ± 標準誤差であり，予算はバッチサイズ32で $ \#\mathrm{grad}/N \le 20 $，512で $ \le 350 $ である．

| 手法 | バッチサイズ | 予算でのテスト精度 | 参考：付録B（`ex004`，ランダム点または平均点，reset） |
|---|---:|---:|---:|
| SVRG（最終点） | 32 | 0.6937 ± 0.0048 | SVRG 0.5916 ± 0.0240 |
| SVRG（最終点） | 512 | 0.7157 ± 0.0038 | SVRG 0.6445 ± 0.0026，ASA-SVRG 0.7098 ± 0.0015 |
| NFG-SVRG（最終点） | 32 | 0.1043 ± 0.0150 | NFG-SVRG 0.1083 ± 0.0108 |
| NFG-SVRG（最終点） | 512 | 0.4480 ± 0.0382 | NFG-SVRG 0.5434 ± 0.0334 |

- **事実（E6）**：SVRG（最終点）は両バッチサイズでランダム点の SVRG より予算でのテスト精度が 7〜10%pt 高く，バッチサイズ512では ASA-SVRG（平均点，reset）と同程度か少し高い．NFG-SVRG（最終点）は，バッチサイズ32では全シードで3エポック目に崩壊して回復せず，512では崩壊と回復を繰り返した．
- **事実（E7，report_075，ImageWoof）**：同じ傾向が ImageWoof でも見られた．SVRG（最終点）はバッチサイズ128・512でランダム点より 5〜6%pt 高い．NFG-SVRG（最終点）は全てのバッチサイズでランダム点より低く，テスト精度の急落を繰り返す．
- **推測**：最終点の SVRG がランダム点より良いこと，最終点の NFG-SVRG が不安定であることは，CIFAR-10 と ImageWoof の両方で成り立つ．データセットに固有の現象ではないと考えられる．ただし，崩壊の原因（近似勾配 $ {\mathbf g}_{s} $ と $ {\mathbf z}_{s} $ での全勾配のずれ）は，どちらの実験でも確かめていない．

## 6. 未解決の論点・次の作業の候補

1. **論文での SVRG の扱い（ユーザーの判断事項）．** SVRG（最終点）は CIFAR-10（E6）と ImageWoof（E7）の両方でランダム点より強く，CIFAR-10 のバッチサイズ512では ASA-SVRG（平均点，reset）と同程度である．付録の図で SVRG を最終点とランダム点のどちらで示すか，平均点の手法の利点をどう説明するかを決める必要がある．
2. **論文用の図（付録の CIFAR-10，5手法）．** ASA-SVRG（平均点，continue）と SGD は既存の実験 `ex0024_cifar10_alexnet_layernorm_longrun` から取る（report_080 11節）．バッチサイズ32では ASA-SVRG（平均点，reset）の発散が縦軸を支配するため，縦軸の範囲の指定か対数軸が必要である．
3. **NFG-SVRG（最終点）の崩壊の時点での近似誤差の確認．** E6・E7 の `log.json` の `approx_error`，`approx_cos`，`omega_snapshot` をエポックごとに見ればよく，追加の学習は不要である．
4. **GPU の違い．** E6 は RTX 4090，付録B（`ex004`）は本PCの RTX PRO 6000 である．同じ図に並べる場合，バッチサイズ32では1エポック目以降に相対 $ 10^{-2} $ 程度の差が生じうる（3.1節）．結論を変える大きさではないが，厳密にそろえるなら本PCで E6 の20試行を取り直す（本PCの GPU は E7 の終了後は空いている）．
5. Git の commit・push は行っていない．`E6_results.zip`，`E7_results.zip` はリポジトリのルートに未追跡のまま残っている．
