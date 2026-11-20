# order_070：E4（付録B.1，a9a の MSE・BCE の再実施）

## ユーザーからの指示（原文）

指示書：`実験指示E4.md`

ユーザーがこのPCで与えた指示：

> 実験指示E4.md　これを実施してください

## 指示の解釈

`実験指示E4.md`（E4 の正本は `実験計画.md` 6章．両者が食い違う場合は `実験指示E4.md` を優先する）に従い，以下を実施する．

1. **前提**：論文は修正しない．E4 の実装・実行・報告のみを行う．プログラムは全て `programs_review/` に置き，`programs_review/` の外のファイルを import・参照しない．既存のログとの比較では，既存のログのパスをコマンドライン引数で受け取る．内側ループの開始点は `inner_start="snapshot"` とし，`last_iterate` は旧実装の ASA-SVRG の再現の確認にのみ用いる．
2. **番号の割り当て**：`実験計画.md` 6.3節の `ex007_e4_a9a` は E2 に割り当て済みのため，`ex008_e4_a9a` とする．指示とレポートは `.orders/order_070.md`，`.reports/report_070.md`．プログラムは `programs_review/programs/ex008_e4_a9a/`，結果は `programs_review/outputs/ex008_e4_a9a/`，集計は `programs_review/programs/analysis_e4.py`，テストは `programs_review/tests/test_e4_programs.py`，可視化は `programs_review/visualize_result_E4.ipynb`，説明文書は `programs_review/programs/ex008_e4_a9a/README.md`，実行ログは `programs_review/outputs/_run_logs/ex008*.log`，データは `programs_review/datasets/a9a/`．
3. **変更してよいファイル**：上記の新規作成のみ．既存のファイル（`programs/`，`outputs/`，`tests/`，`programs_review/programs/common/`，`ex00{1..6}_*`，`analysis_*.py`，`document.md`，`実験計画.md`，`requirements_*.txt`，`CLAUDE.md`，`.ai/` 等）は変更しない．共通化の提案は `report_070.md` の「統合時の注意」に，`document.md` への追記内容は README.md に書く．git の commit・push は行わない．
4. **環境構築**：`.venv_pytorch`（Python 3.12）に `programs_review/requirements_pytorch_gpu.txt` をインストールする．計算は CPU のみ（`torch.set_num_threads(1)`，1試行1プロセス）．a9a は既存と同じ URL から `programs_review/datasets/a9a/raw/a9a` に1回だけ取得する．並列数は並列数1で1試行を実行したときの DRAM の使用量と論理コア数から決める．
5. **実装**：E1（`ex006_e1_mushroom_rho_sampling/`）を `ex008_e4_a9a/` にコピーし，データとモデルを既存の a9a（`ex000_a9a_least_squares`：MSE，`ex0001_a9a_logistic`：BCE）と同一にする．`--loss mse|bce` で切り替える．学習率は $ \eta = 0.99/(3L) $ のみ，サンプリングは `random_reshuffle` のみ，$ K = N_{\mathrm{train}} = 29304 $，ミニバッチサイズ1，100外側ループ，シード0〜4．手法は SGD，SVRG（`SVRGFinalPoint`），NFG-SVRG（`NFGSVRGFinalPoint`），ASA-SVRG（`ASAISVRG`）．試行数は 2 × 4 × 5 = 40．
6. **一致の確認（本実行の前）**：シード0について，SGD（`inner_none`），SVRG・NFG-SVRG（`inner_snapshot`），ASA-SVRG（`inner_last_iterate`）を既存のログと全てのキー（`elapsed_time` を除く）で比較する．比較処理は `ex008_e4_a9a/compare_e4.py` に置き，結果を `programs_review/outputs/ex008_e4_a9a/consistency_with_existing.md` に保存する．一致しない場合は本実行に進まず，原因を調べて報告する．
7. **集計・可視化**：`analysis_e4.py` で表（`programs_review/outputs/e4_summary.md`）と図（MSE：$ \|\nabla f({\mathbf z}_{s})\|^{2} $，テスト精度，近似誤差．BCE：目的関数誤差，テスト精度，近似誤差．旧実装の ASA-SVRG を破線で重ねた確認用の図も）を作る．ノートブックを `nbconvert --execute` で実行済みにする．
8. **テスト**：データの分割と初期値が既存のプログラムと同一（既存のコードで求めた値を定数として埋め込む），$ L $・$ \lambda $・$ \eta $・$ f({\mathbf w}^{*}) $ の値，最終点スナップショットの手法での `inner_start` のビット単位の一致，`snapshot` での ASA-SVRG の開始点，40個の保存先の名前を確認する．既存のテストを含め全て通ることを確認する．
9. **報告と成果物**：`report_070.md` に実行環境，一致の確認，図表，`実験計画.md` 6.6節の5つの論点への回答（事実と推測を分ける），統合時の注意，未解決の論点を書く．最後に `E4_results.zip` を作り，`unzip -l` の結果をレポートの末尾に貼る．

## 作業環境に関する補足

- このクローンでは `.ai/` のサブモジュールが未初期化であったため，`git submodule update --init .ai/ai-dev-kit` で記録されたコミット（`a89fcb4`）を取得し，規約を読み込んだ．`.ai/` の内容は変更していない．
- `tokens.json` は存在しない．本タスクは外部 API を用いないため，作業には影響しない．
