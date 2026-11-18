# order_050：E1（Mushroomにおける $ \rho $ の異なる学習率での収束挙動とサンプリング方式の影響）

## ユーザーからの指示（原文）

指示書：`実験指示E1.md`

ユーザーがこのPCで与えた指示：

> @実験指示E1.md に従って E1 を実施して

## 指示の解釈

`実験指示E1.md` に従い，以下を実施する．

1. **前提**：論文は修正しない．実装・実行・報告のみを行う．プログラムは全て `programs_review/` に置き，
   `programs_review/` の外のファイルを import・参照しない．内側ループの開始点は `inner_start="snapshot"`
   固定とする（`last_iterate` は3節の既存実装との一致確認にのみ用いる）．
2. **番号の割り当て**：指示とレポートは `.orders/order_050.md`，`.reports/report_050.md`．実験のプログラムは
   `programs_review/programs/ex006_e1_mushroom_rho_sampling/`，結果は
   `programs_review/outputs/ex006_e1_mushroom_rho_sampling/`．集計スクリプトは `programs_review/programs/analysis_e1.py`，
   テストは `programs_review/tests/test_e1_programs.py`，可視化ノートブックは `programs_review/visualize_result_E1.ipynb`，
   説明文書は `programs_review/programs/ex006_e1_mushroom_rho_sampling/README.md`．
3. **変更してよいファイル**：上記0.1節の表のファイル・ディレクトリのみ．既存のファイル（`programs_review/programs/common/`，
   `ex001_e0_mushroom_logistic/`，`document.md`，`visualize_result.ipynb`，`tests/test_e0_programs.py`，
   `requirements_*.txt`，`CLAUDE.md`，`.ai/` 等）は変更しない．共通化すべき変更は `report_050.md` の
   「統合時の注意」に提案として書く．`document.md` に追記すべき内容は README.md に書き，統合時に転記する．
   git の commit・push は行わない．成果物は `E1_results.zip` にまとめる．
4. **環境構築**：`.venv_pytorch`（Python 3.12）を作成し，`programs_review/requirements_pytorch_gpu.txt` を
   インストールする．Mushroom はCPUのみで計算する（`torch.set_num_threads(1)`，1試行1プロセス）．
5. **実装**：`ex001_e0_mushroom_logistic/` を `ex006_e1_mushroom_rho_sampling/` にコピーし，
   SGD・7種類の $ \rho $（学習率）・2種類のサンプリング方式（`random_reshuffle`，`uniform_with_replacement`）を
   追加する．条件は `random_reshuffle` が全7 $ \rho $，`uniform_with_replacement` が $ \rho \in \{0.5, 1.0, 298.4\} $
   の3条件．手法はSGD，SVRG（最終点），NFG-SVRG（最終点），ASA-SVRGの4手法．5シード．合計 $ (7+3) \times 4 \times 5 = 200 $ 試行．
6. **既存実装との一致の確認**：本実行の前に，`inner_start="last_iterate"` と `"snapshot"`，$ \rho = 298.4 $，
   シード0の新しい実装を，E0の既存ログおよび既存のSGDのログと比較する．
7. **集計・可視化**：`analysis_e1.py` で実測の縮小率 $ \hat{r} $，誤差床，表（`outputs/e1_summary.md`），
   図（目的関数誤差・近似誤差，sampling・手法ごと，$ \rho $ を重ねる）を作成する．
   `visualize_result_E1.ipynb` で実行済みにする．
8. **テスト**：`tests/test_e1_programs.py` で $ \eta $ の値，乱数消費の一致，`uniform_with_replacement` の性質，
   `snapshot` の開始点，SGDの `oracle_calls` を確認する．
9. **レポート**：`.reports/report_050.md` に実行環境，一致確認の結果，実験計画3.7節の3つの論点への回答
   （事実と推測を分けて記述），図表，統合時の注意を記載する．
10. **成果物**：作業の最後に `E1_results.zip` を作成し，`unzip -l` の結果を `report_050.md` の末尾に貼る．
