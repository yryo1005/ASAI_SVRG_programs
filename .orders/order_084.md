# `.orders/order_084.md`

## 1. ユーザーからの指示（原文）

> @実験指示E11.md 新たな実験指示なのでこれを実施してください

指示書：`実験指示E11.md`（E11：Mushroom における全てのスナップショットと内側ループの開始点の構成，ex006 の追加）．

## 2. 指示の解釈

- 実験1（Mushroom，L2 正則化付きロジスティック回帰）で，スナップショット（random，final，average）× 全勾配の推定（全勾配，平均勾配）× 内側ループの開始点（reset，continue）の10構成をそろえる．
  - 新規の構成は5つ：`SVRG_random`・`NFG_SVRG_random`・`SVRG_average`・`ASAI_SVRG` の continue（`inner_start="last_iterate"`）と，`SVRG_average` の reset（`inner_start="snapshot"`）．
  - 実施済みの5構成（reset の random 2つ，final 2つ，ASA の reset）は再実行しない．
- 学習の条件は ex006（E1）と同じ．ρ は全7値（random reshuffling），一様サンプリングは ρ = 0.5，1.0，298.4．シード0〜4．新規の試行は 175 + 75 = 250．
- プログラムの変更（指示書3節）：`programs_review/programs/ex006_e1_mushroom_rho_sampling/train.py` に `SVRG_average` を加え，docstring の `last_iterate` の位置づけを「continue の構成（E11）に使用する」に改める．
  - 変更の理由は，指示書の指定どおり「order_114，指導教員の指示」としてコメントに残す．order_114 は指示書の作成側の番号と解釈する．本PCの記録としては `order_084` も併記する．
  - 実験の通し番号は ex006 のまま（新しい ex 番号は使わない）．
- テストを `programs_review/tests/` に加える（`SVRG_average` のスナップショットが更新前のパラメータの平均であること，`approx_error` が0であること，`last_iterate` の開始点）．
- 本実行の前の確認（指示書4節）：既存の結果が変わらないこと（ビット単位），SVRG（average）の `oracle_calls` と `approx_error`，並列数1での DRAM と時間．結果は `outputs/_verify/ex006_e11/` と `outputs/ex006_e1_mushroom_rho_sampling/consistency_e11.md`．
- 本実行は指示書5節の2つのコマンド（continue の4構成，SVRG（average）の reset）．並列数は確認の結果で決める．CPU のみの計算である．
- 集計：`programs_review/programs/analysis_e11.py` で `outputs/e11_summary.md` を作る．図は作らない．
  - $ \hat{r} $ は指示書6節の定義（シードごとの最小二乗の傾き $ a_{c} $，$ \hat{r} = \frac{1}{C}\sum_{c}\exp(a_{c}) $）とする．参照先の `make_report090_e1.hat_r_of` は本PCでは `ASAI_SVRG_Paper/programs/` にあり，`compute_hat_r_lsq` の計算（$ s_{\mathrm{end}} < 3 $ は対象外）を確認した．`programs_review/` の外を import しない規約のため，同じ計算を `analysis_e11.py` に書く．
- 番号は本PCの次の空き番号 `order_084`／`report_084` とする．既存の結果は変更しない．git の commit・push は行わない．
- 報告用の zip を作る（新しいログ，実行ログ，集計，consistency，変更したプログラム，テスト）．
