# `.orders/order_051.md`

## 1. ユーザーからの指示（原文）

別のPCでやってる以前の実験指示E2と実験内容はかぶってないよね？
被ってるようであればE3をやってください

## 2. 指示の解釈

- 別のPCでは，旧版の `実験指示E2.md`（6手法，全て `inner_start="snapshot"`，`ex007_e2_cifar10_snapshot_gradient_factorial`）が実行中である．
- 本PCで実行していたE2（order_049）の条件と照合した結果は次の通りである．
  - NFG-final（必須）：重複する．最終点をスナップショットとする手法では，2つの初期値が同じ計算となるためである．
  - SVRG-average・`snapshot`（任意）：重複する．
  - SVRG-final（任意）：重複する．
  - SVRG-average・`last_iterate`（必須）：重複しない．
- 「被ってるようであればE3をやってください」に従い，次のように対応する．
  - 重複する条件を本PCで止め，別のPCの結果を用いる．止めたのは，6エポック目まで進んでいた NFG-final の5試行と，未着手の任意の条件である．NFG-final の途中の結果は削除した．
  - 重複しない SVRG-average・`last_iterate` の5試行は，本PCで実行を続ける．
  - その後，`実験計画.md`（2026-11-19改訂版）5章の E3 を本PCで実施する．
- E3 は実時間の計測であり，他の実験と同時に実行してはならない（5.3節）．そのため，E3 の実装と確認を先に行う．本実行は SVRG-average の5試行が終わり，GPU が空いたことを確かめてから開始する．
- E3 の条件（5.2節）は次の通りである．
  - ImageWoof・ResNet18（LayerNorm），$ \eta = 10^{-3} $，バッチサイズ128．
  - 手法は SGD，SVRG，NFG-SVRG，ASA-SVRG．SVRG系は全て `last_iterate` とする．
  - 予算は $ \#\mathrm{grad}/N \approx 200 $ とする（SGD 200，NFG-SVRG・ASA-SVRG 100，SVRG 67エポック）．
  - シードは0〜2を必須とする．時間に余裕があれば0〜4とする．
  - 計測の規則は5.3節に従う．逐次に実行し，`torch.cuda.synchronize()` を挟んで4区間に分けて計測する．
- 実験番号は `ex009_e3_imagewoof_walltime` とする．`ex008` は別のPCの E4 に割り当てているためである．
- Gitのcommit・pushはユーザーの指示があるまで行わない．

## 3. 成果物

- `programs_review/programs/ex009_e3_imagewoof_walltime/`，`analysis_e3.py`，`tests/test_e3_programs.py`
- `programs_review/outputs/ex009_e3_imagewoof_walltime/`，`outputs/e3_summary.md`
- `.reports/report_051.md`（E3）．E2（order_049）の結果は，別のPCの結果を統合した後に `report_049.md` にまとめる．
