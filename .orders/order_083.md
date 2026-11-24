# `.orders/order_083.md`

## 1. ユーザーからの指示（原文）

> ASA(continue), ASA(reset), SGD, SVRG(final, continue), NFG(final, continue)でImageWoofを分類する問題のバッチサイズ128について、改めて時間を含めて取り直してください
> 先の時間計測と同様に実行順の影響をなくすべくSGD→SVRG→NFGのように順番に実行してください

（直前の相談：上の5手法のうち，全エポックの経過時間を他の学習と同時に動かさずに測ったデータは ASA(continue)・SGD（ex009）だけであり，ASA(reset)・SVRG(final)・NFG(final) は並列実行のため時間が使えないことを確認した．）

## 2. 指示の解釈

- ImageWoof（ResNet18（LayerNorm））・バッチサイズ128 で，次の5手法を全エポック学習し，精度と実時間を同時に記録し直す．
  - SGD，SVRG（最終点，`SVRG_final`），NFG-SVRG（最終点，`NFG_SVRG_final`），ASA-SVRG（continue，`inner_start="last_iterate"`），ASA-SVRG（reset，`inner_start="snapshot"`）．
  - 最終点の手法は開始点によらず同じ計算になるため，`snapshot` で実行する（E7 と同じ）．
- 条件は E3（`ex009`）・E7 と同じとする：$ \eta = 10^{-3} $，$ \lambda = 5\times10^{-4} $，予算 $ \#\mathrm{grad}/N = 200 $ に合わせたエポック数（SGD 200，SVRG 67，NFG-SVRG・ASA-SVRG 100），シード0〜4（5シード）．
- 「先の時間計測と同様に…順番に実行」は，E7 B（`ex013`）と同じく，1試行ずつ新しいプロセスで逐次に実行し，手法を交互に並べることと解釈する．
  - 順序はシードごとに SGD → SVRG → NFG-SVRG → ASA-SVRG（continue）→ ASA-SVRG（reset）とし，これをシード0〜4の5周繰り返す．同じ手法の5回の計測が時間的に離れた5つの時点に分かれる．
  - 各試行の前に GPU のメモリ使用量が十分小さくなるまで待つ．計測中は他の GPU のジョブや重い CPU の計算を行わない．
  - 実時間は ex009 と同じく `time_optimizer + time_svrg_full_grad`（評価と診断の時間を除く）とし，各エポックの累積値を記録する．
- 本実行の前に，シード0の短い実行で学習の値が既存の結果（ex009，ex003，ex012）と一致することを確かめる．
- 番号は `order_083`／`report_083`，実験のディレクトリは `ex016_imagewoof_bs128_walltime` とする（本PCの次の空き番号）．
- 既存のプログラムと結果は変更しない．`programs_review/` の外を import しない．git の commit・push は行わない．
