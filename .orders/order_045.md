# `.orders/order_045.md`

## 1. ユーザーからの指示（原文）

初期値をスナップショットとしてバッチサイズ32と512の実験を実施してください

## 2. 指示の解釈

- 「初期値をスナップショットとして」は，内側ループの開始点を論文のAlgorithm通りとすることと解釈した．
  具体的には `inner_start="snapshot"`（$ {\mathbf w}_{s}^{0} \leftarrow {\mathbf z}_{s} $）である（report_043，report_044）．
- 「バッチサイズ32と512の実験」は，論文の付録B.2（CIFAR-10・AlexNet（LayerNorm））とB.3（ImageWoof・ResNet18（LayerNorm））のミニバッチサイズ32・512の実験と解釈した．
- 条件は既存の実験と同一とし，開始点のみを変える．
  - CIFAR-10（`outputs/ex0024_cifar10_alexnet_layernorm_longrun`）：バッチサイズ32は12エポック，512は192エポック．
  - ImageWoof（`outputs/ex0051_imagewoof_resnet18_bs128_longrun`）：バッチサイズ32は33エポック，512は505エポック．
  - 共通：学習率 $ 10^{-3} $，$ \lambda = 5\times10^{-4} $，シード0〜4．
  - 手法は SVRG，NFG-SVRG，ASA-SVRG．学習損失が非有限となった場合は打ち切る（既存と同じ基準）．
- SGDは開始点の影響を受けないため，再実行しない．比較には既存のログを用いる．
  `last_iterate` も既存のログを用いる．実装が同じことは，短い実行で既存のログと照合して確認する．
- プログラムは `programs_review/` に置き，外のファイルを参照しない．既存のログのパスはコマンドライン引数で受け取る．
  - 新しい実験：`ex004_appb_cifar10_alexnet_layernorm`，`ex005_appb_imagewoof_resnet18_layernorm`
  - 集計：`analysis_appb.py`
- 図の横軸の予算（$ \#\mathrm{grad}/N $）は，論文本文の値（約20，350，30，1000）と既存のSGDの最長の実行が届く範囲から決める．
  - CIFAR-10：バッチサイズ32は24，512は384
  - ImageWoof：バッチサイズ32は33，512は1010

## 3. 成果物

- `programs_review/programs/ex004_appb_*`，`ex005_appb_*`，`analysis_appb.py`，テストの追加
- `programs_review/outputs/ex00{4,5}_appb_*`（結果，図，既存の実装との一致の確認），`programs_review/outputs/appb_summary.md`
- `.reports/report_045.md`
