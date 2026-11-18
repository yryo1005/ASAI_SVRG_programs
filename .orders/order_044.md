# `.orders/order_044.md`

## 1. ユーザーからの指示（原文）

report_044.md としてスナップショットの取り方の影響をSGDも交えてまとめて

## 2. 指示の解釈

- 「スナップショットの取り方」は，E0（report_043）で比較した内側ループの開始点 `inner_start` を指すと解釈した．
  比較するのは `last_iterate`（$ {\mathbf w}_{s}^{0} = {\mathbf w}_{s-1}^{K} $，既存の実装）と `snapshot`（$ {\mathbf w}_{s}^{0} = {\mathbf z}_{s} $，論文のAlgorithm）である．
  各手法のスナップショット $ {\mathbf z}_{s} $ そのもの（最終点・ランダム点・平均点）の決め方は変えない．
- E0の結果（Mushroom，CIFAR-10，ImageWoof の3手法 × 2条件 × 5シード）に SGD を加えて比較し，report_044.md にまとめる．
- SGDは内側ループを持たないため，開始点の影響を受けない．SGDの結果には既存のログを用いる．
  - Mushroom：`outputs/ex001_mushroom_logistic/SGD/eta1.049767e-02_...`
  - CIFAR-10：`outputs/ex0024_.../SGD/lr0.001_bs128_..._epochs96`
  - ImageWoof：`outputs/ex0051_.../SGD/lr0.001_bs128_..._epochs384`
  - 図と同じ予算（#grad/N ≤ 31，90，200）で切り出す．
- 集計スクリプトは `programs_review/programs/analysis_e0_with_sgd.py` とする．`programs_review/` の外を参照しない方針に従い，SGDのログのパスはコマンドライン引数で受け取る．
- 新しい学習は行わない．著者の判断（全実験を `snapshot` で行い，「SGD並みの精度」とは書かない）を前提に，`snapshot` で論文の各主張が成り立つかを整理する．

## 3. 成果物

- `programs_review/programs/analysis_e0_with_sgd.py`
- `programs_review/outputs/{実験}/figures/e0_with_sgd_*.png`
- `programs_review/outputs/e0_with_sgd_summary.md`
- `.reports/report_044.md`
