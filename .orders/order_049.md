# `.orders/order_049.md`

## 1. ユーザーからの指示（原文）

E2を修正しこれを実行してください

## 2. 指示の解釈

- 「E2を修正し」は，旧版の計画に基づく `実験指示E2.md`（2×3の要因配置，全条件 `snapshot`）を，2026-11-19に改訂した `実験計画.md` の4章に合わせて改めることと解釈した．
- 「これを実行してください」は，改めたE2を本PCで実行することと解釈した．別のPCに渡す指示書は不要になる．そのため，`実験指示E2.md`（本セッションで作成し，未コミット）は削除する．
- 実施する条件は，`実験計画.md` 4.2節の必須の2条件とする．
  - NFG-final（`NFGSVRGFinalPoint`，平均勾配・最終点）：最終点では2つの初期値が同じ計算となる．そのため `last_iterate` で1回実行し，5シードとする．
  - SVRG-average（新規の `SVRGAverage`，全勾配・平均点）：`last_iterate`，5シード．
  - 任意の条件（SVRG-average・`snapshot`，`SVRGFinalPoint`，E0の条件の $ \omega $ 付きの再実行）は，実験計画7節の順序に従い，E3・E4の後に回す．
- 実装は `実験計画.md` 4.2節に従う．
  - `SVRGAverage` を `programs_review/programs/common/optimizers/` に追加する．
  - $ \omega $ の記録（1.3節）を `common/vr_image_training.py` に追加する．
  - 全勾配を用いる手法の判定を `method == "SVRG"` から手法の集合に一般化する．
  - これらの変更は既存の手法の計算を変えてはならない．E0のCIFAR-10のログとビット単位で一致することを短い実行で確認する．
- 条件はE0のCIFAR-10と同一である．学習率 $ 10^{-3} $，バッチサイズ128，48エポック，$ \lambda = 5\times10^{-4} $，シード0〜4．
- 実験番号は `ex007_e2_cifar10_ablation` とする．
- 集計は，E0の5条件（`ex002_e0_cifar10_alexnet_layernorm`）と新規の条件を合わせる．4.5節の基準（平均勾配の5パターンの，予算 $ \#\mathrm{grad}/N \le 90 $ の最後の評価点の学習損失）で最良の構成を選び，4.6節の表と図を作る．
- Gitのcommit・pushはユーザーの指示があるまで行わない．

## 3. 成果物

- `programs_review/programs/common/optimizers/svrg_average.py`，`common/vr_image_training.py` の変更
- `programs_review/programs/ex007_e2_cifar10_ablation/`，`analysis_e2.py`，`tests/test_e2_programs.py`
- `programs_review/outputs/ex007_e2_cifar10_ablation/`，`outputs/e2_summary.md`
- `.reports/report_049.md`
