# `.orders/order_042.md`

## 1. ユーザーからの指示（原文）

a9aの実験を4.1と同様に\eta = 0.99/(3L) となるよう，学習率を定め，再度実験する
（MSE，BCEともに

## 2. 指示の解釈

「4.1」はASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）4.1節のMushroom実験（本リポジトリの
実験1，`programs/ex001_mushroom_logistic/`）を指す．実験1では学習率の1つを補題3の上界近傍
$ \eta_a = 0.99/(3L) $ としている（`.reports/report_021.md`）．a9aを用いた2つの実験でも，同じ規則
$ \eta = 0.99/(3L) $ で学習率を定めて実験する．

- MSE（実験0，`programs/ex000_a9a_least_squares/`，非線形最小二乗損失）：既存の学習率は
  $ 1/(3L) $，$ 1/(20L) $，$ 1/(20Ln) $ の3つであり，$ 0.99/(3L) $ は含まれない．そこで
  `build_hyperparameter_list` に $ 0.99/(3L) $（$ L = L_{\text{individual}} $）を4つ目の規則として
  追加し，4手法 × 5Seed = 20条件を新規に学習する．既存の60条件は `is_run_completed` により
  スキップされ，変更されない．
- BCE（実験0-1，`programs/ex0001_a9a_logistic/`，L2正則化付きBCE損失）：`.orders/order_041.md`
  で既に $ \eta_a = 0.99/(3L) $（$ = 8.544\times10^{-2} $）の20条件を実施済みである．学習は
  Seedにより決定的であり，同一条件を再実行しても同じ結果になる．そのため既存の結果を用い，
  Seed 0を数エポック再実行して保存済みのログと一致することを確認する（再現性の確認）．
- 両実験の $ \eta = 0.99/(3L) $ の結果を並べて比較する可視化を `visualize_result.ipynb` に追加し，
  レポートにまとめる．

## 3. 成果物

`programs/ex000_a9a_least_squares/train.py` の更新，`outputs/ex000_a9a_least_squares/` への
20条件の追加，`visualize_result.ipynb` の更新，`.reports/report_042.md`，`document.md` の更新．
