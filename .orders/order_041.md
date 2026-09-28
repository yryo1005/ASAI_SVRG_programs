# `.orders/order_041.md`

## 1. ユーザーからの指示（原文）

@ex000_a9a_least_squares/ ではNFGの論文と全くの同条件で実験を実施しましたが，
これを@ex001_mushroom_logistic/と誤差関数やモデルを揃えた実験としてex0001を実装してください

具体的には誤差関数をBinary Cross Entropy，モデルをシグモイド函数+切片付きの線型結合モデルとします

（追加確認）ex001の誤差関数はL2正則化付きBCE（$\lambda=0.03 L_0$，切片にも正則化）であるが，
ex0001でもL2正則化を揃えるかを確認したところ，「ex001と同じ正則化」との回答を得た．

## 2. 指示の解釈

`programs/ex000_a9a_least_squares/`（NFG SVRG原論文 付録A.1の非線形最小二乗回帰，切片なし・
正則化なし，a9aデータセット）の派生実験として，`programs/ex0001_a9a_logistic/`を新設する．
実験番号は既存の派生実験（ex0021，ex0031等）の命名規則に倣い，ex000の派生であることを示す
`ex0001`とする．

- データ：ex000と同一のa9a（LIBSVM Data，$N=32561$，$d=123$，One-Hot 0/1の特徴量を無加工で
  使用，ラベルは{0,1}，9:1の層化分割）．ex001で行った標準化は，Mushroomの順序符号化に対する
  前処理であるため，One-Hot済みのa9aには適用しない．
- モデル：ex001と同一の切片付き線形結合 + シグモイド関数 $\hat y=\sigma(w^\top x+b)$．
- 誤差関数：ex001と同一のL2正則化付きBCE
  $f(w,b)=\frac1N\sum\ell_{BCE}+\frac\lambda2(\|w\|^2+b^2)$，$\lambda=0.03L_0$，
  $L_0=\frac14\max_i\|[x_i;1]\|^2$．
- 学習率：ex001と同一の3規則（$\eta_a=0.99/(3L)$，$\rho(\eta)=0.5$を満たす$\eta_b$，
  $\eta_c=1/(20L)$）を，a9aから計算した$L$，$\mu=\lambda$，$K=N_{\text{train}}$で決定する．
- その他（比較手法SGD・SVRG・NFG SVRG・ASAI SVRG，バッチサイズ1，ランダムリシャッフル，
  100エポック，5Seed，評価指標`objective_gap`等）はex000・ex001と共通とする．

## 3. 成果物

`programs/ex0001_a9a_logistic/{data,model,train}.py`，`tests/test_ex0001_a9a_logistic.py`，
`outputs/ex0001_a9a_logistic/`，`visualize_result.ipynb`の更新，`.reports/report_041.md`，
`document.md`の更新．
