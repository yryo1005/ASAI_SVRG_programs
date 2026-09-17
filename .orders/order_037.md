# `.orders/order_037.md`

## 1. ユーザーからの指示（原文）

`outputs/ex0023_cifar10_alexnet_groupnorm_longrun/` では正規化としてGroup Normを用いていますが
これをLayer Normに変更した ex0024を実装し実験してください．ほかの実験条件はex0023と同様とします．

## 2. 指示の解釈

`programs/ex0023_cifar10_alexnet_groupnorm_longrun/`（GroupNorm・学習率0.001固定での長期エポック
学習，`.orders/order_026.md`・`report_026.md`）と同一の実験条件（4手法：SGD, SVRG, NFG SVRG,
ASAI SVRG，3バッチサイズ：512, 128, 32，学習率0.001固定，正則化係数$\lambda=5\times10^{-4}$，
5Seed，バッチサイズごとのエポック数：512→192，128→48，32→12）を保ったまま，正規化層のみを
GroupNormからLayerNormへ変更した新規実験`ex0024_cifar10_alexnet_layernorm_longrun`を実装し，
実行する．

`programs/ex0023_cifar10_alexnet_groupnorm_longrun/model.py` の `AlexNetCIFARNorm` は，
`.orders/order_023.md`・`.orders/order_024.md` により正規化層4パターン（なし，LayerNorm，
GroupNorm，SpectralNorm）を`norm_type`引数で選択できるよう既に実装済みである（`ex0021_cifar10_
alexnet_norm/model.py`に由来）。そのため，`model.py`・`data.py`はex0023から変更せず（`data.py`は
`EXPERIMENT_NAME`のみ変更），`train.py`の`NORM_TYPE`定数を`"groupnorm"`から`"layernorm"`に
変更するのみで実装できる。

## 3. 実行前の計算コスト見積もり

`.ai/ai-dev-kit/root_prompt.md`・`machine_learning.md`の指示（並列数1で実測してから並列数を
決定する）に基づき，グリッド中の最大バッチサイズ（512）で1エポック全体を実測し，GroupNorm
（ex0023）との所要時間・VRAM使用量の差を確認した上で並列数を決定する。

## 4. 実行後の成果物

`.reports/report_037.md`に，実装内容・実行前見積もり・実験結果（GroupNorm・ex0023との比較を
含む）をまとめる。`document.md`・`visualize_result.ipynb`も更新する。
