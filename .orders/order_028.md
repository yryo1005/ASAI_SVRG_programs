# `.orders/order_028.md`

# 実験3 Stage B：SGD・SVRGを含めた4手法比較

## 1. 目的

Stage A（`report_027.md`）は，NFG SVRG・ASAI SVRGの2手法について，Tiny Shakespeare・
Decoder-only Transformer（826Kパラメータ）という設定で，バッチサイズ32〜512・学習率
0.001〜0.01・12エポックの範囲で発散・崩壊が一切生じないことを確認した．

実験3の目的は，SVRG系分散削減手法が実務上意味のある規模の系列データ・Transformer
アーキテクチャに対して適用可能であることを示すことである．スケール依存性（モデルサイズ，
$N$の大小，$K$のレンジ）そのものの検証は実験0・実験2で既に行っているため，実験3では
これを繰り返さず，**Stage Aと同一のモデル・データ規模のまま**，SGD・SVRGを比較対象に
加えた4手法比較を行う．

## 2. 実験条件

Stage A（`.orders/order_027.md` 6節，`report_027.md` 6節）と同一の設定を用いる．

- **モデル**：Stage Aと同一（Decoder-only Transformer，4層，隠れ次元128，Attention
  head数4，FFN中間次元512，826,433パラメータ）．変更しない
- **データセット**：Tiny Shakespeare，Stage Aと同一の前処理・分割（`report_027.md` 3節）
- **比較手法**：SGD, SVRG, NFG SVRG, ASAI SVRG の4手法（Stage AのNFG SVRG・ASAI SVRGに，
  SGD・SVRGを追加する）
- **バッチサイズ**：512, 128, 32（Stage Aと同一グリッド）
- **学習率**：0.01, 0.001（Stage Aと同一グリッド）
- **エポック数**：12（Stage Aと同一．長期学習・エポック数の延長は本実験のスコープ外とし，
  まずは同条件での4手法比較の結果を確認してから，必要に応じて別途検討する）
- **Seed数**：3（0〜2，Stage Aと同一）
- **正則化係数**：$\lambda=5\times10^{-4}$（Stage Aと同一）

条件数はバッチサイズ3種×学習率2種×手法4種×Seed3 = 72条件．うちNFG SVRG・ASAI SVRGの
36条件はStage Aで取得済みのため，既存結果を再利用する．

## 3. 既存結果の再利用

NFG SVRG・ASAI SVRGの36条件（Stage A，`outputs/ex003_tinyshakespeare_transformer/`）は，
モデル・データセット・ハイパーパラメータが本実験と完全に同一であるため，再実行せず
既存結果をそのまま用いる．新規に学習が必要なのはSGD・SVRGの2手法×3バッチサイズ×
2学習率×3Seed = 36条件のみである．

`programs/ex003_tinyshakespeare_transformer/train.py` にSGD・SVRGの学習ループを追加し，
Stage Aと同一の出力ディレクトリ（`outputs/ex003_tinyshakespeare_transformer/`）に結果を
格納すること．オラクル呼び出し回数（#grad/N）はSGDが$N$，NFG SVRG・ASAI SVRGが$2N$，
SVRGが$3N$であり，これまでの実験と同一の構造を用いる．

## 4. 実験終了の基準

以下を確認した時点で本実験を完了とする．

1. SGD・SVRGの36条件が完了し，発散の有無を確認すること
2. オラクル呼び出し回数を横軸とした場合の4手法の比較（分類精度に相当する次文字予測精度，
   目的関数値）を確認すること
3. Stage Aで確認された「発散・崩壊が一切生じない」という傾向が，SGD・SVRGを含めても
   維持されるか（SGD・SVRGは実験0〜ex0023を通じて一貫して安定であったため，本実験でも
   同様の傾向が期待される）を確認すること
4. ASAI SVRGが同一オラクル呼び出し回数のもとでSVRGと比べてどの程度の効率性を示すかを
   確認すること．これは`report_022.md`終了基準4，ex0022（`report_025.md`）で確認された
   知見（GroupNorm下のCIFAR-10でASAI SVRGがSVRGを一貫して上回る）が，Transformer・
   系列データでも成立するかを見る，実験3における対応する検証項目である

## 5. レポートへの記載事項

`report_027.md` の構成に倣い，特に以下を明記すること．

- 既存結果（NFG SVRG・ASAI SVRGの36条件）の再利用箇所
- オラクル呼び出し回数を横軸とした4手法の比較図（バッチサイズ・学習率ごと）
- SGD・SVRGの発散の有無（実験0〜ex0023の傾向と整合するか）
- ASAI SVRGのSVRGに対する効率性についての考察
- 12エポック・同条件での比較にとどめた本実験の結果を踏まえ，長期学習（Stage C相当）が
  必要かどうかの判断・提案

## 6. 実行コマンド（想定，実装時に確定）

```bash
# 実験3 Stage Bの学習実行（既存36条件はスキップし，新規36条件を実行する）
.venv_pytorch_gpu/bin/python programs/ex003_tinyshakespeare_transformer/train.py

# 単体テスト
.venv_pytorch_gpu/bin/python -m pytest tests/ -v

# 結果の可視化
.venv_pytorch_gpu/bin/jupyter nbconvert --to notebook --execute --inplace visualize_result.ipynb
```