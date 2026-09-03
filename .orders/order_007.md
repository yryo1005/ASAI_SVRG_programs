Ex002は`@.orders/order_005.md`の状態に戻してください
実験結果も元に戻します

Ex003として`@references/No_Full_Grad_SVRG.pdf`の実験と同様の内容（データ，モデル，誤差関数，ハイパーパラメータ等）を揃え実験し，論文と同様の結果になるか確認してください
この際，比較対象はSGD，SVRG，NFGに加えASAIも追加してください

---

## 解釈・保管メモ

ユーザーからチャットで指示された内容をそのまま上記に記録した．以下のように解釈して作業する．

1. **Ex002の復元**：`.orders/order_006.md` の検証（NFG SVRGのスナップショット構成方法の
   比較）のためにEx002（`programs/ex002_cifar10_cnn/`）に加えた変更（`NFG_SVRG_FinalPoint`
   をEx002の比較対象に含める変更，および対応する実験結果 `outputs/ex002_cifar10_cnn/
   NFG_SVRG_FinalPoint/` と比較図）を，`.orders/order_005.md` 完了時点の状態に戻す．
   ただし，`programs/optimizers/optimizers.py` の `NFGSVRGFinalPoint` クラス自体は，
   Ex003で「NFG SVRG原論文に忠実なNFG」として必要になるため保持し，Ex002側の参照のみを
   取り除く．`.reports/report_006.md` はEx002の実験結果を直接引用する記録であるため，
   参照先の実験結果を移動・整理した上で，レポート自体は歴史的記録として残す．

2. **Ex003の新規実施**：NFG SVRG原論文（`references/No_Full_Grad_SVRG.pdf`）7節の実験
   （ResNet-18によるCIFAR-10分類，min-max敵対的ロバスト性の定式化，学習率0.01，
   正則化係数λ1=λ2=0.0005）に，データセット・モデル・誤差関数・ハイパーパラメータを
   可能な限り揃えた実験を，`programs/ex003_cifar10_resnet_minmax/` として新規に実装する．
   比較手法は，原論文が扱うSGD，SVRG，NFG（原論文Algorithm 1，最終点採用のスナップショット）
   に加え，ASAI SVRGも含めた4手法とする．
