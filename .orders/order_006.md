`@references/No_Full_Grad_SVRG.pdf` の論文の実験を参考に，NFG SVRGが論文と同じような結果になるか確認してください

---

## 解釈・保管メモ

ユーザーからチャットで指示された内容をそのまま上記に記録した．以下のように解釈して作業する．

- `references/No_Full_Grad_SVRG.pdf` は，ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）が
  参考文献[5]として引用するNo Full Grad SVRG（NFG SVRG）の原論文である．
- 本指示は，これまでのEx001（4.1節，マッシュルームデータセット）・Ex002（4.2節，CIFAR-10 CNN）
  とは別に，NFG SVRG原論文が実施した実験設定（データセット，モデル，ハイパーパラメータ，評価方法）
  を参考にして，同様の実験を本リポジトリで再現し，NFG SVRG原論文と同様の傾向の結果
  （収束挙動，誤差床の有無等）が得られるかを検証するタスクであると解釈する．
- まず `references/No_Full_Grad_SVRG.pdf` を読み込み，実験設定を把握した上で，実施する実験の
  詳細（データセット・モデル・比較手法・評価指標・Seed数等）を具体化する．
