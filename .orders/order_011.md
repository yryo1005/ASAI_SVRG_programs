@ASAI_SVRG_programs/references/No_Full_Grad_SVRG.pdf の7章の再現実装は不明な条件が多いので諦めます．
新たな実験として @ASAI_SVRG_programs/references/No_Full_Grad_SVRG.pdf のA.1 LEAST SQUARES REGRESSIONを再現してください．

この実験ではハイパーパラメータを何かしらで最適化したtunedがありますが，これは不要です．

従来の実験通りSGD，SVRG，NFG，ASAIで比較してください．
SVRG，NFGのハイパーパラメータは記載のある限り論文に準拠し，
記載のないパラメータやSGD，ASAIのハイパーパラメータは妥当に定めてください．

論文では2つのデータセットを用いていますが，どちらか1つのみで良いです．

# 実装プロセス
1. 実験条件を調査する
2. 同様のプログラムを実装する
3. プログラムの実行結果と論文の記載が一致するか確認する

---

## 解釈・保管メモ

ユーザーから与えられた指示をそのまま上記に記録した．以下のように解釈して作業する．

- `references/No_Full_Grad_SVRG.pdf` 7節（ResNet-18・CIFAR-10，min-max敵対的摂動）の再現は，
  `.reports/report_006.md`〜`report_010.md` の一連の調査により不明な条件（特にNFG SVRG・
  ASAI SVRGの発散原因）が多いことが判明しているため，指示通り断念する．
- 新たな実験として，同論文の付録A.1（LEAST SQUARES REGRESSION，式(8)の非線形最小二乗回帰問題）
  を再現する．データセットはijcnn1・a9aのうちa9aのみを実装する．
- チューニング済みステップ幅（tuned）は実装せず，理論ステップ幅（原論文Theorem 1）のみを用いる．
- 比較手法は，本リポジトリの従来の実験（Ex001・Ex003〜Ex005）と同様にSGD，SVRG，NFG，
  ASAI SVRGの4手法とする．SVRG・NFGの学習率は原論文Theorem 1の上界に基づき定め，記載のない
  パラメータ（内部ループ長，エポック数）およびSGD・ASAI SVRGの学習率は，SVRG・NFGと同一の
  値を用いることで妥当に定める．
- 実装後，`programs/ex006_a9a_least_squares/` として学習を実行し，結果を`.reports/report_011.md`
  にまとめ，原論文の記載（理論ステップ幅では収束が緩やかであること等）と実行結果を比較する．