# `.orders/order_043.md`

## 1. ユーザーからの指示（原文）

@main.pdf に対する査読コメントが @review.txt です．これを確認してください．

@対応計画.md のように査読に対応します．これを確認してください．
このリポジトリでは論文は修正せずに，@実験計画.md にある実験のみを実装します．

まずはE0について実施してください．
特にパラメータの初期値で実験結果が変わると根幹にかかわるので，急ぎ実施する．

査読後の実験のためのプログラムは @programs_review に実装する
なお，このディレクトリを後に独立したリポジトリとするため，
ディレクトリ外のファイルを参照しないようにする

## 2. 指示の解釈

- 対象は `実験計画.md` 2章のE0（内側ループの開始点 `inner_start` ∈ {`"last_iterate"`, `"snapshot"`} の影響確認）のみとする．E1〜E3は本orderでは実装しない．論文（`main.pdf`）は修正しない．
- 「パラメータの初期値」は，各外側ループの内側ループの開始点 $ {\mathbf w}_{s}^{0} $（既存実装：$ {\mathbf w}_{s-1}^{K} $，論文のAlgorithm：$ {\mathbf z}_{s} $）を指すと解釈した．
- 新しいプログラムは全て `programs_review/` 以下に置き，`programs_review/` の外のファイル（`programs/optimizers/`，`.ai/ai-dev-kit/src/machine_learning_utils.py`，既存の `datasets/`・`outputs/` 等）を import・参照しない．必要なモジュール（最適化手法のクラス，`set_seed`，`ResultLogger`）は `programs_review/` 内にコピーする．データセットも `programs_review/datasets/` に取得し，結果は `programs_review/outputs/` に保存する．
- 既存ログとの一致確認（実験計画2.2節）だけは既存の `outputs/` を読む必要があるため，比較スクリプトは既存ログのパスをコマンドライン引数で受け取る形とし，ディレクトリ外のパスをコードに埋め込まない．
- 問題は (i) Mushroom（ASA-SVRGのみ，$ \eta = 0.99/(3L) $，100エポック），(ii) CIFAR-10・AlexNet（LayerNorm）・バッチサイズ128（SVRG，NFG-SVRG，ASA-SVRG，48エポック），(iii) ImageWoof・ResNet18（LayerNorm）・バッチサイズ128（同3手法，128エポック）とし，各5シード．
- 手順：(1) `last_iterate` が既存ログと一致することを確認 → (2) 両条件の全シードを実行 → (3) 図・表で比較し，方針(a)/(b)の判断材料を報告する．
- 実験計画1.3節の新しいログ項目のうち，E0で必須でない `omega_*`，`time_*` は記録しない．`test_correct`，`full_grad_norm_sq`，`approx_cos`，`terminated` は記録する．

## 3. 成果物

- `programs_review/`（プログラム，`document.md`，`visualize_result.ipynb`，`tests/`）
- `programs_review/outputs/`（E0の結果，図，表）
- `.reports/report_043.md`
