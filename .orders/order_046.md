# `.orders/order_046.md`

## 1. ユーザーからの指示（原文）

@E1_results.zip 実験結果が出たので統合してください

## 2. 指示の解釈

- 別のPCで `実験指示E1.md` に従って実施したE1の成果物（`E1_results.zip`）を，本リポジトリに取り込むことと解釈した．
- `E1_results.zip` の番号（`order_050`，`report_050`，`ex006`）は `実験指示E1.md` で予約したものであり，そのまま用いる．本指示（統合作業）は本PCの番号帯から `order_046` とする．
- 統合の手順は次の通りとする．
  1. zipを作業用ディレクトリに展開し，絶対パス・`..` を含むパスがないこと，既存のファイルと衝突しないことを確認してから，リポジトリのルートへ複写する（既存のファイルは上書きしない）．
  2. `programs_review/` の全テストを実行する．
  3. E1のプログラムが `programs_review/` の外を参照しないことを確認する．
  4. 本PCで1試行を再実行し，zip内のログと一致することを確認する（別のPCとの環境差の確認）．
  5. `ex006_e1_mushroom_rho_sampling/README.md` 5節の転記内容を `programs_review/document.md` に反映する．
- Gitのcommit・pushはユーザーの指示があるまで行わない．

## 3. 成果物

- 取り込んだE1の成果物（`.orders/order_050.md`，`.reports/report_050.md`，`programs_review/` 以下のE1のプログラム・結果）
- `programs_review/document.md` の更新
- `.reports/report_046.md`
