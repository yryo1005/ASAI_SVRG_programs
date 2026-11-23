# `.orders/order_081.md`

## 1. ユーザーからの指示（原文）

指示書：`実験指示E9.md`

> @実験指示E9.md 新たな実験指示です。これを実施してください

## 2. 指示の解釈

- `実験指示E9.md`（E9：ImageWoof・バッチサイズ512 の ASA-SVRG（reset，`inner_start="snapshot"`）のシード1〜4）を本PCで実施する．
  - 本PCは既存のシード0（`ex005_appb_imagewoof_resnet18_layernorm`）を実行した PC（NVIDIA RTX PRO 6000 Blackwell Max-Q）である．
  - 番号は指示書0.3節のとおり `order_081`／`report_081` とする（どちらも未使用であることを確認した）．
- 新しい学習のプログラムは作らない．既存の `ex005_appb_imagewoof_resnet18_layernorm/train.py` を `--batch-sizes 512 --methods ASAI_SVRG --inner-starts snapshot --seeds 1 2 3 4` で実行する．
- 手順は指示書のとおり，(1) 残っている未完了の試行の確認，(2) 一致の確認（シード0，3エポック，`outputs/_verify/e9/`），(3) 並列数の決定，(4) 本実行（4試行），(5) `analysis_e9.py` と `tests/test_e9_programs.py` の作成，(6) 集計（`e9_summary.md`），(7) `report_081.md` と `E9_results.zip` の作成，とする．
- 新規作成は指示書0.3節のファイルのみとする．既存のファイル（シード0，他の手法・バッチサイズの結果，`common/`，`ex005_*`，`document.md`，既存のテスト等）は変更しない．git の commit・push は行わない．
- 一致の確認で不合格（`oracle_calls` の不一致，または相対差が $ 10^{-3} $ を超える）の場合は，本実行に進まずにユーザーに伝える．
