# `.orders/order_077.md`

## 1. ユーザーからの指示（原文）

> @E6_results.zip E6の結果です。統合してください

## 2. 指示の解釈

- 別のPCで実施した E6（`実験指示E6.md`，order_080／report_080，`ex011_e6_cifar10_final_bs32_512`：CIFAR-10 のバッチサイズ32・512における最終点の SVRG・NFG-SVRG）の成果物 `E6_results.zip` を本リポジトリに取り込む．
- 統合の手順は，これまでの統合（order_046，order_052）と同じとする．
  - 作業用ディレクトリ（スクラッチ）に展開する．
  - 絶対パス・`..` を含むパスが無いことと，既存のファイルとの衝突を確認する．E6 のプログラム（`ex011_e6_cifar10_final_bs32_512/`，`analysis_e6.py`，`tests/test_e6_programs.py`）は本PCで作成済み（order_074）であるため，zip 内のものと内容が同一かを確かめる．
  - 上書きしない設定で複写する．内容の異なるファイルがあれば，上書きせずに差分を調べて報告する．
  - テストを実行し，本PCでの短い再実行で zip のログと比較する．集計（`analysis_e6.py`）を本PCで再実行し，zip 内の `e6_summary.md` と一致するかを確かめる．
- 統合時に `programs_review/document.md` に E6 を追記する（order_074 の方針どおり）．E7・E8 の転記内容（report_075・report_076 の「統合時の注意」）も同じ機会に追記する．
- 番号：本PCの次の番号は `order_077`／`report_077` である（E6 の `order_080`・`ex011` は別のPCに予約済み）．
- Git の commit・push はユーザーの指示があるまで行わない．
