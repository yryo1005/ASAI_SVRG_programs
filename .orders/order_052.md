# `.orders/order_052.md`

## 1. ユーザーからの指示（原文）

@E4_results.zip, @E2_results.zip  結果が出たので統合してください

## 2. 指示の解釈

- 別のPCで実施した2つの実験の成果物を，本リポジトリに取り込む．
  - `E4_results.zip`：E4（a9a，付録B.1の再実施）．`実験指示E4.md`，order_070／report_070，`ex008_e4_a9a`．
  - `E2_results.zip`：旧版の `実験指示E2.md` による E2．6手法，全て `inner_start="snapshot"`，order_060／report_060，`ex007_e2_cifar10_snapshot_gradient_factorial`．
- 統合の手順は，E1 の統合（order_046）と同じとする．
  - 作業用ディレクトリに展開する．
  - 絶対パス・`..` を含むパスがないことと，既存のファイルとの衝突を確認する．
  - 上書きしない設定で複写する．
  - テストを実行し，本PCでの再実行でログと一致することを確認する．
- 旧版の E2 の5つのファイルは，本PCの E2（order_049，`ex007_e2_cifar10_ablation`）と名前が衝突する．本PCのファイルを上書きせず，取り込むファイルの名前を `*_factorial` に変えて取り込む．
  - `analysis_e2.py`
  - `tests/test_e2_programs.py`
  - `visualize_result_E2.ipynb`
  - `outputs/e2_summary.md`
  - `outputs/_run_logs/ex007.log`
- 統合後に，order_049 の E2 のアブレーション（実験計画4.2節）を完成させる．
  - 別のPCの NFG-final（必須の条件）を，本PCの `analysis_e2.py` で読めるようにする．任意の条件（SVRG-average・`snapshot`，SVRG-final）も同様とする．
  - 選択規則（予算 $ \#\mathrm{grad}/N \le 90 $ で学習損失が最小の構成）を適用する．
  - `visualize_result_E2.ipynb` を実行し，`.reports/report_049.md` にまとめる．
- 統合そのものの報告は `.reports/report_052.md` とする．
- Gitのcommit・pushはユーザーの指示があるまで行わない．
