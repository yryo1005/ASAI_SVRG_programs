# order_080：E6（最終点のスナップショットの SVRG・NFG-SVRG，CIFAR-10，バッチサイズ32・512）

## ユーザーからの指示（原文）

- 指示書：`実験指示E6.md`
- チャットでの指示の原文：

```text
実験指示E6.md `wo
```

## 指示の解釈

- チャットの指示は入力の途中で送信されたものと考えられるが，指示書 `実験指示E6.md` を指定していることから，「`実験指示E6.md` を実施せよ」という指示であると解釈した．
- `実験指示E6.md` に従い，次を行う．
  1. `.venv_pytorch` の構築（`programs_review/requirements_pytorch_gpu.txt`）．
  2. `programs_review/tests/test_e6_programs.py` と全テストの実行．
  3. 一致の確認（3.1節：E2 のバッチサイズ128，3.2節：付録B のバッチサイズ32・512）を並列数1で実行し，DRAM・VRAM の使用量から並列数を決める．
  4. 本実行（`SVRG_final`，`NFG_SVRG_final` × バッチサイズ32・512 × シード0〜4 = 20試行）．
  5. `analysis_e6.py` による集計（`programs_review/outputs/e6_summary.md` と確認用の図）．
  6. `.reports/report_080.md` の作成と `E6_results.zip` の作成．
- 作成済みのプログラム（`ex011_e6_cifar10_final_bs32_512/`，`analysis_e6.py`，`test_e6_programs.py`）は変更しない．git の commit・push は行わない．
- 番号は `order_080`／`report_080` を用いる（どちらも未使用であることを確認した）．
