# order_060：E2（スナップショットの選択方法と全勾配の推定方法の要因分離，CIFAR-10）

## ユーザーからの指示（原文）

指示書：`実験指示E2.md`

ユーザーがこのPCで与えた指示：

> @実験指示E2.md を実施して

## 指示の解釈

`実験指示E2.md`（正本は `実験計画.md` 4章．食い違う場合は指示書を優先する）に従い，以下を実施する．

1. **前提**：論文は修正しない．実装・実行・報告のみを行う．プログラムは全て `programs_review/` に置き，
   `programs_review/` の外のファイルを import・参照しない．内側ループの開始点は `inner_start="snapshot"` のみとし，
   `last_iterate` は実装しない．git の commit・push は行わない．
2. **番号の割り当て**：order・report は `060`，実験は `ex007_e2_cifar10_snapshot_gradient_factorial`，
   集計は `analysis_e2.py`，テストは `tests/test_e2_programs.py`，可視化は `visualize_result_E2.ipynb`，
   説明文書は `ex007_.../README.md`，実行ログは `outputs/_run_logs/ex007*.log`．
3. **変更してよいファイル**：上記の新規ファイル・ディレクトリのみ．`common/` を含む既存のファイルは変更しない．
   振る舞いを変える必要がある関数は実験のディレクトリにコピーしてから変更する．共通化の提案は report の「統合時の注意」に，
   `document.md` への追記内容は README.md に書く．
4. **環境構築**：`.venv_pytorch`（Python 3.12）に `programs_review/requirements_pytorch_gpu.txt` をインストールする．
   並列数は，並列数1で1試行を数エポック実行したときの DRAM・VRAM の使用量から決める．
5. **実装**：2×3 の要因配置の6手法（`SVRG_final`，`SVRG`，`SVRG_average`（新規 `SVRGAverage`），
   `NFG_SVRG_final`，`NFG_SVRG`，`ASAI_SVRG`）．学習ループ `e2_training.py` は `common/vr_image_training.py`
   の一部の関数をコピーし，全勾配を用いる手法の集合による分岐と $ \omega $（`omega_snapshot`，`omega_average`）
   の記録を加える．並列実行は `e2_runner.py`（`e0_runner.py` のコピー）．
6. **一致の確認**（本実行の前）：(3.1) `SVRG`，`NFG_SVRG`，`ASAI_SVRG`，シード0，3エポックを E0 のログと比較する．
   0エポック目は丸め誤差の範囲で，`oracle_calls` は完全に一致すべきである．(3.2) $ \omega $ の記録の有無で
   `ASAI_SVRG`，シード0，2エポックの記録がビット単位で一致することを確認する．
7. **本実行**：6手法 × 5シード = 30試行（$ \eta = 10^{-3} $，バッチサイズ128，48エポック，$ \lambda = 5\times10^{-4} $）．
   E0 の3手法の結果も流用せず再実行する．
8. **集計・可視化**：`analysis_e2.py` で図（`figures/`）と表（`outputs/e2_summary.md`，対応のある差の表を含む）を作り，
   ノートブックを実行済みにする．
9. **テスト**：指示書2.5節の6項目以上を CPU で数秒で終わる小さな問題で確認し，既存のテストを含め全て通す．
10. **レポート**：`.reports/report_060.md` に実行環境，一致の確認，図表，実験計画4.6節の3論点への回答（事実と推測を分ける），
    予算の扱い（(a) $ \#\mathrm{grad}/N \le 90 $ と (b) 同じエポック数），統合時の注意，未解決の論点を書く．
11. **成果物**：`E2_results.zip` を作り，`unzip -l` の結果を report の末尾に貼る．

補足：
- このPCの GPU（RTX 3090，24GB）は元のPC（RTX PRO 6000，96GB）より小さいため，並列数は実測に基づいて小さめに決める．
- `.ai/ai-dev-kit` のサブモジュールが未初期化であったため，規約を読むために `git submodule update --init .ai/ai-dev-kit` を実行した（ファイルの変更はない）．
