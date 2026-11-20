# `.orders/order_072.md`

## 1. ユーザーからの指示（原文）

実験3：NFG-SVRG だけを restart で計測し直す（5.6節） を実施してください

## 2. 指示の解釈

- `実験計画.md` 5.6節に従い，E3（`ex009_e3_imagewoof_walltime`）の NFG-SVRG を `inner_start="snapshot"`（restart）で再計測する．100エポック，シード0〜4，5.3節の計測の規則（逐次実行，GPU の空きの確認）による．
- 実装：
  - `train.py` に `--inner-start {last_iterate,snapshot}` を追加し，条件名と `config.json` の `inner_start` に反映する．既定値は `last_iterate` とし，既存の試行の条件名とログは変えない．
  - `tests/test_e3_programs.py` に，`--inner-start` を省略したときの条件名とタスクが変更前と同じであることを確認するテストを追記する．
  - `analysis_e3.py` は手法ごとに用いる条件名を指定できるようにし，論文用の図と表では NFG-SVRG に `inner_snapshot_...` を用いる．`last_iterate` の NFG-SVRG は確認用の図にのみ残す．
- 本実行前後の確認（5.6節）：
  1. E0 の ImageWoof の NFG-SVRG（`ex003_e0_imagewoof_resnet18_layernorm`，`snapshot`）と，学習損失・テスト精度が全シードで完全に一致すること．
  2. `time_optimizer`／エポックを当初の NFG-SVRG（continue）の 10.708 ± 0.086 s と比べ，差が3%以内であることを確認する．3%を超える場合は報告し，4手法の再計測（約6.5時間）を行うかはユーザーに判断を仰ぐ．
  3. 使用機器とソフトウェアのバージョンが当初の計測と同じであることを `config.json` で確認する．
- 計測中は他の実験を実行しない．
- 報告は `.reports/report_072.md` とする．Gitのcommit・pushはユーザーの指示があるまで行わない．
