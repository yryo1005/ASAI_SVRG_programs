# `.orders/order_039.md`

## 1. ユーザーからの指示（原文）

ex0051のlr0.001_bs512_lambda0.0005_epochs505と同条件でSGDだけEpoch数を2倍にして学習して
ください。終了の見積もり時間を報告して下さい。

## 2. 指示の解釈

`programs/ex0051_imagewoof_resnet18_bs128_longrun/`の条件B（`CONDITIONS`リストの
`{"batch_size": 512, "learning_rate": 0.001, "epochs": 505}`，`.orders/order_036.md`，
`.reports/report_036.md`）と同一のバッチサイズ・学習率・正則化係数（$\lambda=5\times10^{-4}$）
のまま，SGDのみを対象にエポック数を2倍（505→1010）に延長して追加学習する。他の3手法
（SVRG, NFG SVRG, ASAI SVRG）は対象外とし，既存の60条件（基準条件・条件A・条件B）は変更
しない。5Seed（0〜4）は既存条件と同一とする。

`.orders/order_038.md`（ex0024，SGDのみエポック数を倍にした追加学習）と同種の指示であり，
同じ実装パターン（`run_single_experiment`へのエポック数の明示的な受け渡し，`main`関数への
新規フェーズの追加）を踏襲する。

## 3. 実行前の計算コスト見積もり

条件Bの実測`elapsed_time`（4プロセス並列実行下，資源競合を含む実測値）から，1エポックあたり
所要時間を算出し，2倍のエポック数に適用して見積もる。VRAM使用量は，`.reports/report_036.md`
で実測済みの通りバッチサイズ512では手法によらず約22.5GB（reservedベース）に達するため
（`report_036.md`のパイロット計測でSGD単独でも22.52GB），条件Bと同様に4プロセス並列で実行
する。5Seedに対し4プロセス並列のため，2ラウンド（1ラウンド目4Seed，2ラウンド目1Seed）と
なり，見積もり時間は単一タスクの所要時間の約2倍になる。見積もり結果はチャットで報告する。

## 4. 成果物

学習完了後，`.reports/report_039.md`，`document.md`，`visualize_result.ipynb`を更新する。
