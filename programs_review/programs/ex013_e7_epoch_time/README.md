# ex013_e7_epoch_time：1エポックの時間の短い計測（E7 の B）

## 1. 目的

論文の表で示す1エポックの時間を，全てのデータセットとバッチサイズについて，同じ規則で計測する．学習曲線は作らない．

同じ PC で同じ計測をやり直しても，数時間の学習の実時間は5%程度ばらつく．一方，1つの試行の中では実時間はエポック数にほぼ比例する．
そこで，各手法を数エポックだけ，手法を交互に入れ替えながら繰り返して計測し，1エポックの時間を求める．

## 2. 条件

| 項目 | 設定 |
|---|---|
| データセット・モデル | ImageWoof（ResNet18（LayerNorm），`ex009` と同じ），CIFAR-10（AlexNet（LayerNorm），`ex007` と同じ） |
| バッチサイズ | 32，128，512 |
| 手法 | `ASAI_SVRG`（continue），`NFG_SVRG_final`，`SVRG_final`（以上2つは `snapshot`），`SGD` |
| 学習率・正則化 | $ \eta = 10^{-3} $，$ \lambda = 5\times10^{-4} $ |
| シード | 0 のみ |
| エポック数 | 4（1エポック目は準備運転．2〜4エポック目を計測に使う） |
| 繰り返し | 3周 |

試行の順序：データセットとバッチサイズの組（ImageWoof の32・128・512，CIFAR-10 の32・128・512）ごとに，4手法を ASA-SVRG → NFG-SVRG → SVRG → SGD の順に交互に3周実行する．
失敗した試行は，同じ組の3周が終わった直後に再実行する．

計測の規則は `ex009` と同じである．全ての試行を1つずつ逐次に，新しいプロセスで実行する．各試行の前に GPU のメモリ使用量が `--idle-mib` 以下になるまで待つ．`torch.set_num_threads(1)`，ローダーのワーカー数0，cuDNN の決定論的設定とする．

1エポックの時間は，`time_optimizer + time_svrg_full_grad` の累積値 $ T_{e} $ を用いて $ (T_{4} - T_{1})/3 $ とする．
SVRG の初期の全勾配，近似誤差の評価のためだけの全勾配（`time_diagnostic`），評価（`time_evaluation`）は含めない．

保存先：`outputs/ex013_e7_epoch_time/{dataset}/bs{B}/{手法}/rep{r}/{config.json,log.json}`．
`config.json` には周回の番号 `repeat` と，試行の開始時刻 `started_at`・終了時刻 `finished_at` を記録する．`best_model.pth` は試行の終了後に削除する．

## 3. ファイルとコピー元

| ファイル | 内容 |
|---|---|
| `data.py`，`model.py`，`timed_training.py` | `ex009_e3_imagewoof_walltime/` から変更せずにコピーした |
| `data_cifar10.py`，`model_cifar10.py` | `ex007_e2_cifar10_snapshot_gradient_factorial/` の `data.py`，`model.py` を変更せずにコピーした |
| `train.py` | `ex009` の `train.py` を元に，手法と内側ループの開始点の表，データセット・バッチサイズ・周回のループ，組ごとの再実行，`config.json` への時刻の記録を加えた |

## 4. 実行方法（`programs_review/programs` から）

```bash
# 動作の確認（4試行）
python ex013_e7_epoch_time/train.py --datasets imagewoof --batch-sizes 128 --repeats 1 \
    --output-root ../outputs/_verify/ex013
# 本計測（72試行）
python ex013_e7_epoch_time/train.py > ../outputs/_run_logs/ex013.log 2>&1
# 集計
python analysis_e7.py
```

- 途中で止まった場合は，同じコマンドで再開すると，完了済みの試行を飛ばして同じ順序で残りを実行する．
- 動作の確認の結果は `outputs/ex013_e7_epoch_time/consistency.md` にある．
