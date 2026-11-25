# report_084：E11 Mushroom における全てのスナップショットと内側ループの開始点の構成（ex006 の追加）

指示書：`実験指示E11.md`．指示：`.orders/order_084.md`．
番号は本PCの次の空き番号（`order_084`／`report_084`）とした．指示書3節が変更の理由として挙げる「order_114」は，指示書を作成した側の番号と解釈し，`train.py` のコメントにはそのまま記した．

## 0. 要点

- 新規の5構成（random の SVRG・NFG-SVRG の continue，SVRG（average）の reset・continue，ASA-SVRG の continue）× 全7 ρ（random reshuffling）＋ 3 ρ（一様サンプリング）× 5シードの **250試行が全て完了した**．打ち切り・失敗は無い．実施済みの5構成は再実行していない．
- `ex006_e1_mushroom_rho_sampling/train.py` に `SVRG_average` を加え，`inner_start="last_iterate"` を continue の構成として用いるように docstring を改めた．学習の計算は変更していない（17行の追加，7行の削除）．
- 本実行の前の確認では，既存の構成（`SVRG_random`・reset・ρ = 298.4・シード0）が既存のログと**ビット単位で一致**した．`SVRG_average` の `oracle_calls` は $ N + 3Ns $，`approx_error` は厳密に0であった．
- 集計：`programs_review/outputs/e11_summary.md`（10構成 × 7 ρ の表，一様サンプリングの誤差床，打ち切り，実行の記録）．図は指示書のとおり作っていない．
- **注意（8節）**：既存の `analysis_e1.py` と，論文側の `make_e1_paper_assets.py` は `inner_start` で試行を絞り込まない．そのため，E11 の結果を加えたディレクトリでそのまま実行すると，ASA-SVRG・`SVRG_random`・`NFG_SVRG_random` の各セルに reset と continue の10試行が混ざる．

## 1. 実行環境と並列数

| 項目 | 内容 |
|---|---|
| OS | Windows 上の WSL2（Linux 6.18.33.2-microsoft-standard-WSL2） |
| CPU | Intel Xeon w7-3565X（32コア・64論理コア），メモリ 125 GB |
| Python・主要パッケージ | Python 3.12.15（`.venv_pytorch`），torch 2.11.0+cu128（CPU で計算．GPU は使わない） |

**並列数：32．** 根拠：並列数1の確認で，1プロセスの最大常駐メモリは約 0.90 GB，1外側ループは約 4.6 s であった．各プロセスは1スレッドなので，物理コア数と同じ32とした（メモリは約 29 GB）．32並列でも1外側ループは約 5.0 s（tqdm，評価を含む）であり，遅れは小さかった．

## 2. プログラムの変更（指示書3節）

| ファイル | 内容 |
|---|---|
| `programs_review/programs/ex006_e1_mushroom_rho_sampling/train.py` | `SVRGAverage` の import，`ALL_METHODS`・`FULL_GRADIENT_METHODS`・`_VARIANCE_REDUCED_OPTIMIZER_CLASSES` に `SVRG_average` を追加．docstring の `inner_start` の説明を「`"last_iterate"` は continue の構成（E11）に使用する」に改め，変更の理由（order_114，指導教員の指示．本PCの記録は order_084）を記した． |
| `programs_review/programs/analysis_e11.py` | 新規．集計（`e11_summary.md`）． |
| `programs_review/tests/test_e11_programs.py` | 新規（21件）． |

`SVRGAverage`（`common/optimizers/svrg_average.py`．E2 で追加済み）は変更していない．`begin_epoch` で乱数を消費しないため，既存の試行の添字の列は変わらない．

テストの内容：
1. `SVRG_average` の外側ループの後のスナップショットが，更新前のパラメータ $ {\mathbf w}^{0},\dots,{\mathbf w}^{K-1} $ の平均と一致する（reset・continue）．
2. `SVRG_average` の `approx_error` が $ 10^{-28} $ 未満である．
3. `last_iterate` で，外側ループ2以降の内側ループの開始点が前の内側ループの最終点と完全に一致する（6手法）．最終点以外の手法では，開始点がスナップショットと異なる外側ループがある．
4. `snapshot` では従来どおり開始点が $ {\mathbf z}_{s} $ である（6手法）．
5. `SVRG_average` の `oracle_calls` が $ N + 3Ns $ である．手法の一覧が正しい．
6. 指示書5節のコマンドのタスク数（continue 200，reset 50）．
7. 集計の関数（$ \hat{r} $ の最小二乗，到達の $ \#\mathrm{grad}/N $，最後の20の中央値，`inner_start` での絞り込み）が手計算と一致する．

テストを最初に実行したとき，3の「開始点がスナップショットと異なる」確認が `NFG_SVRG_random` で失敗した．テスト用のデータは $ K = 12 $ と小さく，外側ループ1で偶然に最終点がスナップショットに選ばれていた（確認済み）．プログラムの誤りではないため，「少なくとも1つの外側ループで異なる」という確認に改めた．

```bash
cd programs_review && ../.venv_pytorch/bin/python -m pytest tests -q
```

全体で240件が通過した（`test_e11_programs.py` の21件を含む）．

## 3. 本実行の前の確認（指示書4節）

詳細は `programs_review/outputs/ex006_e1_mushroom_rho_sampling/consistency_e11.md` にある．ログ：`programs_review/outputs/_run_logs/ex006_e11_verify.log`．

1. **既存の結果が変わらないこと**：`SVRG_random`・`snapshot`・random reshuffling・ρ = 298.4・シード0を3外側ループ実行し，既存のログと比べた（`compare_with_existing_logs.py`）．
   - 比較先の既存のログは，本PCで 2026-11-20 に実行したものである（`report_071`）．
   - `oracle_calls`，`objective_gap`，`accuracy`，`approx_error`，`grad_norm_sq` の4点が全てビット単位で一致した．
2. **SVRG（average）の動作**：`SVRG_average` の reset・continue を3外側ループ実行した．
   - `oracle_calls` は 0，29244，51177，73110 であり，$ N + 3Ns $（$ N = 7311 $）と一致した．
   - `approx_error` は全て0であった．
   - reset と continue は外側ループ1で同じ値であり，外側ループ2から異なる．
3. **並列数**：1節のとおり．

## 4. 本実行（指示書5節）

```bash
cd programs_review/programs
python ex006_e1_mushroom_rho_sampling/train.py \
    --methods SVRG_random NFG_SVRG_random SVRG_average ASAI_SVRG --inner-starts last_iterate \
    --samplings random_reshuffle uniform_with_replacement --workers 32 > ../outputs/_run_logs/ex006_e11_continue.log 2>&1
python ex006_e1_mushroom_rho_sampling/train.py \
    --methods SVRG_average --inner-starts snapshot \
    --samplings random_reshuffle uniform_with_replacement --workers 32 > ../outputs/_run_logs/ex006_e11_reset.log 2>&1
```

- 期間：2026-11-25 09:52:39 〜 11:02:45（1時間10分）．
- `ex006_e11_continue.log`：`[done]` 200件．`ex006_e11_reset.log`：`[done]` 50件．どちらも `[skip]` は0件で，「全ての学習が終了しました．」が出力された．エラーは無い．
- 全250試行が外側ループ100まで記録した．目的関数誤差が非有限になって打ち切られた試行は無い．
- 指示書のコマンドは `tee` を用いるが，バックグラウンドで実行したためリダイレクトでログを保存した（内容は同じ）．

## 5. 結果（`programs_review/outputs/e11_summary.md`）

### 5.1 random reshuffling の $ \hat{r} $（5シードの平均 ± 標準誤差）

$ \hat{r} $ は指示書6節の定義である．$ s = 1 $ から $ \Delta_{s} > 10^{-12} $ が続く範囲で，最小二乗の傾きから求め，シードの間で平均した．
この計算は，論文側の `ASAI_SVRG_Paper/programs/make_report090_e1.py`（`compute_hat_r_lsq`）を読んで確認したうえで，`analysis_e11.py` に同じ式を書いた．

| ρ | SVRG random reset | SVRG random cont. | SVRG final | SVRG avg. reset | SVRG avg. cont. | NFG random reset | NFG random cont. | NFG final | ASA reset | ASA cont. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.0027 | 0.0027 | 0.0009 | 0.0144 | 0.0017 | 0.1538 | 0.1810 | 0.1241 | 0.0137 | 0.0034 |
| 0.64 | 0.0022 | 0.0032 | 0.0018 | 0.0068 | 0.0016 | 0.1069 | 0.1214 | 0.0929 | 0.0066 | 0.0029 |
| 0.8 | 0.0027 | 0.0035 | 0.0024 | 0.0040 | 0.0015 | 0.0863 | 0.0895 | 0.0739 | 0.0042 | 0.0027 |
| 1.0 | 0.0031 | 0.0038 | 0.0032 | 0.0027 | 0.0014 | 0.0700 | 0.0740 | 0.0636 | 0.0030 | 0.0025 |
| 2.0 | 0.0039 | 0.0047 | 0.0044 | 0.0009 | 0.0013 | 0.0623 | 0.0637 | 0.0574 | 0.0018 | 0.0023 |
| 10 | 0.0095 | 0.0103 | 0.0084 | 0.0002 | 0.0009 | 0.0818 | 0.0898 | 0.0756 | 0.0016 | 0.0023 |
| 298.4 | 0.0127 | 0.0135 | 0.0090 | —（注） | 0.0009 | 0.1071 | 0.1108 | 0.0850 | 0.0017 | 0.0025 |

（注）SVRG（average）・reset・ρ = 298.4 は，全てのシードで外側ループ3までに $ \Delta_{s} \le 10^{-12} $ となった．そのため直線をあてはめる点が足りず，$ \hat{r} $ を求めていない（参照先の定義と同じ扱い）．シード0の $ \Delta_{s} $ は $ 2.9\times10^{-5} $，$ 3.9\times10^{-9} $，$ 6.2\times10^{-13} $ であった．

標準誤差，$ 10^{-10} $ への到達の $ \#\mathrm{grad}/N $，最後の20外側ループの近似誤差，テスト精度・正解数は `e11_summary.md` 1節にある．

### 5.2 一様サンプリングの誤差床（最後の20外側ループの中央値，5シードの平均）

| ρ | 構成 | 目的関数誤差 | 近似誤差 |
|---:|---|---:|---:|
| 0.5 | NFG random reset / cont. | 1.47e-04 / 1.48e-04 | 4.09e-04 / 4.09e-04 |
| 0.5 | NFG final | 1.73e-04 | 4.15e-04 |
| 0.5 | ASA reset / cont. | 1.38e-04 / 1.38e-04 | 4.04e-04 / 4.04e-04 |
| 1.0 | NFG random reset / cont. | 1.58e-04 / 1.57e-04 | 4.03e-04 / 4.03e-04 |
| 1.0 | NFG final | 1.72e-04 | 4.12e-04 |
| 1.0 | ASA reset / cont. | 1.54e-04 / 1.54e-04 | 4.04e-04 / 4.04e-04 |
| 298.4 | NFG random reset / cont. | 1.69e-04 / 1.69e-04 | 4.06e-04 / 4.06e-04 |
| 298.4 | NFG final | 1.75e-04 | 4.16e-04 |
| 298.4 | ASA reset / cont. | 1.64e-04 / 1.64e-04 | 4.04e-04 / 4.04e-04 |

全勾配の構成（SVRG の random・final・average）は，目的関数誤差が $ 10^{-16} $ 以下（丸め誤差の水準．負の値を含む），近似誤差が0であった．

### 5.3 その他

- テスト精度は，全ての構成・ρ で 88.66 ± 0.36%（正解数 720.8 ± 2.9／813）であった．いずれも最適解に収束したため，最後の点の精度は同じである．
- 打ち切られた試行は無い．

## 6. 所見

### 6.1 事実

- **ASA-SVRG**：continue は小さい ρ で reset より速く，大きい ρ でやや遅い．
  - $ \hat{r} $ は ρ = 0.5 で 0.0034 と 0.0137，ρ = 298.4 で 0.0025 と 0.0017（continue と reset の順）．
  - $ 10^{-10} $ への到達は，ρ = 0.5 で 10.4 と 13.6，ρ = 298.4 で 10.0 と 9.2 の $ \#\mathrm{grad}/N $ であった．
- **SVRG（average）**：同じ傾向がより大きい．
  - $ \hat{r} $ は ρ = 0.5 で continue 0.0017，reset 0.0144 である．
  - ρ ≥ 2 では reset の方が小さく，ρ = 10 で 0.0002 と 0.0009（reset と continue の順）．
- **ランダム点（SVRG・NFG-SVRG）**：continue と reset の差は小さい．$ \hat{r} $ は，SVRG の ρ = 0.5（共に 0.0027）を除き，continue の方がわずかに大きい（例：NFG-SVRG の ρ = 0.5 で 0.1810 と 0.1538）．ただし，ランダム点の $ \hat{r} $ はシードの間の標準誤差が大きく（0.01〜0.04），差の多くはその範囲である．
- **一様サンプリングの誤差床**：continue と reset はほぼ同じである（ASA-SVRG・NFG-SVRG（random）で，目的関数誤差・近似誤差とも有効数字3桁でほぼ一致）．
- **average の全勾配と平均勾配（SVRG（average）と ASA-SVRG．同じ開始点どうし）**：
  - reset の ρ ≤ 1 では $ \hat{r} $ が近い（ρ = 0.5 で 0.0144 と 0.0137，ρ = 1.0 で 0.0027 と 0.0030）．
  - reset の ρ ≥ 2 では SVRG（average）の方が小さい（ρ = 10 で 0.0002 と 0.0016）．
  - continue では全ての ρ で SVRG（average）の方が小さい（ρ = 0.5 で 0.0017 と 0.0034，ρ = 298.4 で 0.0009 と 0.0025）．

### 6.2 推測

- 平均点のスナップショットは，小さい ρ（小さい学習率）では内側ループの途中の点を多く含むため，最終点より最適解から遠い．そのため，平均点から内側ループを再び始める reset は，continue より遅くなると考えられる．大きい学習率では最終点がばらつき，平均の方が良い出発点になると考えられる．
- 一様サンプリングの誤差床が開始点によらないのは，誤差床が平均勾配の近似誤差（サンプリングのばらつき）で決まり，開始点の影響は外側ループを重ねると消えるためと考えられる．
- SVRG（average）と ASA-SVRG の違いはスナップショット勾配（全勾配か平均勾配か）だけである．したがって，両者の $ \hat{r} $ の差は近似誤差の影響と考えられる．その影響は reset の小さい ρ では小さく，大きい ρ と continue では $ \hat{r} $ が2〜8倍になる程度である．ただし，どちらも $ \hat{r} \le 0.015 $ であり，100外側ループの間に丸め誤差の水準に達する．

## 7. 成果物

| ファイル | 内容 |
|---|---|
| `programs_review/outputs/e11_summary.md` | 集計（指示書6節） |
| `programs_review/outputs/ex006_e1_mushroom_rho_sampling/consistency_e11.md` | 本実行の前の確認 |
| `programs_review/outputs/ex006_e1_mushroom_rho_sampling/{SVRG_random,NFG_SVRG_random,ASAI_SVRG}/inner_last_iterate_*`，`SVRG_average/` | 新しい試行のログ（250試行．`log.json`，`config.json`） |
| `programs_review/outputs/_run_logs/ex006_e11_continue.log`，`ex006_e11_reset.log`，`ex006_e11_verify.log` | 実行ログ |
| `programs_review/programs/ex006_e1_mushroom_rho_sampling/train.py`，`analysis_e11.py`，`tests/test_e11_programs.py` | 変更・追加したプログラム |

`outputs/_verify/ex006_e11/` は確認用であり，Git の管理外である（zip にも入れていない）．

## 8. 統合時の注意・未解決の論点

- **既存の集計が reset と continue を混ぜる**：`programs_review/programs/analysis_e1.py` の `select_runs` は，手法・サンプリング・ρ だけで試行を選ぶ．`inner_start` では絞り込まない．
  - 論文側の `ASAI_SVRG_Paper/programs/make_e1_paper_assets.py` の `load_runs` も，手法のディレクトリの下の全ての条件を読む．
  - E11 の結果を含むディレクトリでこれらを実行すると，ASA-SVRG・`SVRG_random`・`NFG_SVRG_random` の表と図（表III を含む）に continue の試行が混ざる．
  - `SVRG_average` は手法の一覧に無いため混ざらない．
  - 既存のプログラムは変更していない．統合の前に `config["inner_start"] == "snapshot"` で絞り込む修正が必要である．
- **PC の違い**：既存の最終点の SVRG・NFG-SVRG と ASA-SVRG の reset は別の PC で実行されており，本PCとは $ f^{*} $ の最後の桁が異なる（`consistency_final_this_pc.md`．差は $ 2.2\times10^{-16} $ 以下）．
  - 新しい試行と，ランダム点の reset は本PCで実行した．
  - $ \hat{r} $（閾値 $ 10^{-12} $）や誤差床（$ 10^{-4} $ の水準）への影響は無視できる．ただし，SVRG の目的関数誤差の誤差床（$ 10^{-16} $ の水準）は PC によって値が変わる．
- `programs_review/document.md` には，E9 以降（E9，E10，ex016，E11）をまだ反映していない．
- 図と論文用の表は，指示書のとおり別の PC で作る．

## 9. 成果物の zip

`E11_results.zip`（リポジトリのルート）．展開先はリポジトリのルートである．

`unzip -l E11_results.zip` の一覧（試行ごとの `log.json`・`config.json` とディレクトリの項目は，下の数にまとめた）：

```text
Archive:  E11_results.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
     3268  2026-11-25 09:48   .orders/order_084.md
    17197  2026-11-25 11:08   .reports/report_084.md
    31120  2026-11-25 09:49   programs_review/programs/ex006_e1_mushroom_rho_sampling/train.py
    21344  2026-11-25 11:06   programs_review/programs/analysis_e11.py
    14807  2026-11-25 09:54   programs_review/tests/test_e11_programs.py
    15779  2026-11-25 11:06   programs_review/outputs/e11_summary.md
     3702  2026-11-25 09:55   programs_review/outputs/ex006_e1_mushroom_rho_sampling/consistency_e11.md
  2339729  2026-11-25 10:46   programs_review/outputs/_run_logs/ex006_e11_continue.log
   565708  2026-11-25 11:02   programs_review/outputs/_run_logs/ex006_e11_reset.log
     4273  2026-11-25 09:52   programs_review/outputs/_run_logs/ex006_e11_verify.log
---------                     -------
  9761327                     811 files
（このほか log.json 250 件，config.json 250 件，ディレクトリの項目 301 件）
```
