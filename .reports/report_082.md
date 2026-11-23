# report_082：E10 付録B.1（a9a）の BCE の独立一様サンプリング

指示書：`実験指示E10.md`．指示：`.orders/order_082.md`．番号は指示書0.3節のとおり（`order_082`，`report_082`，`ex015`）であり，既存の番号との衝突は無かった．

## 0. 要点

- 20試行（4手法 × 5シード，a9a の BCE，独立一様サンプリング）が全て完了した．打ち切り・失敗は無い．本PCで E9（GPU）と同時に CPU で実行し，本実行の実時間は34分であった．
- 3.1節の確認は全て合格した．
  - (a) random reshuffling では，E8 の2手法（`SVRG_random`，`NFG_SVRG_random`）はビット単位で一致した．E4 の2手法（`SGD`，`ASAI_SVRG`）は，E4 の本実行（別のPC）とは実数の量が相対 $ 10^{-12} $ 程度で異なったが，本PCで E4 のプログラム（無変更）を実行した結果とはビット単位で一致した．
  - (b) 一様サンプリングでは，4手法とも記録と `config.json` の `sampling` が正しく，random reshuffling と値が異なった．
- 結果（6節）：
  - **一様サンプリングでは，NFG-SVRG と ASA-SVRG に誤差床が現れた．** 目的関数誤差の床は $ 1.24\times10^{-4} $ と $ 1.07\times10^{-4} $，近似誤差の床は $ 5.17\times10^{-5} $ と $ 4.40\times10^{-5} $ であり，両手法で同程度である．random reshuffling（E4・E8）では，同じ手法の床は倍精度の丸め誤差の水準（目的関数誤差 $ 10^{-17} $，近似誤差 $ 10^{-31} $〜$ 10^{-29} $）である．
  - **近似誤差の床は $ \sigma^{2}/K = 6.71\times10^{-5} $ と同じ桁である．** 比（床 ÷ $ \sigma^{2}/K $）は NFG-SVRG が 0.770 ± 0.029，ASA-SVRG が 0.656 ± 0.040 である．実験1（Mushroom）の 0.98〜1.0 よりは小さい．床の定義が中央値であることと，a9a では勾配の分散が少数の方向に集中していることで説明できると考えられる（7.2節）．
  - SVRG（全勾配）は一様サンプリングでも倍精度の床（$ 10^{-17} $）に達した．SGD は一様サンプリングと random reshuffling で同程度（$ 3.75\times10^{-2} $ と $ 3.81\times10^{-2} $）であった．
  - 以上は実験1（Mushroom，$ \rho = 298.4 $）と同じ傾向である．

## 1. 実行環境と並列数

| 項目 | 内容 |
|---|---|
| OS | Windows 上の WSL2（Linux 6.18.33.2-microsoft-standard-WSL2） |
| CPU | Intel Xeon w7-3565X（64論理コア），メモリ 125 GB |
| Python・主要パッケージ | Python 3.12.15（`.venv_pytorch`），torch 2.11.0+cu128（計算は CPU のみ），numpy 2.4.6，scipy 1.18.1 |
| 実行の設定 | 1試行1プロセス，`torch.set_num_threads(1)`（E4・E8 と同じ） |
| 並列数 | 20（20試行を全て同時に実行） |

並列数の根拠：ai-dev-kit の規約に従い，3.1節 (b) の確認（4試行を並列数1で順に実行）を `/usr/bin/time -v` で計測した．最大常駐メモリは 0.89 GB であった．
本実行と同時に E9（`ex005` の ASA-SVRG（reset）・バッチサイズ512，GPU，4プロセス）が動いており，E9 は CPU 4コア（各約100%）と約 7 GB のメモリ（システム全体の使用量）を使っていた．残りは論理コア約60，空きメモリ約 116 GB である．
E10 の20並列はメモリ約 18 GB，20コアであり，余裕がある．E9 は精度の実験であり実時間を計測しないため，同時に実行しても E9 の結果は変わらない．E10 の `elapsed_time` も結論には用いない．

所要時間：

| 作業 | 期間 | 実時間 |
|---|---|---:|
| 3.1節の確認（(a) 4並列，(b) 並列数1，同時に実行） | 2026-11-23 12:29:30〜12:32:42 | 3分 |
| 本実行（20試行，20並列） | 2026-11-23 12:34:12〜13:08:22 | 34分 |

## 2. 作成したファイル

| ファイル | 内容 |
|---|---|
| `programs_review/programs/ex015_e10_a9a_bce_uniform/train.py` | `ex014_e8_a9a_supplement/train.py` をコピーし，指示書2節の変更（サンプリング方式の選択，保存先，引数）を加えた |
| 同 `data.py`，`model.py` | `ex014_e8_a9a_supplement/` から変更せずにコピーした（テストで同一であることを確認） |
| 同 `compare_e10.py` | 3.1節の確認．(a) 既存の結果との項目ごとの比較，(b) 一様サンプリングの動作の確認を行い，表を `consistency.md` に書く |
| 同 `README.md` | 目的，E8 との違い，実行方法，出力，コピー元 |
| `programs_review/programs/analysis_e10.py` | 誤差床，$ \sigma^{2}/K $ の表（`e10_summary.md`）．E4・E8 の結果のパスは `--e4-root`・`--e8-root` で受け取る |
| `programs_review/tests/test_e10_programs.py` | テスト13件（5節） |
| `programs_review/outputs/ex015_e10_a9a_bce_uniform/` | 20試行の結果，`consistency.md` |
| `programs_review/outputs/e10_summary.md` | 集計表 |
| `programs_review/outputs/_run_logs/ex015.log`，`ex015_verify_{rr,uniform}.log` | 実行ログ |
| `.orders/order_082.md`，`.reports/report_082.md` | 指示と本レポート |

`train.py` の主な変更：

- `ALL_SAMPLINGS = ("random_reshuffle", "uniform_with_replacement")` と `get_epoch_indices(rng, N_train, sampling)`（E1 の `ex006` と同一）を加えた．どちらの方式も外側ループの最初（`begin_epoch` の直前）に1回だけ乱数を消費する．
- `run_variance_reduced`・`run_sgd` に引数 `sampling` を加え，`config.json` の `sampling` にタスクの方式を記録する（E8 では定数）．
- 保存先を `{output_root}/bce/{手法}/inner_{inner_start}_{sampling}_{hp_name}/{seed}/` とした．
- `--part` を削除し，既定値を `--loss bce`，`--methods SGD SVRG_random NFG_SVRG_random ASAI_SVRG`，`--inner-starts snapshot`，`--samplings uniform_with_replacement`，`--workers 20` とした．

## 3. 確認（指示書3.1節）

詳細は `programs_review/outputs/ex015_e10_a9a_bce_uniform/consistency.md` にある．比較の対象はシード0の外側ループ0〜3（4点）である．

### 3.1 (a) random reshuffling での E4・E8 の再現

| 手法 | 比較先 | `oracle_calls` | `objective_gap` | `grad_norm_sq` | `accuracy` | `approx_error` |
|---|---|---|---|---|---|---|
| SVRG（ランダム点） | E8（本PC） | 一致 | 一致 | 一致 | 一致 | 一致 |
| NFG-SVRG（ランダム点） | E8（本PC） | 一致 | 一致 | 一致 | 一致 | 一致 |
| SGD | E4 の本実行（別のPC） | 一致 | 相対 $ 1.9\times10^{-15} $ | 相対 $ 8.6\times10^{-16} $ | 一致 | — |
| ASA-SVRG | E4 の本実行（別のPC） | 一致 | 相対 $ 1.3\times10^{-12} $ | 相対 $ 2.1\times10^{-14} $ | 一致 | 相対 $ 2.8\times10^{-3} $ |
| SGD | 本PCで E4 のプログラムを実行した結果 | 一致 | 一致 | 一致 | 一致 | — |
| ASA-SVRG | 本PCで E4 のプログラムを実行した結果 | 一致 | 一致 | 一致 | 一致 | 一致 |

（「一致」はビット単位の一致である．）

- 指示書は「CPU・単一スレッドなので全ての項目がビット単位で一致する」としている．しかし，E4 の本実行は別のPC（Intel Core Ultra 9 285K，`report_070.md`）で行われたため，E4 の2手法の実数の量はビット単位では一致しなかった．同じ現象は E8 でも確認済みである（`outputs/ex014_e8_a9a_supplement/consistency.md` 3.1節）．
- E8 のときに本PCで E4 のプログラム（`ex008_e4_a9a/train.py`，無変更）を同じ条件（シード0，3外側ループ）で実行した結果（`outputs/_verify/ex014_e4_ref_ex008_this_pc/`）がある．これと比べると，全ての項目がビット単位で一致した．したがって，コピーと変更で既存の手法の計算は変わっていない．E4 の手法の判定にはこちらを用いた．
- ASA-SVRG の `approx_error` の相対差 $ 2.8\times10^{-3} $ は，値そのものが $ 10^{-10} $ 程度と小さく，2つの勾配の差であるため丸め誤差に敏感なことによる．

### 3.2 (b) 一様サンプリングの動作

| 手法 | 記録の点数 | `config.json` の `sampling` | random reshuffling と異なり始める外側ループ（期待値） | それ以降も全て異なる |
|---|---:|---|---|---|
| SGD | 4 | `uniform_with_replacement` | 1（1） | はい |
| SVRG（ランダム点） | 4 | `uniform_with_replacement` | 1（1） | はい |
| NFG-SVRG（ランダム点） | 4 | `uniform_with_replacement` | 2（2） | はい |
| ASA-SVRG | 4 | `uniform_with_replacement` | 2（2） | はい |

- 指示書は「外側ループ1以降の値が異なること」を確かめるとしている．NFG-SVRG と ASA-SVRG は，$ {\mathbf g}_{0} = {\mathbf 0} $ かつ内側ループが $ {\mathbf w}_{0}^{0} = {\mathbf z}_{0} $ から始まるため，1回目の内側ループの補正勾配が $ {\mathbf 0} $ になる．そのため点が動かず，外側ループ1はサンプリングに依らず同じ値になる（E8 の3.2節と同じ）．したがって，この2手法では外側ループ2から異なることを確かめた．
- 外側ループ0は全手法で random reshuffling と一致した（初期値と評価は同じ）．

## 4. 本実行の確認

- ログ（`outputs/_run_logs/ex015.log`）の末尾は「全ての学習が終了しました．」であり，`[done]` が20件，`[failed]` は0件であった．
- 20試行の全てに101点（外側ループ0〜100）の記録があり，打ち切りは無い．
- 20試行の `config.json` の `sampling` は全て `uniform_with_replacement` である．

## 5. テスト

`programs_review/tests/test_e10_programs.py`（13件，CPU で約5秒）．

- `data.py`・`model.py` が `ex014` のものと同一であること．
- `get_epoch_indices` が，同じシードの生成器から E1（`ex006`）と同じ添字の列を返すこと（両方の方式，3外側ループ分）．`random_reshuffle` が E8 の `get_epoch_indices` とも同じであること．
- `uniform_with_replacement` の添字が $ [0, N) $ にあり，重複を含むこと．未知の方式で例外を出すこと．
- 既定の引数で20試行のタスクを作り，保存先が `inner_{inner_start}_{sampling}_{hp_name}/{seed}` の形式であること．
- `random_reshuffle` では，小さなデータでの学習の記録が E8 の実装とビット単位で一致すること（SVRG・NFG-SVRG のランダム点，ASA-SVRG）．
- `uniform_with_replacement` では SGD の記録が random reshuffling と異なること．
- 誤差床（最後の20点の中央値，非有限の値を除く）と平均 ± 標準誤差が手計算と一致すること．
- $ \sigma^{2} $ の計算が，サンプルごとの勾配（正則化項を含む）を素朴に計算した値と相対 $ 10^{-12} $ で一致すること．

全てのテスト（`python -m pytest -q tests`）は214件が通った（2026-11-23）．

## 6. 結果

集計表は `programs_review/outputs/e10_summary.md` にある．誤差床は，有限の値だけを残した系列の最後の20点の中央値である（E1 の表IIIと同じ定義．打ち切りが無いため外側ループ81〜100の中央値と同じ）．値は5シードの平均 ± 標準誤差（`ddof=1`）である．

### 6.1 表1：誤差床（一様サンプリングと random reshuffling）

random reshuffling の値は，SGD・ASA-SVRG が E4（`ex008_e4_a9a`），SVRG・NFG-SVRG のランダム点が E8（`ex014_e8_a9a_supplement`）の結果である．

| 手法 | 指標 | 一様サンプリング（E10） | 参考：random reshuffling（E4・E8） |
|---|---|---:|---:|
| SGD | 目的関数誤差 | $ 3.750\times10^{-2} \pm 3.3\times10^{-3} $ | $ 3.810\times10^{-2} \pm 3.2\times10^{-3} $（E4） |
| SVRG（ランダム点） | 目的関数誤差 | $ -2.220\times10^{-17} \pm 2.8\times10^{-17} $ | $ 0.000 \pm 2.5\times10^{-17} $（E8） |
| SVRG（ランダム点） | 近似誤差 | 0（全勾配） | 0（全勾配） |
| NFG-SVRG（ランダム点） | 目的関数誤差 | $ 1.243\times10^{-4} \pm 4.1\times10^{-6} $ | $ -2.220\times10^{-17} \pm 2.2\times10^{-17} $（E8） |
| NFG-SVRG（ランダム点） | 近似誤差 | $ 5.170\times10^{-5} \pm 2.0\times10^{-6} $ | $ 1.253\times10^{-31} \pm 1.3\times10^{-32} $（E8） |
| ASA-SVRG（reset） | 目的関数誤差 | $ 1.072\times10^{-4} \pm 2.9\times10^{-6} $ | $ 6.106\times10^{-17} \pm 1.8\times10^{-17} $（E4） |
| ASA-SVRG（reset） | 近似誤差 | $ 4.401\times10^{-5} \pm 2.7\times10^{-6} $ | $ 2.770\times10^{-29} \pm 5.0\times10^{-30} $（E4） |

試行数は全て5である．SGD の近似誤差は定義されない．

### 6.2 表2：$ \sigma^{2}/K $ と近似誤差の床

$ \sigma^{2} = \frac{1}{N}\sum_{n}\|\nabla f_{n}({\mathbf w}^{*}) - \nabla f({\mathbf w}^{*})\|^{2} $（重みと切片の124次元），$ K = N = 29304 $．$ {\mathbf w}^{*} $ はシードごとの学習データで L-BFGS-B により求め，$ f({\mathbf w}^{*}) $ は全シードで `config.json` の `f_star` と相対 $ 1.2\times10^{-16} $ 以下で一致した．

| シード | $ \sigma^{2} $ | $ \sigma^{2}/K $ | NFG-SVRG の近似誤差の床 | ASA-SVRG の近似誤差の床 | NFG-SVRG の床 ÷ $ \sigma^{2}/K $ | ASA-SVRG の床 ÷ $ \sigma^{2}/K $ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.964 | $ 6.703\times10^{-5} $ | $ 5.036\times10^{-5} $ | $ 4.439\times10^{-5} $ | 0.751 | 0.662 |
| 1 | 1.966 | $ 6.708\times10^{-5} $ | $ 4.961\times10^{-5} $ | $ 4.010\times10^{-5} $ | 0.739 | 0.598 |
| 2 | 1.969 | $ 6.718\times10^{-5} $ | $ 5.458\times10^{-5} $ | $ 3.673\times10^{-5} $ | 0.812 | 0.547 |
| 3 | 1.968 | $ 6.715\times10^{-5} $ | $ 5.752\times10^{-5} $ | $ 5.207\times10^{-5} $ | 0.857 | 0.775 |
| 4 | 1.970 | $ 6.722\times10^{-5} $ | $ 4.641\times10^{-5} $ | $ 4.678\times10^{-5} $ | 0.690 | 0.696 |
| 平均 ± 標準誤差 | $ 1.967 \pm 0.001 $ | $ 6.713\times10^{-5} \pm 3.4\times10^{-8} $ | $ 5.170\times10^{-5} \pm 2.0\times10^{-6} $ | $ 4.401\times10^{-5} \pm 2.7\times10^{-6} $ | 0.770 ± 0.029 | 0.656 ± 0.040 |

### 6.3 表3：シードごとの値（一様サンプリング）

| 手法 | シード | 目的関数誤差の床 | 近似誤差の床 | 打ち切り | 最終外側ループのテスト精度 |
|---|---:|---:|---:|---|---:|
| SGD | 0 | $ 4.515\times10^{-2} $ | — | なし | 0.7605 |
| SGD | 1 | $ 2.711\times10^{-2} $ | — | なし | 0.7842 |
| SGD | 2 | $ 3.793\times10^{-2} $ | — | なし | 0.7820 |
| SGD | 3 | $ 4.327\times10^{-2} $ | — | なし | 0.8149 |
| SGD | 4 | $ 3.405\times10^{-2} $ | — | なし | 0.7602 |
| SVRG（ランダム点） | 0 | $ -1.110\times10^{-16} $ | — | なし | 0.8010 |
| SVRG（ランダム点） | 1 | $ -5.551\times10^{-17} $ | — | なし | 0.8010 |
| SVRG（ランダム点） | 2 | 0 | — | なし | 0.8060 |
| SVRG（ランダム点） | 3 | $ 5.551\times10^{-17} $ | — | なし | 0.8078 |
| SVRG（ランダム点） | 4 | 0 | — | なし | 0.8109 |
| NFG-SVRG（ランダム点） | 0 | $ 1.150\times10^{-4} $ | $ 5.036\times10^{-5} $ | なし | 0.8041 |
| NFG-SVRG（ランダム点） | 1 | $ 1.280\times10^{-4} $ | $ 4.961\times10^{-5} $ | なし | 0.8023 |
| NFG-SVRG（ランダム点） | 2 | $ 1.147\times10^{-4} $ | $ 5.458\times10^{-5} $ | なし | 0.8032 |
| NFG-SVRG（ランダム点） | 3 | $ 1.286\times10^{-4} $ | $ 5.752\times10^{-5} $ | なし | 0.8023 |
| NFG-SVRG（ランダム点） | 4 | $ 1.355\times10^{-4} $ | $ 4.641\times10^{-5} $ | なし | 0.8142 |
| ASA-SVRG（reset） | 0 | $ 1.082\times10^{-4} $ | $ 4.439\times10^{-5} $ | なし | 0.8038 |
| ASA-SVRG（reset） | 1 | $ 1.022\times10^{-4} $ | $ 4.010\times10^{-5} $ | なし | 0.8020 |
| ASA-SVRG（reset） | 2 | $ 9.916\times10^{-5} $ | $ 3.673\times10^{-5} $ | なし | 0.8044 |
| ASA-SVRG（reset） | 3 | $ 1.117\times10^{-4} $ | $ 5.207\times10^{-5} $ | なし | 0.8063 |
| ASA-SVRG（reset） | 4 | $ 1.150\times10^{-4} $ | $ 4.678\times10^{-5} $ | なし | 0.8133 |

全ての試行で記録は100外側ループである．

### 6.4 追加の確認（集計用の一時的なスクリプトで求めた値．`e10_summary.md` には無い）

- **床に達しているか．** 近似誤差の外側ループ61〜80の中央値（5シードの平均）は NFG-SVRG $ 5.44\times10^{-5} $，ASA-SVRG $ 4.57\times10^{-5} $ であり，81〜100（$ 5.17\times10^{-5} $，$ 4.40\times10^{-5} $）とほぼ同じである．目的関数誤差も同様である（NFG-SVRG $ 1.20\times10^{-4} $ と $ 1.24\times10^{-4} $，ASA-SVRG $ 1.05\times10^{-4} $ と $ 1.07\times10^{-4} $）．したがって，外側ループ60以降は床に達している．
- **中央値と平均．** 外側ループ81〜100の近似誤差の平均（5シードの平均）は NFG-SVRG $ 7.06\times10^{-5} $（$ \sigma^{2}/K $ の1.05倍），ASA-SVRG $ 5.92\times10^{-5} $（0.88倍）であり，中央値（0.77倍，0.66倍）より $ \sigma^{2}/K $ に近い．
- **勾配の分散の有効次元．** シード0の $ {\mathbf w}^{*} $ での1サンプルの勾配の共分散行列 $ {\mathbf C} $（124 × 124，$ \mathrm{tr}\,{\mathbf C} = \sigma^{2} $）は，最大の固有値が全体の53% を占め，有効次元 $ (\mathrm{tr}\,{\mathbf C})^{2}/\mathrm{tr}({\mathbf C}^{2}) $ は 3.4 である．$ K $ 個の独立なサンプルの平均の誤差 $ {\mathbf e} $ を共分散 $ {\mathbf C}/K $ の正規分布で近似すると，$ \|{\mathbf e}\|^{2} $ の中央値は平均の0.74倍である（モンテカルロで20000点）．

## 7. 所見

### 7.1 事実

1. **一様サンプリングで NFG-SVRG と ASA-SVRG に誤差床が現れた．** 目的関数誤差の床は NFG-SVRG $ 1.24\times10^{-4} $，ASA-SVRG $ 1.07\times10^{-4} $，近似誤差の床は $ 5.17\times10^{-5} $ と $ 4.40\times10^{-5} $ である．両手法の床は同じ桁であり，ASA-SVRG が目的関数誤差で0.86倍，近似誤差で0.85倍とやや小さい．5シード全てで同じ傾向である（シード4の近似誤差のみ ASA-SVRG がわずかに大きい）．
2. **近似誤差の床は $ \sigma^{2}/K = 6.71\times10^{-5} $ と同じ桁である．** 比は NFG-SVRG が 0.69〜0.86（平均 0.770），ASA-SVRG が 0.55〜0.78（平均 0.656）である．外側ループ81〜100の平均で比べると 1.05 と 0.88 である（6.4節）．
3. **SVRG（全勾配）は，一様サンプリングでも倍精度の床（目的関数誤差 $ 10^{-17} $ 以下）に達した．** random reshuffling と同じである．**SGD の目的関数誤差は，一様サンプリングで $ 3.75\times10^{-2} $，random reshuffling で $ 3.81\times10^{-2} $ であり，差は標準誤差（$ 3\times10^{-3} $）の範囲である．** SGD の床は分散縮小の手法より2桁以上大きい．
4. **random reshuffling（E4・E8）との違い．** random reshuffling では，NFG-SVRG・ASA-SVRG の目的関数誤差の床は $ 10^{-17} $（倍精度の丸め誤差の水準），近似誤差の床は $ 10^{-31} $〜$ 10^{-29} $ である．一様サンプリングに変えると，この2手法だけが床を持つ（目的関数誤差で約13桁，近似誤差で約25桁大きい）．
5. **最終外側ループのテスト精度**は，分散縮小の3手法で 0.80〜0.81，SGD で 0.76〜0.81 である．床の有無はテスト精度にはほとんど現れない．
6. **実験1（Mushroom，$ \rho = 298.4 $）との比較．** 実験1の一様サンプリングでも，NFG-SVRG・ASA-SVRG は誤差床を持ち（目的関数誤差 $ 1.6\times10^{-4} $〜$ 1.8\times10^{-4} $），SVRG は倍精度の床に達し，random reshuffling では床が消えた（`outputs/e1_summary.md`）．E10 も同じ傾向である．一方，近似誤差の床と $ \sigma^{2}/K $ の比は，実験1では 0.98〜1.0（`report_071.md`）であり，E10 の 0.66〜0.77 より1に近い．

### 7.2 推測

1. NFG-SVRG と ASA-SVRG の近似勾配 $ {\mathbf g}_{s} $ は，内側ループで使った確率的勾配の平均である．床では内側ループの点がほぼ $ {\mathbf w}^{*} $ にあり，$ {\mathbf g}_{s} - \nabla f({\mathbf z}_{s}) $ は $ K $ 個の独立なサンプルの勾配の平均の誤差になる．そのため，近似誤差の期待値は $ \sigma^{2}/K $ になると考えられる．random reshuffling では $ K = N $ 個のサンプルを重複なく1回ずつ使うため，この誤差は打ち消し合って消える．これは実験1と同じ説明である．
2. E10 の比（中央値 ÷ $ \sigma^{2}/K $）が実験1より小さいのは，床の定義が中央値であり，a9a では $ \|{\mathbf e}_{s}\|^{2} $ の分布が右に長い裾を持つためと考えられる．勾配の分散の有効次元は 3.4 と小さく，正規近似では中央値が平均の0.74倍になる．NFG-SVRG の比 0.77 はこれとほぼ一致し，平均で比べると 1.05 である（6.4節）．実験1では有効次元が大きく，中央値と平均の差が小さいため，比が1に近かったと考えられる（実験1の有効次元は確かめていない）．
3. ASA-SVRG の床が NFG-SVRG より少し小さい（平均で0.88倍）理由は，本実験では分からない．実験1でも ASA-SVRG がわずかに小さかった（log10 で $ -0.03 $〜$ -0.005 $，`report_071.md`）．ASA-SVRG の平均点 $ {\mathbf z}_{s} $ と近似勾配の組み合わせにより，誤差の一部が打ち消し合っている可能性がある．
4. 論文の付録B.1 に，「a9a の BCE でも，独立一様サンプリングでは NFG-SVRG・ASA-SVRG に $ \sigma^{2}/K $ と同じ桁の誤差床が現れ，random reshuffling では消える」と書ける．ただし，比を「一致する」と書く場合は，床の定義（中央値か平均か）を明記する必要がある．

## 8. 統合時の注意

- `programs_review/document.md` は変更してはいけないため，統合時に次を転記する．
  - 「現在実装している実験」に E10（`ex015_e10_a9a_bce_uniform`，付録B.1 の a9a の BCE．独立一様サンプリング（`uniform_with_replacement`）．`SGD`，`SVRG_random`，`NFG_SVRG_random`，`ASAI_SVRG`（`snapshot`），各5シード．CPU のみ）を追加する．
  - ディレクトリ構成に `ex015_e10_a9a_bce_uniform/`（`data.py`，`model.py` は ex014 のコピー．`train.py`，`compare_e10.py`，`README.md`），`analysis_e10.py`，`tests/test_e10_programs.py`，`outputs/ex015_e10_a9a_bce_uniform/`（`consistency.md` を含む），`outputs/e10_summary.md` を追加する．
  - 依存関係の図に `ex015 train.py` → `common/optimizers`，`common/machine_learning_utils`，`ex015 data.py`，`ex015 model.py` と，`analysis_e10.py` → `ex015 data.py`・`ex015 train.py`（`compute_optimal_solution`），`outputs/` を追加する．
  - 実行方法に `ex015_e10_a9a_bce_uniform/README.md` 3節の本実行・集計のコマンドを追加する．
  - 条件のディレクトリ名は E1 と同じく `inner_{inner_start}_{sampling}_eta{η}_lambda{λ}_K29304_epochs100` である．
- zip に含めなかったもの：`outputs/_verify/ex015_rr/`，`outputs/_verify/ex015_uniform/`（3.1節の確認の学習結果），`datasets/`，`.venv_*`，`tokens.json`．`consistency.md` は `outputs/ex015_e10_a9a_bce_uniform/` にあり，zip に含まれる．
- 3.1節 (a) の判定には，E8 のときに作った `outputs/_verify/ex014_e4_ref_ex008_this_pc/`（本PCで E4 のプログラムを実行した結果．Git 管理外）を用いた．
- 6.4節の値は一時的なスクリプトで求めたものであり，`analysis_e10.py` には含めていない．
- 図は作っていない（論文用の図は別のPCで作る）．

## 9. 未解決の論点・次の作業の候補

1. **論文での $ \sigma^{2}/K $ との比較の書き方．** 中央値による床は $ \sigma^{2}/K $ の0.66〜0.77倍，平均では0.88〜1.05倍である．実験1と同じ書き方（「ほぼ一致」）を使うなら，平均で比べるか，「同じ桁」と書くかを決める必要がある．
2. **実験1の勾配の分散の有効次元の確認．** 7.2節2の推測（実験1では中央値と平均の差が小さい）は確かめていない．Mushroom の $ {\mathbf w}^{*} $ での共分散の固有値を計算すれば確かめられる（学習は不要）．
3. **ASA-SVRG の床が NFG-SVRG より少し小さい理由．** 実験1と E10 の両方で見られるが，原因は分からない．論文で触れるかはユーザーの判断事項である．
4. 図が必要なら，`ex015` の結果を論文用のリポジトリで描く（実験1の図と同じ形式）．

## 10. 成果物の zip

リポジトリのルートで指示書6.2節のコマンドを実行し，`E10_results.zip` を作った．`unzip -l E10_results.zip` の結果は次のとおりである．

```text
Archive:  E10_results.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
     2220  2026-11-23 12:28   .orders/order_082.md
    39454  2026-11-23 13:10   .reports/report_082.md
        0  2026-11-23 12:36   programs_review/programs/ex015_e10_a9a_bce_uniform/
     9082  2026-11-23 12:28   programs_review/programs/ex015_e10_a9a_bce_uniform/model.py
     4809  2026-11-23 12:28   programs_review/programs/ex015_e10_a9a_bce_uniform/data.py
     5111  2026-11-23 12:36   programs_review/programs/ex015_e10_a9a_bce_uniform/README.md
    36320  2026-11-23 12:28   programs_review/programs/ex015_e10_a9a_bce_uniform/train.py
    13655  2026-11-23 12:34   programs_review/programs/ex015_e10_a9a_bce_uniform/compare_e10.py
    19950  2026-11-23 12:35   programs_review/programs/analysis_e10.py
    10092  2026-11-23 12:36   programs_review/tests/test_e10_programs.py
        0  2026-11-23 12:34   programs_review/outputs/ex015_e10_a9a_bce_uniform/
     8138  2026-11-23 12:34   programs_review/outputs/ex015_e10_a9a_bce_uniform/consistency.md
        0  2026-11-23 12:34   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/
        0  2026-11-23 12:34   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/
        0  2026-11-23 12:34   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/
        0  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/4/
    28648  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/4/log.json
      745  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/4/config.json
        0  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/3/
    28673  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/3/log.json
      746  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/3/config.json
        0  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/2/
    28643  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/2/log.json
      745  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/2/config.json
        0  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/
    28643  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/log.json
      745  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/config.json
        0  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/1/
    28667  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/1/log.json
      746  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/NFG_SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/1/config.json
        0  2026-11-23 12:34   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/
        0  2026-11-23 12:34   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/
        0  2026-11-23 12:52   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/4/
    24603  2026-11-23 12:52   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/4/log.json
      638  2026-11-23 12:52   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/4/config.json
        0  2026-11-23 12:53   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/3/
    24565  2026-11-23 12:53   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/3/log.json
      639  2026-11-23 12:53   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/3/config.json
        0  2026-11-23 12:52   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/2/
    24590  2026-11-23 12:52   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/2/log.json
      638  2026-11-23 12:52   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/2/config.json
        0  2026-11-23 12:52   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/
    24552  2026-11-23 12:52   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/log.json
      638  2026-11-23 12:52   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/config.json
        0  2026-11-23 12:52   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/1/
    24589  2026-11-23 12:52   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/1/log.json
      639  2026-11-23 12:52   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SGD/inner_none_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/1/config.json
        0  2026-11-23 12:34   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/
        0  2026-11-23 12:34   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/
        0  2026-11-23 13:05   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/4/
    24091  2026-11-23 13:05   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/4/log.json
      741  2026-11-23 13:05   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/4/config.json
        0  2026-11-23 13:08   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/3/
    25632  2026-11-23 13:08   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/3/log.json
      742  2026-11-23 13:08   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/3/config.json
        0  2026-11-23 13:05   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/2/
    23998  2026-11-23 13:05   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/2/log.json
      741  2026-11-23 13:05   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/2/config.json
        0  2026-11-23 13:05   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/
    26859  2026-11-23 13:05   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/log.json
      741  2026-11-23 13:05   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/config.json
        0  2026-11-23 13:05   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/1/
    25595  2026-11-23 13:05   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/1/log.json
      742  2026-11-23 13:05   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/SVRG_random/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/1/config.json
        0  2026-11-23 12:34   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/
        0  2026-11-23 12:34   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/
        0  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/4/
    28642  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/4/log.json
      645  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/4/config.json
        0  2026-11-23 13:08   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/3/
    28661  2026-11-23 13:08   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/3/log.json
      646  2026-11-23 13:08   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/3/config.json
        0  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/2/
    28641  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/2/log.json
      645  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/2/config.json
        0  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/
    28638  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/log.json
      645  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0/config.json
        0  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/1/
    28643  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/1/log.json
      646  2026-11-23 13:06   programs_review/outputs/ex015_e10_a9a_bce_uniform/bce/ASAI_SVRG/inner_snapshot_uniform_with_replacement_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/1/config.json
     4641  2026-11-23 13:08   programs_review/outputs/e10_summary.md
   236179  2026-11-23 13:08   programs_review/outputs/_run_logs/ex015.log
     2884  2026-11-23 12:30   programs_review/outputs/_run_logs/ex015_verify_rr.log
     3971  2026-11-23 12:32   programs_review/outputs/_run_logs/ex015_verify_uniform.log
---------                     -------
   945932                     85 files
```
