# ex014_e8_a9a_supplement の確認（実験指示E8.md 3節）

実施日：2026-11-22．本PC：Intel Xeon w7-3565X（64論理コア），WSL2（Linux 6.18.33.2），Python 3.12.15，torch 2.11.0+cu128（計算は CPU のみ），numpy 2.4.6，scipy 1.18.1．
結果は `outputs/_verify/`（Git 管理外，zip に含めない）に置いた．ログ：`outputs/_run_logs/ex014_verify_{e4,random,continue}.log`．

## 3.1 E4 の結果を再現できること

```bash
python ex014_e8_a9a_supplement/train.py --loss mse bce --methods SGD SVRG NFG_SVRG ASAI_SVRG \
    --inner-starts snapshot --seeds 0 --epochs 3 --output-root ../outputs/_verify/ex014_e4
```

### (a) E4 の結果（`outputs/ex008_e4_a9a/{mse,bce}/{手法}/…_epochs100/0/log.json`）の最初の3外側ループとの比較

| 損失 | 手法 | 点数 | 整数の量（`epoch`，`oracle_calls`，`full_grad_computations`，`accuracy`） | 実数の量の最大相対誤差 |
|---|---|---:|---|---:|
| BCE | SGD | 4 | 完全一致 | 1.9e-15 |
| BCE | SVRG | 4 | 完全一致 | 4.0e-12 |
| BCE | NFG-SVRG | 4 | 完全一致 | 8.4e-13 |
| BCE | ASA-SVRG | 4 | 完全一致 | 8.1e-12 |
| MSE | SGD | 4 | 完全一致 | 1.2e-15 |
| MSE | SVRG | 4 | 完全一致 | 3.5e-14 |
| MSE | NFG-SVRG | 4 | 完全一致 | 1.3e-14 |
| MSE | ASA-SVRG | 4 | 完全一致 | 1.0e-13 |

実数の量（目的関数，$ \|\nabla f\|^2 $，近似誤差，コサイン類似度）はビット単位では一致しなかった．
E4 の本実行は別のPC（`report_070.md`：Intel Core Ultra 9 285K）で行われたため，指示書3.1節の「同じ PC ならビット単位で一致すべき」には当たらない．整数の量は完全に一致し，実数の量の差は丸め誤差の水準である．

### (b) 本PCで E4 のプログラム（`ex008_e4_a9a/train.py`，無変更）を実行した結果との比較

コピーと変更で壊していないことを同じ PC で確かめるため，`ex008_e4_a9a/train.py` を同じ引数で本PCで実行し（`outputs/_verify/ex014_e4_ref_ex008_this_pc/`），本実験の結果と比べた．

| 損失 | SGD | SVRG | NFG-SVRG | ASA-SVRG |
|---|---|---|---|---|
| MSE | ビット単位で一致 | ビット単位で一致 | ビット単位で一致 | ビット単位で一致 |
| BCE | ビット単位で一致 | ビット単位で一致 | ビット単位で一致 | ビット単位で一致 |

`elapsed_time` を除く全てのキー（11個）が一致した．したがって，`snapshot_rng` を `begin_epoch` に渡す変更を含めて，既存の4手法の計算は E4 と同一である．

## 3.2 ランダム点の手法の動作

```bash
python ex014_e8_a9a_supplement/train.py --part A --seeds 0 --epochs 3 --output-root ../outputs/_verify/ex014_random
```

| 項目 | `SVRG_random` | `NFG_SVRG_random` |
|---|---|---|
| `config.json` | `snapshot_rule: "uniform_random"`，`snapshot_rng_seed: [0, 1]`，`inner_start: "snapshot"` | 同左 |
| `oracle_calls` | 0，$ 4N $，$ 7N $，$ 10N $（初期 $ N $・1外側ループ $ 3N $．E4 の `SVRG` と同じ） | 0，$ 2N $，$ 4N $，$ 6N $（$ 2N $．E4 の `NFG_SVRG` と同じ） |
| 近似誤差 | 全て0 | — |
| 0外側ループ目 | 本PCの E4 の `SVRG` と全てのキーが一致 | 本PCの E4 の `NFG_SVRG` と全てのキーが一致 |
| 1外側ループ目以降 | 1外側ループ目から異なる（目的関数の誤差 1.430e-02 と 2.233e-02） | 1外側ループ目は一致し，2外側ループ目から異なる（2.042e-02 と 1.423e-02） |

NFG-SVRG の1外側ループ目が最終点版と一致するのは想定どおりである．$ {\mathbf g}_{0} = {\mathbf 0} $ かつ $ {\mathbf w}_{0}^{0} = {\mathbf z}_{0} $ であるため，1回目の内側ループの補正勾配は $ \nabla f_{i}({\mathbf z}_{0}) - \nabla f_{i}({\mathbf z}_{0}) + {\mathbf 0} = {\mathbf 0} $ となり，内側ループの全ての点が $ {\mathbf z}_{0} $ に等しい．
したがって，どの点を選んでも $ {\mathbf z}_{1} = {\mathbf z}_{0} $ となる（目的関数の誤差が0外側ループ目とビット単位で一致すること，1外側ループ目の近似誤差が $ 1.0\times10^{-28} $ であることで確認した）．

0外側ループ目の値を E4 の結果（別のPC）と比べると，$ f({\mathbf w}^{*}) $ などの丸め誤差のためビット単位では一致しない．そのため，本PCで実行した E4 のプログラムの結果（3.1節 (b)）と比べた．

## 3.3 MSE の ASA-SVRG（continue）と旧実装の一致

```bash
python ex014_e8_a9a_supplement/train.py --part B --seeds 0 --output-root ../outputs/_verify/ex014_continue
python ex014_e8_a9a_supplement/compare_e4.py --new-root ../outputs/_verify/ex014_continue \
    --existing mse=../../outputs/ex000_a9a_least_squares \
    --output ../outputs/ex014_e8_a9a_supplement/consistency_continue_with_ex000.md
```

比較先はリポジトリのルートの `outputs/ex000_a9a_least_squares/ASAI_SVRG/eta1.530030e-01_K29304_epochs100/0/`（旧実装．内側ループは直前の最終点から開始）である．
`compare_e4.py` の規則で，ログの7系列（`epoch`，`oracle_calls`，`full_grad_computations`，`objective_value`，`grad_norm_sq`，`accuracy`，`approx_error`．各101点）が**全て完全一致**し，`config.json` の定数（$ \eta $，$ L $，$ K $，外側ループ数など）も一致した．
旧実装のログは本PCで作られたため（`outputs/ex008_e4_a9a/consistency_with_existing_this_pc.md` の見出し「本PC（実験用リポジトリの既存のログを作成したPC）」），ビット単位で一致したと考えられる．詳細は `consistency_continue_with_ex000.md` にある．

### 本実行の全シードと旧実装の一致（追加の確認）

本実行（シード0〜4）の `outputs/ex014_e8_a9a_supplement/mse/ASAI_SVRG/inner_last_iterate_eta1.530030e-01_K29304_epochs100/{seed}/log.json` を，旧実装の同じシードのログと比べた．
5シードとも，`elapsed_time` を除く全てのキーがビット単位で一致した．最終の $ f({\mathbf z}_{s}) $ は 0.24082，0.10327，0.13678，0.10328，0.24082（シード0〜4）である．

## 並列数

3.3節の1試行（並列数1）を `/usr/bin/time -v` で計測した．最大常駐メモリは 0.90 GB，実時間は21分20秒（1外側ループ約11.6秒）であった．
本実行の15試行を15並列とすると，メモリは約13.4 GB（空き約115 GB）であり，論理コア64のうち15を使う（E7 の A が約4コアを使用中）．
GPU は使わない．したがって，15試行を全て同時に実行する15並列とした．
