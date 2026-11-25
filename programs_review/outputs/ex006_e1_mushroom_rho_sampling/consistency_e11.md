# E11 の本実行の前の確認（`実験指示E11.md` 4節）

実施日：2026-11-25．本PC：Intel Xeon w7-3565X（32コア・64論理コア），メモリ 125 GB，WSL2．torch 2.11.0+cu128（CPU で計算）．
結果：`outputs/_verify/ex006_e11/`（Git 管理外）．ログ：`outputs/_run_logs/ex006_e11_verify.log`．

```bash
cd programs_review/programs
python ex006_e1_mushroom_rho_sampling/train.py --methods SVRG_random --inner-starts snapshot \
    --rhos 298.4 --samplings random_reshuffle --seeds 0 --epochs 3 --workers 1 \
    --output-root ../outputs/_verify/ex006_e11
H=inner_snapshot_random_reshuffle_rho298.4_eta1.049767e-02_lambda9.155984e-01_K7311
python compare_with_existing_logs.py --new-root ../outputs/_verify/ex006_e11 \
    --existing-root ../outputs/ex006_e1_mushroom_rho_sampling --hp-map ${H}_epochs3=${H}_epochs100
python ex006_e1_mushroom_rho_sampling/train.py --methods SVRG_average --inner-starts snapshot last_iterate \
    --rhos 298.4 --samplings random_reshuffle --seeds 0 --epochs 3 --workers 1 \
    --output-root ../outputs/_verify/ex006_e11
```

## 1. 既存の結果が変わらないこと

変更後の `train.py` で `SVRG_random`・`snapshot`・`random_reshuffle`・ρ = 298.4・シード0を3外側ループ実行し，既存の100外側ループのログ（本PCで 2026-11-20 に実行．`report_071`）の先頭4点と比べた（`compare_with_existing_logs.py`）．

| 指標 | 比較した点数 | 完全一致 | 最大絶対誤差 |
|---|---:|---|---:|
| `oracle_calls` | 4 | ○ | 0 |
| `objective_gap` | 4 | ○ | 0 |
| `accuracy` | 4 | ○ | 0 |
| `approx_error` | 4 | ○ | 0 |
| `grad_norm_sq` | 4 | ○ | 0 |

全てビット単位で一致した．`SVRGAverage` は `begin_epoch` で乱数を消費しないため，他の手法の添字の列も変わらない．
また，既存の手法（`SVRG`，`NFG_SVRG`，`ASAI_SVRG`）を既存のログと比べる `tests/test_e1_programs.py` も全て通過した．

## 2. SVRG（average）の動作

ρ = 298.4，random reshuffling，シード0，3外側ループ．$ N = 7311 $．

| 外側ループ $ s $ | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| `oracle_calls`（期待値 $ N + 3Ns $，$ s = 0 $ は0） | 0 | 29244 | 51177 | 73110 |
| reset の `approx_error` | 0 | 0 | 0 | 0 |
| continue の `approx_error` | 0 | 0 | 0 | 0 |
| reset の `objective_gap` | 0.2644 | 2.867e-05 | 3.927e-09 | 6.161e-13 |
| continue の `objective_gap` | 0.2644 | 2.867e-05 | 2.744e-07 | 1.029e-11 |

- `oracle_calls` は $ N + 3Ns $（$ 29244 = 7311 + 3 \times 7311 $）と一致した．
- `approx_error` は全て厳密に0であった（スナップショット勾配はスナップショットにおける全勾配であり，評価と同じ関数で計算するため）．
- 外側ループ1は reset と continue で同じ値であり，外側ループ2から異なる（開始点が異なるのは外側ループ2からであるため，期待どおり）．

## 3. 並列数

| 項目 | 値 |
|---|---|
| 1プロセスの最大常駐メモリ（`/usr/bin/time -v`） | 約 0.90 GB（895,680 kB） |
| 1外側ループの時間（並列数1，tqdm．評価を含む） | 約 4.6 s（`SVRG_random`），約 4.3 s（`SVRG_average`） |
| CPU | 32コア・64論理コア |
| 空きメモリ（本実行の前） | 約 117 GB |

並列数は **32** とした．各プロセスは1スレッド（`OMP_NUM_THREADS=1`，`torch.set_num_threads(1)`）であり，物理コア数と同じにした．メモリは $ 32 \times 0.9 \approx 29 $ GB で十分に収まる．
本実行の32並列でも1外側ループは約 5.0 s であり，並列数1（約 4.6 s）からの遅れは小さかった．
