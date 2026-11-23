# ex015_e10_a9a_bce_uniform の一致の確認（実験指示E10.md 3.1節）

実施日：2026-11-23．本PC：Intel Xeon w7-3565X（64論理コア），WSL2（Linux 6.18.33.2），Python 3.12.15，torch 2.11.0+cu128（計算は CPU のみ，`torch.set_num_threads(1)`），numpy 2.4.6，scipy 1.18.1．
E4（`ex008_e4_a9a`）の本実行は別のPC（Intel Core Ultra 9 285K）で行われ，E8（`ex014_e8_a9a_supplement`）は本PCで行われた．
確認の結果は `outputs/_verify/ex015_rr/`，`outputs/_verify/ex015_uniform/`（Git 管理外）にある．ログ：`outputs/_run_logs/ex015_verify_{rr,uniform}.log`．

```bash
# programs_review/programs で実行
python ex015_e10_a9a_bce_uniform/train.py --samplings random_reshuffle --seeds 0 --epochs 3 \
    --output-root ../outputs/_verify/ex015_rr --workers 4
python ex015_e10_a9a_bce_uniform/train.py --seeds 0 --epochs 3 --output-root ../outputs/_verify/ex015_uniform --workers 1
python ex015_e10_a9a_bce_uniform/compare_e10.py --rr-root ../outputs/_verify/ex015_rr \
    --uniform-root ../outputs/_verify/ex015_uniform --e4-this-pc-root ../outputs/_verify/ex014_e4_ref_ex008_this_pc \
    --output ../outputs/ex015_e10_a9a_bce_uniform/consistency.md
```

比較の対象は，シード0の外側ループ0〜3（4点）である．
E4 の本実行は別のPCで行われたため，SGD・ASA-SVRG は実数の量がビット単位では一致しない（(a-1)）．同じ現象は E8 でも確認した（`outputs/ex014_e8_a9a_supplement/consistency.md` 3.1節）．
そのため，E8 のときに本PCで E4 のプログラム（`ex008_e4_a9a/train.py`，無変更）を同じ条件（シード0，3外側ループ）で実行した結果（`outputs/_verify/ex014_e4_ref_ex008_this_pc/`）とも比べ（(a-2)），E4 の手法の判定にはこちらを用いた．

## (a) random reshuffling での E4・E8 の再現

### (a-1) 既存の結果（E4 の本実行，E8）との比較

| 手法 | 比較先 | 点数 | 項目 | ビット単位で一致 | 最大の相対差 |
|---|---|---:|---|---|---:|
| SGD | `ex008_e4_a9a/bce/SGD/inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `oracle_calls` | はい | 0.0e+00 |
| SGD | `ex008_e4_a9a/bce/SGD/inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `objective_gap` | いいえ | 1.9e-15 |
| SGD | `ex008_e4_a9a/bce/SGD/inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `grad_norm_sq` | いいえ | 8.6e-16 |
| SGD | `ex008_e4_a9a/bce/SGD/inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `accuracy` | はい | 0.0e+00 |
| SVRG_random | `ex014_e8_a9a_supplement/bce/SVRG_random/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `oracle_calls` | はい | 0.0e+00 |
| SVRG_random | `ex014_e8_a9a_supplement/bce/SVRG_random/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `objective_gap` | はい | 0.0e+00 |
| SVRG_random | `ex014_e8_a9a_supplement/bce/SVRG_random/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `grad_norm_sq` | はい | 0.0e+00 |
| SVRG_random | `ex014_e8_a9a_supplement/bce/SVRG_random/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `accuracy` | はい | 0.0e+00 |
| SVRG_random | `ex014_e8_a9a_supplement/bce/SVRG_random/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `approx_error` | はい | 0.0e+00 |
| NFG_SVRG_random | `ex014_e8_a9a_supplement/bce/NFG_SVRG_random/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `oracle_calls` | はい | 0.0e+00 |
| NFG_SVRG_random | `ex014_e8_a9a_supplement/bce/NFG_SVRG_random/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `objective_gap` | はい | 0.0e+00 |
| NFG_SVRG_random | `ex014_e8_a9a_supplement/bce/NFG_SVRG_random/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `grad_norm_sq` | はい | 0.0e+00 |
| NFG_SVRG_random | `ex014_e8_a9a_supplement/bce/NFG_SVRG_random/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `accuracy` | はい | 0.0e+00 |
| NFG_SVRG_random | `ex014_e8_a9a_supplement/bce/NFG_SVRG_random/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `approx_error` | はい | 0.0e+00 |
| ASAI_SVRG | `ex008_e4_a9a/bce/ASAI_SVRG/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `oracle_calls` | はい | 0.0e+00 |
| ASAI_SVRG | `ex008_e4_a9a/bce/ASAI_SVRG/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `objective_gap` | いいえ | 1.3e-12 |
| ASAI_SVRG | `ex008_e4_a9a/bce/ASAI_SVRG/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `grad_norm_sq` | いいえ | 2.1e-14 |
| ASAI_SVRG | `ex008_e4_a9a/bce/ASAI_SVRG/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `accuracy` | はい | 0.0e+00 |
| ASAI_SVRG | `ex008_e4_a9a/bce/ASAI_SVRG/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs100/0` | 4 | `approx_error` | いいえ | 2.8e-03 |

### (a-2) 本PCで E4 のプログラム（`ex008_e4_a9a/train.py`，無変更）を実行した結果との比較

| 手法 | 比較先 | 点数 | 項目 | ビット単位で一致 | 最大の相対差 |
|---|---|---:|---|---|---:|
| SGD | `_verify/ex014_e4_ref_ex008_this_pc/bce/SGD/inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs3/0` | 4 | `oracle_calls` | はい | 0.0e+00 |
| SGD | `_verify/ex014_e4_ref_ex008_this_pc/bce/SGD/inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs3/0` | 4 | `objective_gap` | はい | 0.0e+00 |
| SGD | `_verify/ex014_e4_ref_ex008_this_pc/bce/SGD/inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs3/0` | 4 | `grad_norm_sq` | はい | 0.0e+00 |
| SGD | `_verify/ex014_e4_ref_ex008_this_pc/bce/SGD/inner_none_eta8.543689e-02_lambda1.125000e-01_K29304_epochs3/0` | 4 | `accuracy` | はい | 0.0e+00 |
| ASAI_SVRG | `_verify/ex014_e4_ref_ex008_this_pc/bce/ASAI_SVRG/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs3/0` | 4 | `oracle_calls` | はい | 0.0e+00 |
| ASAI_SVRG | `_verify/ex014_e4_ref_ex008_this_pc/bce/ASAI_SVRG/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs3/0` | 4 | `objective_gap` | はい | 0.0e+00 |
| ASAI_SVRG | `_verify/ex014_e4_ref_ex008_this_pc/bce/ASAI_SVRG/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs3/0` | 4 | `grad_norm_sq` | はい | 0.0e+00 |
| ASAI_SVRG | `_verify/ex014_e4_ref_ex008_this_pc/bce/ASAI_SVRG/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs3/0` | 4 | `accuracy` | はい | 0.0e+00 |
| ASAI_SVRG | `_verify/ex014_e4_ref_ex008_this_pc/bce/ASAI_SVRG/inner_snapshot_eta8.543689e-02_lambda1.125000e-01_K29304_epochs3/0` | 4 | `approx_error` | はい | 0.0e+00 |

**判定（E8 の手法は (a-1)，E4 の手法は (a-2) で判定）：合格（全ての項目がビット単位で一致）．**

## (b) 一様サンプリングの動作

| 手法 | 記録の点数（期待値） | `config.json` の `sampling` | (a) と異なり始める外側ループ（期待値） | それ以降も全て異なる | 目的関数誤差（一様，外側ループ0〜） | 目的関数誤差（(a)，外側ループ0〜） |
|---|---|---|---|---|---|---|
| SGD | 4（4） | `uniform_with_replacement` | 1（1） | はい | 2.671891e-01，1.772405e-02，2.895924e-02，1.747175e-02 | 2.671891e-01，2.967099e-02，3.402270e-02，3.382008e-02 |
| SVRG_random | 4（4） | `uniform_with_replacement` | 1（1） | はい | 2.671891e-01，1.486980e-02，3.633696e-04，1.096968e-05 | 2.671891e-01，1.429795e-02，2.786686e-04，1.564914e-05 |
| NFG_SVRG_random | 4（4） | `uniform_with_replacement` | 2（2） | はい | 2.671891e-01，2.671891e-01，3.405025e-02，3.369992e-02 | 2.671891e-01，2.671891e-01，2.042285e-02，2.036590e-02 |
| ASAI_SVRG | 4（4） | `uniform_with_replacement` | 2（2） | はい | 2.671891e-01，2.671891e-01，4.932745e-04，2.578842e-04 | 2.671891e-01，2.671891e-01，4.246056e-05，4.673954e-05 |

**判定：合格．**
