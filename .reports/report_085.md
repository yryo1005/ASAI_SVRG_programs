# report_085：論文（最終版）の実験の公開版リポジトリ `../ASA-SVRG` の作成

指示：`.orders/order_085.md`．対象の論文：`main.pdf`（`ASAI_SVRG_Paper/papers/ASAI_SVRG_v5/main.pdf` と同一）．

## 0. 要点

- `/home/user/workspace/ASA-SVRG` に公開版リポジトリを作成した（約 37 MB，2,027 ファイル．`datasets/` を除く．10節の並べ替えの後）．Git の初期化とコミットはしていない（ユーザーの指定）．
- 論文に載っている全ての実験（図1〜3，図B-1〜B-6，表V〜XI）のプログラム（15実験）と，各試行の `log.json`・`config.json` を収めた．重み（`.pth`）は含めていない．
- **結果の配置（10節，追加の指示）**：`results/{データセット}/{手法間で共通の条件}/{手法}/{シード}/` に統一した．同じ条件・手法の重複は論文で使った1組だけを残し（1,034 → 959試行），全てのプログラムがこの配置に書くように改めた．並べ替えの後も図は画素単位，表は数値が論文と一致し，短い再実行は保存先・値とも一致した．
- 論文の図表を `results/` から再現するスクリプト（`paper/make_figures.py`，`paper/make_tables.py`）を加えた．
  - 9つの図は，論文の図（v5 の `figures/`）と**画素単位で完全に一致**した（matplotlib 3.11.2．150 dpi で比較）．
  - 表V〜XI の数値（計599個）は，論文の TeX の値と**順序まで含めて全て一致**した．本文の $ \sigma^{2}/K $（Mushroom 4.10e-4，a9a 6.7e-5）も再現した．
- 移設したプログラムは，このPCで元のプログラムと同じ条件で短く実行し，**ビット単位で一致**することを確かめた（6節）．
- README（英語）に，論文の図表と実験・結果のディレクトリの対応，環境，再現の手順，結果の形式，再現性の注意を書いた．コードの docstring・コメントは日本語のままである（ユーザーの指定）．

## 1. 確認した事項と決定（order_085 の3節）

| 項目 | 決定 |
|---|---|
| 構成 | 論文の節ごとに再編（`experiments/{データセット}/{実験名}`，`results/` も同じ構造） |
| 言語 | README は英語，コードは日本語のまま |
| 結果 | 各試行のログと設定，論文の図表と再現スクリプト．重みは含めない |
| Git | ファイルの作成のみ |

## 2. 論文の図表とデータの出典の調査

論文の図表は `ASAI_SVRG_Paper/programs/` のスクリプトで作られている（図は `make_report128_figures.py` とその依存モジュール，表の値は `make_report114_e11.py`・`make_report115_tables.py`・`make_report091_e2.py`・`make_report100_e6_e7.py`・`make_report095_time_table.py` の関数．表は TeX に直接書かれている）．
これらのスクリプトが読む結果から，論文に使われている実験は次の15である．ex001，ex003，ex009，ex010 は最終版の図表には使われていないため含めなかった．

| 公開版 | 元の実験 | 論文 |
|---|---|---|
| `mushroom/rho_sampling` | ex006_e1_mushroom_rho_sampling | 図1，表V〜VII |
| `a9a/main` | ex008_e4_a9a | 図B-1，B-2 |
| `a9a/supplement` | ex014_e8_a9a_supplement | 図B-1，B-2 |
| `a9a/bce_uniform` | ex015_e10_a9a_bce_uniform | 図B-2，本文の a9a の誤差床と $ \sigma^{2}/K $ |
| `cifar10/batch_sizes` | ex0024_cifar10_alexnet_layernorm_longrun（旧 `programs/`・`outputs/`） | 図2（SGD），表IX・X，図B-3・B-4 |
| `cifar10/reset_continue_bs128` | ex002_e0_cifar10_alexnet_layernorm | 図2，表VIII，表IX・X（ASA reset） |
| `cifar10/snapshot_factorial_bs128` | ex007_e2_cifar10_snapshot_gradient_factorial | 図2，表VIII，表IX・X |
| `cifar10/svrg_average_continue_bs128` | ex007_e2_cifar10_ablation | 表VIII |
| `cifar10/asa_reset_bs32_512` | ex004_appb_cifar10_alexnet_layernorm | 表IX・X，図B-3・B-4 |
| `cifar10/final_snapshot_bs32_512` | ex011_e6_cifar10_final_bs32_512 | 表IX・X，図B-3・B-4 |
| `imagewoof/batch_sizes` | ex0051_imagewoof_resnet18_bs128_longrun（旧 `programs/`・`outputs/`） | 表IX・X，図B-5・B-6 |
| `imagewoof/asa_reset_bs32_512` | ex005_appb_imagewoof_resnet18_layernorm | 表IX・X，図B-5・B-6 |
| `imagewoof/final_snapshot` | ex012_e7_imagewoof_final | 表IX・X，図B-5・B-6 |
| `imagewoof/walltime_bs128` | ex016_imagewoof_bs128_walltime | 図3，表IX〜XI の ImageWoof・ミニバッチ128 |
| `timing/epoch_time` | ex013_e7_epoch_time | 表XI |

## 3. 公開版の構成

```text
ASA-SVRG/
├── README.md, requirements.txt, requirements_paper.txt, .gitignore
├── experiments/common/           # 最適化手法（optimizers/）と学習の共通モジュール（programs_review/programs/common と同一）
├── experiments/{mushroom,a9a,cifar10,imagewoof,timing}/*/   # 各実験の train.py，data.py，model.py
├── results/                      # 各試行の log.json・config.json（experiments/ と同じ構造）
├── paper/                        # 図表の再現：make_figures.py，make_tables.py と依存モジュール，figures/，tables/
├── tools/compare_logs.py         # 再実行のログと results/ の比較
└── tests/test_reproduction.py    # pytest（7件）
```

公開版の組み立ては `programs_review/programs/build_public_repo.py`（本PCのリポジトリに新規作成）で行った．README，`paper/make_figures.py`，`paper/make_tables.py`，`tools/compare_logs.py`，テスト，requirements，`.gitignore` は公開版に直接作成した．

## 4. プログラムの移設で変更した点

学習の計算は変更していない．変更はパスと実行の入口だけである（`build_public_repo.py` の置換の表）．

1. ディレクトリが1段深くなったため，`_PROGRAMS_DIR`・`_COMMON_DIR` の計算に `os.path.dirname` を1つ加えた．
2. 結果の保存先を `outputs/{実験名}` から `results/{データセット}/{実験名}` に変えた（その後，10節で `results/{データセット}/{共通の条件}/{手法}/{シード}` に改めた）．`config.json` に記録する実験名（`EXPERIMENT_NAME`，例：`ex006_e1_mushroom_rho_sampling`）は変えていない（README の対応表に "Original ID" として記載）．
3. 旧リポジトリの2実験（ex0024，ex0051）は，`.ai/ai-dev-kit/src` と `programs/optimizers` の代わりに `experiments/common` を参照するようにした（`machine_learning_utils.py` は同一，`optimizers/` は `__init__.py` に `SVRGAverage` が加わる以外は同一であることを `diff` で確認）．生データの保存先も他の実験と同じ `datasets/{cifar10,imagewoof}/raw` にした．
4. 旧リポジトリの2実験の `__main__` は，作業の途中で未実行のフェーズだけを実行するように書き換えてあったため，全てのフェーズを実行する（完了済みの条件は飛ばす）ように戻した．
5. 旧リポジトリの結果との比較のための内部の検証スクリプト（`compare_e4.py`，`compare_e10.py`，`check_consistency.py`）と作業記録の README はコピーしていない．

論文の側のモジュールは，名前を内容が分かるものに変えて `paper/` に置いた（`make_report090_e1` → `fig_mushroom`，`make_report091_e2` → `fig_cifar10_bs128`，`make_report098_e8` → `fig_a9a`，`make_report100_e6_e7` → `fig_batch_sizes`，`make_report114` → `fig_imagewoof_walltime`，`aggregate_existing_results` → `nonconvex_results`，`make_e1_paper_assets` → `mushroom_assets`，`analysis_e11` → `mushroom_configs`，`analysis_e10` → `a9a_uniform_floor`，`analyze_e1_gradient_variance` → `mushroom_gradient_variance`）．
import の名前，結果のパス（`programs_review/outputs` と `outputs` → `results`），実験名を置き換えた．TeX の表の組版（`nolta.cls`）に使う `make_report113_tables` の import は除いた（図の関数は使わない）．

## 5. 図表の再現の確認

### 5.1 図

`paper/make_figures.py` で描いた9つの図を，v5 の `figures/` の PDF と，`pdftoppm -r 150` で画像にして画素ごとに比べた．

| 図 | 異なる画素の数（matplotlib 3.9.2） | 異なる画素の数（matplotlib 3.11.2） |
|---|---:|---:|
| 図1 | 40,238 | 0 |
| 図2 | 46,839 | 0 |
| 図3 | 38,497 | 0 |
| 図B-1 | 43,421 | 0 |
| 図B-2 | 41,199 | 0 |
| 図B-3 | 52,731 | 0 |
| 図B-4 | 33,071 | 0 |
| 図B-5 | 43,138 | 0 |
| 図B-6 | 38,350 | 0 |

本PCの `.venv_pytorch`（matplotlib 3.9.2）では，曲線は同じで目盛りの表示だけが異なった．論文の図の PDF の作成者は matplotlib 3.11.2 であり，同じ版（`uv run --with matplotlib==3.11.2`）で描くと全ての図が一致した．公開版の `paper/figures/` には 3.11.2 で描いた図を置き，`requirements_paper.txt` に版を固定した．

### 5.2 表

`paper/make_tables.py` の出力（`paper/tables/tables.md`）の数値を，v5 の `sections/04_experiments.tex` の各表の数値と，出現順に比べた．

| 表 | 数値の数 | 一致 |
|---|---:|---|
| 表V（$ \hat{r} $．¶ の2点の値と $ \eta $ の行を含む） | 151 | 全て一致 |
| 表VI（$ 10^{-12} $ に達する $ \#\mathrm{grad}/N $） | 144 | 全て一致 |
| 表VII（誤差床．見出しを除く） | 66 | 全て一致 |
| 表VIII（CIFAR-10） | 54 | 全て一致 |
| 表IX（最終のテスト精度） | 60 | 全て一致 |
| 表X（精度の変動） | 54 | 全て一致 |
| 表XI（1エポックの時間） | 24 | 全て一致 |

本文の値：Mushroom の $ \sigma^{2}/K = 4.099\times10^{-4} $（論文 $ 4.10\times10^{-4} $），近似誤差の床 ÷ $ \sigma^{2}/K $ は NFG 1.005，ASA 0.986（論文 0.98〜1.02）．a9a の $ \sigma^{2}/K = 6.71\times10^{-5} $（論文 $ 6.7\times10^{-5} $），誤差床は目的関数誤差 $ 1.24\times10^{-4} $・$ 1.07\times10^{-4} $，近似誤差 $ 5.17\times10^{-5} $・$ 4.40\times10^{-5} $（NFG・ASA．ASA が約15%低い．論文と同じ）．

なお，論文の表VI（到達の閾値 $ 10^{-12} $，SVRG 系の評価点を $ 3s $ と数える）は，report_084 の10節で変更した `e11_summary.md` の到達の列（`oracle_calls/N` をそのまま使うため SVRG 系の値が1大きい）とは横軸の数え方が異なる．公開版の表は論文と同じ数え方（`paper/grad_count.py`）である．

## 6. 移設したプログラムの動作の確認

各実験を公開版で短く（2外側ループ，シード0）実行し，`tools/compare_logs.py` で保存された結果と比べた．一致しない場合は，元のプログラム（本PCのリポジトリ）でも同じ条件で実行し，公開版と比べた．データセットは本PCの `programs_review/datasets` へのシンボリックリンクで与えた（元のデータセットのディレクトリには書き込んでいない）．

| 公開版の実験 | 実行した条件 | 保存された結果と | 公開版と元のプログラム |
|---|---|---|---|
| `mushroom/rho_sampling` | 4手法 × reset・continue × 2サンプリング，ρ = 298.4（16条件） | 14条件はビット一致．ASA-SVRG reset の2条件は浮動小数点の値が最大 1.1e-16 異なる（別のPCの結果．$ f^{*} $ の最後の桁が異なる） | ビット一致（ASA reset の2条件，22件） |
| `a9a/main` | MSE・BCE × 4手法 | 整数の量は一致，浮動小数点の値が最大 1.4e-14 異なる（別のPCの結果） | ビット一致（88件） |
| `a9a/supplement` | 全構成 | ビット一致（33件） | — |
| `a9a/bce_uniform` | 全構成 | ビット一致（44件） | — |
| `cifar10/reset_continue_bs128` | ASA-SVRG reset・continue | ビット一致（18件） | — |
| `cifar10/snapshot_factorial_bs128` | SVRG average・final，NFG-SVRG final | 不一致（学習損失で最大 1.8e-4，正解数で最大4．保存された結果は RTX 3090 で計算） | ビット一致（33件） |
| `cifar10/svrg_average_continue_bs128` | SVRG average continue | ビット一致（11件） | — |
| `cifar10/asa_reset_bs32_512` | ASA-SVRG reset，ミニバッチ512 | ビット一致（9件） | — |
| `cifar10/final_snapshot_bs32_512` | SVRG・NFG-SVRG final，ミニバッチ512 | 不一致（学習損失で最大 4.5e-6，正解数で最大2．保存された結果は RTX 4090 で計算） | ビット一致（22件） |
| `imagewoof/asa_reset_bs32_512` | ASA-SVRG reset，ミニバッチ512 | ビット一致（9件） | ビット一致（11件） |
| `imagewoof/final_snapshot` | SVRG・NFG-SVRG final，ミニバッチ512 | ビット一致（22件） | ビット一致（22件） |
| `imagewoof/walltime_bs128` | 5条件 | 時間以外はビット一致（35件） | ビット一致（35件） |
| `timing/epoch_time` | 2データセット × ミニバッチ512 × 4手法，1回 | 時間以外はビット一致（56件） | — |
| `cifar10/batch_sizes`（旧） | ASA-SVRG，ミニバッチ512 | ビット一致（5件） | ビット一致（5件） |
| `imagewoof/batch_sizes`（旧） | ASA-SVRG，ミニバッチ512 | 精度・勾配の計算回数は一致，学習損失 6.4e-7・近似誤差 1.8e-4 の差（保存された結果の GPU は記録なし） | ビット一致（5件） |

比較の件数は，試行と指標の組の数である．比較に使った GPU は RTX PRO 6000 Blackwell Max-Q（本PC）である．
全ての実験で，公開版の実行は元のプログラムの実行とビット単位で一致したため，移設（パスの書き換え）は計算に影響していない．
保存された結果と一致しない場合は，いずれも結果を別の計算機（別の GPU）で計算したものであり，初期点は一致し，学習の後に差が生じる．
旧リポジトリの2実験は，出力先とデータの場所を差し替えて1試行を実行する補助スクリプト（作業用．公開版には含めない）で動かした．
この旨（計算機による違い）は README の Reproducibility に記載した．

## 7. 公開版のテスト

`pytest tests`（公開版）：7件全て通過．最後の実行は `datasets/` を削除した状態で行い，Mushroom のデータが UCI から自動でダウンロードされることも確かめた（テストの後に `datasets/` と `__pycache__` は削除した）．

- `compare_logs` の規則（エポック数を除いた対応，許容誤差，計測時間の除外）2件．
- `make_tables.py` が保存された `tables.md` と同じ表を作ること．
- `make_figures.py` が9つの図を作ること．
- Mushroom の3構成（SVRG random reset，ASA-SVRG continue，SVRG average reset）の短い再実行が保存された結果と一致すること（許容誤差 $ 10^{-12} $）．

## 8. 未解決の論点・ユーザーが判断すべき事項

- **論文の Data availability と脚注の URL**：論文は `https://github.com/yryo1005/ASAI_SVRG_programs.git` を示しているが，公開版のディレクトリ名は `ASA-SVRG` である．GitHub に公開する名前と論文の URL をそろえる必要がある（どちらを変えるかはユーザーが判断する）．
- **ライセンス**：公開版には LICENSE を置いていない．公開する場合はライセンスを決めて加える必要がある．
- **README の雑誌情報**：巻・号・DOI は論文の PDF でも未定（"vol. ??"）のため，README には誌名だけを書いた．
- **コードのコメント**：docstring・コメントは日本語のままで，開発中のディレクトリ名（`programs_review/programs/ex006_...` など）や作業の指示番号（order_xxx）を含む．README に対応を記載したが，公開前に整理するかはユーザーが判断する．
- **`datasets/`**：検証に使ったシンボリックリンクは削除した．公開版の利用者は初回の実行で各データセットをダウンロードする．
- `programs_review/document.md` には，E9 以降（E9，E10，ex016，E11）と本作業をまだ反映していない．

## 9. AI エージェントへの引き継ぎ

- 次の候補：GitHub への公開（リポジトリ名と論文の URL の整合，ライセンス），公開版の README の英文の確認，コメントの英訳の要否の判断．
- `results/` を作り直す場合は `python programs_review/programs/build_public_repo.py --target ../ASA-SVRG` を実行する（結果の並べ替えのみ）．`--code` はプログラムを元のリポジトリからコピーし直すため，10節で公開版に直接加えた変更が失われる（最初の作成専用）．

## 10. 追加の指示：結果の配置の統一（order_085 の4節）

### 10.1 配置

```text
results/{データセット}/{手法間で共通の条件}/{手法}/{シード}/{config.json, log.json}
results/{データセット}/{手法間で共通の条件}/epoch_time/{手法}/rep{r}/   # 表XI の計測
```

- 共通の条件：画像分類 `bs{B}_lr{η}_lambda{λ}`，Mushroom `{サンプリング方式}_rho{ρ}_eta{η}`，a9a `{損失}_{サンプリング方式}_eta{η}`．
- 手法：`SVRG_random_reset`，`SVRG_random_continue`，`SVRG_final`，`SVRG_average_reset`，`SVRG_average_continue`，`NFG-SVRG_random_reset`，`NFG-SVRG_random_continue`，`NFG-SVRG_final`，`ASA-SVRG_reset`，`ASA-SVRG_continue`，`SGD`．
- エポック数は手法ごとに異なるため保存先に含めず，`config.json` に記録する．
- 以前は実験ごとに `SVRG` の意味（凸の問題では final，画像分類では random）が異なっていたが，新しい名前では区別がなくなった．

### 10.2 重複の整理（論文で使った1組だけを残す）

| 除いた試行 | 数 | 残した試行との関係 |
|---|---:|---|
| CIFAR-10 bs128 の SVRG・NFG-SVRG・ASA-SVRG（random・reset）の `snapshot_factorial_bs128` の実行 | 15 | 別の GPU（RTX 3090）の実行．論文は `reset_continue_bs128` の実行を使う |
| CIFAR-10 bs128 の分散縮小法（continue）の `batch_sizes` の実行 | 15 | `reset_continue_bs128` とビット一致 |
| CIFAR-10 の SGD の短い実行（12・48・192エポック） | 15 | 倍のエポック数の実行の先頭部分とビット一致 |
| ImageWoof bs128 の `final_snapshot` の実行 | 10 | `walltime_bs128` とビット一致 |
| ImageWoof bs128 の SGD（128・384エポック）と ASA-SVRG continue の `batch_sizes` の実行 | 15 | 別の実行．論文は `walltime_bs128` を使う |
| ImageWoof bs512 の SGD の短い実行（505エポック） | 5 | 1010エポックの実行（論文で使う）がある |

計75試行を除き，959試行を残した．並べ替えは `build_public_repo.py` の `build_results`（`EXCLUDED_RUNS`）で行い，保存先の衝突が無いことを確かめた．

### 10.3 プログラムの変更（公開版で直接編集）

- `experiments/common/result_layout.py` を新設し，全ての `train.py`（共通の `e0_runner.build_tasks` を含む）がこの規則で保存先を決めるようにした．`--output-root` の既定値は `results`．
- 完了の判定（7か所）を「指定のエポック数と一致」から「指定以上のエポック数の記録がある」に改めた．保存先にエポック数を含めないため，短い再実行で既存の結果を上書きしないためである．
- 既定の実行範囲を，各セルがちょうど1つのプログラムから作られるように調整した：Mushroom の既定の開始点を reset と continue の両方にした（論文の10構成．final は同じ保存先なので1つにまとめる），`snapshot_factorial_bs128` の既定から random の3手法，`svrg_average_continue_bs128` の既定から NFG-SVRG final，`imagewoof/final_snapshot` の既定から bs128，旧 `batch_sizes` 2実験から重複する条件（と結果を公開していない学習率 1e-4 の条件A）を除いた．
- 確認：全プログラムの既定の保存先（967）と `results/` の試行（959）を突き合わせ，全ての試行がちょうど1つのプログラムから作られることを確かめた．既定で作られるのに結果が無い8試行は，元から1シードしか無い ImageWoof bs512 の SVRG・NFG-SVRG（random・reset）である（論文では使っていない．README に記載）．

### 10.4 図表のモジュールの変更と確認

- `paper/` の読み込み（`mushroom_assets`，`mushroom_configs`，`mushroom_gradient_variance`，`fig_cifar10_bs128`，`fig_batch_sizes`，`fig_a9a`，`fig_imagewoof_walltime`，`a9a_uniform_floor`，`make_figures`，`make_tables`）を新しい配置に合わせた．試行の手法は `config.json` の `method`・`inner_start` で判定し，予定のエポック数も `config.json` から読む．`fig_imagewoof_walltime` から TeX の表の関数（未使用）を除いた．
- 9つの図：論文の図と画素単位で一致（matplotlib 3.11.2）．表V〜XI：並べ替えの前の `tables.md` とバイト単位で一致．σ²/K の2つの集計も一致．

### 10.5 短い再実行による確認

| プログラム | 保存先 | 保存された結果と |
|---|---|---|
| `mushroom/rho_sampling`（SVRG random・ASA-SVRG，reset・continue） | `mushroom/random_reshuffle_rho298.4_eta1.049767e-02/{SVRG_random_*,ASA-SVRG_*}/0` | ASA-SVRG reset 以外はビット一致（ASA reset は 6節と同じく別のPCの結果で 1e-16 の差） |
| `a9a/supplement` | `a9a/{mse,bce}_random_reshuffle_eta*/...` | ビット一致（33件） |
| `cifar10/reset_continue_bs128`（ASA-SVRG） | `cifar10/bs128_.../ASA-SVRG_{reset,continue}/0` | ビット一致（18件） |
| `cifar10/svrg_average_continue_bs128` | `cifar10/bs128_.../SVRG_average_continue/0` | ビット一致（11件） |
| `imagewoof/final_snapshot`（bs512） | `imagewoof/bs512_.../SVRG_final/0` | ビット一致（11件） |
| `imagewoof/walltime_bs128`（SGD，ASA reset） | `imagewoof/bs128_.../{SGD,ASA-SVRG_reset}/0` | 時間以外はビット一致（14件） |
| `timing/epoch_time`（CIFAR-10 bs512） | `cifar10/bs512_.../epoch_time/{SGD,ASA-SVRG_continue}/rep0` | 時間以外はビット一致（14件） |
| 旧 `cifar10/batch_sizes`（ASA-SVRG bs512） | `cifar10/bs512_.../ASA-SVRG_continue/0` | ビット一致（5件） |

公開版の `pytest tests`：7件全て通過（`datasets/` を削除した状態で実行）．

## 11. 追加の指示：提案手法の名前を ASA-SVRG に統一（order_085 の5節）

公開版 `../ASA-SVRG` の中の開発時の名前を全て置き換えた（本PCのリポジトリは変えていない）．

| 種類 | 変更前 | 変更後 | 箇所 |
|---|---|---|---|
| 手法名（プログラム，`config.json` の `method`，`condition`） | `ASAI_SVRG`，`ASAI_SVRG_reset`，`ASAI_SVRG_continue` | `ASA_SVRG`，`ASA_SVRG_reset`，`ASA_SVRG_continue` | 336 |
| クラス名 | `ASAISVRG` | `ASASVRG` | 46 |
| モジュール | `optimizers/asai_svrg.py` | `optimizers/asa_svrg.py` | ファイル名と import 4 |
| コメント・docstring | 「ASAI SVRG」，「Averaged Snapshot Approximate Incremental SVRG」，「ASAI SVRG論文（`references/ASAI_SVRG_paper.pdf`）」 | 「ASA-SVRG」，「Averaged Snapshots and Approximated full gradients for SVRG」，「ASA-SVRG論文」 | 34 |
| 開発用のリポジトリ名を含む説明（`ASAI_SVRG_programs`，`ASAI_SVRG_Paper`，`ASAI_SVRG_evaluate.ipynb`，`papers/ASAI_SVRG_v5`） | `paper/` の6モジュールの docstring と使い方 | 公開版での役割と使い方に書き直した | 6ファイル |

- 置き換えの後に `ASAI`・`asai` が残るのは README の著者名（Suguru Asai）だけである．
- 結果：`config.json` 198個の `method`（と ImageWoof bs128 の 10個の `condition`）を置き換えた．`build_public_repo.py` も `config.json` を書き出すときに同じ置き換えをするように改め，`results/` を作り直した．作り直した 1,918ファイルは，元の結果の `ASAI_SVRG` を `ASA_SVRG` に置き換えたものとバイト単位で一致した（改行コードも元のまま．`log.json` は変更なし）．
- 確認：
  - 図9つは論文の図と画素単位で一致，表V〜XI と σ²/K の2つの集計は置き換えの前とバイト単位で一致．
  - 10.5節と同じ8つの短い再実行（`--methods ASA_SVRG` など新しい名前で指定）は，保存先・値とも保存された結果と一致した（Mushroom の ASA-SVRG reset は 6節と同じく別のPCの結果による 1e-16 の差）．再実行の `config.json` の `method` は `ASA_SVRG` である．
  - 公開版の `pytest tests`：7件全て通過．
- 付随：公開版の一部の `.py`（元が CRLF）は，これまでの編集で LF になっていたため，残る `machine_learning_utils.py` も LF にそろえた（動作に影響なし）．
- 注意：コマンドラインの名前も変わった（例：`--methods ASA_SVRG`，`--conditions ASA_SVRG_reset`）．README の例も新しい名前である．

## 12. 追加の指示：README_ja.md の作成

指示（原文）：「README_ja.mdも作っておいて」．

- 公開版に `README_ja.md` を作成した．内容は `README.md`（英語）と同じで，である調・句読点は「，」「．」とした．
- 両方の先頭に互いへのリンク（`English | [日本語](README_ja.md)`）を加えた．
- `README_ja.md` の中のリンク先とディレクトリが全て実在することを確かめた．

## 13. 追加の指示：計算機の記述の削除とコメントの見直し（order_085 の7節）

### 13.1 計算機と複数のPCの記述

- README（英語・日本語）：環境構築の節から GPU・CPU の機種と試行ごとの GPU の記述を消し，ソフトウェアの版だけにした．再現性の節から「保存した結果の一部を別の計算機で計算した」記述とそれに基づく数値を消し，異なる環境で再実行する場合の一般的な注意に書き直した．
- プログラム：コメント・docstring・出力の文字列から，PC・計算機の記述を消した（`paper/mushroom_configs.py` の計算機の構成を書き出す関数，「$f^*$ の値は実行した PC によって…」の文など）．`tools/compare_logs.py` と `tests/` の注意は「実行環境によっては…」とした．
- 結果の `config.json` の計算機の情報（gpu，cpu，platform 等）は，ユーザーの回答により残した．

### 13.2 コメントの見直し

- 各モジュールの冒頭の説明（64モジュール）を，公開版での役割に合わせて書き直した．複製された data・model（a9a，CIFAR-10，ImageWoof）は同じ説明である．
- 各 `train.py`（15個）の冒頭に，対応する論文の節・図表，保存先，リポジトリのルートから実行するコマンド（全ての条件，1つの構成の短い実行，`tools/compare_logs.py` による比較）を書いた．`--help` でも表示される．
- 関数・クラスの docstring とコメントから，開発時の経緯（指示書，order・report の番号，開発時の実験番号，査読，論文の版，複製元，検証の経緯）を指す文と括弧書きを除いた．概要・引数・戻り値は残した．古い `batch_sizes` の2実験の開発時の条件名（基準条件・条件A〜C）と GPU メモリの見積もりの説明は書き直した．
- `paper/` から，図表の作成に使わない関数（論文の執筆中の内部の集計・TeX の表の作成など53関数），定数，import を除いた．`paper/a9a_uniform_floor.py` は開発時の実験の区別（E4・E8・E10）をやめ，`--results-root` の1つのオプションにした（出力の表の見出しが変わり，値は同じ）．
- docstring 中の `\omega` などによる `SyntaxWarning` を，該当する文字列を raw にして解消した．
- 作業は，控えから同じ手順をやり直せるスクリプト（作業用の `clean_comments.py`，`pipeline_post.py`）で行った．

### 13.3 確認

- `experiments/` のコードは，docstring を除いた構文木が見直しの前と同一である（変更は `cifar10/batch_sizes/train.py` の表示の文字列「SGD倍エポック（order_038）」→「SGD のエポック数」だけ）．
- 開発時の言及（order・report，指示書，開発時の実験番号，PC，GPU の機種など）は，コメント・docstring・文字列に0件（`EXPERIMENT_NAME` の値を除く．これは `config.json` に記録される実験の ID である）．
- 図9つは論文と画素単位で一致，表V〜XI と σ²/K の集計は見直しの前と同じ（a9a の集計は見出しの文言だけが変わった）．
- 全ての `train.py` の `--help`（オプションの無い2つは構文の確認），各 `train.py` の docstring に書いた短い実行のコマンド（Mushroom，a9a の3つ，CIFAR-10）をそのまま実行し，保存された結果と一致した（53件）．`pytest tests`：7件全て通過．
