# report_074：E6 の別のPCでの実行の準備（実験指示E6.md の照合，プログラムの作成，事前確認）

## 0. 要点

- `実験指示E6.md` の記述を，リポジトリの実物（関数，引数，パス，既存の結果，番号）と照合した．
  - 参照するファイルと既存の結果（`ex004`，`ex007` の `log.json`・`config.json`）は全て Git で管理されており，クローン先で利用できる．
  - 番号 `order_074`／`report_074` は本PCの次の番号と衝突するため，E6 は `order_080`／`report_080` に改めた．`ex011` は E6 に割り当てたままとし，本PCは今後 `ex012` 以降を使う．
- 指示書2節の実装を本PCで先に作成した．
  - 作成したのは `ex011_e6_cifar10_final_bs32_512/`，`analysis_e6.py`，`tests/test_e6_programs.py` である．
  - クローン先の作業は，環境の構築，一致の確認，並列数の決定，本実行，集計，レポート，zip の作成になる．
- 本PCで短い一致の確認を行った（3節）．
  - 付録B（`ex004`，同じ GPU）とは，バッチサイズ32・512の全項目がビット単位で一致した．
  - E2（`ex007`，RTX 3090）とは，`oracle_calls` が完全に一致し，0エポック目は相対 $ 3.8\times10^{-8} $ で一致した．1エポック目以降は GPU の違いによる差がある．
- テストは全146件が通った（既存132件，E6 の14件）．
- E6 の本実行（20試行）は本PCでは行っていない．
- **クローン先で使うには，本PCの変更の commit・push が必要である（未実施）．**

## 1. 作成・変更したファイル

| ファイル | 内容 |
|---|---|
| `programs_review/programs/ex011_e6_cifar10_final_bs32_512/data.py`，`model.py`，`e2_training.py`，`svrg_average.py` | ex007 から変更せずにコピーした |
| 同 `e2_runner.py` | ex007 のコピー．`--batch-sizes` の追加と，`--methods` の選択肢を6手法とする変更の2点のみ |
| 同 `train.py` | ex007 のコピー．`SPEC`（`SVRG_final`・`NFG_SVRG_final`，`epochs_by_batch_size = {32: 12, 512: 192}`，保存先）のみを変えた |
| 同 `README.md` | 目的，手法，条件，コピー元と変更点，実行方法，事前確認の結果 |
| `programs_review/programs/analysis_e6.py` | 指示書5節の集計．出力は表（予算と最終エポックの学習損失・テスト精度，予算の後半の近似誤差の中央値，打ち切り），ex004 の参考の表，確認用の図6枚 |
| `programs_review/tests/test_e6_programs.py` | テスト14件（2節） |
| `実験指示E6.md` | 実装済みであること，番号（`order_080`／`report_080`），テストの実行，一致の確認の比較のコマンド，事前確認の結果（3.4節），集計の実行方法を反映した |
| `.orders/order_074.md` | 本指示 |

- 変更前の `実験指示E6.md` は Git 管理外であったため，作業前の内容をセッションの作業用ディレクトリに退避した．
- `programs_review/document.md` の E6 の追記は，結果を統合するときに行う．
- 事前確認の結果は `programs_review/outputs/_verify/`（Git 管理外）に置いた．実行ログは `programs_review/outputs/_run_logs/verify_ex011_bs32_512.log` と `verify_ex011_bs128.log` に置いた．

### 1.1 指示書の主な変更点

| 節 | 変更 |
|---|---|
| 冒頭 | プログラムが作成済みであること．作業前に `ex011` 等の存在を確かめ，ない場合は `git pull` が必要と伝えること |
| 0.3，0.4 | 番号を `order_080`／`report_080` とした．作成済みのプログラムは原則として変更しない（誤りがあった場合の最小限の修正は許し，レポートに記録する） |
| 1 | テストの実行を加えた．WSL2 ではプロセスごとの GPU メモリが表示されないため，`memory.used` の差で測る旨を加えた |
| 2 | 「実装（作成済み）」とし，テストの内容を14件に合わせた |
| 3 | 実行のコマンドに比較のコマンド（`compare_with_existing_logs.py`）と出力先を加えた．実行ログの名前を `ex011_verify_*.log` とした（zip の `ex011*.log` に含めるため）．元のPCでの事前確認の結果を3.4節に載せた |
| 5 | `analysis_e6.py` の実行方法と出力（図のファイル名，`--reference-root`）を書いた |
| 6，7 | 番号を改めた．作成済みのプログラムの修正の記録をレポートの項目に加えた．プログラムを zip に含める理由を書いた |

## 2. テスト（`tests/test_e6_programs.py`）

- ex007 と ex011 には同じ名前のモジュール（`e2_runner`，`e2_training`）がある．そのため，ex011 のモジュールはファイルのパスから一意の名前で読み込んだ．同じ pytest のプロセスで E2 のテストと衝突しない．
- 確認した内容は次のとおりである．
  1. 既定の引数で20個のタスクが作られ，保存先とエポック数が指示書2.4節と一致すること．
  2. `--batch-sizes 32` で10個，`--methods SVRG --batch-sizes 32 --seeds 0` で1個，`--batch-sizes 128 --epochs 3` で `..._bs128_..._epochs3` になること．`--methods` の選択肢が6手法であること．
  3. `SVRG_final`・`NFG_SVRG_final` のスナップショットが内側ループの最終点と一致し，次の内側ループがそこから始まること．
  4. `oracle_calls` の増分（`SVRG_final`：初期 $ N $・1エポック $ 3N $，`NFG_SVRG_final`：1エポック $ 2N $）．
  5. コピーした4ファイルが ex007 と同一であること．集計の打ち切りのエポックと予算の後半の中央値．
- `analysis_e6.py` は，ex004 の SVRG・NFG-SVRG を E6 の手法の名前で読み替えた仮の入力でも実行し，表と図が作られることを確かめた．

## 3. 本PCでの事前確認

本PCの GPU は NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition（`ex004` と同じ）である．

| 確認 | 比較先（GPU） | `oracle_calls` | 0エポック目 | 1エポック目以降 |
|---|---|---|---|---|
| SVRG，バッチサイズ32・512，2エポック，シード0 | `ex004`（同じ GPU） | 完全一致 | 完全一致 | `train_loss`，`test_accuracy`，`approx_error` が完全一致 |
| `SVRG_final`，`NFG_SVRG_final`，バッチサイズ128，3エポック，シード0 | `ex007`（RTX 3090） | 完全一致 | `train_loss` の相対差 $ 3.8\times10^{-8} $，`test_accuracy` は一致 | `train_loss` の相対差は最大 $ 2.3\times10^{-4} $，`test_accuracy` の相対差は最大 $ 3.4\times10^{-3} $ |

- 同じ GPU の比較がビット単位で一致した．したがって，コピーしたプログラムは付録B と同じ計算をバッチサイズ32・512で行う．
- E2 との差は，0エポック目では丸め誤差の範囲にあり，学習が進むと広がる．この傾向は GPU の違いによるものと考えられる（推測）．
- クローン先の GPU が `ex007` と同じ RTX 3090 であれば，3.1節の確認はビット単位で一致すると予想される（推測）．

## 4. 未解決の論点・次の作業の候補

- **commit・push：** クローン先で実行するには，次のファイルの commit・push が必要である．
  - `実験指示E6.md`，`.orders/order_074.md`，`.reports/report_074.md`
  - `programs_review/programs/ex011_e6_cifar10_final_bs32_512/`，`programs_review/programs/analysis_e6.py`，`programs_review/tests/test_e6_programs.py`
  - `programs_review/outputs/_run_logs/verify_ex011_*.log`
- **E5 の ImageWoof・512：** シード4が実行中である（11月21日22時半頃に終了の見込み）．終了後に `report_073` を更新する．
- **E6 の統合：** `E6_results.zip` を受け取った後に，`document.md` への追記と論文用の図の作成を行う．
