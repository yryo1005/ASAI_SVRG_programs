# report_002

本レポートは，`.orders/order_002.md` の指示に基づき実施した，最適化手法クラス
（`programs/optimizers/optimizers.py`）の実装方式の修正内容と，修正後もEx001の実験結果が
変化しないことの検証結果をまとめたものである．Ex001の実験条件・結果の詳細な考察は
`.reports/report_001.md` を参照のこと．

## 1. 指示内容の解釈

`.orders/order_002.md` は，`.orders/order_001.md` における「SVRGはPyTorchの公式実装ではなく，
陽に最適化手法のクラスとして実装しこれを使用します」という指示について，以下の2点を明確化する
ものと解釈した．

1. 「PyTorchの公式実装を使わない」とは，`torch.optim.SGD` 等，具体的な最適化アルゴリズムの
   公式実装クラスを使わないという意図であり，`torch.optim.Optimizer` 基底クラス自体の利用を
   排除するものではない．したがって，全ての最適化手法クラスは `torch.optim.Optimizer` の
   サブクラスとして実装する．
2. SVRG，NFG SVRG，ASAI SVRGの3手法には内部ループの更新則など共通する部分があるが，これを
   単一の共通クラス（`SVRGOptimizer`）へ抽出せず，4手法（SGD，SVRG，NFG SVRG，ASAI SVRG）
   それぞれを独立した陽実装として個別のクラスに定義する．

## 2. 実施内容

### 2.1 変更したプログラム

`programs/optimizers/optimizers.py` を全面的に書き直し，以下の4クラスを実装した．いずれも
`torch.optim.Optimizer` のサブクラスであり，共通の基底クラスを持たない．

| クラス | 対応するアルゴリズム | 概要 |
| :--- | :--- | :--- |
| `SGD` | (3)/(4)式 | 通常のミニバッチ確率的勾配降下法． |
| `SVRG` | Algorithm 1 | スナップショット勾配 $ g_s $ を外部（`train.py`）で計算したフル勾配として `set_snapshot_gradient()` により設定する．次エポックのスナップショット点 $ z_{s+1} $ は，内部ループのパラメータ列から一様ランダムに選ぶ処理を，クラス内部（`begin_epoch()`／`step()`）で完結させている． |
| `NFGSVRG` | Algorithm 2 | スナップショット勾配を，内部ループで観測した確率的勾配の逐次平均（式(8)）としてクラス内部で計算・保持する．次エポックのスナップショット点の選択方法は`SVRG`と同様． |
| `ASAISVRG` | Algorithm 3 | スナップショット勾配（式(14)）に加え，スナップショット点そのものも内部ループのパラメータ列の逐次平均（式(13)）としてクラス内部で計算・保持する（外部からの乱数選択が不要）． |

各クラスは，通常の `torch.optim.Optimizer` の流儀（`self.param_groups`，`self.state`，
各パラメータの `.grad` 属性からの勾配読み取り）に従う．SVRG系3クラスは，`step()` に加えて
外部ループの境界で呼び出す `begin_epoch()`／`end_epoch()`，および次エポックのスナップショットを
取得する `get_snapshot_params()`／`get_snapshot_gradient()` を提供する．3クラスが同一の
インターフェースを持つため，`programs/ex001_mushroom_svrg/train.py` の `run_variance_reduced()`
関数は，このインターフェースを介して1つの関数で3手法をまとめて扱っている（Optimizerクラス自体は
指示通りそれぞれ独立に陽実装されている）．

`.grad` 属性には，`model.py` の解析的勾配計算関数（`compute_gradient`）で計算した勾配を
`train.py` 側で明示的に代入している（`p.grad = grad_tensor`）．これは自動微分を経由しないが，
`torch.optim.Optimizer` のインターフェース上は通常の学習ループと同じ扱いとなる．

### 2.2 単体テストの更新

`tests/test_optimizers.py` を新しいクラス設計に合わせて全面的に書き直した．8件のテストを追加し，
全て合格することを確認した．

- 全クラスが `torch.optim.Optimizer` のサブクラスであることの確認
- `SGD` の更新式の正しさ
- `SVRG` の補正勾配（分散削減付き更新式）の正しさ
- `SVRG` のスナップショット選択（`begin_epoch` で選ばれたステップ番号における更新前パラメータの保持）の正しさ
- `NFGSVRG` の平均勾配（式(8)）が単純平均と一致すること
- `ASAISVRG` の平均パラメータ（式(13)）が単純平均と一致すること
- `ASAISVRG` の第1エポックでパラメータが更新されないこと（論文3.1節の性質）
- `SVRG` が分散0の場合に通常の勾配降下法へ一致すること（2次関数の最小化）

```bash
.venv_pytorch/bin/python -m pytest tests/test_optimizers.py -v
# 8 passed
```

### 2.3 Ex001の再実行と結果の検証

修正後のクラスを用いて，Ex001（4手法 x 5Seed = 20条件）を再実行した．修正前後で目的関数の誤差・
分類精度・フル勾配の近似誤差の推移が完全に一致すること（アルゴリズムとしての等価性）を確認した．

| 手法 | 目的関数の誤差（60エポック目，5Seed平均） | 修正前との一致 |
| :--- | ---: | :---: |
| SGD | $ 0.070012 $ | 一致 |
| SVRG | $ 0.000267 $ | 一致 |
| NFG SVRG | $ 0.006375 $ | 一致 |
| ASAI SVRG | $ 0.000889 $ | 一致 |

（「修正前」は `.reports/report_001.md` 作成時点の実装による結果．全指標について小数点以下
十分な桁数まで完全に一致することを確認した．）

なお，オラクル呼び出し回数（`oracle_calls`）の記録方法について，修正の過程で1点の不整合を
発見し合わせて修正した．SVRGはエポック0（学習開始前の初期状態）の評価時点で，次エポックの
学習に必要なフル勾配の計算コストが誤って計上されていた．これを，エポック0では
`oracle_calls = 0`（他手法と同様，学習開始前は計算コストを消費していない状態）となるよう修正し，
フル勾配の計算コストはエポック1の学習コストとして計上するよう変更した．目的関数の誤差・
分類精度・近似誤差の値自体には影響しない，記録上の整合性の修正である．

再実行に伴い，`visualize_result.ipynb` を再実行し，`outputs/ex001_mushroom_svrg/` 以下の
グラフ画像を更新した．図・数値の内容および考察は `.reports/report_001.md` の記載から変更がない
ことを確認済みである．

## 3. 未解決の論点・今後の検討候補

- `.reports/report_001.md` の「4. 未解決の論点」に記載した内容は，本修正後も変更なく有効である．
- SVRG，NFGSVRG，ASAISVRGの3クラスは，本レポートで述べた通り意図的に共通の基底クラスを持たない
  設計とした．4.2節（CNNとCIFAR-10の実験）で同じ設計方針を維持するか（コードの重複が3倍に
  増える），あるいは異なる方針を採るかは，次の指示があった際に確認が必要である．

## 4. 実行コマンド

```bash
# 単体テスト
.venv_pytorch/bin/python -m pytest tests/ -v

# 学習の再実行
.venv_pytorch/bin/python programs/ex001_mushroom_svrg/train.py

# 結果の可視化
.venv_pytorch/bin/jupyter nbconvert --to notebook --execute --inplace visualize_result.ipynb
```
