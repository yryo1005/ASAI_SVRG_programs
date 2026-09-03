# report_003

本レポートは，`.orders/order_003.md` の指示に基づき実施した，勾配計算方式の修正内容
（解析的勾配からPyTorchの標準的な自動微分への変更）と，修正後もEx001の実験結果が変化しない
ことの検証結果をまとめたものである．Ex001の実験条件・結果の詳細な考察は `.reports/report_001.md`
を，最適化手法クラスの設計については `.reports/report_002.md` を参照のこと．

## 1. 指示内容

`.orders/order_003.md` は次の2点を指示するものである．

1. `@.ai/ai-dev-kit/root_prompt.md` に従ってプログラムを作成すること．
2. 勾配はPyTorchの標準的な自動微分（`torch.autograd`，`loss.backward()`）で取得すること．

これまでのEx001の実装（`.reports/report_001.md`，`.reports/report_002.md`）では，ロジスティック
回帰の閉形式（解析的）勾配を直接計算しており，自動微分は使用していなかった．本指示に基づき，
`programs/ex001_mushroom_svrg/model.py` の勾配計算を自動微分による実装へ全面的に置き換えた．

## 2. 実施内容

### 2.1 設計上の課題と対応

SVRG系手法（SVRG，NFG SVRG，ASAI SVRG）は，論文の補正勾配

$$
\mathbf{v}_s^k = \nabla f_{n_s^k}(\mathbf{w}_s^k) - \nabla f_{n_s^k}(\mathbf{z}_s) + \mathbf{g}_s
$$

の計算のため，同一サンプルに対して現在のパラメータ $ \mathbf{w}_s^k $ とスナップショット
$ \mathbf{z}_s $ の双方における勾配を必要とする．自動微分は，実際にその値でforward計算を
行った計算グラフに対してのみ機能するため，1つのモデルインスタンスのパラメータを差し替えながら
2種類の勾配を得ることはできない．そこで，以下の設計とした．

- `model`：現在のパラメータ $ \mathbf{w}_s^k $ を保持し，最適化手法クラス（`SGD`／`SVRG`／
  `NFGSVRG`／`ASAISVRG`）が直接更新するモデルインスタンス．
- `snapshot_model`：スナップショット $ \mathbf{z}_s $ を保持する，`model` と同一構造の別の
  モデルインスタンス．エポック境界でのみ `model.py` の `set_model_params` によりパラメータを
  上書きする．

内部ループの各ステップでは，同一サンプル $ (\mathbf{x}_n, y_n) $ に対して `model` と
`snapshot_model` それぞれにforward／backwardを1回ずつ実行し，2種類の勾配を得る．最適化手法
クラス（`programs/optimizers/optimizers.py`）は，`model.parameters()` の `.grad` 属性から
現在のパラメータにおける勾配を読み取り，スナップショットにおける勾配は引数として受け取る
という，`.orders/order_002.md` の時点から変更していないインターフェースをそのまま利用できる
ため，最適化手法クラス自体の修正は不要であった．

### 2.2 変更したプログラム

- `programs/ex001_mushroom_svrg/model.py`：`compute_gradient` を，解析的勾配の直接計算から
  `model.zero_grad()` → `loss.backward()` → `model.parameters()` の `.grad` を読み取る実装へ
  置き換えた．また，スナップショット専用モデルのパラメータを更新するための `set_model_params`
  関数を追加した．`compute_loss`／`compute_accuracy` も，パラメータのタプルではなくモデル
  インスタンスを受け取る形式に統一した．
- `programs/ex001_mushroom_svrg/train.py`：`run_sgd`／`run_variance_reduced` を，
  パラメータのテンソル列を直接操作する実装から，`model`／`snapshot_model` の2モデル構成へ
  書き換えた．

### 2.3 単体テストの追加

`tests/test_model.py` を新規に作成し，以下を確認した（4件，全て合格）．

- 自動微分による勾配が，ロジスティック回帰の閉形式勾配と数値的に一致すること
  （`atol=1e-10`）．
- `compute_gradient` の呼び出し後，`model.parameters()` の `.grad` が直接更新されていること
  （最適化手法クラスが正しく読み取れることの保証）．
- `compute_loss`／`compute_accuracy` が勾配計算グラフを構築しないこと．
- `set_model_params` がモデルのパラメータを正しく上書きすること．

既存の `tests/test_optimizers.py`（8件）は最適化手法クラス自体の修正を伴わないため無変更であり，
引き続き全て合格することを確認した．

```bash
.venv_pytorch/bin/python -m pytest tests/ -v
# 12 passed
```

### 2.4 Ex001の再実行と結果の検証

修正後の実装でEx001（4手法 x 5Seed = 20条件）を再実行した．60エポック目における目的関数の
誤差は以下の通りであり，解析的勾配による実装（`.reports/report_002.md` 時点）の結果と，
浮動小数点演算の経路の違いに由来する誤差（相対誤差 $ 10^{-10} $ 程度）を除いて完全に一致する
ことを確認した．

| 手法 | 目的関数の誤差（60エポック目，5Seed平均） | 解析的勾配実装との一致 |
| :--- | ---: | :---: |
| SGD | $ 0.070012 $ | 一致 |
| SVRG | $ 0.000267 $ | 一致 |
| NFG SVRG | $ 0.006375 $ | 一致 |
| ASAI SVRG | $ 0.000889 $ | 一致 |

分類精度・フル勾配の近似誤差についても同様に一致を確認した．`visualize_result.ipynb` を
再実行し，`outputs/ex001_mushroom_svrg/` 以下のグラフ画像を更新した．図・数値の内容および
考察は `.reports/report_001.md` の記載から変更がない．

### 2.5 実行時間への影響

自動微分は解析的勾配の直接計算と比べてオーバーヘッドが大きいため，1エポックあたりの実行時間は
手法により約2〜4倍に増加した（60エポックの1条件あたり，SGDで約2.4秒/エポック，SVRG系手法で
約4.4〜5.3秒/エポックの計算時間を要した．マルチプロセス並列実行のため，20条件全体の実行時間の
増加は最も遅い条件の実行時間で決まる）．目的関数の誤差等の実験結果自体には影響しない．

## 3. 未解決の論点・今後の検討候補

- `.reports/report_001.md` の「4. 未解決の論点」に記載した内容は，本修正後も変更なく有効である．
- 4.2節（CNNとCIFAR-10の実験）では非線形なモデルを扱うため，自動微分の利用は今回の変更により
  むしろ自然な選択となる．一方，`model`／`snapshot_model` の2モデル構成は，CNNのような
  パラメータ数の多いモデルでは，モデルの複製・パラメータの上書きにかかるコストが無視できなく
  なる可能性があり，実装時に確認が必要である．

## 4. 実行コマンド

```bash
# 単体テスト
.venv_pytorch/bin/python -m pytest tests/ -v

# 学習の再実行
.venv_pytorch/bin/python programs/ex001_mushroom_svrg/train.py

# 結果の可視化
.venv_pytorch/bin/jupyter nbconvert --to notebook --execute --inplace visualize_result.ipynb
```
