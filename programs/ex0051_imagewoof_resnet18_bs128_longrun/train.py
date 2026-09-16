"""
実験5b（ex0051，バッチサイズ128長期学習とプラトー判定方法の改善）・実験5c（ex0051拡張，
学習率・バッチサイズを変えた崩壊対策グリッド）の学習ループを定義し，実験を実行する
スクリプト．

`.orders/order_035.md` に対応する．Stage B（`report_034.md`）は，バッチサイズ128・学習率
0.001において，ASAI SVRGが64エポック時点で`compute_trailing_relative_change`（隣接エポック
間の相対変化の最大値）による判定で「プラトーに到達した」と結論づけた．しかし，この指標は
隣接ステップ間の変化率のみを見るため，各ステップの変化が小さくても複数エポックにわたって
累積すれば無視できない上昇トレンドとなる「緩やかな単調増加」を検出できないという方法論上の
欠陥がある．

本実験は，(1) バッチサイズ128・学習率0.001の条件についてエポック数を倍（128エポック）に
延長し，(2) プラトー判定の方法自体を，従来の隣接相対変化に加え，末尾区間の正味変化量および
線形回帰の傾きを用いる，より頑健な3指標方式に改善した上で，改めてプラトーへの到達を確認する．

## モデル・データセット（`.orders/order_035.md` 2節）

`data.py`・`model.py`はStage A/B（`programs/ex005_imagewoof_resnet18/`）と完全に同一のもの
（`EXPERIMENT_NAME`も含め無変更）を複製している．これにより，`datasets/
ex005_imagewoof_resnet18/`にキャッシュ済みのImagewoofデータ・チャネル統計量がそのまま
再利用される．

## プラトー判定方法の改善（3節）

以下の3指標を全て算出し，いずれか1つでも「未プラトー」を示す場合は全体として未プラトーと
判定する（`is_plateaued`関数）．閾値は，Stage Bの実測データ（`report_034.md`）を用いた事前
較正により決定した．較正では，明らかに学習途上にあるSGD・バッチサイズ64（Stage B，末尾
window=10で正味変化21.7%，回帰傾き1.5%/エポック）と，Stage Bで「プラトー」と判定された
ASAI SVRG・バッチサイズ128（隣接相対変化のみ0.97%と小さいが，正味変化は5.2%，回帰傾きは
0.53%/エポックであり，決して無視できる水準ではなかった）を比較し，後者を「未プラトー」側に
正しく分類できる閾値として以下を採用した．

1. **隣接相対変化の最大値**（`compute_trailing_relative_change`，既存指標）：閾値2%
2. **末尾$W$エポック区間の正味変化量**（`compute_trailing_net_change`，新規）：
   区間の始点・終点の相対差 $ |v^{(T)}-v^{(T-W)}|/|v^{(T-W)}| $．閾値2%
3. **末尾$W$エポックの線形回帰の傾き**（`compute_trailing_regression_slope`，新規）：
   最小二乗の傾きを区間平均値で正規化した相対傾き．閾値0.1%/エポック

窓幅 $ W=10 $ を採用した（128エポックの約7.8%に相当し，Stage Bの短いエポック数（12〜64）
向けに用いていた$ W=5 $よりも長期のトレンドを捉えられる一方，128エポックに対し極端に長すぎ
ない値である）．

## Optimizerクラスの選択とelapsed_timeの計上方法（`.orders/order_034.md` 2節を踏襲）

実験2系列以降で用いてきた `SGD`／`SVRG`／`NFGSVRG`／`ASAISVRG` をそのまま用いる．
`elapsed_time`の計上方法は`report_032_ex002.md` 8.1節の修正済みルールを踏襲する．

## 既存結果を再利用しない理由（`.orders/order_035.md` 4節）

Stage Bと条件が重複するバッチサイズ128・学習率0.001の20条件についても，Optimizerの内部状態
（running average等）が保存されておらず学習途中から再開できないため，継続学習ではなく
ゼロから再学習する（Stage B（`.orders/order_034.md` 3節）と同様の理由）．Stage Bとの
接続確認（Seed 0の先頭64エポックの一致）は別途実施する．

## 実行前の計算コスト見積もり（`.orders/order_035.md` 6節）

Stage Bで判明した教訓（1イテレーションのforward/backward時間から1エポックあたりの時間を
算出する方式は，`evaluate_epoch`や診断専用フル勾配計算のコストを見落とし，実測が見積もりの
約2.8倍に達する誤差を生んだ）を踏まえ，本実験では**1エポック全体（学習・評価・診断計算を
全て含む）を実際に1回実行し，その所要時間を計測してから全20条件の総所要時間を見積もる**
方式に変更した．実測結果は，SGD 85.1秒／エポック，NFG SVRG 81.8秒／エポック，ASAI SVRG
82.9秒／エポック，SVRG 116.5秒／エポック（SVRGのみ診断専用ではなく実際に学習へ用いる
フル勾配計算を毎エポック行うため他手法より長い）であった．VRAM使用量は全手法で約5.9GB
（reservedベース）であり，Stage Bの実測値と同程度であることを確認した．

単一プロセスでの総所要時間（4手法×5Seed×128エポックの合計）は約65.1時間と見積もられる．
8プロセス並列により理想的には約8.1時間まで短縮される計算だが，Stage Bで観測された並列
実行時のGPU資源競合による補正（実測は理想的な並列短縮の約1.32倍）を適用すると，実際の
所要時間は約10.7〜12.2時間程度になると見込まれる．VRAM使用量には十分な余裕がある
（$8\\times5.9\\text{GB}\\approx47.2\\text{GB}$，総VRAM約102.6GBに対し十分小さい）ため，
8プロセス並列のまま実行する．実測との比較は`.reports/report_035.md`に記載する．

## 実験5c：崩壊対策グリッドへの拡張（`.orders/order_036.md`）

ex0051（基準条件，バッチサイズ128・学習率0.001・128エポック）は，ASAI SVRG・Seed 3で
エポック117付近から近似誤差の急増を伴う終盤崩壊を示した（`.reports/report_035.md` 6.1節）．
本実験は，ASAI SVRG論文のアルゴリズム自体（Algorithm 3, 4）を変更しない範囲での崩壊対策
として，(1) 学習率を一桁下げる（条件A：バッチサイズ128・学習率0.0001），(2) バッチサイズを
拡大する（条件B：バッチサイズ512・学習率0.001）の2条件を追加検証する．

条件A・Bとも，基準条件の総イテレーション数（$128\\times71=9088$）にエポック数を揃えた
（`CONDITIONS`定数）．条件Aはバッチサイズ不変（$K=71$）のため基準条件と同一の128エポック，
条件Bはバッチサイズ512（$K=\\lceil9025/512\\rceil=18$）のため
$\\lceil9088/18\\rceil=505$エポック（総イテレーション数9090，基準条件から+2の端数）とした．

タスク生成部分（`main`関数）を，単一のハイパーパラメータの組ではなく`CONDITIONS`（基準条件・
条件A・条件Bのリスト）に対応するグリッドを生成できるよう修正した．基準条件の既存20条件は
`is_run_completed`により自動的にスキップされ，条件A・Bの40条件のみが新規に学習される．
`data.py`・`model.py`はex0051から変更していない．

### 実験5c：実行前の計算コスト見積もり（`.orders/order_036.md` 6節）

グリッド中の最大バッチサイズ（512，条件B）で1エポック全体（学習・評価・診断計算を全て
含む）を実際に1回実行し計測した結果，SGD 81.6秒／エポック，NFG SVRG 80.1秒／エポック，
ASAI SVRG 83.6秒／エポック，SVRG 114.7秒／エポックであった（バッチサイズ128とほぼ同水準の
1エポックあたり時間だが，バッチサイズ512は1エポックあたりのイテレーション数$K=18$が
バッチサイズ128の$K=71$よりはるかに少ないため，同一エポック数での学習の進み方は大きく
異なる）．

**VRAM使用量はバッチサイズ512で約22.6GB（reservedベース）に達し，バッチサイズ128の
約5.9GBの約3.8倍であった**．8プロセス並列のままバッチサイズ512のタスクを実行すると
合計VRAM使用量が約180.8GBとなり，総VRAM（約102.6GB）を大幅に超過してGPUが応答不能に
なる危険がある（ex004・`.reports/report_030.md` 5.3節と同種のリスク）．そのため，
`main`関数を，バッチサイズ128のタスク（基準条件・条件A，8プロセス並列，合計約47.2GB）と
バッチサイズ512のタスク（条件B，4プロセス並列，合計約90.2GB）を分離した2つの逐次実行
プロセスプールに変更した（`_build_tasks`関数がバッチサイズを引数に取るよう修正）．

条件Bの単一プロセスでの総所要時間（4手法×5Seed×505エポックの合計）は約252.5時間と
見積もられ，4プロセス並列により理想的には約63.1時間まで短縮される計算である．ex0051の
並列実行時の補正係数（実測は理想的な並列短縮の約1.32倍）を適用すると，条件Bの実際の
所要時間は約80〜85時間程度になると見込まれる．これは基準条件・条件A（バッチサイズ128，
8プロセス並列，合計約20〜25時間程度と見込まれる）よりも大幅に長く，本実験全体の所要時間は
条件Bが支配的になる．実測との比較は`.reports/report_036.md`に記載する．
"""

import itertools
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

_PROGRAMS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_PROGRAMS_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))
sys.path.insert(0, _PROGRAMS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from machine_learning_utils import ResultLogger, set_seed  # noqa: E402
from optimizers import ASAISVRG, NFGSVRG, SGD, SVRG  # noqa: E402

from data import load_dataloader  # noqa: E402
from model import (  # noqa: E402
    ResNet18LayerNorm,
    load_model,
    loss_func,
    metrics_func,
    set_model_params,
)

EXPERIMENT_NAME = "ex0051_imagewoof_resnet18_bs128_longrun"
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs", EXPERIMENT_NAME)

SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGD", "SVRG", "NFG_SVRG", "ASAI_SVRG"]
REG_LAMBDA = 5e-4
NUM_CLASSES = 10
NEAR_CHANCE_TOLERANCE = 0.03  # チャンスレベル(0.1)の±0.03，本実験の10クラス設定向け

# --- 実験条件のグリッド（`.orders/order_035.md`：基準条件，`.orders/order_036.md`：
# 条件A・条件B）。総イテレーション数を基準条件（128エポック×K=71=9088）に揃えている
# （条件Bのみ端数により9090で若干上回る，`.reports/report_036.md`参照）。
CONDITIONS = [
    {"batch_size": 128, "learning_rate": 0.001, "epochs": 128},  # 基準条件（order_035）
    {"batch_size": 128, "learning_rate": 0.0001, "epochs": 128},  # 条件A：学習率を1/10に低減
    {"batch_size": 512, "learning_rate": 0.001, "epochs": 505},  # 条件B：バッチサイズ拡大
]

# 後方互換のため，基準条件の値を単体テスト等から参照できるよう維持する。
BATCH_SIZE = CONDITIONS[0]["batch_size"]
LEARNING_RATE = CONDITIONS[0]["learning_rate"]
EPOCHS = CONDITIONS[0]["epochs"]

# --- プラトー判定（order_035 3節）：窓幅と3指標の閾値 ---
PLATEAU_WINDOW = 10
PLATEAU_THRESHOLD_RELATIVE_CHANGE = 0.02  # 隣接相対変化の最大値
PLATEAU_THRESHOLD_NET_CHANGE = 0.02  # 末尾区間の正味相対変化量
PLATEAU_THRESHOLD_REGRESSION_SLOPE = 0.001  # 末尾区間の相対線形回帰傾き（1エポックあたり）

_VARIANCE_REDUCED_OPTIMIZER_CLASSES = {
    "SVRG": SVRG,
    "NFG_SVRG": NFGSVRG,
    "ASAI_SVRG": ASAISVRG,
}


def hp_name(lr: float, batch_size: int, epochs: int) -> str:
    """
    概要: ハイパーパラメータ条件名（ディレクトリ名）を構築する．Stage A/Bと同一の命名規則
        を用いることで，接続確認（同一条件の比較）を行いやすくする．
    引数:
        lr (float)．学習率．
        batch_size (int)．ミニバッチサイズ．
        epochs (int)．エポック数．
    戻り値: name (str)．
    """
    return f"lr{lr}_bs{batch_size}_lambda{REG_LAMBDA}_epochs{epochs}"


def is_stuck_near_chance(accuracies, num_classes: int, window: int = 3, tolerance: float = NEAR_CHANCE_TOLERANCE) -> bool:
    """
    概要: 末尾 `window` エポックの分類精度がすべてチャンスレベル（$ 1/\\text{num\\_classes} $）
        付近に留まっているかを判定する．NaNにはならないがモデルが実質的に機能不全に陥る
        「見えない崩壊」の検出に用いる（Stage A/Bと同一実装）．
    引数:
        accuracies (Sequence[float])．エポックごとの分類精度の列．
        num_classes (int)．クラス数．
        window (int) = 3．判定に用いる末尾のエポック数．
        tolerance (float) = 0.03．チャンスレベルからの許容乖離幅．
    戻り値: stuck (bool)．
    """
    if len(accuracies) < window:
        return False
    chance = 1.0 / num_classes
    tail = accuracies[-window:]
    return all(abs(a - chance) <= tolerance for a in tail)


def compute_trailing_relative_change(values, window: int) -> float:
    """
    概要: 数値列の末尾 `window` 個について，隣接する値の間の相対変化
        $ |v^{(t)}-v^{(t-1)}|/|v^{(t-1)}| $ の最大値を計算する（ex0023・Stage Bと同一実装）．
        各ステップの変化が小さくても，複数エポックにわたる緩やかな単調増加は検出できない
        （`.orders/order_035.md` 1節が指摘する方法論上の欠陥）．
    引数:
        values (Sequence[float])．時系列の数値列．
        window (int)．末尾から何個の値を対象にするか．`window + 1` 個以上の要素が必要．
    戻り値: max_relative_change (float)．非有限値を含む場合は `float("inf")`．
    """
    tail = np.asarray(values[-(window + 1):], dtype=np.float64)
    if not np.all(np.isfinite(tail)):
        return float("inf")
    diffs = np.abs(tail[1:] - tail[:-1])
    denom = np.abs(tail[:-1])
    denom = np.where(denom == 0.0, np.finfo(np.float64).eps, denom)
    return float(np.max(diffs / denom))


def compute_trailing_net_change(values, window: int) -> float:
    """
    概要: 数値列の末尾 `window` エポック区間について，区間の始点（$ t=T-W $）と終点
        （$ t=T $）の値の相対差 $ |v^{(T)}-v^{(T-W)}|/|v^{(T-W)}| $ を計算する
        （`.orders/order_035.md` 3節1項）．隣接ステップの変化が小さくても，区間全体での
        正味の変化が大きい「緩やかな単調増加」を検出するために用いる．
    引数:
        values (Sequence[float])．時系列の数値列．
        window (int)．区間の長さ．`window + 1` 個以上の要素が必要．
    戻り値: net_relative_change (float)．非有限値を含む場合は `float("inf")`．
    """
    tail = np.asarray(values[-(window + 1):], dtype=np.float64)
    if not np.all(np.isfinite(tail)):
        return float("inf")
    v_start, v_end = tail[0], tail[-1]
    denom = abs(v_start) if v_start != 0.0 else np.finfo(np.float64).eps
    return float(abs(v_end - v_start) / denom)


def compute_trailing_regression_slope(values, window: int) -> float:
    """
    概要: 数値列の末尾 `window + 1` 個の値に対し，エポック番号を説明変数とする単純な最小
        二乗の線形回帰を当てはめ，得られた傾き（1エポックあたりの変化量）を区間内の値の
        絶対値の平均で正規化した相対傾きを計算する（`.orders/order_035.md` 3節2項）．
        振動を含むデータに対しても，末尾区間全体のトレンドをロバストに検出できる．
    引数:
        values (Sequence[float])．時系列の数値列．
        window (int)．区間の長さ．`window + 1` 個以上の要素が必要．
    戻り値: relative_slope (float)．1エポックあたりの相対変化量．非有限値を含む場合は
        `float("inf")`．
    """
    tail = np.asarray(values[-(window + 1):], dtype=np.float64)
    if not np.all(np.isfinite(tail)):
        return float("inf")
    x = np.arange(len(tail), dtype=np.float64)
    slope, _ = np.polyfit(x, tail, 1)
    mean_abs = np.mean(np.abs(tail))
    denom = mean_abs if mean_abs != 0.0 else np.finfo(np.float64).eps
    return float(slope / denom)


def is_plateaued(values, window: int = PLATEAU_WINDOW) -> dict:
    """
    概要: 3指標（隣接相対変化，正味変化量，線形回帰の相対傾きの絶対値）を全て算出し，
        いずれか1つでも対応する閾値を超える場合は全体として「未プラトー」と判定する
        （`.orders/order_035.md` 3節）．
    引数:
        values (Sequence[float])．時系列の数値列（例：エポックごとの分類精度）．
        window (int) = `PLATEAU_WINDOW`．
    戻り値: result (dict)．
        {"relative_change": float, "net_change": float, "regression_slope": float,
         "plateaued": bool}．`values` の長さが `window + 1` 未満の場合は
        `plateaued=False`，各指標は `float("nan")` とする．
    """
    if len(values) < window + 1:
        return {
            "relative_change": float("nan"),
            "net_change": float("nan"),
            "regression_slope": float("nan"),
            "plateaued": False,
        }
    relative_change = compute_trailing_relative_change(values, window)
    net_change = compute_trailing_net_change(values, window)
    regression_slope = compute_trailing_regression_slope(values, window)
    plateaued = (
        relative_change < PLATEAU_THRESHOLD_RELATIVE_CHANGE
        and net_change < PLATEAU_THRESHOLD_NET_CHANGE
        and abs(regression_slope) < PLATEAU_THRESHOLD_REGRESSION_SLOPE
    )
    return {
        "relative_change": relative_change,
        "net_change": net_change,
        "regression_slope": regression_slope,
        "plateaued": plateaued,
    }


def iteration(model, inputs, teacher_signals, reg_lambda, optimizer, snapshot_model=None) -> dict:
    """
    概要: 1つのミニバッチのデータを学習する関数．`snapshot_model` が指定される場合（SVRG系
        手法），同一ミニバッチに対して `model`（現在のパラメータ $ w_s^k $）と
        `snapshot_model`（スナップショット $ z_s $）の双方でforward／backwardを実行し，
        2種類の勾配を用いて `optimizer.step(grad_at_snapshot)` を呼び出す．`snapshot_model`
        が `None` の場合（SGD），通常のforward／backward／`optimizer.step()` を実行する．
    引数:
        model (torch.nn.Module)．
        inputs (torch.Tensor)，形状 (B, 3, 224, 224)．入力画像．
        teacher_signals (torch.Tensor)，形状 (B,)．教師ラベル．
        reg_lambda (float)．L2正則化係数．
        optimizer (torch.optim.Optimizer)．
        snapshot_model (torch.nn.Module) = None．SVRG系手法のスナップショットを保持するモデル．
    戻り値: metrics_to_value (dict)．{"loss": ..., "accuracy": ...}．
    """
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_func(outputs, teacher_signals, model, reg_lambda)
    loss.backward()

    if snapshot_model is not None:
        snapshot_model.zero_grad()
        snapshot_outputs = snapshot_model(inputs)
        snapshot_loss = loss_func(snapshot_outputs, teacher_signals, snapshot_model, reg_lambda)
        snapshot_loss.backward()
        grad_at_snapshot = [p.grad.detach().clone() for p in snapshot_model.parameters()]
        optimizer.step(grad_at_snapshot)
    else:
        optimizer.step()

    metrics = metrics_func(outputs, teacher_signals)
    metrics["loss"] = loss.item()
    return metrics


def train_epoch(model, dataloader, device, reg_lambda, optimizer, snapshot_model, desc="") -> None:
    """
    概要: 1エポック分の学習（内部ループ）を実行する．
    引数: `iteration` 参照．desc (str)．tqdmの進捗バーの説明文．
    戻り値: なし
    """
    for inputs, teacher_signals in tqdm(dataloader, desc=desc, leave=False):
        inputs = inputs.to(device)
        teacher_signals = teacher_signals.to(device)
        iteration(model, inputs, teacher_signals, reg_lambda, optimizer, snapshot_model)


def evaluate_epoch(model, dataloader, device, reg_lambda) -> dict:
    """
    概要: 1つのデータローダーの全データを（学習せず）評価する関数．
    引数: model, dataloader, device, reg_lambda．
    戻り値: metrics_to_value (dict)．1データあたりの評価・誤差の平均値の辞書．
    """
    total_metrics = {}
    total_count = 0
    with torch.no_grad():
        for inputs, teacher_signals in dataloader:
            inputs = inputs.to(device)
            teacher_signals = teacher_signals.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, teacher_signals, model, reg_lambda)
            batch_metrics = metrics_func(outputs, teacher_signals)
            batch_metrics["loss"] = loss.item()

            batch_size = inputs.shape[0]
            for key, value in batch_metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value * batch_size
            total_count += batch_size

    return {key: value / total_count for key, value in total_metrics.items()}


def compute_full_gradient_and_metrics(model, dataloader, device, reg_lambda):
    """
    概要: データローダー全体に対するフル勾配，および同じ1回の走査で得られる誤差・分類精度の
        平均値をまとめて計算する．
    引数: model, dataloader, device, reg_lambda．
    戻り値:
        grads (list of torch.Tensor)．
        metrics (dict)．{"loss": ..., "accuracy": ...} の1データあたりの平均値．
    """
    accumulated_grads = [torch.zeros_like(p) for p in model.parameters()]
    total_metrics = {}
    total_count = 0

    for inputs, teacher_signals in dataloader:
        inputs = inputs.to(device)
        teacher_signals = teacher_signals.to(device)
        batch_size = inputs.shape[0]

        model.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, teacher_signals, model, reg_lambda)
        loss.backward()

        for acc, p in zip(accumulated_grads, model.parameters()):
            acc.add_(p.grad.detach(), alpha=batch_size)

        batch_metrics = metrics_func(outputs, teacher_signals)
        batch_metrics["loss"] = loss.item()
        for key, value in batch_metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value * batch_size
        total_count += batch_size

    grads = [acc / total_count for acc in accumulated_grads]
    metrics = {key: value / total_count for key, value in total_metrics.items()}
    return grads, metrics


def compute_approx_error(snapshot_gradient, true_full_gradient) -> float:
    """
    概要: フル勾配の近似誤差 $ \\|e_s\\|^2 = \\|g_s - \\nabla f(z_s)\\|^2 $ を計算する．
    引数: snapshot_gradient, true_full_gradient (list of torch.Tensor)．
    戻り値: approx_error (float)．
    """
    return sum(
        torch.sum((g_s - g_true) ** 2).item()
        for g_s, g_true in zip(snapshot_gradient, true_full_gradient)
    )


def _save_if_best(model, test_accuracy, best_accuracy, target_dir):
    """
    概要: 検証用データに対する分類精度が過去最高を更新した場合，モデルの重みを保存する．
    引数: model, test_accuracy, best_accuracy, target_dir．
    戻り値: best_accuracy (float)．
    """
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), os.path.join(target_dir, "best_model.pth"))
        return test_accuracy
    return best_accuracy


def run_sgd(target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, seed, logger):
    """
    概要: SGDによる学習を実行し，各エポック終了時の評価指標を `logger` に記録する．
    引数:
        target_dir (str)．結果保存先ディレクトリ．
        load_dataloader_func (func)．
        eta (float)．学習率．
        batch_size (int)．ミニバッチサイズ．
        reg_lambda (float)．L2正則化係数．
        epochs (int)．
        device (torch.device)．
        seed (int)．乱数シード．
        logger (ResultLogger)．
    戻り値: なし
    """
    train_dataloader, test_dataloader = load_dataloader_func(seed=seed, batch_size=batch_size)
    N_train = len(train_dataloader.dataset)

    model = load_model(ResNet18LayerNorm, seed=seed).to(device)
    optimizer = SGD(model.parameters(), lr=eta)

    oracle_calls = 0
    best_accuracy = -1.0

    train_metrics = evaluate_epoch(model, train_dataloader, device, reg_lambda)
    test_metrics = evaluate_epoch(model, test_dataloader, device, reg_lambda)
    logger(0, oracle_calls, 0.0, train_metrics["loss"], test_metrics["accuracy"], float("nan"))
    best_accuracy = _save_if_best(model, test_metrics["accuracy"], best_accuracy, target_dir)

    elapsed_time = 0.0
    desc = f"SGD bs={batch_size} eta={eta} seed={seed}"
    for epoch_index in tqdm(range(1, epochs + 1), desc=desc, leave=False):
        start_time = time.time()
        train_epoch(
            model, train_dataloader, device, reg_lambda, optimizer, snapshot_model=None,
            desc=f"{desc} ep{epoch_index}",
        )
        oracle_calls += N_train
        elapsed_time += time.time() - start_time

        start_time = time.time()
        train_metrics = evaluate_epoch(model, train_dataloader, device, reg_lambda)
        test_metrics = evaluate_epoch(model, test_dataloader, device, reg_lambda)
        elapsed_time += time.time() - start_time

        logger(
            epoch_index, oracle_calls, elapsed_time,
            train_metrics["loss"], test_metrics["accuracy"], float("nan"),
        )
        best_accuracy = _save_if_best(model, test_metrics["accuracy"], best_accuracy, target_dir)

        if not np.isfinite(train_metrics["loss"]):
            break


def run_variance_reduced(
    method, target_dir, load_dataloader_func, eta, batch_size, reg_lambda, epochs, device, seed, logger
):
    """
    概要: SVRG系手法（SVRG，NFG SVRG，ASAI SVRG）による学習を実行し，各エポック終了時の
        評価指標を `logger` に記録する．SVRGのみ，スナップショット勾配 $ g_s $ が真のフル
        勾配であり，これを次エポックの学習に用いるため，その計算時間は `elapsed_time` に
        計上する．NFG SVRG・ASAI SVRGは評価指標（近似誤差）算出のためだけに真のフル勾配を
        計算するため，この計算時間は `elapsed_time` から除外する．
    引数:
        method (str)．"SVRG"，"NFG_SVRG"，"ASAI_SVRG" のいずれか．
        target_dir (str)．結果保存先ディレクトリ．
        load_dataloader_func (func)．
        eta (float)．学習率．
        batch_size (int)．ミニバッチサイズ．
        reg_lambda (float)．L2正則化係数．
        epochs (int)．外部ループ数 $ S $．
        device (torch.device)．
        seed (int)．乱数シード．
        logger (ResultLogger)．
    戻り値: なし
    """
    assert method in _VARIANCE_REDUCED_OPTIMIZER_CLASSES

    train_dataloader, test_dataloader = load_dataloader_func(seed=seed, batch_size=batch_size)
    N_train = len(train_dataloader.dataset)
    K = len(train_dataloader)

    model = load_model(ResNet18LayerNorm, seed=seed).to(device)
    snapshot_model = load_model(ResNet18LayerNorm, seed=seed).to(device)

    OptimizerClass = _VARIANCE_REDUCED_OPTIMIZER_CLASSES[method]
    optimizer = OptimizerClass(model.parameters(), lr=eta, K=K)
    rng = np.random.default_rng(seed)

    oracle_calls = 0

    if method == "SVRG":
        snapshot_grad, train_metrics_0 = compute_full_gradient_and_metrics(
            snapshot_model, train_dataloader, device, reg_lambda
        )
        true_full_grad_0 = snapshot_grad
    else:
        snapshot_grad = optimizer.get_snapshot_gradient()  # g_0 = 0
        true_full_grad_0, train_metrics_0 = compute_full_gradient_and_metrics(
            snapshot_model, train_dataloader, device, reg_lambda
        )
    approx_error_0 = compute_approx_error(snapshot_grad, true_full_grad_0)

    best_accuracy = -1.0
    test_metrics = evaluate_epoch(snapshot_model, test_dataloader, device, reg_lambda)
    logger(0, oracle_calls, 0.0, train_metrics_0["loss"], test_metrics["accuracy"], approx_error_0)
    best_accuracy = _save_if_best(snapshot_model, test_metrics["accuracy"], best_accuracy, target_dir)

    if method == "SVRG":
        optimizer.set_snapshot_gradient(snapshot_grad)
        oracle_calls += N_train

    elapsed_time = 0.0
    desc_prefix = f"{method} bs={batch_size} eta={eta} seed={seed}"
    for epoch_index in tqdm(range(1, epochs + 1), desc=desc_prefix, leave=False):
        start_time = time.time()
        optimizer.begin_epoch(rng)
        train_epoch(
            model, train_dataloader, device, reg_lambda, optimizer, snapshot_model,
            desc=f"{desc_prefix} ep{epoch_index}",
        )
        oracle_calls += 2 * N_train

        optimizer.end_epoch()
        set_model_params(snapshot_model, optimizer.get_snapshot_params(), source_model=model)
        elapsed_time += time.time() - start_time

        if method == "SVRG":
            start_time = time.time()
            snapshot_grad, train_metrics = compute_full_gradient_and_metrics(
                snapshot_model, train_dataloader, device, reg_lambda
            )
            oracle_calls += N_train
            optimizer.set_snapshot_gradient(snapshot_grad)
            approx_error = 0.0
            elapsed_time += time.time() - start_time
        else:
            # 評価指標算出のためだけの診断的なフル勾配計算であり，elapsed_timeには計上しない．
            snapshot_grad = optimizer.get_snapshot_gradient()
            true_full_grad, train_metrics = compute_full_gradient_and_metrics(
                snapshot_model, train_dataloader, device, reg_lambda
            )
            approx_error = compute_approx_error(snapshot_grad, true_full_grad)

        start_time = time.time()
        test_metrics = evaluate_epoch(snapshot_model, test_dataloader, device, reg_lambda)
        elapsed_time += time.time() - start_time

        logger(
            epoch_index, oracle_calls, elapsed_time,
            train_metrics["loss"], test_metrics["accuracy"], approx_error,
        )
        best_accuracy = _save_if_best(snapshot_model, test_metrics["accuracy"], best_accuracy, target_dir)

        if not np.isfinite(train_metrics["loss"]):
            break


def is_run_completed(target_dir: str, epochs: int) -> bool:
    """
    概要: 指定した条件の学習が既に正常終了しているか確認する．崩壊により打ち切った条件も
        完了済みとして扱う．
    引数: target_dir (str)．epochs (int)．
    戻り値: completed (bool)．
    """
    log_path = os.path.join(target_dir, "log.json")
    if not os.path.exists(log_path):
        return False
    try:
        logger = ResultLogger(log_path)
        recorded_epochs = logger["epoch"]
        if len(recorded_epochs) == epochs + 1:
            return True
        train_loss = logger["train_loss"]
        return len(train_loss) > 0 and not np.isfinite(train_loss[-1])
    except (json.JSONDecodeError, OSError):
        return False


def run_single_experiment(args):
    """
    概要: 1つの (手法, バッチサイズ, 学習率, エポック数, Seed) の組に対する学習を実行し，
        結果を保存する．`.orders/order_036.md` によりバッチサイズ・学習率・エポック数が
        条件ごとに異なるグリッドへ拡張されたため，これらを明示的にタプルへ含める。
    引数: args (tuple)．(method, batch_size, eta, epochs, seed) のタプル．
    戻り値: なし
    """
    method, batch_size, eta, epochs, seed = args
    torch.set_num_threads(1)

    name = hp_name(eta, batch_size, epochs)
    target_dir = os.path.join(OUTPUT_ROOT, method, name, str(seed))
    if is_run_completed(target_dir, epochs):
        print(f"[skip] {method}/{name}/{seed} は既に完了しています．", flush=True)
        return

    os.makedirs(target_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    logger = ResultLogger()
    logger.set_names("epoch", "oracle_calls", "elapsed_time", "train_loss", "test_accuracy", "approx_error")

    if method == "SGD":
        run_sgd(target_dir, load_dataloader, eta, batch_size, REG_LAMBDA, epochs, device, seed, logger)
    else:
        run_variance_reduced(
            method, target_dir, load_dataloader, eta, batch_size, REG_LAMBDA, epochs, device, seed, logger
        )

    logger.save(os.path.join(target_dir, "log.json"))

    train_dataloader, test_dataloader = load_dataloader(seed=seed, batch_size=batch_size)
    config = {
        "experiment": EXPERIMENT_NAME,
        "method": method,
        "seed": seed,
        "learning_rate": eta,
        "batch_size": batch_size,
        "reg_lambda": REG_LAMBDA,
        "epochs": epochs,
        "K": len(train_dataloader),
        "total_iterations": epochs * len(train_dataloader),
        "N_train": len(train_dataloader.dataset),
        "N_test": len(test_dataloader.dataset),
        "num_classes": NUM_CLASSES,
        "collapsed": len(logger["train_loss"]) < epochs + 1,
    }
    with open(os.path.join(target_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[done] {method}/{name}/{seed}", flush=True)


def _build_tasks(batch_size: int) -> list:
    """
    概要: `CONDITIONS`の中から指定したバッチサイズの条件のみを対象に，×`METHODS`×`SEEDS`の
        タスクリストを構築する．`.orders/order_036.md` 6節の指示に基づき，バッチサイズ512
        （条件B）はバッチサイズ128（基準条件・条件A）よりVRAM使用量が大幅に大きい
        （実測約22.6GB vs 約5.9GB，reservedベース）ため，バッチサイズごとに独立した並列数の
        プロセスプールで実行できるよう，タスク生成をバッチサイズ単位に分離する。
    引数: batch_size (int)．対象とするバッチサイズ。
    戻り値: tasks (list of tuple)．(method, batch_size, eta, epochs, seed) のリスト。
    """
    return [
        (method, cond["batch_size"], cond["learning_rate"], cond["epochs"], seed)
        for method, cond, seed in itertools.product(METHODS, CONDITIONS, SEEDS)
        if cond["batch_size"] == batch_size
    ]


def run_bs128_phase() -> None:
    """
    概要: バッチサイズ128のタスク（基準条件・条件A，計40条件，8プロセス並列で合計VRAM
        約47.2GB）を実行する．基準条件20条件は`is_run_completed`により自動的にスキップ
        される．
    引数: なし
    戻り値: なし
    """
    ctx = multiprocessing.get_context("spawn")
    tasks_bs128 = _build_tasks(128)
    num_workers_bs128 = min(8, len(tasks_bs128))
    print(f"バッチサイズ128タスク数: {len(tasks_bs128)}，並列プロセス数: {num_workers_bs128}")
    with ctx.Pool(processes=num_workers_bs128) as pool:
        pool.map(run_single_experiment, tasks_bs128, chunksize=1)


def run_bs512_phase() -> None:
    """
    概要: バッチサイズ512のタスク（条件B，計20条件，4プロセス並列で合計VRAM約90.2GB）を
        実行する．
    引数: なし
    戻り値: なし
    """
    ctx = multiprocessing.get_context("spawn")
    tasks_bs512 = _build_tasks(512)
    num_workers_bs512 = min(4, len(tasks_bs512))
    print(f"バッチサイズ512タスク数: {len(tasks_bs512)}，並列プロセス数: {num_workers_bs512}")
    with ctx.Pool(processes=num_workers_bs512) as pool:
        pool.map(run_single_experiment, tasks_bs512, chunksize=1)


def main(run_bs128: bool = True, run_bs512: bool = True) -> None:
    """
    概要: 実験5b・5cの条件（`CONDITIONS`：基準条件・条件A・条件B の3条件×4手法×5Seed=
        60条件）を学習する．`run_bs128`・`run_bs512`引数により，バッチサイズ128の
        フェーズ（基準条件・条件A）とバッチサイズ512のフェーズ（条件B）を独立に実行できる。

        `.orders/order_036.md` 6節の指示に基づき，最大バッチサイズ（512，条件B）でVRAM
        使用量を実測した結果，バッチサイズ128（基準条件・条件A）の約5.9GBに対し，バッチ
        サイズ512は約22.6GB（reservedベース）に達することが判明した（`.reports/
        report_036.md`参照）。8プロセス並列のままバッチサイズ512のタスクを実行すると
        合計VRAM使用量が約180.8GBとなり，総VRAM（約102.6GB）を大幅に超過してGPUが応答
        不能になる危険がある（ex004・`report_030.md` 5.3節の教訓）。

        さらに，条件Bのみで推定所要時間が約80〜85時間に達し，条件A・基準条件（バッチサイズ
        128フェーズ，約20〜25時間）と合わせると全体で4日以上を要する見込みとなったため，
        ユーザーの明示的な指示（チャットでの指示，`.orders/order_036.md`追記1節参照）に
        基づき，**まず条件B（バッチサイズを上げて総イテレーション数を揃えた条件）のみを
        単独で実行し，バッチサイズ128フェーズ（条件A）は本実行の完了後，別途の実行で追加
        する**方針とした．デフォルト引数（`run_bs128=True, run_bs512=True`）は両フェーズを
        実行するが，`if __name__ == "__main__"`のブロックでは条件Bのみを実行するよう
        明示的に指定している．条件Aを追加実行する際は，同ブロックを`main(run_bs128=True,
        run_bs512=False)`（または引数なしの`main()`）に変更して再実行すればよく，条件B
        （学習済み）は`is_run_completed`により自動的にスキップされ，条件Aのみが新規に
        学習される。
    引数:
        run_bs128 (bool) = True．バッチサイズ128フェーズ（基準条件・条件A）を実行するか。
        run_bs512 (bool) = True．バッチサイズ512フェーズ（条件B）を実行するか。
    戻り値: なし
    """
    for cond in CONDITIONS:
        print(
            f"バッチサイズ = {cond['batch_size']}, 学習率 = {cond['learning_rate']}, "
            f"epochs = {cond['epochs']}"
        )

    if run_bs128:
        run_bs128_phase()
    if run_bs512:
        run_bs512_phase()

    print("全ての学習が終了しました．")


if __name__ == "__main__":
    # ユーザーの明示的な指示により，まず条件B（バッチサイズ512）のみを実行する．
    # 条件A（バッチサイズ128・学習率0.0001）は，本実行の完了後，`main(run_bs128=True,
    # run_bs512=False)` へ変更して別途実行する（`.orders/order_036.md`追記1節参照）．
    main(run_bs128=False, run_bs512=True)
