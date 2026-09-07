"""
実験ex0021（CIFAR-10における正規化層・バッチサイズの構造的変更による安定性の検証）の
モデルの定義に関するモジュール．

`.orders/order_023.md` 3節，および`.orders/order_024.md`（スペクトル正規化の追加）に対応する．
`programs/ex002_cifar10_alexnet/model.py` の `AlexNetCIFAR` をベースに，各畳み込み層に適用する
正規化を4パターン（なし，LayerNorm，GroupNorm，SpectralNorm）から選択できるよう拡張した
`AlexNetCIFARNorm` を定義する．

## 正規化層の実装位置・パラメータ

LayerNorm・GroupNormは，各畳み込み層について `Conv2d → 正規化層 → ReLU` の順で挿入する
（`.orders/order_023.md` の指示通り，全結合層には正規化層を追加しない）．

- **LayerNorm**：ConvNeXt（Liu et al., 2022）のブロック設計に倣い，各空間位置（$ H\\times W $
  の各画素）において，チャネル方向（$ C $ 次元）に対して正規化する．`torch.nn.LayerNorm` は
  入力の末尾の次元を正規化するため，`(B, C, H, W)` を `(B, H, W, C)` に並べ替えてから
  `LayerNorm(C)` を適用し，元の形状に戻す `LayerNorm2d` を実装する．
- **GroupNorm**：`torch.nn.GroupNorm` を用いる．グループ数は各畳み込み層のチャネル数の
  4分の1（すなわち1グループあたり4チャンネル）とする．本モデルの畳み込み層のチャネル数
  （64, 192, 384, 256, 256）はいずれも4の倍数であり，均等なグループ分割が可能である．
  ImageNet規模のモデルで一般的なグループ数32（1グループあたりのチャンネル数が数十〜百程度）
  という基準は，本モデルのように各層のチャンネル数が64〜384と少ない場合には粗すぎるため，
  1グループあたりのチャンネル数を小さく固定する方針を採用した．
- **SpectralNorm**：`.orders/order_024.md` の指示に基づく．LayerNorm・GroupNormが活性化
  （層の出力）を正規化するのに対し，SpectralNormは畳み込み層自体の重み行列を最大特異値で
  除することで，層のリプシッツ定数を直接抑える．そのため，独立した正規化層を追加するのでは
  なく，`Conv2d` 自体を `SpectralConv2d`（重み行列を最大特異値で除したものを畳み込みに用いる
  モジュール）に置き換える．他の3パターンとの層数・チャンネル数・活性化関数の公平性を保つ
  ため，`SpectralConv2d` を用いる場合は正規化層の位置に `nn.Identity` を置く（`ReLU` の前に
  何も挿入しない）．

## SpectralNormにおけるpower iterationバッファの決定論性（`.orders/order_024.md` 4節）

最大特異値はpower iterationで近似計算するため，反復に用いる補助ベクトル $ u, v $ を
`register_buffer` で保持する．SVRG系手法は，スナップショット $ z_s $ に対する勾配
$ \\nabla f_n(z_s) $ を同一の $ n $ について何度評価しても同じ値になることを要求するが，
$ u, v $ がforwardのたびに更新される標準的な実装ではこれが破られる．そのため
`SpectralConv2d` は `frozen` フラグを持ち，`frozen=True` の間は $ u, v $ を更新せず，
保存された値のみを用いて最大特異値を計算する．`frozen=False`（既定，現在のパラメータ
$ w_s^k $ を保持する `model` 用）の間は，forwardのたびに1回のpower iterationで $ u, v $ を
更新する．スナップショット専用モデル（`snapshot_model`）は，`refresh_and_freeze_spectral_norm`
（モジュール関数）により，新しい重みを受け取るたびに複数回のpower iterationで $ u, v $ を
更新した上で凍結する．

### バーンイン（power iterationの反復回数）の決定根拠

`.orders/order_024.md` 4節の指示に従い，スナップショット固定時は既定の1回更新ではなく，
複数回反復収束させた上で固定する．本実装のconv層の重み行列の形状
（(64,75), (192,1600), (384,1728), (256,3456), (256,2304)）に対し，ランダムな初期値から
power iterationを30回行い，連続する反復間の最大特異値推定値の相対変化
$ |\\sigma^{(t)}-\\sigma^{(t-1)}|/|\\sigma^{(t-1)}| $ を確認したところ，10回目で
$ 1.1\\times10^{-3}\\sim4.5\\times10^{-3} $，20回目で $ 5.3\\times10^{-4}\\sim1.1\\times
10^{-3} $ まで減衰していた．この結果から，`_SNAPSHOT_POWER_ITERATIONS=20` を採用すれば，
以後さらに反復を重ねても推定値がほぼ変化しない（相対変化が$ 10^{-3} $程度以下の）水準に
収束していると判断できる．この収束性は
`tests/test_ex0021_cifar10_alexnet_norm.py::test_spectral_norm_power_iteration_has_converged_after_burn_in`
で，実際のモデルの重みに対し，バーンイン後にさらに反復を重ねても最大特異値の推定値が
ほとんど変化しないこと（相対変化が閾値以下であること）を検証している．
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed  # noqa: E402

NORM_TYPES = ("none", "layernorm", "groupnorm", "spectralnorm")
# 各畳み込み層の出力チャネル数．1グループあたり4チャンネルとなるようGroupNormのグループ数を
# 決定する際に用いる．
_CONV_CHANNELS = (64, 192, 384, 256, 256)
_GROUP_SIZE = 4
# スナップショット更新時（refresh_and_freeze_spectral_norm）に最大特異値の推定に用いる
# power iterationの反復回数（バーンイン）．`model`側の逐次更新（forwardごとに1回）と異なり，
# スナップショットは次の更新まで固定されるため，一度に多めの反復で精度の高い推定を行う．
# 20回で連続する反復間の推定値の相対変化が10^-3程度まで収束することを確認した
# （モジュールdocstring，および `test_spectral_norm_power_iteration_has_converged_after_burn_in`
# 参照）．
_SNAPSHOT_POWER_ITERATIONS = 20


class LayerNorm2d(nn.Module):
    """
    ConvNeXt（Liu et al., 2022）のブロック設計に倣い，`(B, C, H, W)` 形式の特徴マップに対し，
    各空間位置においてチャネル方向（$ C $ 次元）に正規化するLayerNorm．バッチ内の他サンプルに
    依存しない（サンプル単位・空間位置単位で完結する）．
    """

    def __init__(self, num_channels: int):
        """
        概要: チャネル方向のLayerNormを初期化する．
        引数: num_channels (int)．正規化対象のチャネル数 $ C $．
        戻り値: なし
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: `(B, C, H, W)` を `(B, H, W, C)` に並べ替えてLayerNormを適用し，元の形状に戻す．
        引数: x (torch.Tensor)，形状 (B, C, H, W)．
        戻り値: y (torch.Tensor)，形状 (B, C, H, W)．
        """
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        return x.permute(0, 3, 1, 2)


class SpectralConv2d(nn.Module):
    """
    重み行列を最大特異値で除すことで層のリプシッツ定数を制約する畳み込み層．
    `torch.nn.utils.spectral_norm` とは異なり，`frozen` フラグにより power iteration
    バッファ（$ u, v $）の更新を停止できる（モジュールdocstring参照）．SVRG系手法の
    スナップショットモデルで用いることを想定する．
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int, eps: float = 1e-12
    ):
        """
        概要: SpectralConv2dを初期化する．power iterationの補助ベクトル $ u \\in
            \\mathbb{R}^{\\text{out\\_channels}} $，$ v \\in \\mathbb{R}^{\\text{in\\_channels}
            \\times k \\times k} $ を乱数で初期化し，バッファとして保持する．
        引数:
            in_channels (int)．入力チャネル数．
            out_channels (int)．出力チャネル数．
            kernel_size (int)．カーネルサイズ（正方形）．
            padding (int)．パディング幅．
            eps (float) = 1e-12．ベクトルの正規化における零除算防止用の微小値．
        戻り値: なし
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.eps = eps
        self.frozen = False

        weight_matrix = self.conv.weight.reshape(out_channels, -1)
        u = F.normalize(torch.randn(weight_matrix.shape[0]), dim=0, eps=eps)
        v = F.normalize(torch.randn(weight_matrix.shape[1]), dim=0, eps=eps)
        self.register_buffer("u", u)
        self.register_buffer("v", v)

    def _spectral_norm_weight(self, update_buffers: bool, power_iterations: int) -> torch.Tensor:
        """
        概要: 最大特異値で除した重み行列を計算する．`update_buffers=True` の場合，
            power iterationを `power_iterations` 回行いバッファ $ u, v $ を更新してから
            計算する．`update_buffers=False` の場合，保存済みの $ u, v $ をそのまま用いる
            （凍結時の決定論的な評価）．
        引数:
            update_buffers (bool)．バッファを更新するかどうか．
            power_iterations (int)．バッファ更新時のpower iterationの反復回数．
        戻り値: weight (torch.Tensor)．最大特異値で正規化した重み（元の形状）．
        """
        weight_matrix = self.conv.weight.reshape(self.conv.weight.shape[0], -1)
        if update_buffers:
            u, v = self.u, self.v
            with torch.no_grad():
                for _ in range(power_iterations):
                    v = F.normalize(torch.mv(weight_matrix.t(), u), dim=0, eps=self.eps)
                    u = F.normalize(torch.mv(weight_matrix, v), dim=0, eps=self.eps)
                self.u.copy_(u)
                self.v.copy_(v)
        sigma = torch.dot(self.u, torch.mv(weight_matrix, self.v))
        return self.conv.weight / sigma

    def current_sigma(self) -> torch.Tensor:
        """
        概要: 現在保存されているバッファ $ u, v $（更新は行わない）を用いて，最大特異値の
            推定値を計算する．power iterationの収束確認（バーンイン回数の妥当性検証）に用いる．
        引数: なし
        戻り値: sigma (torch.Tensor)，形状 ()．
        """
        weight_matrix = self.conv.weight.reshape(self.conv.weight.shape[0], -1)
        return torch.dot(self.u, torch.mv(weight_matrix, self.v))

    def refresh_and_freeze(self, power_iterations: int = _SNAPSHOT_POWER_ITERATIONS):
        """
        概要: 現在の重みに対しpower iterationを複数回行いバッファ $ u, v $ を更新した上で，
            以後の `forward` 呼び出しでバッファを更新しない状態（`frozen=True`）にする．
            スナップショット $ z_s $ を新しい重みで固定する際に呼び出す．
        引数: power_iterations (int) = `_SNAPSHOT_POWER_ITERATIONS`．power iterationの反復回数．
        戻り値: なし
        """
        with torch.no_grad():
            self._spectral_norm_weight(update_buffers=True, power_iterations=power_iterations)
        self.frozen = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: 最大特異値で正規化した重みを用いて畳み込みを計算する．`frozen=False`（既定）の
            間はforwardのたびに1回のpower iterationでバッファを更新し，`frozen=True`の間は
            保存済みのバッファをそのまま用いる（決定論的）．
        引数: x (torch.Tensor)，形状 (B, C_in, H, W)．
        戻り値: y (torch.Tensor)，形状 (B, C_out, H', W')．
        """
        weight = self._spectral_norm_weight(update_buffers=not self.frozen, power_iterations=1)
        return F.conv2d(x, weight, self.conv.bias, padding=self.conv.padding)


def refresh_and_freeze_spectral_norm(model: nn.Module, power_iterations: int = _SNAPSHOT_POWER_ITERATIONS):
    """
    概要: モデル内の全ての `SpectralConv2d` 層について，現在の重みに基づきpower iteration
        バッファを更新した上で凍結する．スナップショット専用モデルの重みを更新した直後に
        呼び出す．`SpectralConv2d` を含まないモデル（`norm_type` が "spectralnorm" 以外）に
        対しては何も行わない．
    引数:
        model (torch.nn.Module)．対象のモデル．
        power_iterations (int) = `_SNAPSHOT_POWER_ITERATIONS`．power iterationの反復回数．
    戻り値: なし
    """
    for module in model.modules():
        if isinstance(module, SpectralConv2d):
            module.refresh_and_freeze(power_iterations)


def _build_conv_and_norm(
    norm_type: str, in_channels: int, out_channels: int, kernel_size: int, padding: int
) -> tuple:
    """
    概要: 指定した正規化の種類に応じて，畳み込み層と（存在する場合は）正規化層の組を構築する．
        SpectralNormは畳み込み層自体を置き換える方式であるため，正規化層の位置には
        `nn.Identity` を返す（他パターンとの層数を揃えるため）．
    引数:
        norm_type (str)．"none"，"layernorm"，"groupnorm"，"spectralnorm" のいずれか．
        in_channels (int)．入力チャネル数．
        out_channels (int)．出力チャネル数．
        kernel_size (int)．カーネルサイズ．
        padding (int)．パディング幅．
    戻り値: (conv, norm)．畳み込み層と正規化層（`torch.nn.Module` のタプル）．
    """
    if norm_type == "spectralnorm":
        conv = SpectralConv2d(in_channels, out_channels, kernel_size, padding=padding)
        return conv, nn.Identity()

    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
    norm = _build_norm_layer(norm_type, out_channels)
    return conv, norm


def _build_norm_layer(norm_type: str, num_channels: int) -> nn.Module:
    """
    概要: 指定した種類の正規化層を構築する．
    引数:
        norm_type (str)．"none"，"layernorm"，"groupnorm" のいずれか（"spectralnorm" は
            `_build_conv_and_norm` で別途処理するため対象外）．
        num_channels (int)．正規化対象のチャネル数．
    戻り値: layer (torch.nn.Module)．正規化層（`norm_type="none"` の場合は `nn.Identity`）．
    """
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "layernorm":
        return LayerNorm2d(num_channels)
    if norm_type == "groupnorm":
        num_groups = num_channels // _GROUP_SIZE
        return nn.GroupNorm(num_groups, num_channels)
    raise ValueError(f"未知の正規化層の種類です: {norm_type}．{NORM_TYPES} のいずれかを指定してください．")


class AlexNetCIFARNorm(nn.Module):
    """
    AlexNetをCIFAR-10向けに縮小したCNN（`programs/ex002_cifar10_alexnet/model.py` の
    `AlexNetCIFAR` と同一の層構成）に，各畳み込み層の後に正規化層を追加できるよう拡張した
    モデル．Dropoutは使用しない．
    """

    def __init__(self, num_classes: int = 10, norm_type: str = "none"):
        """
        概要: 5つの畳み込み層（各層の後に正規化層とReLUを適用）と3つの全結合層を初期化する．
        引数:
            num_classes (int) = 10．分類先のクラス数 $ C $．
            norm_type (str) = "none"．正規化層の種類．`NORM_TYPES` のいずれか．
        戻り値: なし
        """
        super().__init__()
        if norm_type not in NORM_TYPES:
            raise ValueError(f"norm_typeは{NORM_TYPES}のいずれかである必要があります: {norm_type}")
        self.norm_type = norm_type

        c1, c2, c3, c4, c5 = _CONV_CHANNELS
        conv1, norm1 = _build_conv_and_norm(norm_type, 3, c1, 5, 2)
        conv2, norm2 = _build_conv_and_norm(norm_type, c1, c2, 5, 2)
        conv3, norm3 = _build_conv_and_norm(norm_type, c2, c3, 3, 1)
        conv4, norm4 = _build_conv_and_norm(norm_type, c3, c4, 3, 1)
        conv5, norm5 = _build_conv_and_norm(norm_type, c4, c5, 3, 1)
        self.conv_layers = nn.Sequential(
            conv1,
            norm1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (c1, 32, 32) -> (c1, 16, 16)
            conv2,
            norm2,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (c2, 16, 16) -> (c2, 8, 8)
            conv3,
            norm3,
            nn.ReLU(inplace=True),
            conv4,
            norm4,
            nn.ReLU(inplace=True),
            conv5,
            norm5,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (c5, 8, 8) -> (c5, 4, 4)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(c5 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: 入力画像からロジットを計算する．
        引数: x (torch.Tensor)，形状 (B, 3, 32, 32)．入力画像．
        戻り値: logits (torch.Tensor)，形状 (B, num_classes)．
        """
        features = self.conv_layers(x)
        features = features.flatten(start_dim=1)
        return self.fc_layers(features)


def load_model(
    ModelClass=AlexNetCIFARNorm, weight_path: str = None, seed: int = 0, norm_type: str = "none"
) -> nn.Module:
    """
    概要: モデルをインスタンス化するための関数．
    引数:
        ModelClass (torch.nn.Moduleのクラス) = AlexNetCIFARNorm．
        weight_path (str) = None．学習済み重みのパス．指定した場合はこれを読み込む．
        seed (int) = 0．パラメータ初期値を固定する乱数シード．
        norm_type (str) = "none"．正規化層の種類．
    戻り値: model (torch.nn.Module)．
    """
    set_seed(seed)
    model = ModelClass(norm_type=norm_type)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model


def compute_l2_regularization(model: nn.Module, reg_lambda: float) -> torch.Tensor:
    """
    概要: L2正則化項 $ \\frac{\\lambda}{2}\\|w\\|^2 $ を計算する．畳み込み層・全結合層の重み
        （`weight`）のみを対象とし，切片（`bias`）および正規化層のアフィンパラメータ
        （`weight`という名前を持つが，スケール・シフトのパラメータでありモデルの表現力の
        本体ではない）は正則化しない．`SpectralConv2d` は，正則化を正規化前の生の重み
        （`conv.weight`）に対して課す（他パターンと同様，学習可能パラメータそのものを
        対象とするため）．
    引数:
        model (torch.nn.Module)．正則化対象の重みを保持するモデル．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: reg (torch.Tensor)，形状 ()．正則化項．
    """
    reg = next(model.parameters()).new_zeros(())
    # SpectralConv2dは内部に生のnn.Conv2d（self.conv）を保持するため，model.modules()の
    # 走査ではこの内部Conv2dも独立したnn.Conv2dとして訪問される．二重計上を避けるため，
    # SpectralConv2dが保持するConv2dのidを事前に収集し，直接の走査対象から除外する．
    spectral_conv_ids = {id(m.conv) for m in model.modules() if isinstance(m, SpectralConv2d)}
    for module in model.modules():
        if isinstance(module, SpectralConv2d):
            reg = reg + torch.sum(module.conv.weight ** 2)
        elif isinstance(module, (nn.Conv2d, nn.Linear)) and id(module) not in spectral_conv_ids:
            reg = reg + torch.sum(module.weight ** 2)
    return 0.5 * reg_lambda * reg


def loss_func(
    outputs: torch.Tensor, teacher_signals: torch.Tensor, model: nn.Module, reg_lambda: float
) -> torch.Tensor:
    """
    概要: L2正則化付き多値交差エントロピー損失を計算する．
    引数:
        outputs (torch.Tensor)，形状 (B, C)．モデルの出力（ロジット）．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号（クラスインデックス）．
        model (torch.nn.Module)．正則化項の計算対象のモデル．
        reg_lambda (float)．L2正則化係数 $ \\lambda $．
    戻り値: loss (torch.Tensor)，形状 ()．1データあたりの誤差の平均値＋正則化項．
    """
    cce = nn.functional.cross_entropy(outputs, teacher_signals, reduction="mean")
    return cce + compute_l2_regularization(model, reg_lambda)


def metrics_func(outputs: torch.Tensor, teacher_signals: torch.Tensor) -> dict:
    """
    概要: モデルの出力と教師信号の一致度（分類精度）を評価する関数．
    引数:
        outputs (torch.Tensor)，形状 (B, C)．モデルの出力（ロジット）．
        teacher_signals (torch.Tensor)，形状 (B,)．教師信号（クラスインデックス）．
    戻り値: metrics_to_value (dict)．{"accuracy": 1データあたりの正解率の平均値}．
    """
    with torch.no_grad():
        pred = outputs.argmax(dim=1)
        accuracy = (pred == teacher_signals).to(torch.float32).mean().item()
    return {"accuracy": accuracy}


def set_model_params(model: nn.Module, param_values: list, source_model: nn.Module = None):
    """
    概要: モデルのパラメータを，指定したテンソル列の値で上書きする．
        SVRG系手法におけるスナップショット専用モデルのパラメータを $ z_{s+1} $ で更新する
        際に用いる．
    引数:
        model (torch.nn.Module)．上書き対象のモデル．
        param_values (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の値．
        source_model (torch.nn.Module) = None．他の実験とインターフェースを揃えるための
            引数．LayerNorm・GroupNormは移動平均統計量（バッファ）を持たないため，指定
            されても何も行わない．
    戻り値: なし
    """
    with torch.no_grad():
        for p, v in zip(model.parameters(), param_values):
            p.copy_(v)
