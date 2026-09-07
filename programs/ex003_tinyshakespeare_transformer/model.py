"""
実験3（Tiny Shakespeareを用いた文章生成問題）のモデルの定義に関するモジュール．

`.orders/order_027.md` 4節・2節の指示に基づき，小規模なDecoder-only Transformer
（`DecoderOnlyTransformer`）を定義する．SVRG系手法の理論的前提（同一の $ n $・同一の
パラメータ $ z_s $ に対する勾配評価が常に同じ値になること）を満たすため，Dropout・
BatchNormalizationは一切使用しない．正規化にはLayerNorm（各トークンごとに特徴次元方向へ
正規化するため，バッチ内の他サンプルに依存しない）をPre-LN構成で用いる．位置エンコーディング
は学習可能な埋め込み（`nn.Embedding`，モデルパラメータの一部）とする．Causalマスクは
決定論的な操作であるため問題ない．
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, ".ai", "ai-dev-kit", "src"))

from machine_learning_utils import set_seed  # noqa: E402

D_MODEL = 128
N_HEAD = 4
N_LAYER = 4
D_FF = 512


class CausalSelfAttention(nn.Module):
    """
    概要: Causalマスク付きのMulti-Head Self-Attention．Dropoutは使用しない．
    """

    def __init__(self, d_model: int, n_head: int, max_seq_len: int):
        """
        概要: Causal Self-Attention層を初期化する．
        引数:
            d_model (int)．特徴次元数．
            n_head (int)．Attention headの数．
            max_seq_len (int)．最大系列長（causalマスクのバッファサイズ）．
        戻り値: なし
        """
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: Causal Self-Attentionを適用する．
        引数: x (torch.Tensor)，形状 (B, T, d_model)．
        戻り値: y (torch.Tensor)，形状 (B, T, d_model)．
        """
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:T, :T]
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        y = attn_weights @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    """
    概要: Pre-LN構成のTransformerブロック（Self-Attention + Feed Forward）．Dropoutは
        使用しない．
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, max_seq_len: int):
        """
        概要: Transformerブロックを初期化する．
        引数:
            d_model (int)．特徴次元数．
            n_head (int)．Attention headの数．
            d_ff (int)．Feed Forward層の中間次元数．
            max_seq_len (int)．最大系列長．
        戻り値: なし
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        概要: Transformerブロックを適用する．
        引数: x (torch.Tensor)，形状 (B, T, d_model)．
        戻り値: y (torch.Tensor)，形状 (B, T, d_model)．
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    概要: 小規模なDecoder-only Transformer（文字レベル次文字予測）．
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 128,
        d_model: int = D_MODEL,
        n_head: int = N_HEAD,
        n_layer: int = N_LAYER,
        d_ff: int = D_FF,
    ):
        """
        概要: Decoder-only Transformerを初期化する．
        引数:
            vocab_size (int)．語彙サイズ（文字種数）．
            max_seq_len (int) = 128．最大系列長 $ T $．
            d_model (int) = 128．特徴次元数．
            n_head (int) = 4．Attention headの数．
            n_layer (int) = 4．Transformerブロックの層数．
            d_ff (int) = 512．Feed Forward層の中間次元数．
        戻り値: なし
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, d_ff, max_seq_len) for _ in range(n_layer)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        概要: 次文字の予測ロジットを計算する．
        引数: input_ids (torch.Tensor)，形状 (B, T)，dtype long．
        戻り値: logits (torch.Tensor)，形状 (B, T, vocab_size)．
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.head(x)


def load_model(ModelClass, weight_path: str = None, seed: int = 0, vocab_size: int = 65, **kwargs):
    """
    概要: モデルをインスタンス化するための関数．
    引数:
        ModelClass (torch.nn.Moduleのクラス)．
        weight_path (str) = None．
        seed (int) = 0．
        vocab_size (int) = 65．語彙サイズ．
        **kwargs．`ModelClass` に渡すその他の引数（`max_seq_len` 等）．
    戻り値: model (torch.nn.Module)．
    """
    set_seed(seed)
    model = ModelClass(vocab_size=vocab_size, **kwargs)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model


def compute_l2_regularization(model: nn.Module, reg_lambda: float) -> torch.Tensor:
    """
    概要: L2正則化項 $ \\frac\\lambda2\\|w\\|^2 $ を計算する．`nn.Linear`・`nn.Embedding`の
        重み（バイアスを除く）を対象とし，`nn.LayerNorm`のアフィンパラメータは対象外とする．
    引数:
        model (torch.nn.Module)．
        reg_lambda (float)．正則化係数 $ \\lambda $．
    戻り値: reg (torch.Tensor)．スカラー．
    """
    reg = next(model.parameters()).new_zeros(())
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            reg = reg + torch.sum(module.weight ** 2)
    return 0.5 * reg_lambda * reg


def loss_func(outputs: torch.Tensor, teacher_signals: torch.Tensor, model: nn.Module, reg_lambda: float) -> torch.Tensor:
    """
    概要: モデルの出力と教師信号のカテゴリカルクロスエントロピー誤差にL2正則化項を加えた
        誤差を計算する．
    引数:
        outputs (torch.Tensor)，形状 (B, T, vocab_size)．モデルの出力（次文字予測ロジット）．
        teacher_signals (torch.Tensor)，形状 (B, T)，dtype long．教師信号．
        model (torch.nn.Module)．
        reg_lambda (float)．L2正則化係数．
    戻り値: loss (torch.Tensor)．スカラー．
    """
    B, T, V = outputs.shape
    cce = F.cross_entropy(outputs.reshape(B * T, V), teacher_signals.reshape(B * T))
    return cce + compute_l2_regularization(model, reg_lambda)


def metrics_func(outputs: torch.Tensor, teacher_signals: torch.Tensor) -> dict:
    """
    概要: 次文字予測精度を計算する．
    引数:
        outputs (torch.Tensor)，形状 (B, T, vocab_size)．
        teacher_signals (torch.Tensor)，形状 (B, T)．
    戻り値: metrics_to_value (dict)．{"accuracy": float}．
    """
    predictions = torch.argmax(outputs, dim=-1)
    accuracy = (predictions == teacher_signals).float().mean().item()
    return {"accuracy": accuracy}


def set_model_params(model: nn.Module, param_values, source_model: nn.Module = None) -> None:
    """
    概要: モデルのパラメータを指定した値で置き換える．
    引数:
        model (torch.nn.Module)．更新対象のモデル．
        param_values (list of torch.Tensor)．`model.parameters()` と同じ順序・形状の値．
        source_model (torch.nn.Module) = None．本モデルでは未使用（他モデルとの
            インターフェース統一のために受け取る）．
    戻り値: なし
    """
    with torch.no_grad():
        for p, value in zip(model.parameters(), param_values):
            p.copy_(value)
