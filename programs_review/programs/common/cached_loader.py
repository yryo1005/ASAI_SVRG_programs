"""
前処理済みのテンソルをキャッシュから読み出すデータローダー．

既存の実験（CIFAR-10，ImageWoof）は `torchvision` の `Dataset` に決定論的な前処理
（`ToTensor`，`Normalize`，ImageWoofではさらに `Resize`，`CenterCrop`）を毎回適用していた．
前処理は決定論的であるため，同じ変換を1回だけ適用した結果をキャッシュしておけば，各ミニバッチの
入力テンソルは値として完全に一致する．本モジュールは，そのキャッシュからミニバッチを取り出す
ローダーを提供する．

ミニバッチの順序の再現性：既存の実装は `DataLoader(dataset, batch_size, shuffle=True,
generator=torch.Generator().manual_seed(seed))` を用いていた．`DataLoader` が乱数生成器を
消費する量は `dataset` の長さだけに依存し，`dataset` の中身には依存しない．そこで本モジュールでは
長さ $ N $ の添字の `Dataset` に対して同じ引数の `DataLoader` を構築し，得られた添字で
キャッシュを参照する．これにより，各エポックのデータの順序と，乱数生成器の消費の仕方が
既存の実装と一致する．
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class IndexDataset(Dataset):
    """
    概要: 添字 $ 0, \\dots, N-1 $ をそのまま返す `Dataset`．`DataLoader` の添字の生成（シャッフル）
        だけを利用するために用いる．
    """

    def __init__(self, length: int):
        """
        概要: 添字の `Dataset` を初期化する．
        引数: length (int)．データ数 $ N $．
        戻り値: なし
        """
        self.length = length

    def __len__(self) -> int:
        """
        概要: データ数を返す．
        引数: なし
        戻り値: length (int)．
        """
        return self.length

    def __getitem__(self, index: int) -> int:
        """
        概要: 添字をそのまま返す．
        引数: index (int)．
        戻り値: index (int)．
        """
        return index


class CachedTensorLoader:
    """
    概要: キャッシュされた入力・教師信号からミニバッチを取り出すローダー．`len(loader)` は
        ミニバッチ数，`len(loader.dataset)` はデータ数を返し，`torch.utils.data.DataLoader`
        と同じ使い方ができる．
    """

    def __init__(self, inputs, targets: torch.Tensor, batch_size: int, shuffle: bool,
                 generator: torch.Generator = None, device: torch.device = None):
        """
        概要: ローダーを初期化する．
        引数:
            inputs (torch.Tensor または numpy.ndarray)，形状 (N, ...)．前処理済みの入力．
                `numpy.memmap` を渡した場合は，ミニバッチごとに必要な行だけを読み出す．
            targets (torch.Tensor)，形状 (N,)．教師信号．
            batch_size (int)．ミニバッチサイズ．
            shuffle (bool)．エポックごとにデータの順序をシャッフルするか．
            generator (torch.Generator) = None．シャッフルに用いる乱数生成器．
            device (torch.device) = None．ミニバッチを転送するデバイス．`None` の場合は転送しない．
        戻り値: なし
        """
        self.inputs = inputs
        self.targets = targets
        self.device = device
        self.dataset = IndexDataset(len(targets))
        self.index_loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
        )

    def __len__(self) -> int:
        """
        概要: 1エポックあたりのミニバッチ数 $ K $ を返す．
        引数: なし
        戻り値: num_batches (int)．
        """
        return len(self.index_loader)

    def _gather_inputs(self, indices: torch.Tensor) -> torch.Tensor:
        """
        概要: 指定した添字の入力を取り出す．
        引数: indices (torch.Tensor)，形状 (B,)，dtype long．
        戻り値: inputs (torch.Tensor)，形状 (B, ...)．
        """
        if isinstance(self.inputs, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(self.inputs[indices.numpy()]))
        return self.inputs[indices.to(self.inputs.device)]

    def __iter__(self):
        """
        概要: 1エポック分のミニバッチを順に返す．
        引数: なし
        戻り値: (inputs, targets) の組を返すイテレータ．inputs の形状は (B, ...)，targets の形状は (B,)．
        """
        for indices in self.index_loader:
            inputs = self._gather_inputs(indices)
            targets = self.targets[indices.to(self.targets.device)]
            if self.device is not None:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            yield inputs, targets
