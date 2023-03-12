"""Value headの実装。
"""
import torch
from torch import nn


class ValueHead(nn.Module):
    """Value headの実装クラス。
    """
    def __init__(self, board_size: int, channels: int, momentum: float=0.01):
        """Value headの初期化処理。

        Args:
            board_size (int): 碁盤のサイズ。
            channels (int): 共通ブロック部の畳み込み層のチャネル数。
            momentum (float, optional): バッチ正則化層のモーメンタムパラメータ. Defaults to 0.01.
        """
        super().__init__()

        self.conv_layer = nn.Conv2d(in_channels=channels, out_channels=1, \
            kernel_size=1, padding=0, bias=False)
        self.bn_layer = nn.BatchNorm2d(num_features=1, eps=2e-5, momentum=momentum)
        self.fc_layer = nn.Linear(board_size ** 2, 3)
        self.relu = nn.ReLU()

    def forward(self, input_plane: torch.Tensor) -> torch.Tensor:
        """前向き伝搬処理を実行する。

        Args:
            input_plane (torch.Tensor): Value headへの入力テンソル。

        Returns:
            torch.Tensor: PolicyのLogit出力
        """
        hidden = self.relu(self.bn_layer(self.conv_layer(input_plane)))
        batch_size, _, height, width = hidden.shape
        reshape = hidden.reshape(batch_size, height * width)
        value_out = self.fc_layer(reshape)

        return value_out
