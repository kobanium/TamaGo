
"""Residual Blockの実装。
"""
import torch
from torch import nn



class ResidualBlock(nn.Module):
    """Residual Blockの実装クラス。
    """
    def __init__(self, channels: int, momentum: float=0.01):
        """各レイヤの初期化処理。

        Args:
            channels (int): 畳み込み層のチャネル数。
            momentum (float, optional): バッチ正則化層のモーメンタムパラメータ. Defaults to 0.01.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, \
            kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, \
            kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=channels, eps=2e-5, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(num_features=channels, eps=2e-5, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, input_plane: torch.Tensor) -> torch.Tensor:
        """前向き伝搬処理を実行する。

        Args:
            input_plane (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: ブロックの出力テンソル。
        """
        hidden_1 = self.relu(self.bn1(self.conv1(input_plane)))
        hidden_2 = self.relu(self.bn2(self.conv2(hidden_1)))

        return input_plane + hidden_2
