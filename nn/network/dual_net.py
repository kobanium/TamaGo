
"""Dual Networkの実装。
"""
from typing import NoReturn, Tuple
from torch import nn
import torch


from board.constant import BOARD_SIZE
from nn.network.res_block import ResidualBlock
from nn.network.head.policy_head import PolicyHead
from nn.network.head.value_head import ValueHead


class DualNet(nn.Module): # pylint: disable=R0902
    """Dual Networkの実装クラス。
    """

    def __init__(self, device: torch.device, board_size: int=BOARD_SIZE) -> NoReturn:
        """Dual Networkの初期化処理

        Args:
            device (torch.device): 推論実行デバイス。探索での推論実行時にのみ使用し、学習中には使用しない。
            board_size (int, optional): 碁盤のサイズ。 デフォルト値はBOARD_SIZE。
        """
        super().__init__()
        filters = 32

        self.device = device

        self.conv_layer = nn.Conv2d(in_channels=6, out_channels=filters, \
            kernel_size=3, padding=1, bias=False)
        self.bn_layer = nn.BatchNorm2d(num_features=filters)
        self.relu = nn.ReLU()

        self.res_block_1 = ResidualBlock(filters)
        self.res_block_2 = ResidualBlock(filters)
        self.res_block_3 = ResidualBlock(filters)
        self.res_block_4 = ResidualBlock(filters)
        self.res_block_5 = ResidualBlock(filters)

        self.policy_head = PolicyHead(board_size, filters)
        self.value_head = ValueHead(board_size, filters)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向き伝搬処理を実行する。

        Args:
            input_plane (torch.Tensor): 入力特徴テンソル。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: PolicyとValueのlogit。
        """
        hidden1 = self.relu(self.bn_layer(self.conv_layer(input_plane)))
        hidden2 = self.res_block_1(hidden1)
        hidden3 = self.res_block_2(hidden2)
        hidden4 = self.res_block_3(hidden3)
        hidden5 = self.res_block_4(hidden4)
        hidden6 = self.res_block_5(hidden5)

        return self.policy_head(hidden6), self.value_head(hidden6)

    def forward_for_sl(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向き伝搬処理を実行する。教師有り学習で利用する。

        Args:
            input_plane (torch.Tensor): 入力特徴テンソル。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Softmaxを通したPolicyと, Valueのlogit
        """
        policy, value = self.forward(input_plane)
        return self.softmax(policy), value

    def forward_with_softmax(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向き伝搬処理を実行する。

        Args:
            input_plane (torch.Tensor): 入力特徴テンソル。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy, Valueの推論結果。
        """
        policy, value = self.forward(input_plane)
        return self.softmax(policy), self.softmax(value)


    def inference(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向き伝搬処理を実行する。探索用に使うメソッドのため、デバイス間データ転送も内部処理する。

        Args:
            input_plane (torch.Tensor): 入力特徴テンソル。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy, Valueの推論結果。
        """
        policy, value = self.forward(input_plane.to(self.device))
        return self.softmax(policy).cpu(), self.softmax(value).cpu()
