
"""Dual Networkの実装。
"""
from typing import Tuple
from torch import nn
import torch

from board.constant import BOARD_SIZE
from nn.network.res_block import ResidualBlock
from nn.network.head.policy_head import PolicyHead
from nn.network.head.value_head import ValueHead


class DualNet(nn.Module): # pylint: disable=R0902
    """Dual Networkの実装クラス。
    """
    def __init__(self, device: torch.device, board_size: int=BOARD_SIZE):
        """Dual Networkの初期化処理

        Args:
            device (torch.device): 推論実行デバイス。探索での推論実行時にのみ使用し、学習中には使用しない。
            board_size (int, optional): 碁盤のサイズ。 デフォルト値はBOARD_SIZE。
        """
        super().__init__()
        filters = 64
        blocks = 6

        self.device = device

        self.conv_layer = nn.Conv2d(in_channels=6, out_channels=filters, \
            kernel_size=3, padding=1, bias=False)
        self.bn_layer = nn.BatchNorm2d(num_features=filters)
        self.relu = nn.ReLU()
        self.blocks = make_common_blocks(blocks, filters)
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
        blocks_out = self.blocks(self.relu(self.bn_layer(self.conv_layer(input_plane))))

        return self.policy_head(blocks_out), self.value_head(blocks_out)


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


    def inference_with_policy_logits(self, input_plane: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
        """前向き伝搬処理を実行する。Gumbel AlphaZero用の探索に使うメソッドのため、
        デバイス間データ転送も内部処理する。

        Args:
            input_plane (torch.Tensor): 入力特徴テンソル。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy, Valueの推論結果。
        """
        policy, value = self.forward(input_plane.to(self.device))
        return policy.cpu(), self.softmax(value).cpu()


def make_common_blocks(num_blocks: int, num_filters: int) -> torch.nn.Sequential:
    """DualNetの共通の残差ブロックを構成して返す。

    Args:
        num_blocks (int): 積み上げる残差ブロック数。
        num_filters (int): 残差ブロック内の畳込み層のフィルタ数。

    Returns:
        torch.nn.Sequential: 残差ブロック列。
    """
    blocks = [ResidualBlock(num_filters) for _ in range(num_blocks)]
    return nn.Sequential(*blocks)
