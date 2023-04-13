"""GoGui用のコマンド処理の実装。
"""
import math

import torch

from board.go_board import GoBoard
from board.stone import Stone
from nn.feature import generate_input_planes
from nn.network.dual_net import DualNet

class GoguiAnalyzeCommand: # pylint: disable=R0903
    """Gogui解析コマンドの基本情報クラス。
    """
    def __init__(self, command_type, label, command):
        """コンストラクタ。

        Args:
            command_type (_type_): _description_
            label (_type_): _description_
            command (_type_): _description_
        """
        self.type = command_type
        self.label = label
        self.command = command

    def get_command_information(self) -> str:
        """コマンド情報の取得。gogui-analyze_commandで表示する内容。

        Returns:
            str: コマンド情報文字列。
        """
        return self.type + "/" + self.label + "/" + self.command


def display_policy_distribution(model: DualNet, board: GoBoard, color: Stone) -> str:
    """Policyを色付けして表示するための文字列を生成する。（GoGui解析コマンド）

    Args:
        model (DualNet): Policyを出力するニューラルネットワーク。
        board (GoBoard): 評価する局面情報。
        color (Stone): 評価する手番の色。

    Returns:
        str: 表示用文字列。
    """
    board_size = board.get_board_size()
    input_plane = generate_input_planes(board, color)
    input_plane = torch.tensor(input_plane.reshape(1, 6, board_size, board_size)) #pylint: disable=E1121
    policy, _ = model.forward_with_softmax(input_plane)

    max_policy, min_policy = 0, 1
    log_policies = [math.log(policy[0][i]) for i in range(board_size * board_size)]

    for i, log_policy in enumerate(log_policies):
        pos = board.onboard_pos[i]
        if board.board[pos] is Stone.EMPTY and board.is_legal(pos, color):
            max_policy = max(max_policy, log_policy)
            min_policy = min(min_policy, log_policy)

    scale = max_policy - min_policy
    response = ""

    for i, log_policy in enumerate(log_policies):
        pos = board.onboard_pos[i]
        if board.board[pos] is Stone.EMPTY and board.is_legal(pos, color):
            color_value = int((log_policy - min_policy) / scale * 255)
            response += f"\"#{color_value:02x}{0:02x}{255-color_value:02x}\" "
        else:
            response += "\"\" "
        if (i + 1) % board_size == 0:
            response += "\n"

    return response


def display_policy_score(model: DualNet, board: GoBoard, color: Stone) -> str:
    """Policyを数値で表示するための文字列を生成する。（GoGui解析コマンド）

    Args:
        model (DualNet): Policyを出力するニューラルネットワーク。
        board (GoBoard): 評価する局面情報。
        color (Stone): 評価する手番の色。

    Returns:
        str: 表示用文字列。
    """
    board_size = board.get_board_size()
    input_plane = generate_input_planes(board, color)
    input_plane = torch.tensor(input_plane.reshape(1, 6, board_size, board_size)) #pylint: disable=E1121
    policy_predict, _ = model.forward_with_softmax(input_plane)
    policies = [policy_predict[0][i] for i in range(board_size ** 2)]
    response = ""

    for i, policy in enumerate(policies):
        pos = board.onboard_pos[i]
        if board.is_legal(pos, color):
            response += f"\"{policy:.04f}\" "
        else:
            response += "\"\" "
        if (i + 1) % board_size == 0:
            response += "\n"

    return response
