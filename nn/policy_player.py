"""Policy Networkのみを使用した着手生成処理
"""
import random

import torch

from board.constant import PASS
from board.go_board import GoBoard
from board.stone import Stone
from nn.feature import generate_input_planes
from nn.network.dual_net import DualNet

def generate_move_from_policy(network: DualNet, board: GoBoard, color: Stone) -> int:
    """Policy Networkを使用して着手を生成する。

    Args:
        network (DualNet): ニューラルネットワーク。
        board (GoBoard): 現在の碁盤の情報。
        color (Stone): 手番の色。

    Returns:
        int: 生成した着手の座標。
    """
    board_size = board.get_board_size()
    input_plane = generate_input_planes(board, color)
    input_data = torch.tensor(input_plane.reshape(1, 6, board_size, board_size)) #pylint: disable=E1121
    policy, _ = network.inference(input_data)

    policy = policy[0].numpy().tolist()

    # 合法手のみ候補手としてピックアップ
    candidates = [{"pos": pos, "policy": policy[i]} \
        for i, pos in enumerate(board.onboard_pos) if board.is_legal(pos, color)]

    # パスは候補手確定
    candidates.append({ "pos": PASS, "policy": policy[board_size ** 2] })

    max_policy = max([candidate["policy"] for candidate in candidates])

    sampled_candidates = [candidate for candidate in candidates \
        if candidate["policy"] > max_policy * 0.1]

    sampled_pos = [candidate["pos"] for candidate in sampled_candidates]
    sampled_policy = [candidate["policy"] for candidate in sampled_candidates]

    return random.choices(sampled_pos, weights=sampled_policy, k=1)[0]
