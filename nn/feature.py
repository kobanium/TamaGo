"""ニューラルネットワークの入力特徴生成処理
"""
import numpy as np

from board.constant import PASS
from board.go_board import GoBoard
from board.stone import Stone

def generate_input_planes(board: GoBoard, color: Stone, sym: int=0) -> np.ndarray:
    """ニューラルネットワークの入力データを生成する。

    Args:
        board (GoBoard): 碁盤の情報。
        color (Stone): 手番の色。
        sym (int, optional): 対称形の指定. Defaults to 0.

    Returns:
        numpy.ndarray: ニューラルネットワークの入力データ。
    """
    board_data = board.get_board_data(sym)
    board_size = board.get_board_size()
    # 手番が白の時は石の色を反転する.
    if color is Stone.WHITE:
        board_data = [datum if datum == 0 else (3 - datum) for datum in board_data]

    # 碁盤の各交点の状態
    #     空点 : 1枚目の入力面
    #     自分の石 : 2枚目の入力面
    #     相手の石 : 3枚目の入力面
    board_plane = np.identity(3)[board_data].transpose()

    # 直前の着手を取得
    _, previous_move, _ = board.record.get(board.moves - 1)
    # 直前の着手の座標
    if previous_move == PASS:
        history_plane = np.zeros(shape=(1, board_size ** 2))
    else:
        previous_move_data = [1 if previous_move == board.get_symmetrical_coordinate(pos, sym) \
            else 0 for pos in board.onboard_pos]
        history_plane = np.array(previous_move_data).reshape(1, board_size**2)

    if color == Stone.BLACK:
        color_plane = np.ones(shape=(1, board_size ** 2))
    else:
        color_plane = np.zeros(shape=(1, board_size ** 2))

    input_data = np.concatenate([board_plane, history_plane, color_plane]).reshape(5, board_size, board_size)

    return input_data


def generate_target_data(board:GoBoard, target_pos: int, sym: int=0) -> np.ndarray:
    """

    Args:
        board (GoBoard): 碁盤の情報。
        target_pos (int): 教師データの着手の座標。
        sym (int, optional): 対称系の指定. Defaults to 0.

    Returns:
        np.ndarray: _description_
    """
    target = [1 if target_pos == board.get_symmetrical_coordinate(pos, sym) \
        else 0 for pos in board.onboard_pos]
    # パスだけ対称形から外れた末尾に挿入する。
    if target_pos == PASS:
        target.append(1)
    else:
        target.append(0)

    return np.array(target)