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
    data = np.identity(3)[board_data].transpose()

    # 直前の着手を取得
    _, previous_move, _ = board.record.get(board.moves - 1)
    # 直前の着手の座標
    if previous_move == PASS:
        move_history = np.zeros(shape=(1, board_size ** 2))
    else:
        previous_move_data = [1 if previous_move == board.get_symmetrical_coordinate(pos, sym) \
            else 0 for pos in board.onboard_pos]
        move_history = np.array(previous_move_data).reshape(1, board_size**2)

    input_data = np.concatenate([data, move_history]).reshape(4, board_size, board_size)

    return input_data
