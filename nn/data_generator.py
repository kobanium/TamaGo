"""学習データの生成処理。
"""
import glob
import os
import numpy as np
from board.go_board import GoBoard
from board.stone import Stone
from nn.feature import generate_input_planes, generate_target_data
from sgf.reader import SGFReader
from learning_param import BATCH_SIZE, DATA_SET_SIZE


def _save_data(save_file_path: str, input_data: np.ndarray, policy_data: np.ndarray,\
    value_data: np.ndarray, kifu_counter: int):
    """学習データをnpzファイルとして出力する。

    Args:
        save_file_path (str): 保存するファイルパス。
        input_data (np.ndarray): 入力データ。
        policy_data (np.ndarray): Policyのデータ。
        value_data (np.ndarray): Valueのデータ
        kifu_counter (int): データセットにある棋譜データの個数。
    """
    save_data = {
        "input": np.array(input_data[0:DATA_SET_SIZE]),
        "policy": np.array(policy_data[0:DATA_SET_SIZE]),
        "value": np.array(value_data[0:DATA_SET_SIZE], dtype=np.int32),
        "kifu_count": np.array(kifu_counter)
    }
    np.savez_compressed(save_file_path, **save_data)

def generate_supervised_learning_data(program_dir: str, kifu_dir: str, board_size: int=9):
    """教師あり学習のデータを生成して保存する。

    Args:
        program_dir (str): プログラムのホームディレクトリのパス。
        kifu_dir (str): SGFファイルを格納しているディレクトリのパス。
        board_size (int, optional): 碁盤のサイズ. Defaults to 9.
    """
    board = GoBoard(board_size=board_size)

    input_data = []
    policy_data = []
    value_data = []

    kifu_counter = 1
    data_counter = 0

    for kifu_path in sorted(glob.glob(os.path.join(kifu_dir, "*.sgf"))):
        board.clear()
        sgf = SGFReader(kifu_path, board_size)
        color = Stone.BLACK
        value_label = sgf.get_value_label()

        for pos in sgf.get_moves():
            for sym in range(8):
                input_data.append(generate_input_planes(board, color, sym))
                policy_data.append(generate_target_data(board, pos, sym))
                value_data.append(value_label)
            board.put_stone(pos, color)
            color = Stone.get_opponent_color(color)
            # Valueのラベルを入れ替える。
            value_label = 2 - value_label

        if len(value_data) >= DATA_SET_SIZE:
            _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), \
                input_data, policy_data, value_data, kifu_counter)
            input_data = input_data[DATA_SET_SIZE:]
            policy_data = policy_data[DATA_SET_SIZE:]
            value_data = value_data[DATA_SET_SIZE:]
            kifu_counter = 1
            data_counter += 1

        kifu_counter += 1

    # 端数の出力
    n_batches = len(value_data) // BATCH_SIZE
    if n_batches > 0:
        _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), \
            input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], \
            value_data[0:n_batches*BATCH_SIZE], kifu_counter)
