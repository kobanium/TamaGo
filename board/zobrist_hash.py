"""Zobrist Hashの実装。
"""
import numpy as np

from board.constant import BOARD_SIZE, OB_SIZE


hash_bit_mask = np.random.randint(low=0, high=np.iinfo(np.uint64).max, \
    size=[4, (BOARD_SIZE + OB_SIZE * 2) ** 2], dtype=np.uint64)


def affect_stone_hash(hash_value, pos, color):
    """1つの石のハッシュ値を作用させる。

    Args:
        hash_value (np.array): 現局面のハッシュ値。
        pos (int): 座標。
        color (Stone): 石の色。

    Returns:
        np.array : 作用後のハッシュ値。
    """
    return hash_value ^ hash_bit_mask[color.value][pos]


def affect_string_hash(hash_value, pos_list, color):
    """複数の石のハッシュ値を作用させる。

    Args:
        hash_value (np.array): 現局面のハッシュ値。
        pos_list (list[int]): 座標列。
        color (Stone): 石の色。

    Returns:
        np.array: 作用後のハッシュ値。
    """
    for pos in pos_list:
        hash_value = hash_value ^ hash_bit_mask[color.value][pos]

    return hash_value
