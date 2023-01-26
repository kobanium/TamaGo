"""着手の履歴の保持。
"""
import numpy as np

from board.constant import PASS, MAX_RECORDS
from board.stone import Stone
from common.print_console import print_err


class Record:
    """着手の履歴を保持するクラス。
    """
    def __init__(self):
        """Recordクラスのコンストラクタ。
        """
        self.color = [Stone.EMPTY] * MAX_RECORDS
        self.pos = [PASS] * MAX_RECORDS
        self.hash_value = np.zeros(shape=MAX_RECORDS, dtype=np.uint64)

    def clear(self):
        """データを初期化する。
        """
        self.color = [Stone.EMPTY] * MAX_RECORDS
        self.pos = [PASS] * MAX_RECORDS
        self.hash_value.fill(0)

    def save(self, moves, color, pos, hash_value):
        """着手の履歴の記録する。

        Args:
            moves (int): 着手数。
            color (Stone): 着手する石の色。
            pos (int): 着手する座標。
            hash_value (np.array): 局面のハッシュ値。
        """
        if moves < MAX_RECORDS:
            self.color[moves] = color
            self.pos[moves] = pos
            self.hash_value[moves] = hash_value
        else:
            print_err("Cannnot save move record.")

    def has_same_hash(self, hash_value):
        """同じハッシュ値があるかを確認する。

        Args:
            hash_value (np.array): ハッシュ値。

        Returns:
            bool: 同じハッシュ値がある場合はTrue、なければFalse。
        """
        return np.any(self.hash_value == hash_value)

    def get(self, moves):
        """指定した着手を取得する。

        Args:
            moves (int): 着手数。

        Returns:
            (Stone, int, np.array): 着手の色、座標、ハッシュ値。
        """
        return (self.color[moves], self.pos[moves], self.hash_value[moves])
