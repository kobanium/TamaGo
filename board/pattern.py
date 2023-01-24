import numpy as np

from board.coordinate import Coordinate
from board.constant import OB_SIZE
from board.stone import Stone
from common.print_console import print_err

pattern_mask = np.array([
    [0x3fff, 0x00004000, 0x00008000],
    [0xcfff, 0x00001000, 0x00002000],
    [0xf3ff, 0x00000400, 0x00000800],
    [0xfcff, 0x00000100, 0x00000200],
    [0xff3f, 0x00000040, 0x00000080],
    [0xffcf, 0x00000010, 0x00000020],
    [0xfff3, 0x00000004, 0x00000008],
    [0xfffc, 0x00000001, 0x00000002],
], dtype=np.uint32)

class Pattern:
    def __init__(self, board_size, POS):
        """Patternクラスのコンストラクタ。

        Args:
            board_size (int): 碁盤の大きさ。
            POS (func): 座標変換用の関数。
        """
        self.board_size = board_size
        board_size_with_ob = board_size + OB_SIZE * 2
        self.pat3 = np.empty(shape=board_size_with_ob ** 2, dtype=np.uint32)
        self.POS = POS
        self.update_pos = [
            -board_size_with_ob - 1, -board_size_with_ob, -board_size_with_ob + 1,
            -1, 1, board_size_with_ob - 1, board_size_with_ob, board_size_with_ob + 1
        ]

        self.clear()

    def clear(self):
        """周囲の石のパターンを初期状態にする。
        """
        board_start = OB_SIZE
        board_end = self.board_size + OB_SIZE - 1
        self.pat3.fill(0)

        for y in range(board_start, board_end + 1):
            self.pat3[self.POS(y, board_start)] = self.pat3[self.POS(y, board_start)] | 0x003f
            self.pat3[self.POS(board_end, y)] = self.pat3[self.POS(board_end, y)] | 0xc330
            self.pat3[self.POS(y, board_end)] = self.pat3[self.POS(y, board_end)] | 0xfc00
            self.pat3[self.POS(board_start, y)] = self.pat3[self.POS(board_start, y)] | 0x0cc3

    def remove_stone(self, pos):
        """周囲の石のパターンから石を取り除く。

        Args:
            pos (int): 石を取り除く座標。
        """
        for i, shift in enumerate(self.update_pos):
            self.pat3[pos + shift] = self.pat3[pos + shift] & pattern_mask[i][0]

    def put_stone(self, pos, color):
        """周囲の石のパターンの石を追加する。

        Args:
            pos (int): 打つ石の座標。
            color (Stone): 打つ石の色。
        """
        if color == Stone.BLACK:
            color_index = 1
        elif color == Stone.WHITE:
            color_index = 2
        else:
            return
        for i, shift in enumerate(self.update_pos):
            self.pat3[pos + shift] = self.pat3[pos + shift] | pattern_mask[i][color_index]


    def display(self, pos):
        """指定した座標の周囲の石のパターンを表示する。（デバッグ用)

        Args:
            pos (int): 表示する対象の座標。
        """
        coordinate = Coordinate(self.board_size)
        stone = ["+", "@", "O", "#"]
        print_err(self.pat3[pos])
        message = ""
        message += stone[np.bitwise_and(self.pat3[pos], 0x3)]
        message += stone[np.bitwise_and(np.right_shift(self.pat3[pos], 2), 0x3)]
        message += stone[np.bitwise_and(np.right_shift(self.pat3[pos], 4), 0x3)] + '\n'
        message += stone[np.bitwise_and(np.right_shift(self.pat3[pos], 6), 0x3)]
        message += "*"
        message += stone[np.bitwise_and(np.right_shift(self.pat3[pos], 8), 0x3)] +'\n'
        message += stone[np.bitwise_and(np.right_shift(self.pat3[pos], 10), 0x3)] 
        message += stone[np.bitwise_and(np.right_shift(self.pat3[pos], 12), 0x3)]
        message += stone[np.bitwise_and(np.right_shift(self.pat3[pos], 14), 0x3)] + '\n'

        print_err(message)