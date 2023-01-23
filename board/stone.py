from enum import Enum


class Stone(Enum):
    """石の色を表すクラス。
    """
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    OUT_OF_BOARD = 3

    @classmethod
    def get_opponent_color(cls, color):
        """相手の手番の色を取得する。

        Args:
            color (Stone): 手番の色。

        Returns:
            Stone: 相手の色。
        """
        if color == Stone.BLACK:
            return Stone.WHITE

        if color == Stone.WHITE:
            return Stone.BLACK

        return color

    @classmethod
    def get_char(cls, color):
        """色に対応する文字を取得する。

        Args:
            color (Stone): 色。

        Returns:
            str: 色に対応する文字。
        """
        if color == Stone.EMPTY:
            return '+'

        if color == Stone.BLACK:
            return '@'

        if color == Stone.WHITE:
            return 'O'

        if color == Stone.OUT_OF_BOARD:
            return '#'

        return '!'
