from enum import Enum


class Stone(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    OUT_OF_BOARD = 3

    @classmethod
    def get_opponent_color(cls, color):
        if color == Stone.BLACK:
            return Stone.WHITE
        elif color == Stone.WHITE:
            return Stone.BLACK
        else:
            return color

    @classmethod
    def get_char(cls, color):
        if color == Stone.EMPTY:
            return '+'
        elif color == Stone.BLACK:
            return '@'
        elif color == Stone.WHITE:
            return 'O'
        elif color == Stone.OUT_OF_BOARD:
            return '#'
        else:
            return '!'
