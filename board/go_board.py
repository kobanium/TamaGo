import sys

from board.constant import PASS, OB_SIZE, GTP_X_COORDINATE
from board.coordinate import Coordinate
from board.stone import Stone
from board.string import StringData
from common.print_console import print_err



class GoBoard:
    """碁盤クラス
    """
    def __init__(self, board_size):
        """碁盤クラスの初期化

        Args:
            board_size (int): 碁盤の大きさ。
        """
        self.board_size = board_size
        self.board_size_with_ob = board_size + OB_SIZE * 2

        self.board = [Stone.EMPTY] * (self.board_size_with_ob ** 2)
        self.strings = StringData(board_size=board_size)
        self.onboard_pos = [0] * (self.board_size ** 2)
        self.coordinate = Coordinate(board_size=board_size)
        self.ko_move = 0
        self.ko_pos = PASS
        self.prisoner = [0] * 2

        self.clear()


    def clear(self):
        """盤面の初期化
        """
        self.moves = 1
        self.position_hash = 0
        self.ko_move = 0
        self.ko_pos = 0
        self.prisoner = [0] * 2

        i = 0
        for y in range(1, self.board_size + OB_SIZE):
            for x in range(1, self.board_size + OB_SIZE):
                pos = x + y * self.board_size_with_ob
                self.board[pos] = Stone.EMPTY
                self.onboard_pos[i] = pos
                i += 1

        for y in range(self.board_size_with_ob):
            for x in range(OB_SIZE):
                self.board[x + y * self.board_size_with_ob] = Stone.OUT_OF_BOARD
                self.board[y + x * self.board_size_with_ob] = Stone.OUT_OF_BOARD
                self.board[y + (self.board_size_with_ob - 1 - x) * self.board_size_with_ob] = Stone.OUT_OF_BOARD
                self.board[(self.board_size_with_ob - 1 - x) + y * self.board_size_with_ob] = Stone.OUT_OF_BOARD

        self.strings.clear()

    def put_stone(self, pos, color):
        """指定された座標に指定された色の石を石を置く。

        Args:
            pos (int): 石を置く座標。
            color (Stone): 置く石の色。
        """
        if pos == PASS:
            return

        opponent_color = Stone.get_opponent_color(color)

        self.board[pos] = color

        neighbor4 = [pos - self.board_size_with_ob, pos - 1, pos + 1, pos + self.board_size_with_ob]

        connection = []
        prisoner = 0

        for neighbor in neighbor4:
            if self.board[neighbor] == color:
                self.strings.remove_liberty(neighbor, pos)
                connection.append(self.strings.get_id(neighbor))
            elif self.board[neighbor] == opponent_color:
                self.strings.remove_liberty(neighbor, pos)
                if self.strings.get_num_liberties(neighbor) == 0:
                    prisoner += self.strings.remove_string(self.board, neighbor)

        if color == Stone.BLACK:
            self.prisoner[0] += prisoner
        elif color == Stone.WHITE:
            self.prisoner[1] += prisoner

        if len(connection) == 0:
            self.strings.make_string(self.board, pos, color)
            if prisoner == 1:
                self.ko_move = self.moves
                self.ko_pos = self.strings.string[self.strings.get_id(pos)].lib[0]
        elif len(connection) == 1:
            self.strings.add_stone(self.board, pos, color, connection[0])
        else:
            self.strings.connect_string(self.board, pos, color, connection)

        self.moves += 1

    def _is_suicide(self, pos, color):
        """自殺手か否かを判定する。
        自殺手ならTrue、そうでなければFalseを返す。

        Args:
            pos (int): 確認する座標。
            color (Stone): 着手する石の色。

        Returns:
            bool: 自殺手の判定結果。自殺手ならTrue、そうでなければFalse。
        """
        other = Stone.get_opponent_color(color)

        neighbor4 = [pos - self.board_size_with_ob,
                     pos - 1, pos + 1,
                     pos + self.board_size_with_ob]

        for neighbor in neighbor4:
            if self.board[neighbor] == other and self.strings.get_num_liberties(neighbor) == 1:
                return False
            if self.board[neighbor] == color and self.strings.get_num_liberties(neighbor) > 1:
                return False

        return True

    def is_legal(self, pos, color):
        """合法手か否かを判定する。
        合法手ならTrue、そうでなければFalseを返す。

        Args:
            pos (int): 確認する座標。
            color (Stone): 着手する石の色。

        Returns:
            bool: 合法手の判定結果。合法手ならTrue、そうでなければFalse。
        """
        # 既に石がある
        if self.board[pos] != Stone.EMPTY:
            return False

        # 自殺手
        neighbor4 = [pos - self.board_size_with_ob,
                     pos - 1, pos + 1,
                     pos + self.board_size_with_ob]
        if self.board[neighbor4[0]] != Stone.EMPTY and \
           self.board[neighbor4[1]] != Stone.EMPTY and \
           self.board[neighbor4[2]] != Stone.EMPTY and \
           self.board[neighbor4[3]] != Stone.EMPTY and \
           self._is_suicide(pos, color):
            return False

        # 劫
        if (self.ko_pos == pos) and (self.ko_move == (self.moves - 1)):
            return False

        return True

    def is_eye(self, pos, color):
        """眼形か否かを判定する。
        上下左右が同一の色の石で囲われていれば、その色を返す。

        Args:
            pos (int): 確認する座標。
            color (Stone): 手番の色。

        Returns:
            Stone: 眼を構成する石の色。眼でなければ空点を色として返す。
        """
        other = Stone.get_opponent_color(color)
        neighbor4 = [pos - self.board_size_with_ob,
                     pos - 1, pos + 1,
                     pos + self.board_size_with_ob]

        if (self.board[neighbor4[0]] == Stone.OUT_OF_BOARD or self.board[neighbor4[0]] == color) and \
           (self.board[neighbor4[1]] == Stone.OUT_OF_BOARD or self.board[neighbor4[1]] == color) and \
           (self.board[neighbor4[2]] == Stone.OUT_OF_BOARD or self.board[neighbor4[2]] == color) and \
           (self.board[neighbor4[3]] == Stone.OUT_OF_BOARD or self.board[neighbor4[3]] == color):
            return color
        if (self.board[neighbor4[0]] == Stone.OUT_OF_BOARD or self.board[neighbor4[0]] == other) and \
             (self.board[neighbor4[1]] == Stone.OUT_OF_BOARD or self.board[neighbor4[1]] == other) and \
             (self.board[neighbor4[2]] == Stone.OUT_OF_BOARD or self.board[neighbor4[2]] == other) and \
             (self.board[neighbor4[3]] == Stone.OUT_OF_BOARD or self.board[neighbor4[3]] == other):
            return other

        return Stone.EMPTY

    def is_legal_not_eye(self, pos, color):
        """合法手かつ眼でないか否かを確認する。
        合法手かつ眼でなければTrue、そうでなければFalseを返す。

        Args:
            pos (int): 確認する座標。
            color (Stone): 手番の色。

        Returns:
            bool: 判定結果。合法手かつ眼でなければTrue、そうでなければFalse。
        """
        neighbor4 = [pos - self.board_size_with_ob,
                     pos - 1, pos + 1,
                     pos + self.board_size_with_ob]

        if self.is_eye(pos, color) is not color or \
           self.strings.get_num_liberties(neighbor4[0]) == 1 or \
           self.strings.get_num_liberties(neighbor4[1]) == 1 or \
           self.strings.get_num_liberties(neighbor4[2]) == 1 or \
           self.strings.get_num_liberties(neighbor4[3]) == 1:
            return self.is_legal(pos, color)

        return False

    def get_all_legal_pos(self, color):
        """全ての合法手の座標を取得する。ただし眼は除く。

        Args:
            color (Stone): 手番の色

        Returns:
            list[int]: 合法手の座標列。
        """
        legal_pos = []
        for pos in self.onboard_pos:
            if self.is_legal_not_eye(pos, color):
                legal_pos.append(pos)
        return legal_pos

    def display(self):
        """盤面を表示する。
        """
        board_string = f"Move : {self.moves}\n"
        board_string += f"Prisoner(Black) : {self.prisoner[0]}\n"
        board_string += f"Prisoner(White) : {self.prisoner[1]}\n"

        board_string += "   "
        for i in range(self.board_size):
            board_string += " " + GTP_X_COORDINATE[i + 1]
        board_string += "\n"

        board_string += "  +" + "-" * (self.board_size * 2 + 1) + "+\n"

        for y in range(1, self.board_size + 1):
            output = "{:>2d}|".format(self.board_size - y + 1)
            for x in range(1, self.board_size + 1):
                pos = x + y * self.board_size_with_ob
                output += " " + Stone.get_char(self.board[pos])
            output += " |\n"
            board_string += output

        board_string += "  +" + "-" * (self.board_size * 2 + 1) + "+\n"

        print_err(board_string)

    def get_board_size(self):
        """碁盤の大きさを取得する。

        Returns:
            int: 碁盤の大きさ
        """
        return self.board_size