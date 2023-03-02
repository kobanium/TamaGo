"""碁盤のデータ定義と操作処理。
"""
from typing import List, NoReturn
import numpy as np
from board.constant import PASS, OB_SIZE, GTP_X_COORDINATE
from board.coordinate import Coordinate
from board.pattern import Pattern, copy_pattern
from board.record import Record, copy_record
from board.stone import Stone
from board.string import StringData, copy_strings
from board.zobrist_hash import affect_stone_hash, affect_string_hash
from common.print_console import print_err


class GoBoard: # pylint: disable=R0902
    """碁盤クラス
    """
    def __init__(self, board_size: int, check_superko: bool=False) -> NoReturn:
        """碁盤クラスの初期化

        Args:
            board_size (int): 碁盤の大きさ。
            check_superko (bool): 超劫の判定有効化。
        """
        self.board_size = board_size
        self.board_size_with_ob = board_size + OB_SIZE * 2

        def pos(x_coord, y_coord):
            return x_coord + y_coord * self.board_size_with_ob

        def get_neighbor4(pos):
            return [pos - self.board_size_with_ob, pos - 1, pos + 1, pos + self.board_size_with_ob]

        self.board = [Stone.EMPTY] * (self.board_size_with_ob ** 2)
        self.pattern = Pattern(board_size, pos)
        self.strings = StringData(board_size, pos, get_neighbor4)
        self.record = Record()
        self.onboard_pos = [0] * (self.board_size ** 2)
        self.coordinate = Coordinate(board_size=board_size)
        self.ko_move = 0
        self.ko_pos = PASS
        self.prisoner = [0] * 2
        self.positional_hash = np.zeros(1, dtype=np.uint64)
        self.check_superko = check_superko
        self.board_start = OB_SIZE
        self.board_end = board_size + OB_SIZE - 1
        self.sym_map = [[0 for i in range(self.board_size_with_ob ** 2)] for j in range(8)]

        self.POS = pos # pylint: disable=C0103
        self.get_neighbor4 = get_neighbor4

        idx = 0
        for y_coord in range(self.board_start, self.board_end + 1):
            for x_coord in range(self.board_start, self.board_end + 1):
                coord = pos(x_coord, y_coord)
                self.onboard_pos[idx] = coord

                # そのまま
                self.sym_map[0][coord] = coord
                # 左右対称
                self.sym_map[1][coord] = pos(self.board_size_with_ob - (x_coord + 1), y_coord)
                # 上下対称
                self.sym_map[2][coord] = pos(x_coord, self.board_size_with_ob - (y_coord + 1))
                # 上下左右対称
                self.sym_map[3][coord] = pos(self.board_size_with_ob - (x_coord + 1),\
                    self.board_size_with_ob - (y_coord + 1))
                # 左上から右下方向の軸に対称
                self.sym_map[4][coord] = pos(x_coord=y_coord, y_coord=x_coord)
                # 90度反時計回りに回転
                self.sym_map[5][coord] = pos(y_coord, self.board_size_with_ob - (x_coord + 1))
                # 90度時計回りに回転
                self.sym_map[6][coord] = pos(self.board_size_with_ob - (y_coord + 1), x_coord)
                # 左下から右上方向の軸に対称
                self.sym_map[7][coord] = pos(self.board_size_with_ob - (y_coord + 1),\
                    self.board_size_with_ob - (x_coord + 1))
                idx += 1

        self.clear()


    def clear(self) -> NoReturn:
        """盤面の初期化
        """
        self.moves = 1
        self.position_hash = 0
        self.ko_move = 0
        self.ko_pos = 0
        self.prisoner = [0] * 2
        self.positional_hash.fill(0)

        for i, _ in enumerate(self.board):
            self.board[i] = Stone.OUT_OF_BOARD

        for y_coord in range(self.board_start, self.board_end + 1):
            for x_coord in range(self.board_start, self.board_end + 1):
                pos = self.POS(x_coord, y_coord)
                self.board[pos] = Stone.EMPTY

        self.pattern.clear()
        self.strings.clear()
        self.record.clear()

    def put_stone(self, pos: int, color: Stone) -> NoReturn:
        """指定された座標に指定された色の石を石を置く。

        Args:
            pos (int): 石を置く座標。
            color (Stone): 置く石の色。
        """
        if pos == PASS:
            self.record.save(self.moves, color, pos, self.positional_hash)
            self.moves += 1
            return

        opponent_color = Stone.get_opponent_color(color)


        self.board[pos] = color
        self.pattern.put_stone(pos, color)
        self.positional_hash = affect_stone_hash(self.positional_hash, pos, color)

        neighbor4 = self.get_neighbor4(pos)

        connection = []
        prisoner = 0

        for neighbor in neighbor4:
            if self.board[neighbor] == color:
                self.strings.remove_liberty(neighbor, pos)
                connection.append(self.strings.get_id(neighbor))
            elif self.board[neighbor] == opponent_color:
                self.strings.remove_liberty(neighbor, pos)
                if self.strings.get_num_liberties(neighbor) == 0:
                    removed_stones = self.strings.remove_string(self.board, neighbor)
                    prisoner += len(removed_stones)
                    for removed_pos in removed_stones:
                        self.pattern.remove_stone(removed_pos)
                    self.positional_hash = affect_string_hash(self.positional_hash, \
                        removed_stones, color)

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

        # 着手した時に記録
        self.record.save(self.moves, color, pos, self.positional_hash)
        self.moves += 1

    def _is_suicide(self, pos: int, color: Stone) -> bool:
        """自殺手か否かを判定する。
        自殺手ならTrue、そうでなければFalseを返す。

        Args:
            pos (int): 確認する座標。
            color (Stone): 着手する石の色。

        Returns:
            bool: 自殺手の判定結果。自殺手ならTrue、そうでなければFalse。
        """
        other = Stone.get_opponent_color(color)

        neighbor4 = self.get_neighbor4(pos)

        for neighbor in neighbor4:
            if self.board[neighbor] == other and self.strings.get_num_liberties(neighbor) == 1:
                return False
            if self.board[neighbor] == color and self.strings.get_num_liberties(neighbor) > 1:
                return False

        return True

    def is_legal(self, pos: int, color: Stone) -> bool:
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
        if self.pattern.get_n_neighbors_empty(pos) == 0 and \
           self._is_suicide(pos, color):
            return False

        # 劫
        if (self.ko_pos == pos) and (self.ko_move == (self.moves - 1)):
            return False

        # 超劫の確認
        if self.check_superko and pos != PASS:
            opponent = Stone.get_opponent_color(color)
            neighbor4 = self.get_neighbor4(pos)
            neighbor_ids = [self.strings.get_id(neighbor) for neighbor in neighbor4]
            unique_ids = list(set(neighbor_ids))
            current_hash = self.positional_hash

            # 打ち上げる石があれば打ち上げたと仮定
            for string_id in unique_ids:
                if self.strings.get_num_liberties(self.strings.string[string_id].get_origin) == 1:
                    stones = self.strings.get_stone_coordinates(string_id)
                    current_hash = affect_string_hash(current_hash, stones, opponent)
            # 石を置く
            current_hash = affect_stone_hash(current_hash, pos=pos, color=color)

            if self.record.has_same_hash(current_hash):
                return False


        return True

    def is_legal_not_eye(self, pos: int, color: Stone) -> bool:
        """合法手かつ眼でないか否かを確認する。
        合法手かつ眼でなければTrue、そうでなければFalseを返す。

        Args:
            pos (int): 確認する座標。
            color (Stone): 手番の色。

        Returns:
            bool: 判定結果。合法手かつ眼でなければTrue、そうでなければFalse。
        """
        neighbor4 = self.get_neighbor4(pos)
        if self.pattern.get_eye_color(pos) is not color or \
           self.strings.get_num_liberties(neighbor4[0]) == 1 or \
           self.strings.get_num_liberties(neighbor4[1]) == 1 or \
           self.strings.get_num_liberties(neighbor4[2]) == 1 or \
           self.strings.get_num_liberties(neighbor4[3]) == 1:
            return self.is_legal(pos, color)

        return False

    def get_all_legal_pos(self, color: Stone) -> List[int]:
        """全ての合法手の座標を取得する。ただし眼は除く。

        Args:
            color (Stone): 手番の色

        Returns:
            list[int]: 合法手の座標列。
        """
        return [pos for pos in self.onboard_pos if self.is_legal(pos, color)]

    def display(self, sym: int=0) -> NoReturn:
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

        for y_coord in range(self.board_start, self.board_end + 1):
            output = f"{self.board_size - y_coord + 1:>2d}|"
            for x_coord in range(self.board_start, self.board_end + 1):
                pos = self.get_symmetrical_coordinate(self.POS(x_coord, y_coord), sym)
                output += " " + Stone.get_char(self.board[pos])
            output += " |\n"
            board_string += output

        board_string += "  +" + "-" * (self.board_size * 2 + 1) + "+\n"

        print_err(board_string)

    def get_board_size(self) -> NoReturn:
        """碁盤の大きさを取得する。

        Returns:
            int: 碁盤の大きさ
        """
        return self.board_size

    def get_board_data(self, sym: int) -> List[int]:
        """ニューラルネットワークの入力用の碁盤情報を取得する。

        Returns:
            list[int]: 空点は0, 黒石は1, 白石は2のリスト。
        """
        return [self.board[self.get_symmetrical_coordinate(pos, sym)].value \
            for pos in self.onboard_pos]

    def get_symmetrical_coordinate(self, pos: int, sym: int) -> int:
        """8対称のいずれかの座標を取得する。

        Args:
            pos (int): 元の座標。
            sym (int): 対称形の指定。（0〜7）

        Returns:
            int: 指定した対称の座標。
        """
        return self.sym_map[sym][pos]


def copy_board(dst: GoBoard, src: GoBoard):
    """盤面の情報をコピーする。

    Args:
        dst (GoBoard): コピー先の盤面情報のデータ。
        src (GoBoard): コピー元の盤面情報のデータ。
    """
    dst.board = src.board[:]
    copy_pattern(dst.pattern, src.pattern)
    copy_strings(dst.strings, src.strings)
    copy_record(dst.record, src.record)
    dst.ko_move = src.ko_move
    dst.ko_pos = src.ko_pos
    dst.prisoner = src.prisoner[:]
    dst.positional_hash = src.positional_hash.copy()
    dst.moves = src.moves
