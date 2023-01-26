"""連の定義と処理の実装。
"""
from board.constant import STRING_END, LIBERTY_END, NEIGHBOR_END, OB_SIZE
from board.coordinate import Coordinate
from board.stone import Stone
from common.print_console import print_err

class String:
    """連の実装クラス。
    """
    def __init__(self, board_size):
        """連クラスのコンストラクタ。

        Args:
            board_size (int): 碁盤のサイズ。
        """
        self.color = 0
        self.libs = 0
        self.lib = [0] * ((board_size + 2) ** 2)
        self.neighbors = 0
        self.neighbor = [0] * int(0.8 * board_size * (board_size - 1) + 5)
        self.origin = 0
        self.size = 0
        self.flag = False

    def initialize(self, pos, color):
        """連の生成処理。

        Args:
            pos (int): 連を構成する石の座標。
            color (Stone): 連を構成する石の色。
        """
        for i, _ in enumerate(self.lib):
            self.lib[i] = 0

        for i, _ in enumerate(self.neighbor):
            self.neighbor[i] = 0

        self.color = color
        self.lib[0] = LIBERTY_END
        self.neighbor[0] = NEIGHBOR_END
        self.libs = 0
        self.origin = pos
        self.size = 1
        self.neighbors = 0
        self.flag = True

    def has_liberty(self, pos):
        """指定した座標を呼吸点として持っているか確認する。

        Args:
            pos (int): 確認する座標。

        Returns:
            bool: 指定した座標を呼吸点として持つ場合はTrue、そうでなければFalse。
        """
        return self.lib[pos] > 0

    def has_neighbor(self, neighbor):
        """指定した連IDと隣接する敵連として持っているかを確認する。

        Args:
            neighbor (int): 確認する連ID

        Returns:
            _bool: 指定した連IDを敵連として持つ場合はTrue、そうでなければFalse。
        """
        return self.neighbor[neighbor] != 0

    def remove(self):
        """連を削除する。
        """
        self.flag = False

    def add_stone(self):
        """連を構成する石の数を1つ増やす。
        """
        self.size += 1

    def add_size(self, size):
        """連を構成する石の個数を加算する。

        Args:
            size (int): 加算する石の個数。
        """
        self.size += size

    def get_size(self):
        """連を構成する石の個数を取得する。

        Returns:
            int: 連を構成する石の個数。
        """
        return self.size

    def exist(self):
        """連が存在するか確認する。

        Returns:
            bool: 連の存在フラグ。存在していればTrue、存在していなければFalse。
        """
        return self.flag

    def get_num_liberties(self):
        """連が持つ呼吸点の個数を取得する。

        Returns:
            int: 連が持つ呼吸点の個数。
        """
        return self.libs

    def get_origin(self):
        """連を構成する石の始点を取得する。

        Returns:
            int: 連を構成する石の始点の座標。
        """
        return self.origin

    def get_neighbor_origin(self):
        """隣接する敵連IDの最小値を取得する。

        Returns:
            int: 隣接する敵連IDの最小値。
        """
        return self.neighbor[0]

    def set_origin(self, pos):
        """連を構成する石の始点を設定する。

        Args:
            pos (int): 設定する始点の座標。
        """
        self.origin = pos

    def add_liberty(self, pos, head):
        """呼吸点を1つ追加する。

        Args:
            pos (int): 追加する呼吸点の座標。
            head (int): 探索開始点

        Returns:
            int: 追加した呼吸点の座標。
        """
        # 追加済みなので何もしない
        if self.has_liberty(pos):
            return pos

        lib = head
        while self.lib[lib] < pos:
            lib = self.lib[lib]

        self.lib[pos] = self.lib[lib]
        self.lib[lib] = pos
        self.libs += 1

        return pos

    def remove_liberty(self, pos):
        """指定した座標の呼吸点を取り除く。

        Args:
            pos (int): 取り除く座標。
        """
        if not self.has_liberty(pos):
            return

        lib = 0
        while self.lib[lib] != pos:
            lib = self.lib[lib]

        self.lib[lib] = self.lib[self.lib[lib]]
        self.lib[pos] = 0
        self.libs -= 1

    def add_neighbor(self, string_id):
        """隣接する敵連IDを追加する。

        Args:
            string_id (int): 追加する敵連ID
        """
        # 追加済みなので何もしない
        if self.neighbor[string_id] != 0:
            return

        # 追加場所を見つける
        neighbor = 0
        while self.neighbor[neighbor] < string_id:
            neighbor = self.neighbor[neighbor]

        # 隣接する連IDを挿入する追加
        self.neighbor[string_id] = self.neighbor[neighbor]
        self.neighbor[neighbor] = string_id
        self.neighbors += 1

    def remove_neighbor(self, remove_id):
        """指定した隣接する敵連IDを除去する。

        Args:
            remove_id (int): 除去する隣接する敵連ID
        """
        if self.neighbor[remove_id] == 0:
            return

        neighbor = 0
        while self.neighbor[neighbor] != remove_id:
            neighbor = self.neighbor[neighbor]

        self.neighbor[neighbor] = self.neighbor[self.neighbor[neighbor]]
        self.neighbor[remove_id] = 0
        self.neighbors -= 1

    def get_color(self):
        """連を構成する石の色を取得する。

        Returns:
            Stone: 連を構成する石の色。
        """
        return self.color

    def get_liberties(self):
        """連が持つ呼吸点の座標を全て取得する。

        Returns:
            list[int]: 連が持つ呼吸の点の座標列。
        """
        liberties = []
        lib = self.lib[0]
        while lib != LIBERTY_END:
            liberties.append(lib)
            lib = self.lib[lib]
        return liberties

    def get_neighbors(self):
        """隣接する敵連IDを全て取得する。

        Returns:
            list[int]: 隣接する敵連ID列。
        """
        neighbors = []
        neighbor = self.neighbor[0]
        while neighbor != NEIGHBOR_END:
            neighbors.append(neighbor)
            neighbor = self.neighbor[neighbor]
        return neighbors


class StringData:
    """碁盤上の全ての連を管理するクラス
    """
    def __init__(self, board_size, pos_func, get_neighbor4):
        """コンストラクタ。

        Args:
            board_size (int): 碁盤のサイズ。
        """
        board_max = (board_size + OB_SIZE * 2) ** 2
        self.string = [String(board_size=board_size) \
            for i in range(int(0.8 * board_size * (board_size - 1) + 5))]
        self.string_id = [0] * board_max
        self.string_next = [0] * board_max
        self.board_size = board_size
        self.POS = pos_func
        self.get_neighbor4 = get_neighbor4

    def clear(self):
        """全ての連を削除する。
        """
        for string in self.string:
            string.remove()

    def remove_liberty(self, pos, lib):
        """指定した座標の連の呼吸点を除去する。

        Args:
            pos (int): 処理する連の座標。
            lib (int): 除去する呼吸点の座標。
        """
        self.string[self.get_id(pos)].remove_liberty(lib)

    def remove_string(self, board, remove_pos):
        """連を盤上から除去する。

        Args:
            board (Stone): 碁盤上の交点の状態。
            remove_pos (int): 削除する連の座標。

        Returns:
            int: 取り除いた連を構成していた石の個数。
        """
        remove_id = self.get_id(remove_pos)

        pos = self.string[remove_id].get_origin()

        removed_stone = []

        while pos != STRING_END:
            board[pos] = Stone.EMPTY
            removed_stone.append(pos)

            neighbor4 = self.get_neighbor4(pos)

            for neighbor_pos in neighbor4:
                neighbor_id = self.get_id(neighbor_pos)
                if self.string[neighbor_id].exist():
                    self.string[neighbor_id].add_liberty(pos, 0)

            next_pos = self.string_next[pos]
            self.string_next[pos] = 0
            self.string_id[pos] = 0
            pos = next_pos

        neighbor_id = self.string[remove_id].neighbor[0]
        while neighbor_id != NEIGHBOR_END:
            self._remove_neighbor_string(neighbor_id, remove_id)
            neighbor_id = self.string[remove_id].neighbor[neighbor_id]

        self.string[remove_id].remove()

        return removed_stone

    def get_id(self, pos):
        """指定した座標の連IDを取得する。

        Args:
            pos (int): 連IDを取得したい座標。

        Returns:
            int: 連ID。
        """
        return self.string_id[pos]

    def get_stone_coordinates(self, string_id):
        """連を構成する石の座標列を取得する。

        Args:
            string_id (int): 座標を取得したい連ID。

        Returns:
            list[int]: 連を構成する石の座標列。
        """
        pos = self.string[string_id].get_origin()
        stones = []

        while pos != STRING_END:
            stones.append(pos)
            pos = self.string_next[pos]

        return stones

    def get_num_liberties(self, pos):
        """指定した座標の連の呼吸点数を取得する。

        Args:
            pos (int): 呼吸点数を取得したい連の座標。

        Returns:
            int: 呼吸点数。
        """
        return self.string[self.get_id(pos)].get_num_liberties()

    def make_string(self, board, pos, color):
        """連を作成する。

        Args:
            board (Stone): 碁盤の交点の状態。
            pos (int): 作成する連を構成する石の座標。
            color (Stone): 作成する連の色。
        """
        opponent_color = Stone.get_opponent_color(color)
        lib_add = 0

        string_id = 1

        while self.string[string_id].exist():
            string_id += 1

        self.string[string_id].initialize(pos, color)
        self.string_id[pos] = string_id
        self.string_next[pos] = STRING_END

        neighbor4 = self.get_neighbor4(pos)

        for neighbor in neighbor4:
            if board[neighbor] == Stone.EMPTY:
                lib_add = self.string[string_id].add_liberty(neighbor, lib_add)
            elif board[pos] == opponent_color:
                neighbor_id = self.string_id[neighbor]
                self.string[string_id].add_neighbor(neighbor_id)
                self.string[neighbor_id].add_neighbor(string_id)

    def _add_stone_to_string(self, string_id, pos):
        """指定した座標を連に追加する。

        Args:
            string_id (int): 石を追加する連ID。
            pos (int): 追加する石の座標。
        """
        if pos == STRING_END:
            return

        if self.string[string_id].get_origin() > pos:
            self.string_next[pos] = self.string[string_id].get_origin()
            self.string[string_id].set_origin(pos)
        else:
            str_pos = self.string[string_id].get_origin()
            while self.string_next[str_pos] < pos:
                str_pos = self.string_next[str_pos]
            self.string_next[pos] = self.string_next[str_pos]
            self.string_next[str_pos] = pos

        self.string[string_id].add_stone()

    def add_stone(self, board, pos, color, string_id):
        """連に石を1つ追加する。

        Args:
            board (Stone): 碁盤の交点の状態。
            pos (int): 追加する石の座標。
            color (Stone): 追加する石の色。
            string_id (int): 石を追加する連ID。
        """
        opponent_color = Stone.get_opponent_color(color)
        self.string_id[pos] = string_id

        self._add_stone_to_string(string_id, pos)

        neighbor4 = self.get_neighbor4(pos)

        for neighbor in neighbor4:
            if board[neighbor] == Stone.EMPTY:
                self.string[string_id].add_liberty(neighbor, 0)
            elif board[neighbor] == opponent_color:
                neighbor_id = self.string_id[neighbor]
                self.string[string_id].add_neighbor(neighbor_id)
                self.string[neighbor_id].add_neighbor(string_id)

    def connect_string(self, board, pos, color, ids):
        """連を接続する。

        Args:
            board (Stone): 碁盤の交点の座標。
            pos (int): 追加する石の座標。
            color (Stone): 追加する石の色。
            ids (list[int]): 接続する連IDの候補列。
        """
        unique_ids = sorted(list(set(ids)))

        self.add_stone(board, pos, color, unique_ids[0])

        if len(unique_ids) > 1:
            self._merge_string(unique_ids[0], unique_ids[1:])

    def _merge_string(self, dst_id, src_ids):
        """複数の連を接続する。

        Args:
            dst_id (int): 接続先の連ID
            src_ids (list[int]): 接続元の連ID
        """
        for src_id in src_ids:
            self._merge_liberty(dst_id, src_id)
            self._merge_stones(dst_id, src_id)
            self._merge_neighbor(dst_id, src_id)
            self.string[src_id].remove()

    def _merge_stones(self, dst_id, src_id):
        """連を構成する石の座標を連結する。

        Args:
            dst_id (int): 接続先の連ID
            src_id (int): 接続元の連ID
        """
        dst_pos = self.string[dst_id].get_origin()
        src_pos = self.string[src_id].get_origin()

        if dst_pos > src_pos:
            pos = self.string_next[src_pos]
            self.string_next[src_pos] = dst_pos
            self.string_id[src_pos] = dst_id
            self.string[dst_id].set_origin(src_pos)
            dst_pos = src_pos
            src_pos = pos

        while src_pos != STRING_END:
            self.string_id[src_pos] = dst_id
            pos = self.string_next[src_pos]
            while self.string_next[dst_pos] < src_pos:
                dst_pos = self.string_next[dst_pos]

            self.string_next[src_pos] = self.string_next[dst_pos]
            self.string_next[dst_pos] = src_pos
            src_pos = pos

        self.string[dst_id].add_size(self.string[src_id].get_size())

    def _merge_liberty(self, dst_id, src_id):
        """連が持つ呼吸点の座標を連結する。

        Args:
            dst_id (int): 接続先の連ID
            src_id (int): 接続元の連ID
        """
        dst_lib = 0
        src_lib = 0

        while src_lib != LIBERTY_END:
            if not self.string[dst_id].has_liberty(src_lib):
                while self.string[dst_id].lib[dst_lib] < src_lib:
                    dst_lib = self.string[dst_id].lib[dst_lib]
                self.string[dst_id].lib[src_lib] = self.string[dst_id].lib[dst_lib]
                self.string[dst_id].lib[dst_lib] = src_lib
                self.string[dst_id].libs += 1
            src_lib = self.string[src_id].lib[src_lib]

    def _merge_neighbor(self, dst_id, src_id):
        """隣接する敵連IDを連結する。

        Args:
            dst_id (int): 接続先の連ID
            src_id (int): 接続元の連ID
        """
        src_neighbor = 0
        dst_neighbor = 0

        while src_neighbor != NEIGHBOR_END:
            if not self.string[dst_id].has_neighbor(src_neighbor):
                while self.string[dst_id].neighbor[dst_neighbor] < src_neighbor:
                    dst_neighbor = self.string[dst_id].neighbor[dst_neighbor]
                self.string[dst_id].neighbor[src_neighbor] = \
                    self.string[dst_id].neighbor[dst_neighbor]
                self.string[dst_id].neighbor[dst_neighbor] = src_neighbor
            src_neighbor = self.string[src_id].neighbor[src_neighbor]

        neighbor = self.string[src_id].get_neighbor_origin()
        while neighbor != NEIGHBOR_END:
            self._remove_neighbor_string(neighbor, src_id)
            self._add_neighbor(neighbor, dst_id)
            neighbor = self.string[src_id].neighbor[neighbor]


    def _remove_neighbor_string(self, neighbor_id, remove_id):
        """指定した敵連IDを削除する。

        Args:
            neighbor_id (int): 処理する連ID。
            remove_id (int): 削除する連ID。
        """
        self.string[neighbor_id].remove_neighbor(remove_id)

    def _add_neighbor(self, neighbor_id, add_id):
        """隣接する敵連IDを追加する。

        Args:
            neighbor_id (int): 処理する連ID。
            add_id (int): 追加する連ID。
        """
        self.string[neighbor_id].add_neighbor(add_id)

    def display(self):
        """盤上に存在する全ての連の情報を表示する。（デバッグ用）
        """
        coordinate = Coordinate(self.board_size)
        for string in self.string:
            if string.exist():
                # 連ID
                print_err(f"String ID : {self.string_id[string.get_origin()]}")
                # 座標
                position = "\tPosition :"
                pos = string.get_origin()
                while pos != STRING_END:
                    position += " " + coordinate.convert_to_gtp_format(pos)
                    pos = self.string_next[pos]
                print_err(position)
                color = string.get_color()
                liberties = string.get_liberties()
                neighbors = string.get_neighbors()
                if color is Stone.BLACK:
                    print_err("\tColor : Black")
                elif color is Stone.WHITE:
                    print_err("\tColor : White")
                else:
                    print_err("Error Color")
                liberty = ""
                for lib in liberties:
                    liberty += " " + coordinate.convert_to_gtp_format(lib)
                print_err(f"\tLiberty {len(liberties)} : {liberty}")
                neighbor = ""
                for nei in neighbors:
                    neighbor += " " + str(nei)
                print_err(f"\tNeighbor {len(neighbors)} : {neighbors}")
