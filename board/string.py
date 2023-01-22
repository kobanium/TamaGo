from board.constant import STRING_END, LIBERTY_END, NEIGHBOR_END, OB_SIZE
from board.coordinate import Coordinate
from board.stone import Stone
from common.print_console import print_err

class String:
    
    def __init__(self, board_size):
        self.color = 0
        self.libs = 0
        self.lib = [0] * ((board_size + 2) ** 2)
        self.neighbors = 0
        self.neighbor = [0] * int(0.8 * board_size * (board_size - 1) + 5)
        self.origin = 0
        self.size = 0
        self.flag = False

    def initialize(self, pos, color):
        for i in range(len(self.lib)):
            self.lib[i] = 0

        for i in range(len(self.neighbor)):
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
        return self.lib[pos] > 0

    def has_neighbor(self, neighbor):
        return self.neighbor[neighbor] != 0

    def remove(self):
        self.flag = False
        
    def add_stone(self):
        self.size += 1

    def add_liberty(self):
        self.libs += 1

    def add_size(self, size):
        self.size += size

    def get_size(self):
        return self.size

    def exist(self):
        return self.flag

    def get_num_liberties(self):
        return self.libs
    
    def get_origin(self):
        return self.origin

    def get_neighbor_origin(self):
        return self.neighbor[0]

    def set_origin(self, pos):
        self.origin = pos

    def add_liberty(self, pos, head):
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
        if not self.has_liberty(pos):
            return

        lib = 0
        while self.lib[lib] != pos:
            lib = self.lib[lib]

        self.lib[lib] = self.lib[self.lib[lib]]
        self.lib[pos] = 0
        self.libs -= 1
        
    
    def add_neighbor(self, string_id):
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
        if self.neighbor[remove_id] == 0:
            return

        neighbor = 0
        while self.neighbor[neighbor] != remove_id:
            neighbor = self.neighbor[neighbor]

        self.neighbor[neighbor] = self.neighbor[self.neighbor[neighbor]]
        self.neighbor[remove_id] = 0
        self.neighbors -= 1


    def get_color(self):
        return self.color

    def get_liberties(self):
        liberties = []
        lib = self.lib[0]
        while lib != LIBERTY_END:
            liberties.append(lib)
            lib = self.lib[lib]
        return liberties

    def get_neighbors(self):
        neighbors = []
        neighbor = self.neighbor[0]
        while neighbor != NEIGHBOR_END:
            neighbors.append(neighbor)
            neighbor = self.neighbor[neighbor]
        return neighbors
        
        
        
class StringData:
    def __init__(self, board_size):
        self.board_max = (board_size + 2) ** 2
        self.string = [String(board_size=board_size) for i in range(int(0.8 * board_size * (board_size - 1) + 5))]
        self.string_id = [0] * self.board_max
        self.string_next = [0] * self.board_max
        self.board_size = board_size
        self.board_size_with_ob = board_size + OB_SIZE * 2
        self.coordinate = Coordinate(board_size=board_size)

    def clear(self):
        for string in self.string:
            string.remove()

    def remove_liberty(self, pos, lib):
        self.string[self.get_id(pos)].remove_liberty(lib)

    def remove_string(self, board, remove_pos):
        remove_id = self.get_id(remove_pos)

        pos = self.string[remove_id].get_origin()

        while pos != STRING_END:
            board[pos] = Stone.EMPTY

            neighbor4 = [pos - self.board_size_with_ob,
                         pos - 1,
                         pos + 1,
                         pos + self.board_size_with_ob]

            for neighbor_pos in neighbor4:
                neighbor_id = self.get_id(neighbor_pos)
                if self.string[neighbor_id].exist():
                    self.string[neighbor_id].add_liberty(pos, 0)

            next_pos = self.string_next[pos]
            self.string_next[pos] = 0
            self.string_id[pos] = 0
            pos = next_pos
            #if pos == STRING_END:
            #    break

        neighbor_id = self.string[remove_id].neighbor[0]
        while neighbor_id != NEIGHBOR_END:
            self._remove_neighbor_string(neighbor_id, remove_id)
            neighbor_id = self.string[remove_id].neighbor[neighbor_id]
        
        self.string[remove_id].remove()

        return self.string[remove_id].get_size()

    def get_id(self, pos):
        return self.string_id[pos]

    def get_num_liberties(self, pos):
        return self.string[self.get_id(pos)].get_num_liberties()


    def make_string(self, board, pos, color):
        opponent_color = Stone.get_opponent_color(color)
        lib_add = 0

        string_id = 1

        while self.string[string_id].exist():
            string_id += 1

        self.string[string_id].initialize(pos, color)
        self.string_id[pos] = string_id
        self.string_next[pos] = STRING_END

        neighbor4 = [pos - self.board_size_with_ob,
                     pos - 1, pos + 1,
                     pos + self.board_size_with_ob]

        for p in neighbor4:
            if board[p] == Stone.EMPTY:
                lib_add = self.string[string_id].add_liberty(p, lib_add)
            elif board[pos] == opponent_color:
                neighbor_id = self.string_id[p]
                self.string[string_id].add_neighbor(neighbor_id)
                self.string[neighbor_id].add_neighbor(string_id)

    def _add_stone_to_string(self, string_id, pos):
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
        opponent_color = Stone.get_opponent_color(color)
        self.string_id[pos] = string_id

        self._add_stone_to_string(string_id, pos)

        neighbor4 = [pos - self.board_size_with_ob,
                     pos - 1, pos + 1,
                     pos + self.board_size_with_ob]

        for p in neighbor4:
            if board[p] == Stone.EMPTY:
                self.string[string_id].add_liberty(p, 0)
            elif board[p] == opponent_color:
                neighbor_id = self.string_id[p]
                self.string[string_id].add_neighbor(neighbor_id)
                self.string[neighbor_id].add_neighbor(string_id)

    def connect_string(self, board, pos, color, ids):
        unique_ids = sorted(list(set(ids)))

        self.add_stone(board, pos, color, unique_ids[0])

        if len(unique_ids) > 1:
            self._merge_string(unique_ids[0], unique_ids[1:])
        

    def _merge_string(self, dst_id, src_ids):
        for src_id in src_ids:
            rm_id = self.get_id(self.string[src_id].get_origin())
            self._merge_liberty(dst_id, src_id)
            self._merge_stones(dst_id, src_id)
            self._merge_neighbor(dst_id, src_id)
            self.string[src_id].remove()

    def _merge_stones(self, dst_id, src_id):
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
        src_neighbor = 0
        dst_neighbor = 0

        while src_neighbor != NEIGHBOR_END:
            if not self.string[dst_id].has_neighbor(src_neighbor):
                while self.string[dst_id].neighbor[dst_neighbor] < src_neighbor:
                    dst_neighbor = self.string[dst_id].neighbor[dst_neighbor]
                self.string[dst_id].neighbor[src_neighbor] = self.string[dst_id].neighbor[dst_neighbor]
                self.string[dst_id].neighbor[dst_neighbor] = src_neighbor
            src_neighbor = self.string[src_id].neighbor[src_neighbor]

        neighbor = self.string[src_id].get_neighbor_origin()
        while neighbor != NEIGHBOR_END:
            self._remove_neighbor_string(neighbor, src_id)
            self._add_neighbor(neighbor, dst_id)
            neighbor = self.string[src_id].neighbor[neighbor]


    def _remove_neighbor_string(self, neighbor_id, remove_id):
        self.string[neighbor_id].remove_neighbor(remove_id)

    def _add_neighbor(self, neighbor_id, add_id):
        self.string[neighbor_id].add_neighbor(add_id)

    def display(self):
        for string in self.string:
            if string.exist():
                # 連ID
                print_err("String ID : {}".format(self.string_id[string.get_origin()]))
                # 座標
                position = "\tPosition :"
                pos = string.get_origin()
                while pos != STRING_END:
                    position += " " + self.coordinate.convert_to_gtp_format(pos)
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
                    liberty += " " + self.coordinate.convert_to_gtp_format(lib)
                print_err("\tLiberty {} : {}".format(len(liberties), liberty))
                neighbor = ""
                for nei in neighbors:
                    neighbor += " " + str(nei)
                print_err("\tNeighbor {} : {}".format(len(neighbors), neighbors))
