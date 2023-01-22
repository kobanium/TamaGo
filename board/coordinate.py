from board.constant import PASS, RESIGN, OB_SIZE, GTP_X_COORDINATE

class Coordinate:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board_size_with_ob = board_size + OB_SIZE * 2


    def convert_from_gtp_format(self, pos):
        if pos.upper() == "PASS":
            return PASS
        elif pos.upper() == "RESIGN":
            return RESIGN
        else:
            alphabet = pos.upper()[0]
            x = 0
            for i in range(self.board_size):
                if GTP_X_COORDINATE[i + 1] is alphabet:
                    x = i
            y = self.board_size - int(pos[1:])
            
            pos = x + OB_SIZE + (y + OB_SIZE) * self.board_size_with_ob

            return pos

    def convert_to_gtp_format(self, pos):
        if pos == PASS:
            return "PASS"
        elif pos == RESIGN:
            return "RESIGN"
        else:
            x = pos % self.board_size_with_ob - OB_SIZE + 1
            y = self.board_size - (pos // self.board_size_with_ob - OB_SIZE)
            return (GTP_X_COORDINATE[x] + str(y))
