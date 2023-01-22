import random
import sys

from program import PROGRAM_NAME, VERSION, PROTOCOL_VERSION
from board.constant import BOARD_SIZE, PASS, RESIGN
from board.coordinate import Coordinate
from board.go_board import GoBoard
from board.stone import Stone



class GtpClient:
    def __init__(self, board_size):
        self.gtp_commands = [
            "version",
            "protocol_version",
            "name",
            "quit",
            "known_command",
            "list_commands",
            "play",
            "genmove",
            "clear_board",
            "boardsize",
            "time_left",
            "time_settings",
            "get_komi",
            "komi",
            "showboard",
        ]
        self.board = GoBoard(board_size=board_size)
        self.coordinate = Coordinate(board_size=board_size)


    def _respond_success(self, response):
        print("= " + response + '\n')

    def _respond_failure(self, response):
        print("= ? " + response + '\n')
        
    def _version(self):
        self._respond_success(VERSION)

    def _protocol_version(self):
        self._respond_success(PROTOCOL_VERSION)

    def _name(self):
        self._respond_success(PROGRAM_NAME)

    def _quit(self):
        self._respond_success("")
        sys.exit(0)

    def _known_command(self, command):
        if command in self.gtp_commands:
            self._respond_success("true")
        else:
            self._respond_failure("unknown command")

    def _list_commands(self):
        response = ""
        for command in self.gtp_commands:
            response += '\n' + command
        self._respond_success(response)

    def _komi(self, s_komi):
        komi = float(s_komi)
        self._respond_success("")

    def _play(self, color, pos):
        if color.lower()[0] is 'b':
            play_color = Stone.BLACK
        elif color.lower()[0] is 'w':
            play_color = Stone.WHITE
        else:
            self._respond_failure("play color pos")
            return
        
        coord = self.coordinate.convert_from_gtp_format(pos)

        if not self.board.is_legal(coord, play_color):
            print("illigal {} {}".format(color, pos))

        if pos.upper != "RESIGN":
            self.board.put_stone(coord, play_color)

        self._respond_success("")

    def _genmove(self, color):
        if color.lower()[0] is 'b':
            genmove_color = Stone.BLACK
        elif color.lower()[0] is 'w':
            genmove_color = Stone.WHITE
        else:
            self._respond_failure("genmove color")
            return
        

        legal_pos = self.board.get_all_legal_pos(genmove_color)

        if len(legal_pos) > 0:
            pos = random.choice(legal_pos)
        else:
            pos = PASS

        if pos != RESIGN:
            self.board.put_stone(pos, genmove_color)

        self._respond_success(self.coordinate.convert_to_gtp_format(pos))

    def _boardsize(self, size):
        board_size = int(size)
        self.board = GoBoard(board_size=board_size)
        self.coordinate = Coordinate(board_size=board_size)
        self._respond_success("")

    def _clear_board(self):
        self.board.clear()
        self._respond_success("")

    def _time_settings(self):
        self._respond_success("")

    def _time_left(self):
        self._respond_success("")

    def _get_komi(self):
        self._respond_success("7.0")

    def _showboard(self):
        self.board.display()
        self._respond_success("")

            
    def run(self):
        while True:
            command = input()

            command_list = command.split(' ')

            input_gtp_command = command_list[0]
            
            if input_gtp_command == "version":
                self._version()
            elif input_gtp_command == "protocol_version":
                self._protocol_version()
            elif input_gtp_command == "name":
                self._name()
            elif input_gtp_command == "quit":
                self._quit()
            elif input_gtp_command == "known_command":
                self._known_command(command_list[1])
            elif input_gtp_command == "list_commands":
                self._list_commands()
            elif input_gtp_command == "komi":
                self._komi(command_list[1])
            elif input_gtp_command == "play":
                self._play(command_list[1], command_list[2])
            elif input_gtp_command == "genmove":
                self._genmove(command_list[1])
            elif input_gtp_command == "boardsize":
                self._boardsize(command_list[1])
            elif input_gtp_command == "clear_board":
                self._clear_board()
            elif input_gtp_command == "time_settings":
                self._time_settings()
            elif input_gtp_command == "time_left":
                self._time_left()
            elif input_gtp_command == "get_komi":
                self._get_komi()
            elif input_gtp_command == "showboard":
                self._showboard()
            elif input_gtp_command == "showstring":
                self.board.strings.display()
                self._respond_success("")
            else:
                self._respond_failure("unknown_command")

