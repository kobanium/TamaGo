"""Go Text Protocolクライアントの実装。
"""
import os
import random
import sys
from typing import List, NoReturn

from program import PROGRAM_NAME, VERSION, PROTOCOL_VERSION
from board.constant import PASS, RESIGN
from board.coordinate import Coordinate
from board.go_board import GoBoard
from board.stone import Stone
from common.print_console import print_err
from gtp.gogui import GoguiAnalyzeCommand, display_policy_distribution, \
    display_policy_score
from mcts.time_manager import TimeControl, TimeManager
from mcts.tree import MCTSTree
from nn.policy_player import generate_move_from_policy
from nn.utility import load_network
from sgf.reader import SGFReader



class GtpClient: # pylint: disable=R0902,R0903
    """_Go Text Protocolクライアントの実装クラス
    """
    # pylint: disable=R0913
    def __init__(self, board_size: int, superko: bool, model_file_path: str, \
        use_gpu: bool, policy_move: bool, use_sequential_halving: bool, \
        komi: float, mode: TimeControl, visits: int, const_time: float, \
        time: float, batch_size: int): # pylint: disable=R0913
        """Go Text Protocolクライアントの初期化をする。

        Args:
            board_size (int): 碁盤の大きさ。
            superko (bool): 超劫判定の有効化。
            model_file_path (str): ネットワークパラメータファイルパス。
            use_gpu (bool): GPU使用フラグ。
            policy_move (bool): Policyの分布に従って着手するフラグ。
            use_sequential_halving (bool): Gumbel AlphaZeroの探索手法で着手生成するフラグ。
            komi (float): コミの値。
            mode (TimeControl): 思考時間制御モード。
            visits (int): 1手あたりの探索回数。
            const_time (float): 1手あたりの探索時間。
            time (float): 持ち時間。
            batch_size (int): 探索時のニューラルネットワークのミニバッチサイズ。
        """
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
            "load_sgf",
            "gogui-analyze_commands",
            "lz-analyze",
            "lz-genmove_analyze"
        ]
        self.superko = superko
        self.board = GoBoard(board_size=board_size, komi=komi, check_superko=superko)
        self.coordinate = Coordinate(board_size=board_size)
        self.gogui_analyze_command = [
            GoguiAnalyzeCommand("cboard", "Display policy distribution (Black)", \
                "display_policy_black_color"),
            GoguiAnalyzeCommand("cboard", "Display policy distribution (White)", \
                "display_policy_white_color"),
            GoguiAnalyzeCommand("sboard", "Display policy score (Black)", \
                "display_policy_black"),
            GoguiAnalyzeCommand("sboard", "Display policy score (White)", \
                "display_policy_white"),
        ]
        self.policy_move = policy_move
        self.use_sequential_halving = use_sequential_halving
        self.use_network = False

        if mode is TimeControl.CONSTANT_PLAYOUT:
            self.time_manager = TimeManager(mode=mode, constant_visits=visits)
        if mode is TimeControl.CONSTANT_TIME:
            self.time_manager = TimeManager(mode=mode, constant_time=const_time)
        if mode is TimeControl.TIME_CONTROL:
            self.time_manager = TimeManager(mode=mode, remaining_time=time)

        try:
            self.network = load_network(model_file_path, use_gpu)
            self.use_network = True
            self.mcts = MCTSTree(network=self.network, batch_size=batch_size)
        except FileNotFoundError:
            print_err(f"Model file {model_file_path} is not found")
        except RuntimeError:
            print_err(f"Failed to load {model_file_path}")


    def _known_command(self, command: str) -> NoReturn:
        """known_commandコマンドを処理する。
        対応しているコマンドの場合は'true'を表示し、対応していないコマンドの場合は'unknown command'を表示する

        Args:
            command (str): 対応確認をしたいGTPコマンド。
        """
        if command in self.gtp_commands:
            respond_success("true")
        else:
            respond_failure("unknown command")

    def _list_commands(self) -> NoReturn:
        """list_commandsコマンドを処理する。
        対応している全てのコマンドを表示する。
        """
        response = ""
        for command in self.gtp_commands:
            response += '\n' + command
        respond_success(response)

    def _komi(self, s_komi: str) -> NoReturn:
        """komiコマンドを処理する。
        入力されたコミを設定する。

        Args:
            s_komi (str): 設定するコミ。
        """
        komi = float(s_komi)
        self.board.set_komi(komi)
        respond_success("")

    def _play(self, color: str, pos: str) -> NoReturn:
        """playコマンドを処理する。
        入力された座標に指定された色の石を置く。

        Args:
            color (str): 手番の色。
            pos (str): 着手する座標。
        """
        if color.lower()[0] == 'b':
            play_color = Stone.BLACK
        elif color.lower()[0] == 'w':
            play_color = Stone.WHITE
        else:
            respond_failure("play color pos")
            return

        coord = self.coordinate.convert_from_gtp_format(pos)

        if coord != PASS and not self.board.is_legal(coord, play_color):
            print(f"illigal {color} {pos}")

        if pos.upper != "RESIGN":
            self.board.put_stone(coord, play_color)

        respond_success("")

    def _genmove(self, color: str) -> NoReturn:
        """genmoveコマンドを処理する。
        入力された手番で思考し、着手を生成する。

        Args:
            color (str): 手番の色。
        """
        if color.lower()[0] == 'b':
            genmove_color = Stone.BLACK
        elif color.lower()[0] == 'w':
            genmove_color = Stone.WHITE
        else:
            respond_failure("genmove color")
            return

        if self.use_network:
            if self.policy_move:
                # Policy Networkから着手生成
                pos = generate_move_from_policy(self.network, self.board, genmove_color)
                _, previous_move, _ = self.board.record.get(self.board.moves - 1)
                if self.board.moves > 1 and previous_move == PASS:
                    pos = PASS
            else:
                # モンテカルロ木探索で着手生成
                if self.use_sequential_halving:
                    pos = self.mcts.generate_move_with_sequential_halving(self.board, \
                        genmove_color, self.time_manager, False)
                else:
                    pos = self.mcts.search_best_move(self.board, genmove_color, self.time_manager)
        else:
            # ランダムに着手生成
            legal_pos = [pos for pos in self.board.onboard_pos \
                if self.board.is_legal_not_eye(pos, genmove_color)]
            if legal_pos:
                pos = random.choice(legal_pos)
            else:
                pos = PASS

        if pos != RESIGN:
            self.board.put_stone(pos, genmove_color)

        respond_success(self.coordinate.convert_to_gtp_format(pos))

    def _boardsize(self, size: str) -> NoReturn:
        """boardsizeコマンドを処理する。
        指定したサイズの碁盤に設定する。

        Args:
            size (str): 設定する碁盤のサイズ。
        """
        board_size = int(size)
        self.board = GoBoard(board_size=board_size, check_superko=self.superko)
        self.coordinate = Coordinate(board_size=board_size)
        self.time_manager.initialize()
        respond_success("")

    def _clear_board(self) -> NoReturn:
        """clear_boardコマンドを処理する。
        盤面を初期化する。
        """
        self.board.clear()
        self.time_manager.initialize()
        respond_success("")

    def _time_settings(self, arg_list: List[str]) -> NoReturn:
        """time_settingsコマンドを処理する。
        持ち時間のみを設定する。

        Args:
            arg_list (List[str]): コマンドの引数リスト（持ち時間、秒読み、秒読みの手数）。
        """
        time = float(arg_list[0])
        self.time_manager.set_remaining_time(Stone.BLACK, time)
        self.time_manager.set_remaining_time(Stone.WHITE, time)
        respond_success("")

    def _time_left(self, arg_list: List[str]) -> NoReturn:
        """time_leftコマンドを処理する。
        指定した手番の残りの時間を設定する。

        Args:
            arg_list (List[str]): コマンドの引数リスト（手番の色、残り時間）。
        """
        if arg_list[0][0] in ['B', 'b']:
            color = Stone.BLACK
        elif arg_list[0][0] in ['W', 'w']:
            color = Stone.WHITE
        else:
            respond_failure("invalid color")

        self.time_manager.set_remaining_time(color, float(arg_list[1]))
        respond_success("")

    def _get_komi(self) -> NoReturn:
        """get_komiコマンドを処理する。
        """
        respond_success(str(self.board.get_komi()))

    def _showboard(self) -> NoReturn:
        """showboardコマンドを処理する。
        現在の盤面を表示する。
        """
        self.board.display()
        respond_success("")

    def _load_sgf(self, arg_list: List[str]) -> NoReturn:
        """load_sgfコマンドを処理する。
        指定したSGFファイルの指定手番まで進めた局面にする。

        Args:
            arg_list (List[str]): コマンドの引数リスト（ファイル名（必須）、手数（任意））
        """
        if not os.path.exists(arg_list[0]):
            respond_failure(f"cannot load {arg_list[0]}")

        sgf_data = SGFReader(arg_list[0], board_size=self.board.get_board_size())

        if len(arg_list) < 2:
            moves = sgf_data.get_n_moves()
        else:
            moves = int(arg_list[1])

        self.board.clear()

        for i in range(moves):
            pos = sgf_data.get_move_data(i)
            color = sgf_data.get_color(i)
            self.board.put_stone(pos, color)

        respond_success("")

    def _lz_analyze(self, arg_list: List[str]) -> NoReturn:
        color = arg_list[0]
        interval = 0
        if len(arg_list) >= 2:
            interval = int(arg_list[1])/100

        if color.lower()[0] == 'b':
            to_move = Stone.BLACK
        elif color.lower()[0] == 'w':
            to_move = Stone.WHITE
        else:
            respond_failure("lz-analyze color")
            return

        analysis_query = {
            "mode" : "lz",
            "interval" : interval,
            "ponder" : True
        }
        self.mcts.ponder(self.board, to_move, analysis_query)

    def _lz_genmove_analyze(self, arg_list: List[str]) -> int:
        color = arg_list[0]
        interval = 0
        if len(arg_list) >= 2:
            interval = int(arg_list[1])/100

        if color.lower()[0] == 'b':
            genmove_color = Stone.BLACK
        elif color.lower()[0] == 'w':
            genmove_color = Stone.WHITE
        else:
            respond_failure("genmove color")
            return

        if self.use_network:
            if self.policy_move:
                # Policy Networkから着手生成
                pos = generate_move_from_policy(self.network, self.board, genmove_color)
                _, previous_move, _ = self.board.record.get(self.board.moves - 1)
                if self.board.moves > 1 and previous_move == PASS:
                    pos = PASS
            else:
                # モンテカルロ木探索で着手生成
                if self.use_sequential_halving:
                    pos = self.mcts.generate_move_with_sequential_halving(self.board, \
                        genmove_color, self.time_manager, False)
                else:
                    analysis_query = {
                        "mode" : "lz",
                        "interval" : interval,
                        "ponder" : False
                    }
                    pos = self.mcts.search_best_move(self.board, genmove_color, \
                        self.time_manager, analysis_query)
        else:
            # ランダムに着手生成
            legal_pos = [pos for pos in self.board.onboard_pos \
                if self.board.is_legal_not_eye(pos, genmove_color)]
            if legal_pos:
                pos = random.choice(legal_pos)
            else:
                pos = PASS

        if pos != RESIGN:
            self.board.put_stone(pos, genmove_color)
        return pos


    def run(self) -> NoReturn: # pylint: disable=R0912,R0915
        """Go Text Protocolのクライアントの実行処理。
        入力されたコマンドに対応する処理を実行し、応答メッセージを表示する。
        """
        while True:
            command = input()

            command_list = command.rstrip().split(' ')

            input_gtp_command = command_list[0]

            if input_gtp_command == "version":
                _version()
            elif input_gtp_command == "protocol_version":
                _protocol_version()
            elif input_gtp_command == "name":
                _name()
            elif input_gtp_command == "quit":
                _quit()
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
                self._time_settings(command_list[1:])
            elif input_gtp_command == "time_left":
                self._time_left(command_list[1:])
            elif input_gtp_command == "get_komi":
                self._get_komi()
            elif input_gtp_command == "showboard":
                self._showboard()
            elif input_gtp_command == "load_sgf":
                self._load_sgf(command_list[1:])
            elif input_gtp_command == "final_score":
                respond_success("?")
            elif input_gtp_command == "showstring":
                self.board.strings.display()
                respond_success("")
            elif input_gtp_command == "showpattern":
                coordinate = Coordinate(self.board.get_board_size())
                self.board.pattern.display(coordinate.convert_from_gtp_format(command_list[1]))
                respond_success("")
            elif input_gtp_command == "eye":
                coordinate = Coordinate(self.board.get_board_size())
                coord = coordinate.convert_from_gtp_format(command_list[1])
                print_err(self.board.pattern.get_eye_color(coord))
            elif input_gtp_command == "gogui-analyze_commands":
                response = ""
                for cmd in self.gogui_analyze_command:
                    response += cmd.get_command_information() + '\n'
                respond_success(response)
            elif input_gtp_command == "display_policy_black_color":
                respond_success(display_policy_distribution(
                    self.network, self.board, Stone.BLACK))
            elif input_gtp_command == "display_policy_white_color":
                respond_success(display_policy_distribution(
                    self.network, self.board, Stone.WHITE))
            elif input_gtp_command == "display_policy_black":
                respond_success(display_policy_score(
                    self.network, self.board, Stone.BLACK
                ))
            elif input_gtp_command == "display_policy_white":
                respond_success(display_policy_score(
                    self.network, self.board, Stone.WHITE
                ))
            elif input_gtp_command == "self-atari":
                self.board.display_self_atari(Stone.BLACK)
                self.board.display_self_atari(Stone.WHITE)
                respond_success("")
            elif input_gtp_command == "lz-analyze":
                print("= ")
                self._lz_analyze(command_list[1:])
                print("")
            elif input_gtp_command == "lz-genmove_analyze":
                print("= ")
                pos = self._lz_genmove_analyze(command_list[1:])
                print("play {}\n".format(self.coordinate.convert_to_gtp_format(pos)))
            else:
                respond_failure("unknown_command")

def respond_success(response: str) -> NoReturn:
    """コマンド処理成功時の応答メッセージを表示する。

    Args:
        response (str): 表示する応答メッセージ。
    """
    print("= " + response + '\n')

def respond_failure(response: str) -> NoReturn:
    """コマンド処理失敗時の応答メッセージを表示する。

    Args:
        response (str): 表示する応答メッセージ。
    """
    print("= ? " + response + '\n')

def _version() -> NoReturn:
    """versionコマンドを処理する。
    プログラムのバージョンを表示する。
    """
    respond_success(VERSION)

def _protocol_version() -> NoReturn:
    """protocol_versionコマンドを処理する。
    GTPのプロトコルバージョンを表示する。
    """
    respond_success(PROTOCOL_VERSION)

def _name() -> NoReturn:
    """nameコマンドを処理する。
    プログラム名を表示する。
    """
    respond_success(PROGRAM_NAME)

def _quit() -> NoReturn:
    """quitコマンドを処理する。
    プログラムを終了する。
    """
    respond_success("")
    sys.exit(0)
