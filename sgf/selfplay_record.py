"""強化学習用の自己対戦データの記録と出力
"""
from typing import NoReturn
import os

from board.constant import MAX_RECORDS
from board.coordinate import Coordinate
from board.stone import Stone
from mcts.node import MCTSNode
from program import PROGRAM_NAME


class SelfPlayRecord:
    """自己対戦のデータの記録と出力をするクラス
    """

    def __init__(self, save_dir: str, coord: Coordinate):
        """SelfPlayRecordクラスのコンストラクタ。

        Args:
            save_dir (str): 保存先のディレクトリパス。
            coord (Coordinate): 座標変換処理クラスのインスタンス。
        """
        self.record_moves = 0
        self.color = [Stone.EMPTY] * MAX_RECORDS
        self.pos = [0] * MAX_RECORDS
        self.coord = coord
        self.policy_target = [""] * MAX_RECORDS
        self.save_dir = save_dir
        self.file_index = 1

    def clear(self) -> NoReturn:
        """レコードの初期化。
        """
        self.record_moves = 0

    def set_index(self, index: int) -> NoReturn:
        """ファイルのインデックスを設定する。

        Args:
            index (int): 出力ファイルのインデックス。
        """
        self.file_index = index

    def save_record(self, root: MCTSNode, pos: int, color: Stone) -> NoReturn:
        """着手とImproved Policyを記録する。

        Args:
            root (MCTSNode): 探索実行時のルートのデータ。
            pos (int): 着手した座標。
            color (Stone): 手番の色。
        """
        self.color[self.record_moves] = color
        self.pos[self.record_moves] = self.coord.convert_to_sgf_format(pos)

        improved_policy = root.calculate_improved_policy()

        policy_target = f"{root.get_num_children()}"
        for i in range(root.get_num_children()):
            pos = self.coord.convert_to_gtp_format(root.get_child_move(i))
            policy_target += f" {pos}:{improved_policy[i]:.3e}"

        self.policy_target[self.record_moves] = policy_target

        self.record_moves += 1

    def write_record(self, winner: Stone, komi: float, is_resign: bool, score: float) -> NoReturn:
        """自己対戦のファイルを出力する。

        Args:
            winner (Stone): 勝った手番の色。
            komi (float): 対局実行時のコミ。
            is_resign (bool): 投了による決着か否か。
            score (float): 黒から見た目数。
        """
        sgf_string = f"(;FF[4]GM[1]SZ[{self.coord.board_size}]\n"
        sgf_string += f"AP[{PROGRAM_NAME}]"
        sgf_string += f"PB[{PROGRAM_NAME}-Black]"
        sgf_string += f"PW[{PROGRAM_NAME}-White]"

        if winner is Stone.BLACK:
            if is_resign:
                sgf_string += "RE[B+R]"
            else:
                sgf_string += f"RE[B+{score:.1f}]"
        elif winner is Stone.WHITE:
            if is_resign:
                sgf_string += "RE[W+R]"
            else:
                sgf_string += f"RE[W+{-score:.1f}]"
        else:
            sgf_string += "RE[0]"

        sgf_string += f"KM[{komi}]"

        for i in range(self.record_moves):
            if self.color[i] is Stone.BLACK:
                sgf_string += f";B[{self.pos[i]}]"
            else:
                sgf_string += f";W[{self.pos[i]}]"
            sgf_string += "C[" + self.policy_target[i] + "]"

        sgf_string += "\n)"

        out_file_path = os.path.join(self.save_dir, f"{self.file_index}.sgf")

        with open(out_file_path, mode='w', encoding="utf-8") as out_file:
            out_file.write(sgf_string)

        self.file_index += 1
