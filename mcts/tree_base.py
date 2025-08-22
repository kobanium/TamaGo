"""モンテカルロ木探索用の基底クラス定義
"""
from typing import Any, Dict, List
import sys

from board.constant import PASS
from board.coordinate import Coordinate
from board.go_board import GoBoard
from board.stone import Stone
from mcts.constant import MCTS_TREE_SIZE, NN_BATCH_SIZE, NOT_EXPANDED
from mcts.dump import dump_mcts_to_json
from mcts.node import MCTSNode
from nn.tentative_policy import get_tentative_policy


class MCTSTreeBase:
    """モンテカルロ木探索の基底クラス。
    """

    def __init__(
            self,
            tree_size: int = MCTS_TREE_SIZE,
            batch_size: int = NN_BATCH_SIZE,
            cgos_mode: bool = False
    ):
        self.node = [MCTSNode() for i in range(tree_size)]
        self.num_nodes = 0
        self.root = 0
        # MCTSTree Class
        #self.network = network
        #self.batch_queue = BatchQueue()
        # MCTSTreeAsync Class
        #self.nneval = nneval
        self.current_root = 0
        self.batch_size = batch_size
        self.cgos_mode = cgos_mode
        self.to_move = Stone.BLACK

        
    def expand_node(self, board: GoBoard, color: Stone) -> int:
        """ノードを展開する。

        Args:
            board (GoBoard): 現在の局面情報。
            color (Stone): 現在の手番の色。
        """
        node_index = self.num_nodes
        tree_size = len(self.node)
        if node_index >= tree_size:
            self.node.extend([MCTSNode() for i in range(tree_size)])
            sys.stderr.write(f"Tree is full. Allocate new space {tree_size} -> {len(self.node)}\n")

        candidates = board.get_all_legal_pos(color)
        candidates = [candidate for candidate in candidates \
            if (board.check_self_atari_stone(candidate, color) < 7) \
                and not board.is_complete_eye(candidate, color)]
        candidates.append(PASS)

        policy = get_tentative_policy(candidates)
        self.node[node_index].expand(policy)

        self.num_nodes += 1
        return node_index


    def get_root(self) -> MCTSNode:
        """木のルートを返す。

        Returns:
            MCTSNode: モンテカルロ木探索で使用する木のルート。
        """
        return self.node[self.current_root]


    def get_pv_lists(self, root: MCTSNode, coord: Coordinate) -> Dict[str, List[str]]:
        """探索した手の最善応手系列を取得する。

        Args:
            coordinate (Coordinate): 座標変換処理インスタンス。

        Returns:
            Dict[str, List[str]]: 各手の最善応手系列を記録した辞書。
        """
        pv_dict: Dict[str, List[str]] = {}

        for i in range(root.num_children):
            if root.children_visits[i] > 0:
                pv_list = self.get_best_move_sequence(
                    [root.action[i]], root.children_index[i]
                )
                pv_dict[coord.convert_to_gtp_format(root.action[i])] = [
                    coord.convert_to_gtp_format(pv) for pv in pv_list
                ]

        return pv_dict


    def get_best_move_sequence(self, pv_list: List[int], index: int) -> List[int]:
        """最善応手系列を取得する。

        Args:
            pv_list (List[str]): 今までの経路の最善応手系列。
            index (int): ノードのインデックス。

        Returns:
            List[str]: 最善応手系列。
        """
        node = self.node[index]

        if node.node_visits == 0:
            return pv_list

        next_index = node.get_child_index(node.get_best_move_index())
        next_action = node.get_best_move()
        pv_list.append(next_action)

        if next_index == NOT_EXPANDED:
            return pv_list

        return self.get_best_move_sequence(pv_list, next_index)


    def dump_to_json(self, board: GoBoard, superko: bool) -> str:
        """MCTSの状態を表すJSON文字列を返す。

        Args:
            board (GoBoard): 現在の碁盤。
            superko (bool): 超劫判定の有効化。

        Returns:
            str: MCTSの状態を表すJSON文字列。
        """
        return dump_mcts_to_json(self.to_dict(), board, superko)



    def to_dict(self) -> Dict[str, Any]:
        """ツリーの状態を辞書化して返す。

        Returns:
            Dict[str, Any]: ツリーの状態を表す辞書。
        """
        state = {
            "node": [self.node[i].to_dict() for i in range(self.num_nodes)],
            "num_nodes": self.num_nodes,
            "root": self.root,
            "current_root": self.current_root,
            "batch_size": self.batch_size,
            "cgos_mode": self.cgos_mode,
            "to_move": 'black' if self.to_move == Stone.BLACK else 'white',
        }
        return state
