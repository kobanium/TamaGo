"""モンテカルロ木探索の実装。
"""
from typing import Dict, List, NoReturn, Tuple
import copy
import time
import numpy as np
import torch

from board.constant import PASS
from board.go_board import GoBoard, copy_board
from board.stone import Stone
from common.print_console import print_err
from nn.feature import generate_input_planes
from nn.network.dual_net import DualNet
from mcts.batch_data import BatchQueue
from mcts.constant import NOT_EXPANDED, PLAYOUTS, NN_BATCH_SIZE
from mcts.node import MCTSNode


class MCTSTree:
    """モンテカルロ木探索の実装クラス。
    """
    def __init__(self, network: DualNet, tree_size=65536) -> NoReturn:
        """MCTSTreeクラスのコンストラクタ。

        Args:
            network (DualNet): 使用するニューラルネットワーク。
            tree_size (int, optional): 木を構成するノードの最大個数. Defaults to 65536.
        """
        self.node = [MCTSNode() for i in range(tree_size)]
        self.num_nodes = 0
        self.root = 0
        self.network = network
        self.batch_queue = BatchQueue()
        self.current_root = 0


    def search_best_move(self, board: GoBoard, color: Stone) -> int:
        """モンテカルロ木探索を実行して最善手を返す。

        Args:
            board (GoBoard): 評価する局面情報。
            color (Stone): 評価する局面の手番の色。

        Returns:
            int: 着手する座標。
        """
        self.num_nodes = 0

        start_time = time.time()

        self.current_root = self.expand_node(board, color)
        input_plane = generate_input_planes(board, color, 0)
        self.batch_queue.push(input_plane, [], self.current_root)

        self.process_mini_batch(board)

        # 候補手が1つしかない場合はPASSを返す
        if self.node[self.current_root].get_num_children() == 1:
            return PASS

        # 探索を実行する
        self.search(board, color)

        # 最善手を取得する
        next_move = self.node[self.current_root].get_best_move()

        # 探索結果と探索にかかった時間を表示する
        self.node[self.current_root].print_search_result(board)
        search_time = time.time() - start_time
        po_per_sec = self.node[self.current_root].node_visits / search_time
        print_err(f"{search_time:.2f} seconds, {po_per_sec:.2f}")

        return next_move


    def search(self, board: GoBoard, color: Stone) -> NoReturn:
        """探索を指定回数実行する。

        Args:
            board (GoBoard): 現在の局面情報。
            color (Stone): 現局面の手番の色。
        """
        search_board = copy.deepcopy(board)
        for _ in range(PLAYOUTS):
            copy_board(dst=search_board,src=board)
            start_color = color
            self.search_mcts(search_board, start_color, self.current_root, [])


    def search_mcts(self, board: GoBoard, color: Stone, current_index: int, \
        path: List[Tuple[int, int]]) -> NoReturn:
        """モンテカルロ木探索を実行する。

        Args:
            board (GoBoard): 現在の局面情報。
            color (Stone): 現局面の手番の色。
            current_index (int): 評価するノードのインデックス。
            path (List[Tuple[int, int]]): ルートからcurrent_indexに対応するノードに到達するまでの経路。
        """

        # UCB値最大の手を求める
        next_index = self.node[current_index].select_next_action()
        next_move = self.node[current_index].get_child_move(next_index)

        path.append((current_index, next_index))

        # 1手進める
        board.put_stone(pos=next_move, color=color)
        color = Stone.get_opponent_color(color)

        # Virtual Lossの加算
        self.node[current_index].add_virtual_loss(next_index)

        if self.node[current_index].children_visits[next_index] < 1:
            # ニューラルネットワークの計算

            input_plane = generate_input_planes(board, color, 0)
            next_node_index = self.node[current_index].get_child_index(next_index)
            self.batch_queue.push(input_plane, path, next_node_index)

            if len(self.batch_queue.node_index) >= NN_BATCH_SIZE:
                self.process_mini_batch(board)
        else:
            if self.node[current_index].get_child_index(next_index) == NOT_EXPANDED:
                child_index = self.expand_node(board, color)
                self.node[current_index].set_child_index(next_index, child_index)
            next_node_index = self.node[current_index].get_child_index(next_index)
            self.search_mcts(board, color, next_node_index, path)


    def expand_node(self, board: GoBoard, color: Stone) -> NoReturn:
        """ノードを展開する。

        Args:
            board (GoBoard): 現在の局面情報。
            color (Stone): 現在の手番の色。
        """
        node_index = self.num_nodes

        candidates = board.get_all_legal_pos(color)
        candidates.append(PASS)

        policy = get_tentative_policy(candidates)
        self.node[node_index].expand(policy)

        self.num_nodes += 1
        return node_index


    def process_mini_batch(self, board: GoBoard): # pylint: disable=R0914
        """ニューラルネットワークの入力をミニバッチ処理して、計算結果を探索結果に反映する。

        Args:
            board (GoBoard): 碁盤の情報。
        """

        input_planes = torch.Tensor(np.array(self.batch_queue.input_plane))

        raw_policy, value_data = self.network.inference(input_planes)

        policy_data = []
        for policy in raw_policy:
            policy_dict = {}
            for i, pos in enumerate(board.onboard_pos):
                policy_dict[pos] = policy[i]
            policy_dict[PASS] = policy[board.get_board_size() ** 2]
            policy_data.append(policy_dict)

        for policy, value_dist, path, node_index in zip(policy_data, \
            value_data, self.batch_queue.path, self.batch_queue.node_index):

            self.node[node_index].update_policy(policy)

            if path:
                value = value_dist[0] + value_dist[1] * 0.5

                reverse_path = list(reversed(path))
                leaf = reverse_path[0]

                self.node[leaf[0]].set_leaf_value(leaf[1], value)

                for index, child_index in reverse_path:
                    self.node[index].update_child_value(child_index, value)
                    self.node[index].update_node_value(value)
                    value = 1.0 - value

        self.batch_queue.clear()


def get_tentative_policy(candidates: List[int]) -> Dict[int, float]:
    """ニューラルネットワークの計算が行われるまでに使用するPolicyを取得する。

    Args:
        candidates (List[int]): パスを含む候補手のリスト。

    Returns:
        Dict[int, float]: 候補手の座標とPolicyの値のマップ。
    """
    score = np.random.dirichlet(alpha=np.ones(len(candidates)))
    return dict(zip(candidates, score))
