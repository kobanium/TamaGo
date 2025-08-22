"""モンテカルロ木探索の実装。
"""
from typing import Any, Dict, List, Tuple, Callable
import sys
import select
import copy
import time
import numpy as np
import torch

from board.constant import PASS, RESIGN
from board.go_board import GoBoard, copy_board
from board.stone import Stone
from common.print_console import print_err
from mcts.batch_data import BatchQueue
from mcts.constant import NOT_EXPANDED, PLAYOUTS, NN_BATCH_SIZE, \
    MAX_CONSIDERED_NODES, RESIGN_THRESHOLD, MCTS_TREE_SIZE
from mcts.sequential_halving import get_candidates_and_visit_pairs
from mcts.tree_base import MCTSTreeBase
from mcts.time_manager import TimeControl, TimeManager
from nn.feature import generate_input_planes
from nn.network.dual_net import DualNet


class MCTSTree(MCTSTreeBase): # pylint: disable=R0902
    """モンテカルロ木探索の実装クラス。
    """
    def __init__(self, network: DualNet, tree_size: int=MCTS_TREE_SIZE, \
        batch_size: int=NN_BATCH_SIZE, cgos_mode: bool=False):
        """MCTSTreeクラスのコンストラクタ。

        Args:
            network (DualNet): 使用するニューラルネットワーク。
            tree_size (int, optional): 木を構成するノードの最大個数。デフォルトは65536。
            batch_size (int, optional): ニューラルネットワークの前向き伝搬処理のミニバッチサイズ。デフォルトはNN_BATCH_SIZE。
        """
        super().__init__(tree_size=tree_size, batch_size=batch_size, cgos_mode=cgos_mode)
        self.network = network
        self.batch_queue = BatchQueue()


    def _initialize_search(self, board: GoBoard, color: Stone) -> None:
        self.num_nodes = 0
        self.current_root = self.expand_node(board, color)
        input_plane = generate_input_planes(board, color, 0)
        self.batch_queue.push(input_plane, [], self.current_root)
        self.process_mini_batch(board)


    def search_best_move(self, board: GoBoard, color: Stone, time_manager: TimeManager, \
        analysis_query: Dict[str, Any]) -> int:
        """モンテカルロ木探索を実行して最善手を返す。

        Args:
            board (GoBoard): 評価する局面情報。
            color (Stone): 評価する局面の手番の色。
            time_manager (TimeManager): 思考時間管理インスタンス。

        Returns:
            int: 着手する座標。
        """
        self._initialize_search(board, color)

        time_manager.start_timer()

        root = self.node[self.current_root]

        # 候補手が1つしかない場合はPASSを返す
        if root.get_num_children() == 1:
            return PASS

        # 探索を実行する
        self.search(board, color, time_manager, analysis_query)

        if len(self.batch_queue.node_index) > 0:
            self.process_mini_batch(board)

        # 最善手を取得する
        next_move = root.get_best_move()
        next_index = root.get_best_move_index()

        # 探索結果と探索にかかった時間を表示する
        pv_list = self.get_pv_lists(self.get_root(), board.coordinate)
        root.print_search_result(board, pv_list)
        search_time = time_manager.calculate_consumption_time()
        po_per_sec = root.node_visits / search_time

        time_manager.set_search_speed(root.node_visits, search_time)
        time_manager.substract_consumption_time(color, search_time)

        print_err(f"{search_time:.2f} seconds, {po_per_sec:.2f} visits/s")

        value = root.calculate_value_evaluation(next_index)

        if value < RESIGN_THRESHOLD:
            return RESIGN

        return next_move


    def ponder(self, board: GoBoard, color: Stone, analysis_query: Dict[str, Any]) -> None:
        """探索回数の制限なく探索を実行する。

        Args:
            board (GoBoard): 局面情報。
            color (Stone): 思考する手番の色。
            analysis_query (Dict): 解析情報。
        """
        self._initialize_search(board, color)

        # 探索を実行する
        max_visits = 999999999
        mode = TimeControl.CONSTANT_PLAYOUT
        time_manager = TimeManager(mode=mode, constant_visits=max_visits)
        time_manager.initialize()
        time_manager.start_timer()
        self.search(board, color, time_manager, analysis_query)

        if len(self.batch_queue.node_index) > 0:
            self.process_mini_batch(board)


    def search(self, board: GoBoard, color: Stone, time_manager: TimeManager, \
        analysis_query: Dict[str, Any]) -> None: # pylint: disable=R0914
        """探索を実行する。
        Args:
            board (GoBoard): 現在の局面情報。
            color (Stone): 現局面の手番の色。
            time_manager (TimeManager): 思考時間管理インスタンス。
            analysis_query (Dict[str, Any]) : 解析情報。
        """
        self.to_move = color
        analysis_clock = time.time()
        search_board = copy.deepcopy(board)

        interval = analysis_query.get("interval", 0)
        threshold = time_manager.get_num_visits_threshold(color)

        for counter in range(threshold):
            copy_board(dst=search_board,src=board)
            start_color = color
            self.search_mcts(search_board, start_color, self.current_root, [])
            if time_manager.is_time_over() or \
                time_manager.is_move_decided(self.get_root(), threshold):
                break

            if len(analysis_query) > 0:
                elapsed = time.time() - analysis_clock
                root = self.node[self.current_root]

                if interval > 0 and \
                       (counter == threshold - 1 or elapsed > interval):
                    analysis_clock = time.time()
                    mode = analysis_query.get("mode", "lz")
                    sys.stdout.write(root.get_analysis(board, mode, self.get_pv_lists))
                    sys.stdout.flush()

                if analysis_query.get("ponder", False):
                    rlist, _, _ = select.select([sys.stdin], [], [], 0)
                    if rlist:
                        break

        if len(analysis_query) > 0 and interval == 0:
            root = self.node[self.current_root]
            mode = analysis_query.get("mode", "lz")
            sys.stdout.write(root.get_analysis(board, mode, self.get_pv_lists))
            sys.stdout.flush()


    def search_with_callback(self, board: GoBoard, color: Stone, \
        callback: Callable[[List[Tuple[int, int]]], bool]) -> None:
        """探索を実行し、探索系列をコールバック関数へ渡す動作をくり返す。
        コールバック関数の戻り値が真になれば終了する。
        Args:
            board (GoBoard): 現在の局面情報。
            color (Stone): 現局面の手番の色。
            callback (Callable[List[Tuple[int, int]], bool]): コールバック関数。
        """
        original_batch_size = self.batch_size
        self.batch_size = 1
        self._initialize_search(board, color)
        search_board = copy.deepcopy(board)
        while True:
            path: List[Tuple[int, int]] = []
            copy_board(dst=search_board, src=board)
            self.search_mcts(search_board, color, self.current_root, path)
            finished = callback(path)
            if finished:
                break
        self.batch_size = original_batch_size


    def search_mcts(self, board: GoBoard, color: Stone, current_index: int, \
        path: List[Tuple[int, int]]) -> None:
        """モンテカルロ木探索を実行する。

        Args:
            board (GoBoard): 現在の局面情報。
            color (Stone): 現局面の手番の色。
            current_index (int): 評価するノードのインデックス。
            path (List[Tuple[int, int]]): ルートからcurrent_indexに対応するノードに到達するまでの経路。
        """

        # UCB値最大の手を求める
        next_index = self.node[current_index].select_next_action(self.cgos_mode)
        next_move = self.node[current_index].get_child_move(next_index)

        path.append((current_index, next_index))

        # 1手進める
        board.put_stone(pos=next_move, color=color)
        color = Stone.get_opponent_color(color)

        # Virtual Lossの加算
        self.node[current_index].add_virtual_loss(next_index)

        # 既に2回連続パスしている場合は新しいノードを展開しないようにする
        expand_threshold = 1
        if board.moves > 2:
            _, pm1, _ = board.record.get(board.moves - 1)
            _, pm2, _ = board.record.get(board.moves - 2)
            if pm1 == PASS and pm2 == PASS:
                expand_threshold = 10000000

        if self.node[current_index].children_visits[next_index] \
            + self.node[current_index].children_virtual_loss[next_index] < expand_threshold + 1:
            if self.node[current_index].children_index[next_index] == NOT_EXPANDED:
                child_index = self.expand_node(board, color)
                self.node[current_index].set_child_index(next_index, child_index)
            else:
                child_index = self.node[current_index].get_child_index(next_index)
            input_plane = generate_input_planes(board, color, 0)
            self.batch_queue.push(input_plane, path, child_index)
            if len(self.batch_queue.node_index) >= self.batch_size:
                self.process_mini_batch(board)
        else:
            next_node_index = self.node[current_index].get_child_index(next_index)
            self.search_mcts(board, color, next_node_index, path)


    def process_mini_batch(self, board: GoBoard, use_logit: bool=False): # pylint: disable=R0914
        """ニューラルネットワークの入力をミニバッチ処理して、計算結果を探索結果に反映する。

        Args:
            board (GoBoard): 碁盤の情報。
            use_logit (bool): Policyの出力をlogitにするフラグ
        """
        input_planes = torch.Tensor(np.array(self.batch_queue.input_plane))

        if use_logit:
            raw_policy, value_data = self.network.inference_with_policy_logits(input_planes)
        else:
            raw_policy, value_data = self.network.inference(input_planes)

        policy_data = []
        for policy in raw_policy:
            policy_dict = {}
            for i, pos in enumerate(board.onboard_pos):
                policy_dict[pos] = policy[i]
            policy_dict[PASS] = policy[board.get_board_size() ** 2]
            if use_logit:
                policy_dict[PASS] -= 0.5
            policy_data.append(policy_dict)

        for policy, value_dist, path, node_index in zip(policy_data, \
            value_data, self.batch_queue.path, self.batch_queue.node_index):
            self.node[node_index].update_policy(policy)
            self.node[node_index].set_raw_value(value_dist[1] * 0.5 + value_dist[2])

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


    def generate_move_with_sequential_halving(self, board: GoBoard, color: Stone, \
        time_manager: TimeManager, never_resign: bool) -> int:
        """SHOTで探索して着手生成する。

        Args:
            board (GoBoard): 局面情報。
            color (Stone): 思考する手番の色。
            time (TimeManager): 思考時間管理用インスタンス。

        Returns:
            int: 生成した着手の座標。
        """
        self.num_nodes = 0
        start_time = time.time()
        self.current_root = self.expand_node(board, color)
        input_plane = generate_input_planes(board, color)
        self.batch_queue.push(input_plane, [], self.current_root)
        self.process_mini_batch(board, use_logit=True)
        self.node[self.current_root].set_gumbel_noise()

        # 探索を実行
        self.search_by_sequential_halving(board, color, \
            time_manager.get_num_visits_threshold(color))

        # 最善の手を取得
        root = self.node[self.current_root]
        next_index = root.select_move_by_sequential_halving_for_root(PLAYOUTS)

        # 勝率に基づいて投了するか否かを決める
        value = root.calculate_value_evaluation(next_index)

        search_time = time.time() - start_time

        time_manager.set_search_speed(self.node[self.current_root].node_visits, search_time)

        if not never_resign and value < 0.05:
            return RESIGN

        return root.get_child_move(next_index)


    def search_by_sequential_halving(self, board: GoBoard, color: Stone, \
        threshold: int) -> None:
        """指定された探索回数だけSequential Halving探索を実行する。

        Args:
            board (GoBoard): 評価したい局面。
            color (Stone): 評価したい局面の手番の色。
            threshold (int): 実行する探索回数。
        """
        search_board = copy.deepcopy(board)

        num_root_children = self.node[self.current_root].get_num_children()
        base_num_considered = num_root_children \
            if num_root_children < MAX_CONSIDERED_NODES else MAX_CONSIDERED_NODES
        search_control_dict = get_candidates_and_visit_pairs(base_num_considered, threshold)

        for num_considered, max_count in search_control_dict.items():
            for count_threshold in range(max_count):
                for _ in range(num_considered):
                    copy_board(search_board, board)
                    start_color = color

                    # 探索する
                    self.search_sequential_halving(search_board, start_color, \
                        self.current_root, [], count_threshold + 1)
            self.process_mini_batch(search_board, use_logit=True)


    def search_sequential_halving(self, board: GoBoard, color: Stone, current_index: int, \
        path: List[Tuple[int, int]], count_threshold: int) -> None: # pylint: disable=R0913
        """Sequential Halving探索を実行する。

        Args:
            board (GoBoard): 現在の局面。
            color (Stone): 現在の手番の色。
            current_index (int): 現在のノードのインデックス。
            path (List[Tuple[int, int]]): 現在のノードまで辿ったインデックス。
            count_threshold (int): 評価対象とする探索回数の閾値。
        """
        current_node = self.node[current_index]
        if current_index == self.current_root:
            next_index = current_node.select_move_by_sequential_halving_for_root(count_threshold)
        else:
            next_index = current_node.select_move_by_sequential_halving_for_node()
        next_move = self.node[current_index].get_child_move(next_index)

        path.append((current_index, next_index))

        board.put_stone(pos=next_move, color=color)
        color = Stone.get_opponent_color(color)

        self.node[current_index].add_virtual_loss(next_index)

        if self.node[current_index].children_visits[next_index] < 1:
            # ニューラルネットワークの計算
            input_plane = generate_input_planes(board, color)
            next_node_index = self.node[current_index].get_child_index(next_index)
            self.batch_queue.push(input_plane, path, next_node_index)
        else:
            if self.node[current_index].get_child_index(next_index) == NOT_EXPANDED:
                child_index = self.expand_node(board, color)
                self.node[current_index].set_child_index(next_index, child_index)
            next_node_index = self.node[current_index].get_child_index(next_index)
            self.search_sequential_halving(board, color, next_node_index, path, count_threshold)
