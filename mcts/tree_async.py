"""モンテカルロ木探索の実装。"""

from typing import Any, Dict, List, Tuple
import select
import sys
import time

import torch

from board.constant import PASS, RESIGN
from board.go_board import GoBoard, copy_board
from board.stone import Stone
from common.print_console import print_err
from mcts.constant import (
    MAX_CONSIDERED_NODES,
    MCTS_TREE_SIZE,
    NN_BATCH_SIZE,
    NOT_EXPANDED,
    PLAYOUTS,
    RESIGN_THRESHOLD,
)
from mcts.nneval import NNEval
from mcts.sequential_halving import get_candidates_and_visit_pairs
from mcts.time_manager import TimeManager
from mcts.tree_base import MCTSTreeBase
from nn.feature import generate_input_planes


class MCTSTreeAsync(MCTSTreeBase):  # pylint: disable=R0902
    """モンテカルロ木探索の実装クラス。"""

    def __init__(
        self,
        nneval: NNEval,
        tree_size: int = MCTS_TREE_SIZE,
        batch_size: int = NN_BATCH_SIZE,
        cgos_mode: bool = False,
    ):
        """MCTSTreeクラスのコンストラクタ。

        Args:
            network (DualNet): 使用するニューラルネットワーク。
            tree_size (int, optional): 木を構成するノードの最大個数。デフォルトは65536。
            batch_size (int, optional): ニューラルネットワークの前向き伝搬処理のミニバッチサイズ。デフォルトはNN_BATCH_SIZE。
        """
        super().__init__(tree_size=tree_size, batch_size=batch_size, cgos_mode=cgos_mode)
        self.nneval = nneval

    async def _initialize_search(self, board: GoBoard, color: Stone) -> None:
        self.num_nodes = 0
        self.current_root = self.expand_node(board, color)
        input_plane = generate_input_planes(board, color, 0)

        fut = self.nneval.push_eval(input_plane)
        policy, value = await fut
        self.apply_policy_and_value(board, policy, value, [], self.current_root)

    async def search_best_move(
        self,
        board: GoBoard,
        color: Stone,
        time_manager: TimeManager,
        analysis_query: Dict[str, Any],
    ) -> int:
        """モンテカルロ木探索を実行して最善手を返す。

        Args:
            board (GoBoard): 評価する局面情報。
            color (Stone): 評価する局面の手番の色。
            time_manager (TimeManager): 思考時間管理インスタンス。

        Returns:
            int: 着手する座標。
        """
        await self._initialize_search(board, color)

        time_manager.start_timer()

        root = self.node[self.current_root]

        # 候補手が1つしかない場合はPASSを返す
        if root.get_num_children() == 1:
            return PASS

        # 探索を実行する
        await self.search(board, color, time_manager, analysis_query)

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

    async def search(
        self,
        board: GoBoard,
        color: Stone,
        time_manager: TimeManager,
        analysis_query: Dict[str, Any],
    ) -> None:  # pylint: disable=R0914
        """探索を実行する。
        Args:
            board (GoBoard): 現在の局面情報。
            color (Stone): 現局面の手番の色。
            time_manager (TimeManager): 思考時間管理インスタンス。
            analysis_query (Dict[str, Any]) : 解析情報。
        """
        self.to_move = color
        analysis_clock = time.time()
        search_board = GoBoard(board_size=board.get_board_size(), \
                               komi=board.get_komi(), check_superko=board.check_superko)

        interval = analysis_query.get("interval", 0)
        threshold = time_manager.get_num_visits_threshold(color)

        for counter in range(threshold):
            copy_board(dst=search_board, src=board)
            start_color = color
            await self.search_mcts(search_board, start_color, self.current_root, [])
            if time_manager.is_time_over() or time_manager.is_move_decided(
                self.get_root(), threshold
            ):
                break

            if len(analysis_query) > 0:
                elapsed = time.time() - analysis_clock
                root = self.node[self.current_root]

                if interval > 0 and (counter == threshold - 1 or elapsed > interval):
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

    async def search_mcts(
        self,
        board: GoBoard,
        color: Stone,
        current_index: int,
        path: List[Tuple[int, int]],
    ) -> None:
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

        if (
            self.node[current_index].children_visits[next_index]
            + self.node[current_index].children_virtual_loss[next_index]
            < expand_threshold + 1
        ):
            if self.node[current_index].children_index[next_index] == NOT_EXPANDED:
                child_index = self.expand_node(board, color)
                self.node[current_index].set_child_index(next_index, child_index)
            else:
                child_index = self.node[current_index].get_child_index(next_index)
            input_plane = generate_input_planes(board, color, 0)
            policy, value = await self.nneval.push_eval(input_plane)
            self.apply_policy_and_value(board, policy, value, path, child_index)
        else:
            next_node_index = self.node[current_index].get_child_index(next_index)
            await self.search_mcts(board, color, next_node_index, path)

    def apply_policy_and_value( #pylint: disable=R0913, R0914
        self,
        board: GoBoard,
        raw_policy: torch.Tensor,
        value_dist: List[float],
        path: List[Tuple[int, int]],
        node_index: int,
        use_logit: bool = False,
    ) -> None:
        """ニューラルネットワークの入力をミニバッチ処理して、計算結果を探索結果に反映する。

        Args:
            board (GoBoard): 碁盤の情報。
            use_logit (bool): Policyの出力をlogitにするフラグ
        """

        if use_logit:
            policy = raw_policy
        else:
            # calc softmax(raw_policy) using numpy
            # policy = np.exp(raw_policy - np.max(raw_policy))
            # policy /= np.sum(policy)
            # print(raw_policy.shape)
            policy = torch.softmax(raw_policy, dim=0)

        policy_dict = {}
        for i, pos in enumerate(board.onboard_pos):
            policy_dict[pos] = float(policy[i])
        policy_dict[PASS] = float(policy[board.get_board_size() ** 2])
        if use_logit:
            policy_dict[PASS] -= 0.5

        self.node[node_index].update_policy(policy_dict)
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

        # self.batch_queue.clear()

    async def generate_move_with_sequential_halving(
        self,
        board: GoBoard,
        color: Stone,
        time_manager: TimeManager,
        never_resign: bool,
    ) -> int:
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
        nn_policy, nn_value = await self.nneval.push_eval(input_plane)
        # self.batch_queue.push(input_plane, [], self.current_root)
        self.apply_policy_and_value(
            board, nn_policy, nn_value, [], self.current_root, use_logit=True
        )
        self.node[self.current_root].set_gumbel_noise()

        # 探索を実行
        await self.search_by_sequential_halving(
            board, color, time_manager.get_num_visits_threshold(color)
        )

        # 最善の手を取得
        root = self.node[self.current_root]
        next_index = root.select_move_by_sequential_halving_for_root(PLAYOUTS)

        # 勝率に基づいて投了するか否かを決める
        value = root.calculate_value_evaluation(next_index)

        search_time = time.time() - start_time

        time_manager.set_search_speed(
            self.node[self.current_root].node_visits, search_time
        )

        if not never_resign and value < 0.05:
            return RESIGN

        return root.get_child_move(next_index)

    async def search_by_sequential_halving(
        self, board: GoBoard, color: Stone, threshold: int
    ) -> None:
        """指定された探索回数だけSequential Halving探索を実行する。

        Args:
            board (GoBoard): 評価したい局面。
            color (Stone): 評価したい局面の手番の色。
            threshold (int): 実行する探索回数。
        """
        search_board = GoBoard(board_size=board.get_board_size(), \
                               komi=board.get_komi(), check_superko=board.check_superko)

        num_root_children = self.node[self.current_root].get_num_children()
        base_num_considered = (
            num_root_children
            if num_root_children < MAX_CONSIDERED_NODES
            else MAX_CONSIDERED_NODES
        )
        search_control_dict = get_candidates_and_visit_pairs(
            base_num_considered, threshold
        )

        for num_considered, max_count in search_control_dict.items():
            for count_threshold in range(max_count):
                for _ in range(num_considered):
                    copy_board(search_board, board)
                    start_color = color

                    # 探索する
                    await self.search_sequential_halving(
                        search_board,
                        start_color,
                        self.current_root,
                        [],
                        count_threshold + 1,
                    )

    async def search_sequential_halving( #pylint: disable=R0913
        self,
        board: GoBoard,
        color: Stone,
        current_index: int,
        path: List[Tuple[int, int]],
        count_threshold: int,
    ) -> None:
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
            next_index = current_node.select_move_by_sequential_halving_for_root(
                count_threshold
            )
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
            # self.batch_queue.push(input_plane, path, next_node_index)
            policy, value = await self.nneval.push_eval(input_plane)
            self.apply_policy_and_value(board, policy, value, path, next_node_index)
        else:
            if self.node[current_index].get_child_index(next_index) == NOT_EXPANDED:
                child_index = self.expand_node(board, color)
                self.node[current_index].set_child_index(next_index, child_index)
            next_node_index = self.node[current_index].get_child_index(next_index)
            await self.search_sequential_halving(
                board, color, next_node_index, path, count_threshold
            )
