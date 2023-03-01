from typing import Dict, NoReturn

import numpy as np
from board.constant import BOARD_SIZE
from board.go_board import GoBoard
from common.print_console import print_err
from mcts.constant import NOT_EXPANDED
from mcts.pucb.pucb import calculate_pucb_value


MAX_ACTIONS = BOARD_SIZE ** 2 + 1
PUCT_WEIGHT = 1.0

class MCTSNode:
    def __init__(self, num_actions=MAX_ACTIONS) -> NoReturn:
        """_summary_

        Args:
            num_actions (_type_, optional): _description_. Defaults to MAX_ACTIONS.
        """
        self.node_visits = 0
        self.virtual_loss = 0
        self.node_value_sum = 0.0
        self.action = [0] * num_actions
        self.children_index = np.zeros(num_actions, dtype=np.int32)
        self.children_value = np.zeros(num_actions, dtype=np.float64)
        self.children_visits = np.zeros(num_actions, dtype=np.int32)
        self.children_policy = np.zeros(num_actions, dtype=np.float64)
        self.children_virtual_loss = np.zeros(num_actions, dtype=np.int32)
        self.children_value_sum = np.zeros(num_actions, dtype=np.float64)
        self.noise = np.zeros(num_actions, dtype=np.float64)
        self.num_children = 0

    def expand(self, policy: Dict[int, float]) -> NoReturn:
        """_summary_

        Args:
            policy (Dict[int, float]): _description_

        Returns:
            NoReturn: _description_
        """
        self.node_visits = 0
        self.node_value_sum = 0.0
        self.virtual_loss = 0
        self.action = [0] * MAX_ACTIONS
        self.children_index.fill(NOT_EXPANDED)
        self.children_value.fill(0.0)
        self.children_visits.fill(0)
        self.children_virtual_loss.fill(0)
        self.children_value_sum.fill(0.0)
        self.noise.fill(0.0)

        self.set_policy(policy)


    def set_policy(self, policy: Dict[int, float]) -> NoReturn:
        """着手候補の座標とPolicyの値を設定する。

        Args:
            policy (Dict[int, float]): Keyが着手座標, Valueが着手のPolicy。
        """
        index = 0
        for pos, policy in policy.items():
            self.action[index] = pos
            self.children_policy[index] = policy
            index += 1
        self.num_children = index

    def add_virtual_loss(self, index) -> NoReturn:
        self.virtual_loss += 1
        self.children_virtual_loss[index] += 1

    def update_policy(self, policy: Dict[int, float]) -> NoReturn:
        for i in range(self.num_children):
            self.children_policy[i] = policy[self.action[i]]

    def set_leaf_value(self, index: int, value: float) -> NoReturn:
        self.children_value[index] = value

    def update_child_value(self, index: int, value: float) -> NoReturn:
        self.children_value_sum[index] += value
        self.children_visits[index] += 1
        self.children_virtual_loss[index] -= 1

    def update_node_value(self, value: float) -> NoReturn:
        self.node_value_sum += value
        self.node_visits += 1
        self.virtual_loss -= 1


    def select_next_action(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        pucb_values = calculate_pucb_value(self.node_visits + self.virtual_loss, \
            self.children_visits + self.children_virtual_loss, \
            self.children_value_sum, self.children_policy + self.noise)

        return np.argmax(pucb_values[:self.num_children])

    def get_num_children(self) -> int:
        return self.num_children

    def get_best_move_index(self) -> int:
        return np.argmax(self.children_visits[:self.num_children])

    def get_best_move(self) -> int:
        return self.action[self.get_best_move_index()]

    def get_child_move(self, index: int) -> int:
        return self.action[index]

    def get_child_index(self, index: int) -> int:
        return self.children_index[index]

    def set_child_index(self, index: int, child_index: int) -> NoReturn:
        self.children_index[index] = child_index


    def print_search_result(self, board: GoBoard) -> NoReturn:
        value = np.divide(self.children_value_sum, self.children_visits, out=np.zeros_like(self.children_value_sum), where=(self.children_visits != 0)) 
        for i in range(self.num_children):
            if self.children_visits[i] > 0:
                pos = board.coordinate.convert_to_gtp_format(self.action[i])
                print_err(f"pos={pos}, visits={self.children_visits[i]}, value={value[i]:.4f}")