"""モンテカルロ木探索で使用するノードの実装。
"""
import json
from typing import Any, Callable, Dict, List, NoReturn

import numpy as np
import torch
from board.constant import BOARD_SIZE
from board.go_board import GoBoard
from common.print_console import print_err
from mcts.constant import NOT_EXPANDED, C_VISIT, C_SCALE
from mcts.pucb.pucb import calculate_pucb_value
from nn.utility import apply_softmax

MAX_ACTIONS = BOARD_SIZE ** 2 + 1
PUCT_WEIGHT = 1.0

class MCTSNode: # pylint: disable=R0902, R0904
    """モンテカルロ木探索で使うノード情報のクラス。
    """
    def __init__(self, num_actions: int=MAX_ACTIONS):
        """_MCTSNodeクラスのコンストラクタ

        Args:
            num_actions (int, optional): 候補手の最大数. Defaults to MAX_ACTIONS.
        """
        self.node_visits = 0
        self.virtual_loss = 0
        self.node_value_sum = 0.0
        self.raw_value = 0.0
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
        """ノードを展開し、初期化する。

        Args:
            policy (Dict[int, float]): 候補手に対応するPolicyのマップ。
        """
        self.node_visits = 0
        self.virtual_loss = 0
        self.node_value_sum = 0.0
        self.raw_value = 0.0
        self.action = [0] * MAX_ACTIONS
        self.children_index.fill(NOT_EXPANDED)
        self.children_value.fill(0.0)
        self.children_visits.fill(0)
        self.children_virtual_loss.fill(0)
        self.children_value_sum.fill(0.0)
        self.noise.fill(0.0)

        self.set_policy(policy)


    def set_policy(self, policy_map: Dict[int, float]) -> NoReturn:
        """着手候補の座標とPolicyの値を設定する。

        Args:
            policy_map (Dict[int, float]): Keyが着手座標, Valueが着手のPolicy。
        """
        index = 0
        for pos, policy in policy_map.items():
            self.action[index] = pos
            self.children_policy[index] = policy
            index += 1
        self.num_children = index


    def add_virtual_loss(self, index) -> NoReturn:
        """Virtual Lossを加算する。

        Args:
            index (_type_): 加算する対象の子ノードのインデックス。
        """
        self.virtual_loss += 1
        self.children_virtual_loss[index] += 1


    def update_policy(self, policy: Dict[int, float]) -> NoReturn:
        """Policyを更新する。

        Args:
            policy (Dict[int, float]): 候補手と対応するPolicyのマップ。
        """
        for i in range(self.num_children):
            self.children_policy[i] = policy[self.action[i]]


    def set_leaf_value(self, index: int, value: float) -> NoReturn:
        """末端のValueを設定する。

        Args:
            index (int): Valueを設定する対象の子ノードのインデックス。
            value (float): 設定するValueの値。

        Returns:
            NoReturn: _description_
        """
        self.children_value[index] = value


    def set_raw_value(self, value: float) -> NoReturn:
        """ノードに対応する局面のValueを設定する。

        Args:
            value (float): 設定するValueの値。
        """
        self.raw_value = value


    def update_child_value(self, index: int, value: float) -> NoReturn:
        """子ノードにValueを加算し、Virtual Lossを元に戻す。

        Args:
            index (int): 更新する対象の子ノードのインデックス。
            value (float): 加算するValueの値。
        """
        self.children_value_sum[index] += value
        self.children_visits[index] += 1
        self.children_virtual_loss[index] -= 1


    def update_node_value(self, value: float) -> NoReturn:
        """ノードにValueを加算し、Virtual Lossを元に戻す。

        Args:
            value (float): 加算するValueの値。
        """
        self.node_value_sum += value
        self.node_visits += 1
        self.virtual_loss -= 1


    def select_next_action(self, cgos_mode: bool) -> int:
        """PUCB値に基づいて次の着手を選択する。

        Args:
            cgos_mode (bool): 全ての石を打ち上げるまでパスを抑制するモード。
        Returns:
            int: 次の着手として選ぶ子ノードのインデックス。
        """
        pucb_values = calculate_pucb_value(self.node_visits + self.virtual_loss, \
            self.children_visits + self.children_virtual_loss, \
            self.children_value_sum, self.children_policy + self.noise)

        if cgos_mode:
            # PASSのValueを0.1だけ引く
            pucb_values[self.num_children - 1] -= 0.1

        return np.argmax(pucb_values[:self.num_children])


    def get_num_children(self) -> int:
        """子ノードの個数を取得する。

        Returns:
            int: 子ノードの個数。
        """
        return self.num_children


    def get_best_move_index(self) -> int:
        """探索回数最大の子ノードのインデックスを取得する。

        Returns:
            int: 探索回数最大の子ノードのインデックス。
        """
        return np.argmax(self.children_visits[:self.num_children])


    def get_best_move(self) -> int:
        """探索回数最大の着手を取得する。

        Returns:
            int: 探索回数が最大の着手の座標。
        """
        return self.action[self.get_best_move_index()]


    def get_child_move(self, index: int) -> int:
        """指定した子ノードに対応する着手の座標を取得する。

        Args:
            index (int): 指定する子ノードのインデックス。

        Returns:
            int: 着手の座標。
        """
        return self.action[index]


    def get_child_index(self, index: int) -> int:
        """指定した子ノードの遷移先のインデックスを取得する。

        Args:
            index (int): 指定する子ノードのインデックス。

        Returns:
            int: 遷移先のインデックス。
        """
        return self.children_index[index]


    def set_child_index(self, index: int, child_index: int) -> NoReturn:
        """指定した子ノードの遷移先のインデックスを設定する。

        Args:
            index (int): 指定した子ノードのインデックス。
            child_index (int): 遷移先のノードのインデックス。
        """
        self.children_index[index] = child_index


    def to_dict(self) -> Dict[str, Any]:
        """ノードの状態を辞書化して返す。

        Returns:
            Dict[str, Any]: ノードの状態を表す辞書。
        """
        state = {
            "node_visits": self.node_visits,
            "virtual_loss": self.virtual_loss,
            "node_value_sum": self.node_value_sum,
            "raw_value": self.raw_value,
            "action": self.action,
            "children_index": self.children_index,
            "children_value": self.children_value,
            "children_visits": self.children_visits,
            "children_policy": self.children_policy,
            "children_virtual_loss": self.children_virtual_loss,
            "children_value_sum": self.children_value_sum,
            "noise": self.noise,
            "num_children": self.num_children,
        }
        self._make_serializable(state)
        return state

    def _make_serializable(self, dic):
        for key in dic:
            val = dic[key]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            elif isinstance(val, torch.Tensor):
                val = val.item()
            dic[key] = val

    def print_search_result(self, board: GoBoard, pv_dict: Dict[str, List[str]]) -> NoReturn:
        """探索結果を表示する。探索した手の探索回数とValueの平均値を表示する。

        Args:
            board (GoBoard): 現在の局面情報。
        """
        value = np.divide(self.children_value_sum, self.children_visits, \
            out=np.zeros_like(self.children_value_sum), where=(self.children_visits != 0))
        print_err(f"raw_value={self.raw_value:.4f}")
        for i in range(self.num_children):
            if self.children_visits[i] > 0:
                pos = board.coordinate.convert_to_gtp_format(self.action[i])
                msg = f"pos={pos}, "
                msg += f"visits={self.children_visits[i]:5d}, "
                msg += f"policy={self.children_policy[i]:.4f}, "
                msg += f"value={value[i]:.4f}, "
                msg += f"raw_value={self.children_value[i]:.4f}, "
                msg += f"pv={','.join(pv_dict[pos])}"
                print_err(msg)


    def set_gumbel_noise(self) -> NoReturn:
        """Gumbelノイズを設定する。
        """
        self.noise = np.random.gumbel(loc=0.0, scale=1.0, size=self.noise.size)


    def calculate_completed_q_value(self, use_mixed_value :bool=True) -> np.array:
        """Completed-Q valueを計算する。

        Args:
            use_mixed_value (bool, optional): Mixed value approximation使用フラグ. デフォルトはTrue.

        Returns:
            np.array: Completed-Q value.
        """
        policy = apply_softmax(self.children_policy[:self.num_children])

        q_value = np.divide(self.children_value_sum, self.children_visits, \
            out=np.zeros_like(self.children_value_sum), \
            where=(self.children_visits > 0))[:self.num_children]

        sum_prob = np.sum(policy)
        v_pi = np.sum(policy * q_value)

        if use_mixed_value:
            value = (self.raw_value * np.ones_like(self.children_policy[:self.num_children]) + \
                self.node_visits * v_pi / sum_prob) / (self.node_visits + 1.0)
        else:
            value = self.raw_value

        return np.where(self.children_visits[:self.num_children] > 0, q_value, value)


    def calculate_improved_policy(self) -> np.array:
        """Improved Policyを計算する。

        Returns:
            np.array: Improved Policy.
        """
        max_visit = np.max(self.children_visits)

        sigma_base = (C_VISIT + max_visit) * C_SCALE
        completed_q_value = self.calculate_completed_q_value()

        improved_logits = self.children_policy[:self.num_children] + sigma_base * completed_q_value

        return apply_softmax(improved_logits)


    def select_move_by_sequential_halving_for_root(self, count_threshold: int) -> int:
        """Gumbel AlphaZeroの探索手法を使用して次の着手を選択する。Rootのみで使用する。。

        Args:
            count_threshold (int): 探索回数閾値。

        Returns:
            int: 選択した子ノードのインデックス。
        """
        max_count = max(self.children_visits[:self.num_children])

        sigma_base = (C_VISIT + max_count) * C_SCALE

        counts = self.children_visits[:self.num_children] \
            + self.children_virtual_loss[:self.num_children]
        q_mean = np.divide(self.children_value_sum, self.children_visits, \
            out=np.zeros_like(self.children_value_sum), \
            where=(self.children_visits > 0))[:self.num_children]

        evaluation_value = np.where(counts >= count_threshold, -10000.0, \
            self.children_policy[:self.num_children] + self.noise[:self.num_children] \
            + sigma_base * q_mean)
        return np.argmax(evaluation_value)


    def select_move_by_sequential_halving_for_node(self) -> int:
        """Gumbel AlphaZeroの探索手法を使用して次の着手を選択する。Root以外で使用する。。

        Returns:
            int: 選択した子ノードのインデックス。
        """

        improved_policy = self.calculate_improved_policy()

        evaluation_value = improved_policy \
            - (self.children_visits[:self.num_children] / (1.0 + self.node_visits))

        return np.argmax(evaluation_value)


    def calculate_value_evaluation(self, index: int) -> float:
        """指定した子ノードのValueを計算する。

        Args:
            index (int): 子ノードのインデックス。

        Returns:
            float: 指定した子ノードのValueの値。
        """
        if self.children_visits[index] == 0:
            return 0.5
        return self.children_value_sum[index] / self.children_visits[index]


    def print_all_node_info(self) -> NoReturn:
        """子ノードの情報を全て表示する。
        """
        msg = ""
        msg += f"node_visits : {self.node_visits}\n"
        msg += f"virtual_loss : {self.virtual_loss}\n"
        msg += f"node_value_sum : {self.node_value_sum}\n"
        msg += f"num_children : {self.num_children}\n"
        for i in range(self.num_children):
            msg += f"\tindex : {i}\n"
            msg += f"\taction : {self.action[i]}\n"
            msg += f"\tchildren_index : {self.children_index[i]}\n"
            msg += f"\tchildren_value : {self.children_value[i]}\n"
            msg += f"\tchildren_visits : {self.children_visits[i]}\n"
            msg += f"\tchildren_policy : {self.children_policy[i]}\n"
            msg += f"\tchildren_virtual_loss : {self.children_virtual_loss[i]}\n"
            msg += f"\tchildren_value_sum : {self.children_value_sum[i]}\n"
            msg += f"\tnoise : {self.children_value_sum[i]}\n"
        print_err(msg)


    def get_analysis(self, board: GoBoard, mode: str, \
        pv_lists_func: Callable[[List[str], int], List[str]]) -> str: # pylint: disable=R0914
        """解析結果文字列を生成する。

        Args:
            board (GoBoard): 局面情報。
            mode (str): 解析モード。"lz"か"cgos"。
            pv_lists_func (Callable[[List[str], int], List[str]]): PV情報生成関数。

        Returns:
            str: GTP応答用解析結果文字列。
        """
        children_status_list = self.get_analysis_status_list(board, pv_lists_func)
        return self.get_analysis_from_status_list(mode, children_status_list)


    def get_analysis_status_list(self, board: GoBoard, \
        pv_lists_func: Callable[[List[str], int], List[str]]):
        sorted_list = []
        for i in range(self.num_children):
            sorted_list.append((self.children_visits[i], i))
        sorted_list.sort(reverse=True)

        coordinate = board.coordinate
        pv_lists = pv_lists_func(self, coordinate)

        children_status_list = []
        order = 0
        for visits, i in sorted_list:
            if visits != 0:
                pos = self.action[i]
                winrate = self.children_value_sum[i] / visits
                prior = self.children_policy[i]

                pv_list = pv_lists[coordinate.convert_to_gtp_format(pos)]
                pv_str = " ".join([f"{p}" for p in pv_list])

                children_status_list.append(
                    {
                        "move" : coordinate.convert_to_gtp_format(pos),
                        "visits" : int(visits),
                        "winrate" : float(winrate),
                        "prior" : float(prior),
                        "lcb" : float(winrate),
                        "order" : int(order),
                        "pv" : pv_str
                    }
                )
                order += 1
        return children_status_list


    def get_analysis_from_status_list(self, mode, children_status_list):
        out = ""
        if mode == "cgos":
            cgos_dict = {
                "winrate" : float(self.node_value_sum) / self.node_visits,
                "visits" : self.node_visits,
                "moves" : []
            }

        for status in children_status_list:
            if mode == "lz":
                out += f"info move {status['move']} "
                out += f"visits {status['visits']} "
                out += f"winrate {int(10000 * status['winrate'])} "
                out += f"prior {int(10000 * status['prior'])} "
                out += f"lcb {int(10000 * status['lcb'])} "
                out += f"order {status['order']} "
                out += f"pv {status['pv']}"
                out += " "
            elif mode == "cgos":
                cgos_dict["moves"].append(status)

        if mode == "cgos":
            out = json.dumps(cgos_dict, indent=None, separators=(',', ':'))
        else:
            out = out[:-1]
        out += '\n'
        return out
