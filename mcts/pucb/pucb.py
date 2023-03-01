import math
import numpy as np

from mcts.constant import PUCB_SECOND_TERM_WEIGHT

def calculate_pucb_value(node_visits: int, children_visits: np.ndarray, \
    value_sum: np.ndarray, policy: np.ndarray) -> np.ndarray:
    """全ての手のPUCB値を計算する。

    Args:
        node_visits (int): ノードの探索回数。
        children_visits (np.ndarray): 子ノードの探索回数。
        value (np.ndarray): 子ノードのValueの合計値。
        policy (np.ndarray): 子ノードのPolicy。

    Returns:
        np.ndarray: 子ノードのPUCB値。
    """
    # Valueの平均値の算出
    exploration = np.divide(value_sum, children_visits, out=np.zeros_like(value_sum), where=(children_visits != 0))

    # PUCTの第2項の算出
    exploitation = PUCB_SECOND_TERM_WEIGHT * policy \
        * math.sqrt(node_visits + 1) / (children_visits + 1) \

    return exploration + exploitation
