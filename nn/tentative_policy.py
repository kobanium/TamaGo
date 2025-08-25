"""仮のPolicyの値
"""
from typing import Dict, List
import numpy as np


def get_tentative_policy(candidates: List[int]) -> Dict[int, float]:
    """ニューラルネットワークの計算が行われるまでに使用するPolicyを取得する。

    Args:
        candidates (List[int]): パスを含む候補手のリスト。

    Returns:
        Dict[int, float]: 候補手の座標とPolicyの値のマップ。
    """
    score = np.random.dirichlet(alpha=np.ones(len(candidates)))
    return dict(zip(candidates, score))
