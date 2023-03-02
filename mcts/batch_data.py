"""ニューラルネットワーク計算用のキュー。
"""
from typing import List, Tuple
import numpy as np


class BatchQueue:
    """ミニバッチデータを保持するキュー。
    """
    def __init__(self):
        """BatchQueueクラスのコンストラクタ。
        """
        self.input_plane = []
        self.path = []
        self.node_index = []

    def push(self, input_plane: np.array, path: List[Tuple[int, int]], node_index: int):
        """キューにデータをプッシュする。

        Args:
            input_plane (np.array): ニューラルネットワークへの入力データ。
            path (List[Tuple[int, int]]): ルートから評価ノードへまでの経路。
            node_index (int): ニューラルネットワークが評価する局面に対応するノードのインデックス。
        """
        self.input_plane.append(input_plane)
        self.path.append(path)
        self.node_index.append(node_index)

    def clear(self):
        """キューのデータを全て削除する。
        """
        self.input_plane = []
        self.path = []
        self.node_index = []
