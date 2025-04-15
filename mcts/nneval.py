"""モンテカルロ木探索の実装。"""

import asyncio
import copy
import queue
import select
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from board.constant import PASS, RESIGN
from board.coordinate import Coordinate
from board.go_board import GoBoard, copy_board
from board.stone import Stone
from common.print_console import print_err
from mcts.batch_data import BatchQueue
from mcts.constant import (
    MAX_CONSIDERED_NODES,
    MCTS_TREE_SIZE,
    NN_SELFPLAY_BATCH_SIZE,
    NOT_EXPANDED,
    PLAYOUTS,
    RESIGN_THRESHOLD,
)
from mcts.dump import dump_mcts_to_json
from mcts.node import MCTSNode
from mcts.sequential_halving import get_candidates_and_visit_pairs
from mcts.time_manager import TimeControl, TimeManager
from mcts.tree import MCTSTree
from nn.feature import generate_input_planes
from nn.network.dual_net import DualNet


class NNEval:  # pylint: disable=R0902
    """NNの評価と利用"""

    network: DualNet
    batch_queue: queue.Queue[
        Tuple[asyncio.Future[Tuple[torch.Tensor, List[float]]], np.ndarray]
    ]
    batch_size: int

    def __init__(self, network: DualNet, batch_size: int = NN_SELFPLAY_BATCH_SIZE):
        """MCTSTreeクラスのコンストラクタ。

        Args:
            network (DualNet): 使用するニューラルネットワーク。
            batch_size (int, optional): ニューラルネットワークの前向き伝搬処理のミニバッチサイズ。デフォルトはNN_BATCH_SIZE。
        """
        self.network = network
        self.batch_queue = queue.Queue()
        self.batch_size = batch_size

    def push_eval(
        self, input_planes: np.ndarray
    ) -> asyncio.Future[Tuple[torch.Tensor, List[float]]]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Tuple[torch.Tensor, List[float]]] = loop.create_future()
        self.batch_queue.put((fut, input_planes))
        return fut

    def process_mini_batch(self):
        """ニューラルネットワークの入力をミニバッチ処理して、計算結果を探索結果に反映する。

        Args:
            board (GoBoard): 碁盤の情報。
            use_logit (bool): Policyの出力をlogitにするフラグ
        """
        future_list = []
        input_planes_list = []

        # print("Queue", self.batch_queue.qsize(), self.batch_size)

        for _ in range(self.batch_size):
            try:
                fut, input = self.batch_queue.get_nowait()
            except queue.Empty:
                break

            input_planes_list.append(input)
            future_list.append(fut)

        if len(input_planes_list) == 0:
            return

        input_planes = torch.Tensor(np.array(input_planes_list))

        raw_policy, value_data = self.network.inference_with_policy_logits(input_planes)

        for fut, policy, value_dist in zip(future_list, raw_policy, value_data):
            fut.set_result((policy, value_dist))
