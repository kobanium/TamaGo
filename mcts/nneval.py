"""自己対戦用のニューラルネットワークのミニバッチ処理の実装。
"""

import asyncio
import queue
from typing import List, Tuple

import numpy as np
import torch

from mcts.constant import NN_SELFPLAY_BATCH_SIZE
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
        """ニューラルネットワークの推論処理を実行するイベントループを取得する。

        Args:
            input_planes (np.ndarray): ニューラルネットワークの入力データ。

        Returns:
            asyncio.Future[Tuple[torch.Tensor, List[float]]]: 実行中のイベントループ。
        """
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
                fut, input_data = self.batch_queue.get_nowait()
            except queue.Empty:
                break

            input_planes_list.append(input_data)
            future_list.append(fut)

        if len(input_planes_list) == 0:
            return

        input_planes = torch.Tensor(np.array(input_planes_list))

        raw_policy, value_data = self.network.inference_with_policy_logits(input_planes)

        for fut, policy, value_dist in zip(future_list, raw_policy, value_data):
            fut.set_result((policy, value_dist))
