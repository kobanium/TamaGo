"""探索時間を制御する処理。
"""
from enum import Enum
from typing import NoReturn
import time

from board.stone import Stone
from mcts.constant import CONST_VISITS, CONST_TIME, REMAINING_TIME, VISITS_PER_SEC


class TimeControl(Enum):
    """思考時間の管理モードを表すクラス。
    """
    CONSTANT_PLAYOUT = 0
    CONSTANT_TIME = 1
    TIME_CONTROL = 2


class TimeManager:
    """時間管理クラス。
    """
    # pylint: disable=R0902
    def __init__(self, mode: TimeControl, constant_visits: int=CONST_VISITS, constant_time: \
        float=CONST_TIME, remaining_time: float=REMAINING_TIME):
        """TimeManagerクラスのコンストラクタ。

        Args:
            mode (TimeControl): 探索時間の管理モード
            constant_visits (int, optional): 1手あたりの探索回数。デフォルト値はCONST_VISITS。
            constant_time (float, optional): 1手あたりの探索時間。デフォルト値はCONST_TIME。
            remaining_time (float, optional): 持ち時間。デフォルト値REMAINING_TIME。
        """
        self.mode = mode
        self.constant_visits = constant_visits
        self.constant_time = constant_time
        self.default_time = remaining_time
        self.search_speed = VISITS_PER_SEC
        self.remaining_time = [remaining_time] * 2
        self.time_limit = 0
        self.start_time = 0


    def initialize(self):
        """持ち時間の初期設定をする。
        """
        self.remaining_time = [self.default_time] * 2


    def set_search_speed(self, visits: int, consumption_time: float) -> NoReturn:
        """探索速度を設定する。

        Args:
            visits (int): 実行した探索回数。
            consumption_time (float): 探索にかかった時間(秒)。
        """
        self.search_speed = visits / consumption_time if visits > 0 else VISITS_PER_SEC


    def get_num_visits_threshold(self, color: Stone) -> int:
        """探索回数の閾値を取得する。

        Args:
            color (Stone): 手番の色。

        Returns:
            int: 探索回数の閾値。
        """
        if self.mode == TimeControl.CONSTANT_PLAYOUT:
            self.time_limit = 10000.0
            return int(self.constant_visits)
        if self.mode == TimeControl.CONSTANT_TIME:
            self.time_limit = self.constant_time
            threshold = int(self.search_speed * self.constant_time)
            return threshold if threshold > 0 else 1
        if self.mode == TimeControl.TIME_CONTROL:
            remaining_time = self.remaining_time[0] \
                if color is Stone.BLACK else self.remaining_time[1]
            self.time_limit = remaining_time / 10.0
            threshold = int(self.search_speed * self.time_limit)
            return threshold if threshold > 0 else 1
        return int(self.constant_visits)


    def set_remaining_time(self, color: Stone, remaining_time: float) -> NoReturn:
        """残り時間を設定する。

        Args:
            color (Stone): 残り時間を設定する手番の色。
            remaining_time (float): 設定する残り時間。
        """
        if color is Stone.BLACK:
            self.remaining_time[0] = remaining_time
        if color is Stone.WHITE:
            self.remaining_time[1] = remaining_time


    def substract_consumption_time(self, color: Stone, consumption_time: float):
        """消費した時間を持ち時間から引く。

        Args:
            color (Stone): 思考した手番の色。
            consumption_time (float): 消費した時間。
        """
        if color is Stone.BLACK:
            self.remaining_time[0] -= consumption_time
        if color is Stone.WHITE:
            self.remaining_time[1] -= consumption_time


    def set_mode(self, mode:TimeControl) -> NoReturn:
        """思考時間管理の設定を変更する。

        Args:
            mode (TimeControl): 指定する思考モード
        """
        self.mode = mode


    def start_timer(self) -> NoReturn:
        """思考時間の計測を開始する。
        """
        self.start_time = time.time()


    def calculate_consumption_time(self) -> float:
        """探索にかかった時間を算出する。

        Returns:
            float: 探索で消費した時間。
        """
        return time.time() - self.start_time

    def is_time_over(self) -> bool:
        """思考時間が予定をオーバーしたか判定する。

        Returns:
            bool: 思考時間超過判定結果。
        """
        if time.time() - self.start_time > self.time_limit:
            return True
        return False
