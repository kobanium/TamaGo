#!/usr/bin/python3
"""GTPクライアントのエントリーポイント。
"""
import os
import click

from gtp.client import GtpClient
from board.constant import BOARD_SIZE
from mcts.constant import NN_BATCH_SIZE, MCTS_TREE_SIZE
from mcts.time_manager import TimeControl

default_model_path = os.path.join("model", "model.bin")

# pylint: disable=R0913, R0914

@click.command()
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, \
    help=f"碁盤のサイズを指定。デフォルトは{BOARD_SIZE}。")
@click.option('--superko', type=click.BOOL, default=False, help="超劫の有効化フラグ。デフォルトはFalse。")
@click.option('--model', type=click.STRING, default=default_model_path, \
    help=f"使用するニューラルネットワークのモデルパスを指定する。プログラムのホームディレクトリの相対パスで指定。\
    デフォルトは{default_model_path}。")
@click.option('--use-gpu', type=click.BOOL, default=False, \
    help="ニューラルネットワークの計算にGPUを使用するフラグ。デフォルトはFalse。")
@click.option('--policy-move', type=click.BOOL, default=False, \
    help="Policyの分布に従った着手生成処理フラグ。デフォルトはFalse。")
@click.option('--sequential-halving', type=click.BOOL, default=False, \
    help="Gumbel AlphaZeroの探索手法で着手生成するフラグ。デフォルトはFalse。")
@click.option('--komi', type=click.FLOAT, default=7.0, \
    help="コミの値の設定。デフォルトは7.0。")
@click.option('--visits', type=click.IntRange(min=1), default=1000, \
    help="1手あたりの探索回数の指定。デフォルトは1000。\
    --strict-visitsオプション、--const-timeオプション、または--timeオプションが指定された時は無視する。")
@click.option('--strict-visits', type=click.IntRange(min=1), \
    help="1手あたりの探索回数の厳密指定（着手が確定しても打ち切らない）。\
    --const-timeオプション、または--timeオプションが指定された時は無視する。")
@click.option('--const-time', type=click.FLOAT, \
    help="1手あたりの探索時間の指定。--timeオプションが指定された時は無視する。")
@click.option('--time', type=click.FLOAT, \
    help="持ち時間の指定。")
@click.option('--batch-size', type=click.IntRange(min=1), default=NN_BATCH_SIZE, \
    help=f"探索時のミニバッチサイズ。デフォルトはNN_BATCH_SIZE = {NN_BATCH_SIZE}。")
@click.option('--tree-size', type=click.IntRange(min=1), default=MCTS_TREE_SIZE, \
    help=f"探索木を構成するノードの最大数。デフォルトはMCTS_TREE_SIZE = {MCTS_TREE_SIZE}。")
@click.option('--cgos-mode', type=click.BOOL, default=False, \
    help="全ての石を打ち上げるまでパスしないモード設定。デフォルトはFalse。")
def gtp_main(size: int, superko: bool, model:str, use_gpu: bool, sequential_halving: bool, \
    policy_move: bool, komi: float, visits: int, strict_visits: int, const_time: float, time: float, \
    batch_size: int, tree_size: int, cgos_mode: bool):
    """GTPクライアントの起動。

    Args:
        size (int): 碁盤の大きさ。
        superko (bool): 超劫の有効化フラグ。
        model (str): プログラムのホームディレクトリからのモデルファイルの相対パス。
        use_gpu (bool):  ニューラルネットワークでのGPU使用フラグ。デフォルトはFalse。
        policy_move (bool): Policyの分布に従った着手生成処理フラグ。デフォルトはFalse。
        sequential_halving (bool): Gumbel AlphaZeroの探索手法で着手生成するフラグ。デフォルトはFalse。
        komi (float): コミの値。デフォルトは7.0。
        visits (int): 1手あたりの探索回数。デフォルトは1000。
        strict_visits (int): 1手あたりの厳密な探索回数（着手が確定しても打ち切らない）。
        const_time (float): 1手あたりの探索時間。
        time (float): 対局時の持ち時間。
        batch_size (int): 探索実行時のニューラルネットワークのミニバッチサイズ。デフォルトはNN_BATCH_SIZE。
        tree_size (int): 探索木を構成するノードの最大数。デフォルトはMCTS_TREE_SIZE。
        cgos_mode (bool): 全ての石を打ち上げるまでパスしないモード設定。デフォルトはFalse。
    """
    mode = TimeControl.CONSTANT_PLAYOUT

    if strict_visits is not None:
        mode = TimeControl.STRICT_PLAYOUT
        visits = strict_visits
    if const_time is not None:
        mode = TimeControl.CONSTANT_TIME
    if time is not None:
        mode = TimeControl.TIME_CONTROL

    program_dir = os.path.dirname(__file__)
    client = GtpClient(size, superko, os.path.join(program_dir, model), use_gpu, policy_move, \
        sequential_halving, komi, mode, visits, const_time, time, batch_size, tree_size, \
        cgos_mode)
    client.run()


if __name__ == "__main__":
    gtp_main() # pylint: disable=E1120
