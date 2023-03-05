"""GTPクライアントのエントリーポイント。
"""
import os
import click

from gtp.client import GtpClient
from board.constant import BOARD_SIZE

default_model_path = os.path.join("model", "model.bin")

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
@click.option('--komi', type=click.FLOAT, default=7.0, \
    help="コミの値の設定。デフォルトは7.0。")
def gtp_main(size: int, superko: bool, model:str, use_gpu: bool, policy_move: bool, komi: float): # pylint: disable=R0913
    """GTPクライアントの起動。

    Args:
        size (int): 碁盤の大きさ。
        superko (bool): 超劫の有効化フラグ。
        model (str): プログラムのホームディレクトリからのモデルファイルの相対パス。
        use_gpu (bool):  ニューラルネットワークでのGPU使用フラグ。デフォルトはFalse。
        policy_move (bool): Policyの分布に従った着手生成処理フラグ。デフォルトはFalse。
        komi (float): コミの値。デフォルトは7.0。
    """
    program_dir = os.path.dirname(__file__)
    client = GtpClient(size, superko, os.path.join(program_dir, model), use_gpu, policy_move, komi)
    client.run()


if __name__ == "__main__":
    gtp_main() # pylint: disable=E1120
