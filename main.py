"""GTPクライアントのエントリーポイント。
"""
import os
import click

from gtp.client import GtpClient
from board.constant import BOARD_SIZE


@click.command()
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, help="")
@click.option('--superko', type=click.BOOL, default=False, help="")
@click.option('--model', type=click.STRING, default=os.path.join("model", "model.bin"), help="")
@click.option('--use-gpu', type=click.BOOL, default=False, help="")
def gtp_main(size: int, superko: bool, model:str, use_gpu: bool):
    """GTPクライアントの起動。

    Args:
        size (int): 碁盤の大きさ。
        superko (bool): 超劫の有効化フラグ。
    """
    program_dir = os.path.dirname(__file__)
    client = GtpClient(size, superko, os.path.join(program_dir, model), use_gpu)
    client.run()


if __name__ == "__main__":
    gtp_main() # pylint: disable=E1120
