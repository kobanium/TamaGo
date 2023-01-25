"""GTPクライアントのエントリーポイント。
"""
import click

from gtp.client import GtpClient
from board.constant import BOARD_SIZE


@click.command()
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, help="")
def gtp_main(size):
    """GTPクライアントの起動。

    Args:
        size (int): 碁盤の大きさ。
    """
    client = GtpClient(size)
    client.run()


if __name__ == "__main__":
    gtp_main()
