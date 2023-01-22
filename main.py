from gtp.client import GtpClient
from board.constant import BOARD_SIZE
from board.go_board import GoBoard
from board.string import StringData
from board.stone import Stone

import click


@click.command()
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, help="")
def gtp_main(size):
    #call_gtp_client()
    client = GtpClient(size)
    client.run()


if __name__ == "__main__":
    gtp_main()
