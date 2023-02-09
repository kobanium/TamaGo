"""教師あり学習のエントリーポイント。
"""
import os
import click
from learning_param import BATCH_SIZE, EPOCHS
from board.constant import BOARD_SIZE
from nn.learn import train
from nn.data_generator import generate_supervised_learning_data


@click.command()
@click.option('--kifu-dir', type=click.STRING, \
    help="学習データの棋譜ファイルを格納したディレクトリのパス。指定がない場合はデータ生成を実行しない。")
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, \
    help=f"碁盤の大きさ。最小2, 最大{BOARD_SIZE}")
def train_main(kifu_dir: str, size: int):
    """教師あり学習のデータ生成と学習を実行する。

    Args:
        kifu_dir (str): 学習する棋譜ファイルを格納したディレクトリパス。
        size (int): 碁盤の大きさ。
    """
    program_dir = os.path.dirname(__file__)

    # 学習データの指定がある場合はデータを生成する
    if kifu_dir is not None:
        generate_supervised_learning_data(program_dir=program_dir, \
            kifu_dir=kifu_dir, board_size=size)

    train(program_dir=program_dir,board_size=size, \
        batch_size=BATCH_SIZE, epochs=EPOCHS, use_gpu=True)


if __name__ == "__main__":
    train_main() # pylint: disable=E1120
