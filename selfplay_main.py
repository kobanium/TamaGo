"""自己対戦のエントリーポイント。
"""
import glob
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
import click
from board.constant import BOARD_SIZE
from selfplay.worker import selfplay_worker
from learning_param import SELF_PLAY_VISITS, NUM_SELF_PLAY_WORKERS, \
    NUM_SELF_PLAY_GAMES

# pylint: disable=R0913, R0914
@click.command()
@click.option('--save-dir', type=click.STRING, default="archive", \
    help="棋譜ファイルを保存するディレクトリ。デフォルトはarchive。")
@click.option('--process', type=click.IntRange(min=1), default=NUM_SELF_PLAY_WORKERS, \
    help=f"自己対戦実行ワーカ数。デフォルトは{NUM_SELF_PLAY_WORKERS}。")
@click.option('--num-data', type=click.IntRange(min=1), default=NUM_SELF_PLAY_GAMES, \
    help="生成するデータ(棋譜)の数。デフォルトは10000。")
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, \
    help=f"碁盤のサイズ。デフォルトは{BOARD_SIZE}。")
@click.option('--use-gpu', type=click.BOOL, default=True, \
    help="GPU使用フラグ。デフォルトはTrue。")
@click.option('--visits', type=click.IntRange(min=2), default=SELF_PLAY_VISITS, \
    help=f"自己対戦時の探索回数。デフォルトは{SELF_PLAY_VISITS}。")
@click.option('--model', type=click.STRING, default=os.path.join("model", "rl-model.bin"), \
    help="ニューラルネットワークのモデルファイルパス。デフォルトはmodelディレクトリ内のrl-model.bin。")
def selfplay_main(save_dir: str, process: int, num_data: int, size: int, \
    use_gpu: bool, visits: int, model: str):
    """自己対戦を実行する。

    Args:
        save_dir (str): 棋譜ファイルを保存するディレクトリ。デフォルトはarchive。
        process (int): 実行する自己対戦プロセス数。デフォルトは4。
        num_data (int): 生成するデータ数。デフォルトは10000。
        size (int): 碁盤のサイズ。デフォルトはBOARD_SIZE。
        use_gpu (bool): GPU使用フラグ。デフォルトはTrue
        visits (int): 自己対戦実行時の探索回数。デフォルトはSELF_PLAY_VISITS。
        model (str): 使用するモデルファイルのパス。デフォルトはmodel/model.bin。
    """
    file_index_list = list(range(1, num_data + 1))
    split_size = math.ceil(num_data / process)
    file_indice = [file_index_list[i:i+split_size] \
        for i in range(0, len(file_index_list), split_size)]
    kifu_dir_index_list = [int(os.path.split(dir_path)[-1]) \
        for dir_path in glob.glob(os.path.join(save_dir, "*"))]
    kifu_dir_index_list.append(0)
    kifu_dir_index = max(kifu_dir_index_list) + 1

    start_time = time.time()
    os.mkdir(os.path.join(save_dir, str(kifu_dir_index)))

    print(f"Self play visits : {visits}")

    with ProcessPoolExecutor(max_workers=process) as executor:
        futures = [executor.submit(selfplay_worker, os.path.join(save_dir, str(kifu_dir_index)), \
            model, file_list, size, visits, use_gpu) for file_list in file_indice]
        for future in futures:
            future.result()

    finish_time = time.time() - start_time

    print(f"{finish_time:3f} seconds, {(3600.0 * num_data / finish_time):3f} games/hour")


if __name__ == "__main__":
    selfplay_main() # pylint: disable=E1120
