import random
import torch

from board.constant import PASS, RESIGN
from board.go_board import GoBoard, copy_board
from board.stone import Stone

from sgf.selfplay_record import SelfPlayRecord
from mcts.tree import MCTSTree
from mcts.time_manager import TimeManager, TimeControl
from nn.network.dual_net import DualNet
from nn.utility import load_network
from learning_param import SELF_PLAY_VISITS






def selfplay_worker():
    use_gpu = True
    visits = SELF_PLAY_VISITS


    board = GoBoard(board_size=9, komi=7.0, check_superko=True)
    init_board = GoBoard(board_size=9, komi=7.0, check_superko=True)
    record = SelfPlayRecord(board.coordinate)
    network = load_network(model_file_path="model/model.bin", use_gpu=use_gpu)

    mcts = MCTSTree(network, tree_size=SELF_PLAY_VISITS * 10)
    time_manager = TimeManager(TimeControl.CONSTANT_PLAYOUT, constant_visits=visits)

    for _ in range(2):
        copy_board(board, init_board)
        color = Stone.BLACK
        record.clear()
        pass_count = 0
        never_resign = True if random.randint(1, 10) == 1 else False
        is_resign = False
        score = 0.0
        for _ in range(9 * 9 * 3):
            pos = mcts.generate_move_with_sequential_halving(board=board, color=color, \
                time_manager=time_manager, never_resign=never_resign)

            if pos == RESIGN:
                winner = Stone.get_opponent_color(color)
                is_resign = True
                break

            board.put_stone(pos, color)

            if pos == PASS:
                pass_count += 1
            else:
                pass_count = 0

            record.save_record(mcts.get_root(), pos, color)

            color = Stone.get_opponent_color(color)

            if pass_count == 2:
                winner = Stone.EMPTY
                break

        if pass_count == 2:
            score = board.count_score() - board.get_komi()
            if score > 0.1:
                winner = Stone.BLACK
            elif score < -0.1:
                winner = Stone.WHITE
            else:
                winner = Stone.OUT_OF_BOARD
            print(score)

        record.write_record(winner, board.get_komi(), is_resign, score)
