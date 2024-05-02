import sys
import select
import time


def animate_mcts(mcts, board, to_move, pv_wait_sec, move_wait_sec):
    previous_pv = []
    def callback(path):
        _animate_path(path, mcts, board, pv_wait_sec, move_wait_sec, previous_pv)
        finished = _stdin_has_data()
        return finished
    mcts.search_with_callback(board, to_move, callback)


def _stdin_has_data():
    rlist, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(rlist)


def _animate_path(path, mcts, board, pv_wait_sec, move_wait_sec, previous_pv):
    # 今回探索した系列の属性値
    root_index, i = path[0]
    root = mcts.node[root_index]
    if root.children_visits[i] == 0:
        return
    coordinate = board.coordinate
    move = coordinate.convert_to_gtp_format(root.action[i])
    pv = [coordinate.convert_to_gtp_format(mcts.node[index].action[child_index]) for (index, child_index) in path]
    pv_visits = [str(mcts.node[index].children_visits[child_index]) for (index, child_index) in path]
    pv_winrate = [str(int(10000 * _get_winrate(mcts, index, child_index, depth))) for depth, (index, child_index) in enumerate(path)]

    # lz-analyze の本来の出力内容を加工
    children_status_list = root.get_analysis_status_list(board, mcts.get_pv_lists)
    fake_status_list = [status.copy() for status in children_status_list]
    target = next((status for status in fake_status_list if status["move"] == move), None)
    if target is None:
        return  # can't happen
    # 今回探索した系列の初手を最善手と偽って順位をふり直す
    target["order"] = -1
    fake_status_list.sort(key=lambda status: status["order"])
    for order, status in enumerate(fake_status_list):
        status["order"] = order

    # PV 欄を差しかえながら複数回出力することで一手ずつアニメーション
    for k in range(1, len(pv) + 1):
        # 前回の系列と共通な手順はスキップ
        if pv[:k] == previous_pv[:k]:
            continue

        target["pv"] = " ".join(pv[:k])
        target["pvVisits"] = " ".join(pv_visits[:k])
        target["pvWinrate"] = " ".join(pv_winrate[:k])

        sys.stdout.write(root.get_analysis_from_status_list("lz", fake_status_list))
        sys.stdout.flush()
        time.sleep(max(move_wait_sec, 0.0))

    previous_pv[:] = pv
    time.sleep(max(pv_wait_sec, 0.0))


def _get_winrate(mcts, index, child_index, depth):
    node = mcts.node[index]
    i = child_index
    visits = node.children_visits[i]
    value = node.children_value_sum[i] / visits if visits > 0 else node.children_value[i]
    winrate = value if depth % 2 == 0 else 1.0 - value
    return winrate
