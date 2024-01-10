import json
from typing import Any, Dict, NoReturn

from program import PROGRAM_NAME, VERSION, PROTOCOL_VERSION
from board.go_board import GoBoard
from board.coordinate import Coordinate
from board.stone import Stone
from mcts.constant import NOT_EXPANDED

def dump_mcts_to_json(tree_dict: Dict[str, Any], board: GoBoard, superko: bool) -> str:
    """MCTSの状態を表すJSON文字列を返す。

    Args:
        tree_dict (Dict[str, Any]): 辞書化された「ツリーの状態」。
        board (GoBoard): 現在の碁盤。
        superko (bool): 超劫判定の有効化。

    Returns:
        str: MCTSの状態を表すJSON文字列。
    """
    state = {
        "dump_version": 1,
        "tree": tree_dict,
        "board_size": board.get_board_size(),
        "komi": board.get_komi(),
        "superko": superko,
        "name": PROGRAM_NAME,
        "version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
    }
    return json.dumps(state)

def enrich_mcts_dict(state: Dict[str, Any]) -> NoReturn:
    """MCTSの状態を表す辞書に便利項目をいろいろ追加する。

    Args:
        state (Dict[str, Any]): MCTSの状態を表す辞書。
    """
    coord = Coordinate(board_size=state["board_size"])
    tree = state["tree"]
    node = tree["node"]

    # index, parent_index, index_in_brother の逆引き
    for index, item in enumerate(node):
        item["index"] = index
        for index_in_brother, child_index in enumerate(item["children_index"]):
            if child_index == NOT_EXPANDED:
                continue
            child = node[child_index]
            child["parent_index"] = index
            child["index_in_brother"] = index_in_brother
            # 以下のコードは次の条件を前提としている。
            # （将来もし tree.node の仕様が変わったら見落さないようにチェック）
            assert index < child_index, "Parent index must be less than child index."
            assert child_index < tree["num_nodes"], "Child index must be less than num_nodes."

    # 「親は子より前」「兄弟は order の小さい方が前」を保証したリスト
    sorted_indices_list = []
    tree["sorted_indices_list"] = sorted_indices_list

    # expanded_children_index, sorted_indices_list, 兄弟内 order
    root_node = node[tree["current_root"]]
    nodes_pool = [root_node]
    while nodes_pool:
        item = nodes_pool.pop(0)
        sorted_indices_list.append(item["index"])
        expanded_children_index = [i for i in item["children_index"] if i != NOT_EXPANDED]
        item["expanded_children_index"] = expanded_children_index
        expanded_children = [node[i] for i in expanded_children_index]
        expanded_children.sort(key=lambda item: item["node_visits"], reverse=True)
        for order, child in enumerate(expanded_children):
            child["order"] = order
        nodes_pool += expanded_children

    # その他いろいろな便利項目を追加
    for item in node:
        is_root = "parent_index" not in item
        if is_root:
            item["level"] = 0
            item["orders_along_path"] = []
            item["to_move"] = tree["to_move"]
            continue
        parent = node[item["parent_index"]]
        item["level"] = parent["level"] + 1
        item["orders_along_path"] = [*parent["orders_along_path"], item["order"]]
        item["to_move"] = _opposite_color(parent["to_move"])
        # ルートノードは以下の項目を持たないことに注意
        index_in_brother = item["index_in_brother"]
        item["policy"] = parent["children_policy"][index_in_brother]
        item["visits"] = parent["children_visits"][index_in_brother]
        item["value"] = parent["children_value"][index_in_brother]
        item["value_sum"] = parent["children_value_sum"][index_in_brother]
        item["gtp_move"] = coord.convert_to_gtp_format(parent["action"][index_in_brother])
        item["mean_value"] = item["value_sum"] / item["visits"]
        last_move_color = _opposite_color(item["to_move"])
        item["raw_black_winrate"] = _black_winrate(item["value"], last_move_color)
        item["mean_black_winrate"] = _black_winrate(item["mean_value"], last_move_color)

def _opposite_color(color):
    return 'white' if color == 'black' else 'black'

def _black_winrate(value, last_move_color):
    return value if last_move_color == "black" else 1.0 - value
