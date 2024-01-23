import json
from typing import Any, Tuple, List, Dict, NoReturn

from program import PROGRAM_NAME, VERSION, PROTOCOL_VERSION
from board.go_board import GoBoard, copy_board
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
        "dump_version": 2,
        "tree": tree_dict,
        "board_size": board.get_board_size(),
        "komi": board.get_komi(),
        "move_history": _serializable_move_history(board.get_move_history()),
        "handicap_history": board.get_handicap_history(),
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
    root_board = GoBoard(board_size=state["board_size"], komi=state["komi"], \
                         check_superko=state["superko"])
    root_board.set_history(_recovered_move_history(state["move_history"]), \
                           state["handicap_history"])

    coord = Coordinate(board_size=root_board.get_board_size())
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
    initial_move_color = _str_to_stone(tree["to_move"])
    for item in node:
        is_root = "parent_index" not in item
        if is_root:
            item["level"] = 0
            item["orders_along_path"] = []
            item["gtp_moves_along_path"] = []
            item["to_move"] = tree["to_move"]
            item["board_string"] = root_board.get_board_string()
            continue
        parent = node[item["parent_index"]]
        index_in_brother = item["index_in_brother"]
        gtp_move = coord.convert_to_gtp_format(parent["action"][index_in_brother])
        item["level"] = parent["level"] + 1
        item["orders_along_path"] = [*parent["orders_along_path"], item["order"]]
        item["to_move"] = _opposite_color(parent["to_move"])
        item["gtp_moves_along_path"] = [*parent["gtp_moves_along_path"], gtp_move]
        item["board_string"] = _get_updated_board_string(root_board, initial_move_color, \
                                                         item["gtp_moves_along_path"])
        # ルートノードは以下の項目を持たないことに注意
        item["policy"] = parent["children_policy"][index_in_brother]
        item["visits"] = parent["children_visits"][index_in_brother]
        item["value"] = parent["children_value"][index_in_brother]
        item["value_sum"] = parent["children_value_sum"][index_in_brother]
        item["gtp_move"] = gtp_move
        item["mean_value"] = item["value_sum"] / item["visits"]
        last_move_color = _opposite_color(item["to_move"])
        item["raw_black_winrate"] = _black_winrate(item["value"], last_move_color)
        item["mean_black_winrate"] = _black_winrate(item["mean_value"], last_move_color)

def _opposite_color(color):
    return 'white' if color == 'black' else 'black'

def _black_winrate(value, last_move_color):
    return value if last_move_color == "black" else 1.0 - value

def _serializable_move_history(move_history: List[Tuple[Stone, int, Any]]) -> List[Tuple[str, int]]:
    """着手の履歴をシリアライズ可能な値に変換する。ただしハッシュ値は廃棄する。

    Args:
        move_history (List[Tuple[Stone, int, np.array]]): 着手の履歴。

    Returns:
        Lizt[Tuple[str, int]]: シリアライズ可能なよう変換された着手履歴。
    """
    return [(_stone_to_str(color), pos) for (color, pos, _) in move_history]

def _recovered_move_history(converted_move_history: List[Tuple[str, int]]) -> List[Tuple[Stone, int, Any]]:
    """_serializable_move_historyで変換された着手履歴から元の着手履歴を復元する。
ただしハッシュ値はNoneに置きかえられる。

    Args:
        converted_move_history (Lizt[Tuple[str, int]]): 変換された着手履歴。

    Returns:
        List[Tuple[Stone, int, Any]]: 復元された着手履歴。
    """
    return [(_str_to_stone(color_str), pos, None) for (color_str, pos) in converted_move_history]

def _stone_to_str(color: Stone) -> str:
    return 'black' if color == Stone.BLACK else 'white'

def _str_to_stone(color_str: str) -> str:
    return Stone.BLACK if color_str == 'black' else Stone.WHITE

def _get_updated_board_string(root_board: GoBoard, initial_move_color: Stone, gtp_moves_along_path: List[str]) -> str:
    """一連の着手後の盤面を表わす文字列を返す。

    Args:
        root_board (GoBoard): 着手前の盤面。
        initial_move_color (Stone): 最初の着手の色。
        gtp_moves_along_path (List[str]): 着手位置のリスト。

    Returns:
        str: 着手後の盤面を表わす文字列。
    """
    coord = Coordinate(board_size=root_board.get_board_size())
    move_color = initial_move_color
    # 「board = copy.deepcopy(root_board)」は遅いので避ける。
    board = GoBoard(board_size=root_board.get_board_size(), komi=root_board.get_komi(), check_superko=root_board.check_superko)
    copy_board(dst=board, src=root_board)
    for (k, move) in enumerate(gtp_moves_along_path):
        pos = coord.convert_from_gtp_format(move)
        board.put_stone(pos, move_color)
        move_color = Stone.get_opponent_color(move_color)
    return board.get_board_string()
