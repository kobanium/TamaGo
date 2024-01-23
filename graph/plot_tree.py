#!/bin/env python3

import sys
import click
import math
import json
import gzip
import graphviz
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 親ディレクトリからたどって import するための準備
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from mcts.dump import enrich_mcts_dict

@click.command()
@click.argument('input_json_path', type=click.Path(exists=True))
@click.argument('output_image_path', type=click.Path())
@click.option('--around-pv', type=click.BOOL, default=False, \
    help="主分岐のまわりのみ表示するフラグ。デフォルトはFalse。")
def plot_tree_main(input_json_path: str, output_image_path: str, around_pv: bool):
    # docstring 中の \b は click による rewrapping の抑止（入れないと改行が無視される）
    # https://click.palletsprojects.com/en/8.1.x/documentation/#preventing-rewrapping
    """MCTSツリーを可視化。

\b
    Args:
        input_json_path (str): MCTSの状態を表すJSONファイルのパス。
        output_image_path (str): 可視化結果を保存する画像ファイルのパス。
        around_pv (bool): 最善応手系列の周辺のみ表示するフラグ。デフォルトはFalse。

\b
    Example:
        cd tamago
        (echo 'tamago-readsgf (;SZ[9]KM[7];B[fe];W[de];B[ec])';
         echo 'lz-genmove_analyze 7777777';
         echo 'undo';
         echo 'tamago-dump_tree') \\
        | python3 main.py --model model/model.bin --strict-visits 100 \\
        | grep dump_version | gzip > tree.json.gz
        python3 graph/plot_tree.py tree.json.gz tree_graph
        display tree_graph.svg
    """

    opener = gzip.open if input_json_path.endswith('.gz') else open
    with opener(input_json_path, 'r') as file:
        state = json.load(file)

    enrich_mcts_dict(state)
    tree = state["tree"]
    node = tree["node"]
    sorted_indices_list = tree["sorted_indices_list"]

    # colormap = plt.cm.get_cmap('coolwarm_r')
    colormap = plt.cm.get_cmap('Spectral')
    # colormap = plt.cm.get_cmap('RdYlBu')
    # colormap = plt.cm.get_cmap('viridis')

    dot = graphviz.Digraph(comment='Visualization of MCTS Tree')

    for index in sorted_indices_list:
        item = node[index]
        item_id = get_graphviz_id(index, node)
        # ルートノードの場合
        if "parent_index" not in item:
            dot.node(item_id, label=f"root\n{item['node_visits']} visits")
            continue

        parent_index = item['parent_index']
        parent = node[parent_index]
        parent_id = get_graphviz_id(parent_index, node)
        # around_pv が指定された場合は、PV とその直下の子のみ表示する。
        if around_pv and any(order > 0 for order in parent["orders_along_path"]):
            continue

        # ノードの作成
        move = item['gtp_move']
        visits = item['visits']
        winrate = item['mean_black_winrate']
        raw_winrate = item['raw_black_winrate']
        node_color = get_color(winrate, colormap)
        border_color = get_color(raw_winrate, colormap)
        text_color = 'black' if abs(winrate - 0.5) < 0.25 else 'white'
        # 黒の着手（次が白番）は□、白の着手は○でノードを描く
        shape = 'square' if item["to_move"] == 'white' else 'circle'
        wr = int(winrate * 100)
        raw_wr = int(raw_winrate * 100)
        label = f"{move}\n{wr}%" if visits < 10 else f"{move}\n{wr}% (raw {raw_wr}%)\n{visits} visits"
        dot.node(
            item_id,
            label=label,
            color=border_color,
            fillcolor=node_color,
            fontcolor=text_color,
            style='filled',
            penwidth='5.0',
            height=get_size(visits, shape),
            fixedsize='true',
            shape=shape,
        )

        # エッジの作成
        freshness = (item['index'] + 1) / len(node)
        whiteness = 0.9
        c = f"{int(freshness * whiteness * 255):02x}"
        color = f"#{c}{c}{c}"
        penwidth = max(0.5, item['policy'] * 10)
        dot.edge(parent_id, item_id, color=color, penwidth=f"{penwidth}")

    dot.render(output_image_path, format='svg', view=False, cleanup=True)

def get_color(value, colormap):
    emphasis = 1.5  # 色の違いを強調
    v = 0.5 + (value - 0.5) * emphasis
    return mcolors.to_hex(colormap(v))

def get_size(visits, shape):
    size0 = 0.5 + math.log10(visits)
    # 正方形と円の面積が同じになるように
    size = size0 if shape == 'square' else size0 * 2 / (math.pi ** 0.5)
    return str(size)

def get_graphviz_id(index, node):
    max_board_str_len = 400  # 9路盤が340文字程度
    index_str = f"node{index}"
    # IDが文字「:」を含むと、graphvizで不具合が生じる。
    board_str = node[index]['board_string'].replace(':', ' ')
    too_long = len(board_str) > max_board_str_len
    return index_str if too_long else f"{index_str}\n{board_str}"

if __name__ == "__main__":
    plot_tree_main()
