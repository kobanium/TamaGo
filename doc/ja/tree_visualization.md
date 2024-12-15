# 探索木の可視化について
TamaGoは探索木の可視化機能をサポートしています。

## 可視化機能の実行例
```
(echo 'tamago-readsgf (;SZ[9]KM[7];B[fe];W[de];B[ec])';
 echo 'lz-genmove_analyze 7777777';
 echo 'undo';
 echo 'tamago-dump_tree') \
| python3 main.py --model model/model.bin --strict-visits 100 \
| grep dump_version | gzip > tree.json.gz
python3 graph/plot_tree.py tree.json.gz tree_graph
display tree_graph.svg
```

![探索木の可視化結果](../../img/tree_graph.png)

## graph/plot_tree.pyのコマンドライン引数

| 引数 | 概要 | 設定する値 | 設定値の例 | 備考 |
|---|---|---|---|---|
| INPUT_JSON_PATH | tamago-dump_treeコマンドを実行した結果のJSONファイルのパス | tree.json.gz | |
| OUTPUT_IMAGE_PATH | 可視化結果を保持する画像ファイルのパス | tree_graph | | 拡張子(.svg)が自動的に付与される |

## graph/plot_tree.pyのオプション

| オプション | 概要 | 設定する値 | 設定値の例 | デフォルト値 | 備考 |
|---|---|---|---|---|---|
| `--around-pv` | 主分岐のまわりのみ表示するフラグ | true または false | true | false | |
