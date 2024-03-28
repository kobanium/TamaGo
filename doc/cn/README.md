# TamaGo

* 此翻譯對應 v0.9.2

TamaGo 是以一個用純 python 撰寫的圍棋程式。它有以下的功能

 - 通過監督學習（Supervised Learning）學習 SGF 格式的棋譜。
 - 使用 Gumbel AlphaZero 實做強化學習（Reinforcement Learning）系統。
 - 使用神經網路搭配蒙地卡羅樹搜尋

TamaGo 需要 python 3.6 或更高的版本，但以下命令都簡稱為 python

* [需求](#需求)
* [安裝套件](#安裝套件)
* [如何啟用 GTP 引擎](#如何啟用-GTP-引擎)
* [如何使用監督學習](#如何使用監督學習)
* [如何使用強化學習](#如何使用強化學習)
* [授權條款](#授權條款)

# 需求
| 套件 | 用途 |
|---|---|
| click | 支援內建選項功能 |
| numpy | 用於計算和其它 |
| pytorch | 構成神經網路和學習 |

# 安裝套件

可通過 Python 輸入下列指令直接安裝依賴的套件。
```
pip install -r requirements.txt
```

TamaGo 可以僅使用 CPU 執行，但推薦在有 GPU 機器上執行之，執行神經網路運算會快數倍。


# 如何啟用 GTP 引擎

TamaGo 支援 GTP 協議的界面軟體（GoGui, Sabaki, Lizzie 等），使用下列指令即可啟動之

```
python main.py
```

TamaGo 支援以下的指令

| 選項 | 描述 | 設定值 | 預設值 | 設定範例 | 備註 |
|---|---|---|---|---|---|
| `--size` | 棋盤大小 | 2 以上 BOARD_SIZE 以下 | 9 | BOARD_SIZE | BOARD_SIZE 定義在 board/constant.py|
| `--superko` | 是否使用禁全同規則 | true 或是 false | true | false | 僅支援 positional superko |
| `--model` | 使用的網路權重 |  | model/model.bin | 無 |  指定的路徑必須要在 TamaGo 的路徑之下 |
| `--use-gpu` | 是否使用 GPU | true 或 false | true | false | |
| `--policy-move` | 是否只使用 Policy 網路下棋 | true 或 false | true | false | 主要是為了確認 Policy 網路的強度 |
| `--sequential-halving` | 搜尋時是否使用 Sequential Halving | true 或 false | true | false | 主要是為了 debug |
| `--visits` | 每手棋的訪問數 | 1 以上的整數 | 1000 | 1000 | 當使用 --const-time 或 --time 參數時，此選項會被忽略 |
| `--const-time` | 每手棋的思考時間 (秒) | 0 以上的任意數 | 10.0 |  | 當使用 --time 參數時，此選項會被忽略 |
| `--time` | 總思考時間 (秒) | 0 以上的任意數 | 600.0 | |
| `--batch-size` | 搜尋使用的 batch 大小 | 大於零的整數 | 1 | NN_BATCH_SIZE | NN_BATCH_SIZE 定義在 mcts/constant.py |
| `--tree-size` | MCTS 的節點上限 | 大於 0 的整數 | 65536 | MCTS_TREE_SIZE | MCTS_TREE_SIZE 定義在  mcts/constant.py. |
| `--cgos-mode` | 嘗試提光所有死棋 | true or false | true | false | |

## 執行範例

1) 設定棋盤大小為 5，使用的權重為 model/model.bin ，禁止使用 GPU
```
python main.py --size 5 --model model/model.bin --use-gpu false
```

2) 啟用禁全同規則
```
python main.py --superko true
```

3) 使用的權重為 model/model.bin，使用 policy 網路產生下一手
```
python main.py --model model/sl-model.bin --policy-move true
```

4) 設定總思考時間為 10 分鐘
```
python main.py --time 600
```

5) 設定每手的訪問數為 500
```
python main.py --visits 500
```

6) 設定每手的思考時間為 10 秒
```
python main.py --const-time 10.0
```

7) 運行在 CGOS 上
```
python main.py --model model/sl-model.bin --use-gpu true --cgos-mode true --superko true --batch-size 13 --time 600 --komi 7 --tree-size 200000
```

## 使用預先訓練好的權重

你可以從[這裡](https://github.com/kobanium/TamaGo/releases)下載預先訓練好的權重，將權重改名成 ```model.bin``` 後放置在 ```model``` 路徑之下即可使用。請注意每不同版本的網路結構可能不一樣，不同版本的 TamaGo 可能需要對應對不同版本的權重。

* 0.3.0 版本，不使用搜索的條件下強 GNU Go（lv10） 約 90 elo，搜索的條件下大概強約 160 elo。
* 0.6.0 版本後，網路的結構有改變，早於此版本的網路無法使用。
* 0.6.3 版本，強 GNU Go（lv10） 約 420 elo。搜索的條件下（100 visits/move），強 [Ray](https://github.com/kobanium/Ray)（10k playouts/move） 約 180 elo。

# 如何使用監督學習
監督學習教程請看[這裡](supervised_learning.md)。

# 如何使用強化學習
強化學習教程請看[這裡](reinforcement_learning.md)。

# GoGui 分析指令
[GoGui](https://sourceforge.net/projects/gogui/) 可以用顏色顯示 policy 網路輸出的分佈，或是用數值直接顯示。

![Display policy value](../../img/gogui_analyze_policy.png)

下圖裡的越紅位置代表其 policy 的數值越高。

![Coloring policy value](../../img/gogui_analyze_policy_color.png)

# Analyze 指令

自 0.7.0 版本後，TamaGo 支援 lz-analyze, lz-genmove_analyze 分析指令，可以在 Sabaki 或 Lizzie 上使用。

![lz-analyze-sample](../../img/lz_analyze_sample.png)

# CGOS 分析模式

自 0.7.0 版本後，TamaGo 支援 cgos-analyze, cgos-genmove_analyze 分析指令，可以在 [Computer Go Server (CGOS)](http://www.yss-aya.com/cgos/) 上顯示分析結果。由於 CGOS 使用的規則需要提光所有死棋才能正確計地，因此建議 --cgos-mode 選項。

![cgos-analyze](../../img/cgos-analyze.png)

![cgos-analyze-pv](../../img/cgos-analyze-pv.png)


# 授權條款
TamaGo 使用 [Apache License 2.0](LICENSE) 授權。
