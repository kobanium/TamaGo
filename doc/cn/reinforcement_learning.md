# 關於強化學習

此教程是關於如何使用 TamaGo 的 Gumbel AlphaZero 強化學習系統。

# 前置作業

GNU Go 將會被用於確認自對戰的最終結果，它並不是必須的，但依然建議安裝下載，因為 TamaGo 本身對的自對戰的判斷有點不準確。

在 Ubuntu 上安裝 GNU Go 請輸入

~~~
apt install gnugo
~~~

# 使用的參數

強化學習的參數定義在 [learning_param.py](../../learning_param.py)。

| 選項 | 描述 | 預設值 | 備註 |
| --- | --- | --- | --- |
| RL_LEARNING_RATE | 強化學習使用的學習率 | 0.01 | 學習到一定程度後，可以調降學習率繼續訓練。 |
| BATCH_SIZE | 強化學習使用的 batch size | 256 | 如果 GPU 記憶體不夠，請使用較小的值。 |
| MOMENTUM | 優化器使用的 Momentum 參數 | 0.9 | 基本上不需要修改 |
| WEIGHT_DECAY | 優化器使用的 weight decay 參數 | 1e-4 (0.0001) | 基本上不需要修改  |
| DATA_SET_SIZE | 每個 npz 檔案包含的資料個數 | BATCH_SIZE * 4000 | 請根據記憶體大小修改 |
| RL_VALUE_WEIGHT | value loss 相對於 policy loss 的平衡參數 | 1.0 | 必須大於 0.0 |
| SELF_PLAY_VISITS | 每手棋使用的訪問數 | 16 | 必須大於 2 |
| NUM_SELF_PLAY_WORKERS | 自對戰使用的 worker 數 | 4 | 請根據自身的 CPU 改變參數 |
| NUM_SELF_PLAY_GAMES | 每回合自對戰的盤數 | 10000 | 數值太小容易學習到錯誤下法 |

# 網路結構的定義

下列 4 個檔案定義網路結構

| 檔案 | 定義 |
| --- | --- |
| [nn/network/dual_net.py](../../nn/network/dual_net.py) | 定義整體網路 |
| [nn/network/res_block.py](../../nn/network/res_block.py) | 定義 Residual Block |
| [nn/network/head/policy_head.py](../../nn/network/head/policy_head.py) | 定義 Policy Head |
| [nn/network/head/value_head.py](../../nn/network/head/value_head.py) | 定義 Value Head |

# TamaGo 的強化學習步驟

請通過以下順序執行 TamaGo 的強化學習

1. 使用已有的網路進行自對戰產生棋譜（[selfplay_main.py](../../selfplay_main.py)）
2. 用 GNU Go 確認並調整棋譜的勝負結果（可選）
3. 用棋譜和當前網路訓練新的權重（[selfplay_main.py](../../selfplay_main.py)）
4. 重複 1-3 過程

可以直接用 [pipeline.sh](../../pipeline.sh) 執行整個過程。


## 自對戰的參數選項([selfplay_main.py](../../selfplay_main.py))

| 選項 | 描述 | 設定範例 | 預設值 | 備註 |
| --- | --- | --- | --- | --- |
| `--save-dir` | 儲存自對戰棋譜的路徑 | save_dir | archive | |
| `--process` | 自對戰的 worker 數 | 2 | NUM_SELF_PLAY_WORKERS | |
| `--num-data` | 每回合自對戰的盤數 | 5000 | NUM_SELF_PLAY_GAMES | |
| `--size` | 棋盤大小 | 9 | 9 | |
| `--use-gpu` | 是否使用 GPU | true | true | 數值 True 或 False |
| `--visits` | 每手棋使用的訪問數 | 100 | SELF_PLAY_VISITS |訪問數值越高，則棋譜品質越好，但自對戰效率越差 |
| `--model` |  權重存放的路徑 | model/rl-model.bin | model/rl-model.bin | |

## 訓練時的參數選項([train.py](../../train.py))

| 選項 | 描述 | 設定範例 | 預設值 | 備註 |
| --- | --- | --- | --- | --- |
| `--kifu-dir` | 訓練時使用的棋譜的路徑 | /home/user/sgf_files | None | |
| `--size` | 棋盤大小 | 5 | 9 |  |
| `--use-gpu` | 是否使用 GPU | true | true |  true 或是 false. |
| `--rl` | 是否為強化學習 | false | false |  |
| `--window-size` | 學習時使用的資料數目（最新的優先讀取）  | 500000 | 300000 | |

