# 關於監督學習

此篇是關於如何使用 TamaGo 的監督學習系統。

#前置作業

TamaGo 可以使用 SGF 格式的棋譜作為訓練資料，關於 SGF 的更多資訊請看[這裡](https://www.red-bean.com/sgf/)。所使用的棋譜必須符合以下條件

- 棋譜中的下一手不能有分支
- 必須要有 ```B``` 和 ```W``` 標籤
- 必須要有 ```RE``` 標籤
- 編碼格式必須是 UTF-8 或 ASCII

# 使用的參數

監督學習的參數定義在 [learning_param.py](../../learning_param.py)。

| 選項 | 描述 | 預設值 | 備註 |
| --- | --- | --- | --- |
| SL_LEARNING_RATE | 監督學習使用的初始學習率 | 0.01 |  |
| BATCH_SIZE | 監督學習使用的 batch size | 256 | 如果 GPU 記憶體不夠，請使用較小的值。 |
| MOMENTUM | 優化器使用的 Momentum 參數 | 0.9 | 基本上不需要修改 |
| WEIGHT_DECAY | 優化器使用的 weight decay 參數 | 1e-4 (0.0001) | 基本上不需要修改  |
| EPOCHS | 監督學習的 epochs 總數 | 15 |  |
| LEARNING_SCHEDULE | 監督學習的學習率排班 | 參考 learning_param.py |  |
| DATA_SET_SIZE | 每個 npz 檔案包含的資料個數 | BATCH_SIZE * 4000 | 請根據記憶體大小修改 |
| SL_VALUE_WEIGHT | value loss 和 policy loss 的平衡值 | 0.02 | 必須大於 0.0 |

# 網路結構的定義

下列 4 個檔案定義網路結構

| 檔案 | 定義 |
| --- | --- |
| [nn/network/dual_net.py](../../nn/network/dual_net.py) | 定義整體網路 |
| [nn/network/res_block.py](../../nn/network/res_block.py) | 定義 Residual Block |
| [nn/network/head/policy_head.py](../../nn/network/head/policy_head.py) | 定義 Policy Head |
| [nn/network/head/value_head.py](../../nn/network/head/value_head.py) | 定義 Value Head |

如果你想改進網路結構，建議可以先簡單的提昇 filter 和 block 數目，它們定義在 dual_net.py 裡

# TamaGo 的監督學習步驟

請通過以下順序執行 TamaGo 的監督學習

1. 從指定路徑讀取 SGF 棋譜並轉換成 ```sl_data_*.npz``` 儲存到 ```data``` 路徑下。
2. 從 ```data``` 中讀取 ```sl_data_*.npz``` 訓練網路。

## 產生訓練資料

每次訓練時，產生訓練資料的過程不是必須的，除非

- 當你想要更換訓練資料
- 當你想要更新網路輸入的特徵

## 監督學習的步驟

當執行監督學習時，會產生 ```sl_data_*.npz``` 檔案到 ```data``` 路徑下，並以 8:2 切分訓練集和驗證集。如果你想要改變切分比例，請到 [learn.py](../../nn/learn.py) 改變之。

```
train_data_set, test_data_set = split_train_test_set(data_set, 0.8)
```

# 如何執行監督學習

假設棋譜位於 ```home/user/sgf_files``` 路徑下，執行以下指令即可訓練

```
python train.py --kifu-dir /home/user/sgf_files
```

如果已經有 ```sl_data_*.npz``` 檔案，則不需要指定路徑

```
python train.py
```

訓練完成後，會產生 ```sl-model.bin``` 在 ```model``` 路徑下，使用方法請看[這裡](README.md)。

## 訓練時的參數選項

| 選項 | 描述 | 設定範例 | 預設值 | 備註 |
| --- | --- | --- | --- | --- |
| `--kifu-dir` | 訓練時使用的棋譜的路徑 | /home/user/sgf_files | None | |
| `--size` | 棋盤大小 | 9 | 9 |  |
| `--use-gpu` | 是否使用 GPU | true | true | Value is true or false. |
| `--rl` | 是否為強化學習 | false | false |  |
| `--window-size` | 學習時使用的資料數目（最新的優先讀取） | 500000 | 300000 | |
