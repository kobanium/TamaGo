


# 学習率
LEARNING_RATE = 0.02

# ミニバッチサイズ
BATCH_SIZE = 256

# 学習器のモーメンタムパラメータ
MOMENTUM=0.9

#
WEIGHT_DECAY =4e-5

# 学習率を減らすepochと減らす割合
LEARNING_SCHEDULE = {
    "decay_epoch" : [5, 8, 10],
    "decay_rate": [0.1, 0.1, 0.1]
}


# npzファイル1つに格納するデータの個数
DATA_SET_SIZE = BATCH_SIZE * 1000
