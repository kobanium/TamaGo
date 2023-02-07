"""学習用の各種ハイパーパラメータの設定。
"""

# 学習率
LEARNING_RATE = 0.02

# ミニバッチサイズ
BATCH_SIZE = 256

# 学習器のモーメンタムパラメータ
MOMENTUM=0.9

#
WEIGHT_DECAY = 1e-4

EPOCHS = 15

# 学習率を変更するエポック数と辺豪語の学習率
LEARNING_SCHEDULE = {
    "decay_epoch": [5, 8, 10],
    "learning_rate": {
        5: 0.002,
        8: 0.0002,
        10: 0.00002,
    }
}

# npzファイル1つに格納するデータの個数
DATA_SET_SIZE = BATCH_SIZE * 4000

# Policyのlossに対するValueのlossの重み比率
SL_VALUE_WEIGHT = 0.02
