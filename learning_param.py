"""学習用の各種ハイパーパラメータの設定。
"""

# 学習率
LEARNING_RATE = 0.02

# ミニバッチサイズ
BATCH_SIZE = 256

# 学習器のモーメンタムパラメータ
MOMENTUM=0.9

# L2正則化の重み
WEIGHT_DECAY = 1e-4

EPOCHS = 15

# 学習率を変更するエポック数と変更後の学習率
LEARNING_SCHEDULE = {
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
