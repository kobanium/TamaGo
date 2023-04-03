"""学習用の各種ハイパーパラメータの設定。
"""

# 教師あり学習実行時の学習率
SL_LEARNING_RATE = 0.02

# 強化学習実行時の学習率
RL_LEARNING_RATE = 0.02

# ミニバッチサイズ
BATCH_SIZE = 1024

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
DATA_SET_SIZE = BATCH_SIZE * 1000

# Policyのlossに対するValueのlossの重み比率
SL_VALUE_WEIGHT = 0.02

# Policyのlossに対するValueのlossの重み比率
RL_VALUE_WEIGHT = 1.0

# 自己対戦時の探索回数
SELF_PLAY_VISITS = 16

# 自己対戦実行ワーカ数
NUM_SELF_PLAY_WORKER = 4
