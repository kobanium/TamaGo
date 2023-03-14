"""探索用のパラメータ設定
"""

# 未展開の子ノードのインデックス
NOT_EXPANDED = -1

# PUCBの第2項の重みパラメータ
PUCB_SECOND_TERM_WEIGHT = 1.0

# 1手ごとの探索回数
PLAYOUTS = 100

# 探索時のミニバッチサイズ
NN_BATCH_SIZE = 1

# Gumbel AlphaZero用のパラメータ(C_visit)
C_VISIT = 50

# Gumbel AlphaZero用のパラメータ(C_scale)
C_SCALE = 1.0

# Sequential Halvingで考慮する着手の最大数
MAX_CONSIDERED_NODES = 16

# 1手あたりの探索回数のデフォルト値
CONST_VISITS = 1000

# 1手あたりの探索時間のデフォルト値
CONST_TIME = 5.0

# 持ち時間のデフォルト値
REMAINING_TIME = 60.0

# 探索速度のデフォルト値
VISITS_PER_SEC = 200

# 投了の閾値
RESIGN_THRESHOLD = 0.05
