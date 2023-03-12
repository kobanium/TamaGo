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
