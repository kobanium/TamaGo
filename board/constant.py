"""碁盤に関する定数
"""
# 碁盤のサイズ
BOARD_SIZE = 9
# 盤外のサイズ
OB_SIZE = 1
# 連の最大数
STRING_MAX = int(0.8 * BOARD_SIZE * (BOARD_SIZE - 1) + 5)
# 隣接する連の最大数
NEIGHBOR_MAX = STRING_MAX
# 連を構成する石の最大数
STRING_POS_MAX = (BOARD_SIZE + OB_SIZE * 2) ** 2
# 呼吸点の最大数
STRING_LIB_MAX = (BOARD_SIZE + OB_SIZE * 2) ** 2
# 連を構成する石の座標の番兵
STRING_END = STRING_POS_MAX - 1
# 呼吸点の番兵
LIBERTY_END = STRING_LIB_MAX - 1
# 隣接する敵連の番兵
NEIGHBOR_END = NEIGHBOR_MAX - 1

# 着手に関する定数
# パスに対応する座標
PASS = 0
# 投了に対応する座標
RESIGN = -1
# Go Text ProtocolのX座標の文字
GTP_X_COORDINATE = 'IABCDEFGHJKLMNOPQRSTUVWXYZ'

# 着手履歴の最大数
MAX_RECORDS = (BOARD_SIZE ** 2) * 3
