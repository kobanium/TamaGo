from enum import Enum

class MatchResult(Enum):
    """勝敗の結果を表すクラス。
    """
    DRAW = 0
    BLACK_WIN = 1
    WHITE_WIN = 2

    @classmethod
    def get_winner_string(cls, result):
        """対局結果を表す文字列を取得する。

        Args:
            result (MatchResult): 対局結果。

        Returns:
            str: 対局結果の文字列。
        """
        if result == MatchResult.DRAW:
            return "Draw"

        if result == MatchResult.BLACK_WIN:
            return "Black"

        if result == MatchResult.WHITE_WIN:
            return "White"

        return "Undefined"
