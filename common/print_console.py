import sys

def print_out(message):
    """メッセージを標準出力に出力する。

    Args:
        message (str): 表示するメッセージ。
    """
    print(message)

def print_err(message):
    """メッセージを標準エラー出力に出力する。

    Args:
        message (str): 表示するメッセージ。
    """
    print(message, file=sys.stderr)
