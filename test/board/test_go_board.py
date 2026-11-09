import pytest

from board.go_board import GoBoard
from board.stone import Stone


@pytest.mark.parametrize(
    "center_color, corner_color", [
        (Stone.BLACK, Stone.WHITE),
        (Stone.WHITE, Stone.BLACK)
    ]
)
def test_count_score_connected_empty_region_is_neutral(center_color, corner_color):
    """B01: 黒白双方に接する連続した空点領域を中立として数える。"""
    board = GoBoard(board_size=9)
    center = board.coordinate.convert_from_gtp_format("E5")
    corner = board.coordinate.convert_from_gtp_format("A1")
    board.put_stone(center, center_color)
    board.put_stone(corner, corner_color)

    # 空点はすべてつながり、黒白双方に接しているため地にならない。
    # 黒石と白石が1個ずつなので、コミを含まない面積差は0。
    # 石の位置を非対称にし、各空点の隣だけを見る誤実装を検出する。

    assert board.count_score() == 0

@pytest.mark.parametrize(
    "move_list, expected_score", [
        (
            ["E5", "D7", "C6", "F7", "F3", "G6", "C7", "D3", "C4", "F2",
             "E2", "E3", "G2", "F4", "G3", "D2", "F1", "E4", "E8", "G4",
             "E7", "H4", "H3", "H1", "G1", "D5", "D6", "C3", "G8", "B5",
             "C5", "B4", "H7", "A6", "B8", "D1", "E1", "F8", "F9", "G7",
             "H8", "F5", "E6", "J2", "J3", "H6", "B3", "B2", "D4", "J4",
             "B6", "A7", "B7", "A8", "B9", "A3", "J6", "J5", "J7", "F6",
             "G9", "A5", "H9", "A9", "E9", "B3", "C9", "D8", "D9", "B1",
             "C8", "D8", "PASS", "G5", "D7", "C2", "J8", "A2", "PASS", "PASS"], 41-39
        ),
        (
            ["F5", "D5", "E7", "D3", "C6", "C5", "B6", "H5", "F3", "G7",
             "H4", "G5", "G4", "F6", "E6", "E5", "F4", "J4", "D2", "E2",
             "J3", "D6", "D7", "C7", "C8", "J5", "C2", "E3", "F2", "B5",
             "B7", "C3", "H2", "F8", "E1", "F1", "G1", "B2", "B1", "A2",
             "H8", "H7", "D1", "E4", "A5", "A4", "F7", "G8", "A6", "B4",
             "G6", "H6", "E8", "E9", "J7", "J8", "J9", "H9", "F9", "J8",
             "D9", "A1", "F6", "G9", "E9", "C1", "F1", "B1", "PASS", "PASS"], 44-37
        )
    ]
)
def test_count_score_end_position(move_list, expected_score):
    board = GoBoard(board_size=9, komi=7.0, check_superko=True)

    color = Stone.BLACK
    
    for pos in move_list:
        coord = board.coordinate.convert_from_gtp_format(pos)
        board.put_stone(coord, color)
        color = Stone.get_opponent_color(color)
    
    board.display()
    board.strings.display()

    assert board.count_score() == expected_score
