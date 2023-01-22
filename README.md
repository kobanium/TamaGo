# TamaGo
TamaGoはPythonで実装された囲碁の思考エンジンです。  
人間の棋譜を利用した教師あり学習とGumbel AlphaZero方式の強化学習をお試しできるプログラムとなる予定です。  
現在はランダムな着手を返すプログラムとなっています。

# Requirements
|使用するパッケージ|用途|
|---|---|
|click|コマンドライン引数の実装|

# License
ライセンスはApache License ver 2.0です。

# Todo list
- 碁盤の実装
  - [x] 連のデータ構造の実装
  - [ ] 3x3パターンのデータ構造の実装
  - [ ] 着手履歴の実装
- 探索部の実装
  - [ ] 木とノードのデータ構造の実装
  - [ ] モンテカルロ木探索の実装
    - [ ] PUCT探索の実装
    - [ ] Sequential Halving applied to tree探索の実装
- 学習の実装
  - [ ] SGFファイルの読み込み処理の実装
  - [ ] PyTorchを利用した教師あり学習の実装
  - [ ] PyTorchを利用したGumbel AlphaZero方式の強化学習の実装
- GTPクライアントの実装
  - 基本的なコマンド
    - [x] プログラム情報の表示 : name, version, protocol_version
    - [x] プログラムの終了 : quit
    - [x] 碁盤の操作 : boardsize, clear_board
    - [x] 碁盤の表示 : showboard, showstring
    - [x] 着手 : play, genmove
    - [ ] コミの設定と取得 : komi, get_komi
    - [x] コマンドの確認 : known_command, list_commands
    - [ ] SGFファイルの読み込み : load_sgf
  - 大会参加時に必要なコマンド
    - [ ] 持ち時間の初期化 : time_settings
    - [ ] 持ち時間の読み込み : time_left
  - 分析用のコマンド

etc...
