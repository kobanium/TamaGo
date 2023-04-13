# 強化学習について
TamaGoはニューラルネットワークを使用したGumbel AlphaZero方式の強化学習をサポートしています。

# 強化学習の前提条件
強化学習実行時にはTamaGo以外に自己対戦の対局結果を補正するためにGNUGoを利用します。  
GNUGoを利用しなくても強化学習は進むため、GNUGoのインストールはオプショナルとしていますが、TamaGoの終局時の地合判定が非常に雑なため、GNUGoの使用を推奨します。  

GNUGoのインストール方法はUbuntuであれば下記コマンドを実行するだけです。
~~~
apt install gnugo
~~~

# 強化学習で使用するハイパーパラメータの定義
強化学習で使用するハイパーパラメータは[learning_param.py](../../learning_param.py)に定義してあります。

| パラメータ | 概要 | 設定値の例 | 備考 |
| --- | --- | --- | --- |
| RL_LEARNING_RATE | 強化学習で使用する学習率 | 0.01 | 学習がある程度進んだ時に小さな値に変更すると良いです。 |
| BATCH_SIZE | 学習時のミニバッチサイズ | 256 | GPUメモリが小さい場合はこの値を小さめに設定してください。 |
| MOMENTUM | 学習器のモーメンタムパラメータ | 0.9 | 基本的に変更する必要はありません。 |
| WEIGHT_DECAY | L2正則化の重み | 1e-4 (0.0001) | 基本的に変更する必要はありませんが、過学習している場合は大きめの値を設定すると解消する場合があります。 |
| DATA_SET_SIZE | npzファイル1つに格納するデータ数 | BATCH_SIZE * 4000 | メモリサイズに合わせて値を変更してください。 |
| RL_VALUE_WEIGHT | Policyのlossに対するValueのlossの重み | 1.0 | 0.0以上の値を設定してください。値を大きくし過ぎるとValueを過学習する傾向にあります。 |
| SELF_PLAY_VISITS | 自己対戦時の1手あたりの探索回数 | 16 | 2以上の値を設定してください。値を大きくすると棋譜の質は向上しますが、棋譜の生成速度は低下します。 |
| NUM_SELF_PLAY_WORKERS | 自己対戦実行ワーカ数 | 4 | CPUやメモリに合わせて数を変更してください。 |
| NUM_SELF_PLAY_GAMES | 1回の自己対戦ワーカ実行で生成する棋譜の数 | 10000 | 小さい値を設定すると化学臭しやすくなります。 |

学習がうまく進むことを確認しているパラメータのため、試行錯誤する際は最初は設定値そのまま利用し、徐々に値を変更して学習状況を確認してください。

# ニューラルネットワークの定義
ニューラルネットワークの定義は下記4ファイルを使用して定義しています。
| ファイル | 定義内容 |
| --- | --- |
| [nn/network/dual_net.py](../../nn/network/dual_net.py) | ニューラルネットワーク全体の定義 |
| [nn/network/res_block.py](../../nn/network/res_block.py) | Residual Blockの定義 |
| [nn/network/head/policy_head.py](../../nn/network/head/policy_head.py) | Policy Headの定義 |
| [nn/network/head/value_head.py](../../nn/network/head/value_head.py) | Value Headの定義 |

ネットワークの構造を試行錯誤する場合は、まずはdual_net.pyで使用するResidual Blockの個数やfilters変数の値を変更して見ることをお勧めします。

# TamaGoの強化学習のプロセス
TamaGoの強化学習パイプラインは以下の順番で実行されます。
1. 既存のニューラルネットワークのモデルを利用し、指定した数だけ自己対局を実行する。([selfplay_main.py](../../selfplay_main.py)の実行)
2. 自己対戦の結果をGNUGoの判定で補正する。(オプショナル)
3. 自己対戦で生成した棋譜ファイルを使用してニューラルネットワークの学習を実行する。([train.py](../../train.py)の実行)
4. 1〜3を繰り返す。

強化学習パイプラインは[pipeline.sh](../../pipeline.sh)に定義しています。


## 自己対戦実行スクリプト([selfplay_main.py](../../selfplay_main.py))のコマンドラインオプション

| オプション | 概要 | 設定する値の例 | デフォルト値 | 備考 |
| --- | --- | --- | --- | --- |
| `--save-dir` | 自己対戦で生成したSGFファイルを保存するディレクトリパス | save_dir | archive | |
| `--process` | 自己対戦実行ワーカ数 | 2 | NUM_SELF_PLAY_WORKERS | |
| `--num-data` | 1回の学習するごとに生成する棋譜の数 | 5000 | NUM_SELF_PLAY_GAMES | |
| `--size` | 碁盤のサイズ | 9 | 9 | |
| `--use-gpu` | GPU使用フラグ | true | true | GPUを使用して自己対戦を実行する設定のフラグ。trueかfalseで指定 |
| `--visits` | 1手あたりの探索回数 | 100 | SELF_PLAY_VISITS | 探索回数を増やすと棋譜の質が向上しますが、生成速度は遅くなります。 |
| `--model` | 使用するネットワークパラメータファイル | model/rl-model.bin | model/model.bin | |

## 強化学習実行スクリプト([train.py](../../train.py))のコマンドラインオプション

| オプション | 概要 | 設定する値の例 | デフォルト値 | 備考 |
| --- | --- | --- | --- | --- |
| `--kifu-dir` | 教師データとして使用するSGFファイルを格納したディレクトリパスの指定 | /home/user/sgf_files | なし | |
| `--size` | 碁盤の大きさの指定 | 5 | 9 | 教師データとして使用するSGFファイルのSZタグの値と一致させてください |
| `--use-gpu` | GPU使用フラグ | true | true | GPUを使用して学習する設定のフラグ。trueかfalseで指定 |
| `--rl` | 強化学習実行フラグ | false | false | 教師あり学習を実行するときにはfalseを指定する。 |
| `--window-size` | 強化学習時のウィンドウサイズ | 500000 | 300000 | 教師あり学習では使用しないオプション。 |
