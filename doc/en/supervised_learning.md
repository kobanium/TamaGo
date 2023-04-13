# About supervised learning
TamaGo has a function to execute supervised learning with neural networks.


# Prerequisites for training data files
TamaGo can load SGF (Smart Game Format) files. For more information about SGF, please check [here](https://www.red-bean.com/sgf/)  
The following prerequisites are required.
- There are no branches for a move sequence.
- There are B-tags and W-tags for each player's moves (For policy target).
- There is RE-tag with the game result (For value target).
- Character code is UTF-8 or ASCII.


# Hyperparameters for supervised learning
Hyperparameters for supervised learning are defined in [learning_param.py](../../learning_param.py).

| Hyperparameters | Description | Example of value | Note |
| --- | --- | --- | --- |
| SL_LEARNING_RATE | Initial learning rate. | 0.01 | |
| BATCH_SIZE | Mini-batch size for training. | 256 | |
| MOMENTUM | Momentum parameter for an optimizer. | 0.9 | |
| WEIGHT_DECAY | Weight of L2-regularization. | 1e-4 (0.0001) | |
| EPOCHS | The number of training epochs. | 15 | |
| LEARNING_SCHEDULE | Learning rate decaying schedule. | see learning_param.py | |
| DATA_SET_SIZE | Number of data to be stored in a npz file. | BATCH_SIZE * 4000 | |
| SL_VALUE_WEIGHT | Weight of value loss against policy loss. | 0.02 | This must be more than 0.0. |

I have already confirmed that with data from 50000 games, supervised learning works well with the default hyperparameters. If you try to adjust hyperparameters, I recommend you to adjust LEARNING_RATE, LEARNING_SCHEDULE, EPOCHS, or BATCH_SIZE at first.

# Definition of neural network structure.
Neural network is defined using the following four files.
| File | Definition |
| --- | --- |
| [nn/network/dual_net.py](../../nn/network/dual_net.py) | Neural network definition. |
| [nn/network/res_block.py](../../nn/network/res_block.py) | Residual block definition. |
| [nn/network/head/policy_head.py](../../nn/network/head/policy_head.py) | Policy head definition. |
| [nn/network/head/value_head.py](../../nn/network/head/value_head.py) | Value head definition. |

If you try to change structure, I recommend you to change value of filters or blocks at first.

# TamaGo's supervised learning process
Supervised learning process runs in the following order,
1. Reading SGF files from the specified directory and outputting sl_data_*.npz files as training data under the "data" directory.
2. Executing supervised learning using sl_data_*.npz files under tha "data" directory.

## Training data generation step
Running the output of training data files is optional, but
- When you want to change training data source
- When you change input features for neural network

you will need to rerun the data file generation process.

## Supervised learning execution step
When executing supervised learning, sl_data_*.npz files under the "data" directory is split at a ratio of 8:2 into training data tobe used for training and validation data to evaluate the training progress. 
If you want to change this ratio, please change the following code in [learn.py](../../nn/learn.py).

```
train_data_set, test_data_set = split_train_test_set(data_set, 0.8)
```

# How to execute supervised learning

If you store SGF files to be used as supervised learining data under /home/user/sgf_files directory, execute following command in TamaGo home directory.
```
python train.py --kifu-dir /home/user/sgf_files
```

if you already have sl_data_*.npz files in the "data" directory and want to run supervised learning using those files, execute the following command in TamaGo home directory.
```
python train.py
```
When the supervised learning process is completed, a trained model file is output under the "model" directory with the name "sl-model.bin". 
How to use the trained model file is [here](../../README.md).


## Command line options

| Option | Description | Example of value | Default value | Note |
| --- | --- | --- | --- | --- |
| `--kifu-dir` | Path to the directory contains SGF files. | /home/user/sgf_files | None | |
| `--size` | Go board size. | 5 | 9 |  |
| `--use-gpu` | Flag to use a GPU. | true | true | Value is true or false. |
| `--rl` | Flag to execute reinforcement learning. | false | false |  |
| `--window-size` | Window size for reinforcement learning. | 500000 | 300000 | This is an ignored option for supervised learning. |
