"""深層学習の実装。
"""
from typing import NoReturn
import glob
import os
import time
import torch
from nn.network.dual_net import DualNet
from nn.loss import calculate_policy_loss, calculate_value_loss
from nn.utility import get_torch_device, print_learning_process, \
    print_evaluation_information, save_model, load_data_set

from learning_param import LEARNING_RATE, MOMENTUM, \
    WEIGHT_DECAY, SL_VALUE_WEIGHT, LEARNING_SCHEDULE


def train(program_dir: str, board_size: int, batch_size: \
    int, epochs: int ,use_gpu: bool) -> NoReturn: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
        epochs (int): 実行する最大エポック数。
        use_gpu (bool): GPU使用フラグ。
    """
    dual_net = DualNet(board_size=board_size)

    # 学習処理を行うデバイスの設定
    device = get_torch_device(use_gpu=use_gpu)

    dual_net.to(device)
    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY,
                                nesterov=True)

    scaler = torch.cuda.amp.GradScaler()

    data_set = sorted(glob.glob(os.path.join(program_dir, "data", "sl_data_*.npz")))

    train_data_set, test_data_set = \
        data_set[:int(len(data_set) * 0.8)], data_set[int(len(data_set) * 0.8):]
    print(f"Training data set : {train_data_set}")
    print(f"Testing data set  : {test_data_set}")

    current_lr = LEARNING_RATE

    for epoch in range(epochs):
        for data_index, train_data_path in enumerate(train_data_set):
            plane_data, policy_data, value_data = load_data_set(train_data_path)
            train_loss = {
                "loss": 0.0,
                "policy": 0.0,
                "value": 0.0,
            }
            iteration = 0
            dual_net.train()
            epoch_time = time.time()
            for i in range(0, len(value_data) - batch_size + 1, batch_size):
                with torch.cuda.amp.autocast(enabled=True):
                    plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
                    policy = torch.tensor(policy_data[i:i+batch_size]).to(device)
                    value = torch.tensor(value_data[i:i+batch_size]).to(device)

                    policy_predict, value_predict = dual_net.forward_for_sl(plane)

                    policy_loss = calculate_policy_loss(policy_predict, policy)
                    value_loss = calculate_value_loss(value_predict, value)

                    loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss["loss"] += loss.item()
                train_loss["policy"] += policy_loss.mean().item()
                train_loss["value"] += value_loss.mean().item()
                iteration += 1

            print_learning_process(train_loss, epoch, data_index, iteration, epoch_time)

        test_loss = {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }
        test_iteration = 0
        testing_time = time.time()
        for data_index, test_data_path in enumerate(test_data_set):
            dual_net.eval()
            plane_data, policy_data, value_data = load_data_set(test_data_path)
            with torch.no_grad():
                for i in range(0, len(value_data) - batch_size + 1, batch_size):
                    plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
                    policy = torch.tensor(policy_data[i:i+batch_size]).to(device)
                    value = torch.tensor(value_data[i:i+batch_size]).to(device)

                    policy_predict, value_predict = dual_net.forward_for_sl(plane)

                    policy_loss = calculate_policy_loss(policy_predict, policy)
                    value_loss = calculate_value_loss(value_predict, value)

                    loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                    test_loss["loss"] += loss.item()
                    test_loss["policy"] += policy_loss.mean().item()
                    test_loss["value"] += value_loss.mean().item()
                    test_iteration += 1

        print_evaluation_information(test_loss, epoch, test_iteration, testing_time)

        if epoch in LEARNING_SCHEDULE["decay_epoch"]:
            previous_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_SCHEDULE["learning_rate"][epoch]
            current_lr = LEARNING_SCHEDULE["learning_rate"][epoch]
            print(f"Epoch {epoch}, learning rate has changed {current_lr} -> {previous_lr}")

    save_model(dual_net, os.path.join("model", "model.bin"))
