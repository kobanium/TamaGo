"""深層学習の実装。
"""
from typing import NoReturn
import glob
import os
import time
import torch
from nn.network.dual_net import DualNet
from nn.loss import calculate_policy_loss, calculate_value_loss, \
    calculate_policy_kld_loss
from nn.utility import get_torch_device, print_learning_process, \
    print_evaluation_information, save_model, load_data_set, \
    split_train_test_set

from learning_param import SL_LEARNING_RATE, RL_LEARNING_RATE, \
    MOMENTUM, WEIGHT_DECAY, SL_VALUE_WEIGHT, RL_VALUE_WEIGHT, \
    LEARNING_SCHEDULE



def train_on_cpu(program_dir: str, board_size: int, batch_size: \
    int, epochs: int) -> NoReturn: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
        epochs (int): 実行する最大エポック数。
    """
    # 学習データと検証用データの分割
    data_set = sorted(glob.glob(os.path.join(program_dir, "data", "sl_data_*.npz")))

    train_data_set, test_data_set = split_train_test_set(data_set, 0.8)

    # 学習処理を行うデバイスの設定
    device = get_torch_device(use_gpu=False)

    dual_net = DualNet(device=device, board_size=board_size)

    dual_net.to(device)
    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=SL_LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY,
                                nesterov=True)

    current_lr = SL_LEARNING_RATE

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
                plane = torch.tensor(plane_data[i:i+batch_size])
                policy = torch.tensor(policy_data[i:i+batch_size])
                value = torch.tensor(value_data[i:i+batch_size])

                policy_predict, value_predict = dual_net.forward_for_sl(plane)

                policy_loss = calculate_policy_loss(policy_predict, policy)
                value_loss = calculate_value_loss(value_predict, value)

                dual_net.zero_grad()

                loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

        if epoch in LEARNING_SCHEDULE["learning_rate"]:
            previous_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_SCHEDULE["learning_rate"][epoch]
            current_lr = LEARNING_SCHEDULE["learning_rate"][epoch]
            print(f"Epoch {epoch}, learning rate has changed {previous_lr} -> {current_lr}")

    save_model(dual_net, os.path.join("model", "sl-model.bin"))


def train_on_gpu(program_dir: str, board_size: int, batch_size: int, \
    epochs: int) -> NoReturn: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
        epochs (int): 実行する最大エポック数。
    """
    # 学習データと検証用データの分割
    data_set = sorted(glob.glob(os.path.join(program_dir, "data", "sl_data_*.npz")))
    train_data_set, test_data_set = split_train_test_set(data_set, 0.8)

    # 学習処理を行うデバイスの設定
    device = get_torch_device(use_gpu=True)

    dual_net = DualNet(device=device, board_size=board_size)

    dual_net.to(device)

    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=SL_LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY,
                                nesterov=True)

    scaler = torch.cuda.amp.GradScaler()

    current_lr = SL_LEARNING_RATE

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

                    dual_net.zero_grad()

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

        if epoch in LEARNING_SCHEDULE["learning_rate"]:
            previous_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_SCHEDULE["learning_rate"][epoch]
            current_lr = LEARNING_SCHEDULE["learning_rate"][epoch]
            print(f"Epoch {epoch}, learning rate has changed {previous_lr} -> {current_lr}")

    save_model(dual_net, os.path.join("model", "sl-model.bin"))


def train_with_gumbel_alphazero_on_cpu(program_dir: str, board_size: int, \
    batch_size: int) -> NoReturn: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。CPUで実行。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
    """
    data_set = sorted(glob.glob(os.path.join(program_dir, "data", "rl_data_*.npz")))

    # 学習処理を行うデバイスの設定
    device = get_torch_device(use_gpu=False)

    dual_net = DualNet(device=device, board_size=board_size)

    dual_net.to(device)

    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=RL_LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY,
                                nesterov=True)
    num_trained_batches = 0

    model_file_path = os.path.join(program_dir, "model", "rl-model.bin")
    if os.path.exists(model_file_path):
        print(f"load {model_file_path}")
        dual_net.load_state_dict(torch.load(model_file_path))

    state_file_path = os.path.join(program_dir, "model", "rl-state.ckpt")
    if os.path.exists(state_file_path):
        print(f"load {state_file_path}")
        checkpoint = torch.load(state_file_path, map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        num_trained_batches = checkpoint["num_trained_batches"]
        print(f"num_trained_batches : {num_trained_batches}")

    for data_index, train_data_path in enumerate(data_set):
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
            plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
            policy = torch.tensor(policy_data[i:i+batch_size]).to(device)
            value = torch.tensor(value_data[i:i+batch_size]).to(device)

            policy_predict, value_predict = dual_net.forward(plane)

            dual_net.zero_grad()
            policy_loss = calculate_policy_kld_loss(policy_predict, policy)
            value_loss = calculate_value_loss(value_predict, value)

            loss = (policy_loss + RL_VALUE_WEIGHT * value_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_trained_batches += 1

            train_loss["loss"] += loss.item()
            train_loss["policy"] += policy_loss.mean().item()
            train_loss["value"] += value_loss.mean().item()
            iteration += 1

        print_learning_process(train_loss, 0, data_index, iteration, epoch_time)

    save_model(dual_net, model_file_path)

    state = {
        "num_trained_batches": num_trained_batches,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, state_file_path)


def train_with_gumbel_alphazero_on_gpu(program_dir: str, board_size: int, \
    batch_size: int) -> NoReturn: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。GPUで実行。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
    """
    data_set = sorted(glob.glob(os.path.join(program_dir, "data", "rl_data_*.npz")))

    # 学習処理を行うデバイスの設定
    device = get_torch_device(use_gpu=True)

    dual_net = DualNet(device=device, board_size=board_size)

    dual_net.to(device)

    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=RL_LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY,
                                nesterov=True)

    scaler = torch.cuda.amp.GradScaler()

    num_trained_batches = 0

    model_file_path = os.path.join(program_dir, "model", "rl-model.bin")
    if os.path.exists(model_file_path):
        print(f"load {model_file_path}")
        dual_net.load_state_dict(torch.load(model_file_path))

    state_file_path = os.path.join(program_dir, "model", "rl-state.ckpt")
    if os.path.exists(state_file_path):
        print(f"load {state_file_path}")
        checkpoint = torch.load(state_file_path, map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        num_trained_batches = checkpoint["num_trained_batches"]
        print(f"num_trained_batches : {num_trained_batches}")

    for data_index, train_data_path in enumerate(data_set):
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

                policy_predict, value_predict = dual_net.forward(plane)

                dual_net.zero_grad()
                policy_loss = calculate_policy_kld_loss(policy_predict, policy)
                value_loss = calculate_value_loss(value_predict, value)

                loss = (policy_loss + RL_VALUE_WEIGHT * value_loss).mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                num_trained_batches += 1

            train_loss["loss"] += loss.item()
            train_loss["policy"] += policy_loss.mean().item()
            train_loss["value"] += value_loss.mean().item()
            iteration += 1

        print_learning_process(train_loss, 0, data_index, iteration, epoch_time)

    save_model(dual_net, model_file_path)

    state = {
        "num_trained_batches": num_trained_batches,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict()
    }
    torch.save(state, state_file_path)
