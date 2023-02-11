"""深層学習に関するユーティリティ。
"""
from typing import NoReturn, Dict, Tuple
import time
import torch
import numpy as np

from common.print_console import print_err

def get_torch_device(use_gpu: bool) -> torch.device:
    """torch.deviceを取得する。

    Args:
        use_gpu (bool): GPU使用フラグ。

    Returns:
        torch.device: デバイス情報。
    """
    if use_gpu:
        torch.cuda.set_device(0)
        return torch.device("cuda")
    return torch.device("cpu")


def _calculate_losses(loss: Dict[str, float], iteration: int) \
    -> Tuple[float, float, float]:
    """各種損失関数値を算出する。

    Args:
        loss (Dict[str, float]): 損失関数値の情報。
        iteration (int): イテレーション数。

    Returns:
        Tuple[float, float, float]: Total loss, Policy loss, Value loss。
    """
    return loss["loss"] / iteration, loss["policy"] / iteration, \
        loss["value"] / iteration



def print_learning_process(loss_data: Dict[str, float], epoch: int, index: int, \
    iteration: int, start_time: float) -> NoReturn:
    """学習経過情報を表示する。

    Args:
        loss_data (Dict[str]): 損失関数値の情報。
        epoch (int): 学習エポック数。
        index (int): データセットインデックス。
        iteration (int): バッチサイズの学習イテレーション数。
        start_time (float): 学習開始時間。
    """
    loss, policy_loss, value_loss = _calculate_losses(loss_data, iteration)
    training_time = time.time() - start_time

    print_err(f"epoch {epoch}, data-{index} : loss = {loss}, time = {training_time} sec.")
    print_err(f"\tpolicy loss : {policy_loss}")
    print_err(f"\tvalue loss  : {value_loss}")


def print_evaluation_information(loss_data: Dict[str, float], epoch: int, \
    iteration: int, start_time: float) -> NoReturn:
    """テストデータの評価情報を表示する。

    Args:
        loss_data (Dict[str, float]): 損失関数値の情報。
        epoch (int): 学習エポック数。
        iteration (int): テストイテレーション数。
        start_time (float): 評価開始時間。
    """
    loss, policy_loss, value_loss = _calculate_losses(loss_data, iteration)
    testing_time = time.time() - start_time

    print_err(f"Test {epoch} : loss = {loss}, time = {testing_time} sec.")
    print_err(f"\tpolicy loss : {policy_loss}")
    print_err(f"\tvalue loss  : {value_loss}")


def save_model(network: torch.nn.Module, path: str) -> NoReturn:
    """ニューラルネットワークのパラメータを保存する。

    Args:
        network (torch.nnModel): ニューラルネットワークのモデル。
        path (str): パラメータファイルパス。
    """
    torch.save(network.to("cpu").state_dict(), path)


def load_data_set(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """学習データセットを読み込む。

    Args:
        path (str): データセットのファイルパス。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 入力データ、Policy、Value。
    """
    data = np.load(path)
    perm = np.random.permutation(len(data["value"]))
    return data["input"][perm], data["policy"][perm].astype(np.float32), \
        data["value"][perm].astype(np.int64)
