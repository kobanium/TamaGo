"""深層学習に関するユーティリティ。
"""
from typing import NoReturn, Dict, List, Tuple
import time
import torch
import numpy as np
import onnxruntime as ort

from common.print_console import print_err
from nn.network.dual_net import DualNet


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

    print_err(f"epoch {epoch}, data-{index} : loss = {loss:6f}, time = {training_time:3f} seconds.")
    print_err(f"\tpolicy loss : {policy_loss:6f}")
    print_err(f"\tvalue loss  : {value_loss:6f}")


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

    print_err(f"Test {epoch} : loss = {loss:6f}, time = {testing_time:3f} seconds.")
    print_err(f"\tpolicy loss : {policy_loss:6f}")
    print_err(f"\tvalue loss  : {value_loss:6f}")


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


def split_train_test_set(file_list: List[str], train_data_ratio: float) \
    -> Tuple[List[str], List[str]]:
    """学習に使用するデータと検証に使用するデータファイルを分割する。

    Args:
        file_list (List[str]): 学習に使用するnpzファイルリスト。
        train_data_ratio (float): 学習に使用するデータの割合。

    Returns:
        Tuple[List[str], List[str]]: 学習データセットと検証データセット。
    """
    train_data_set = file_list[:int(len(file_list) * train_data_ratio)]
    test_data_set = file_list[int(len(file_list) * train_data_ratio):]

    print(f"Training data set : {train_data_set}")
    print(f"Testing data set  : {test_data_set}")

    return train_data_set, test_data_set


def apply_softmax(logits: np.array) -> np.array:
    """Softmax関数を適用する。

    Args:
        logits (np.array): Softmax関数の入力値。

    Returns:
        np.array: Softmax関数適用後の値。
    """
    shift_exp = np.exp(logits - np.max(logits))

    return shift_exp / np.sum(shift_exp)


def load_network(model_file_path: str, use_gpu: bool) -> DualNet:
    """ニューラルネットワークをロードして取得する。

    Args:
        model_file_path (str): ニューラルネットワークのパラメータファイルパス。
        use_gpu (bool): GPU使用フラグ。

    Returns:
        DualNet: パラメータロード済みのニューラルネットワーク。
    """
    if model_file_path.endswith(".onnx"):
        return OrtWrapper(model_file_path, use_gpu)
    device = get_torch_device(use_gpu=use_gpu)
    network = DualNet(device)
    network.to(device)
    try:
        network.load_state_dict(torch.load(model_file_path))
    except: # pylint: disable=W0702
        print(f"Failed to load {model_file_path}.")
    network.eval()
    torch.set_grad_enabled(False)

    return network

class OrtWrapper:
    def __init__(self, model_file_path: str, use_gpu: bool):
        providers = []
        if use_gpu:
            providers += [
                "DmlExecutionProvider",
                "CUDAExecutionProvider",
            ]
        providers.append("CPUExecutionProvider")
        self.ort_sess = ort.InferenceSession(model_file_path, providers=providers)


    def _run(self, input):
        outputs = self.ort_sess.run(None, {'input': input})
        return outputs[0], outputs[1]


    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


    def inference(self, input_plane: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """前向き伝搬処理を実行する。探索用に使うメソッドのため、デバイス間データ転送も内部処理する。

        Args:
            input_plane (np.ndarray): 入力特徴テンソル。

        Returns:
            Tuple[np.ndarray, np.ndarray]: Policy, Valueの推論結果。
        """
        policy, value = self._run(input_plane)
        return self._softmax(policy), self._softmax(value)


    def inference_with_policy_logits(self, input_plane: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
        """前向き伝搬処理を実行する。Gumbel AlphaZero用の探索に使うメソッドのため、
        デバイス間データ転送も内部処理する。

        Args:
            input_plane (np.ndarray): 入力特徴テンソル。

        Returns:
            Tuple[np.ndarray, np.ndarray]: Policy, Valueの推論結果。
        """
        policy, value = self._run(input_plane)
        return policy, self._softmax(value)
