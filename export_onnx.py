import click
import torch

from board.constant import BOARD_SIZE
from nn.utility import load_network
from nn.network.dual_net import DualNet

@click.command()
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE,
    help=f"碁盤のサイズを指定。デフォルトは{BOARD_SIZE}。")
@click.option('--model-path', type=click.STRING,
              help=f"使用するPyTorchモデルのパスを指定する。プログラムのホームディレクトリの相対パスで指定。")
@click.option('--output-path', type=click.STRING,
              help=f"保存するONNXモデルのパスを指定する。プログラムのホームディレクトリの相対パスで指定。")
def convert_to_onnx(model_path: str, output_path: str, size: int) -> None:
    """Convert to ONNX format.
    """
    net = load_network(model_path, False)
    if not isinstance(net, torch.nn.Module):
        print("Model is not instance of torch.nn.Module.")
        return

    input_tensor = torch.rand((1, 6, size, size), dtype=torch.float32)

    torch.onnx.export(
        net,
        (input_tensor,),
        output_path,
        input_names=["input"],
        output_names=["policy", "value",],
        dynamic_axes={
            "input": {
                0: "batch",
            },
            "policy": {
                0: "batch",
            },
            "value": {
                0: "batch",
            },
        }
    )


if __name__ == "__main__":
    convert_to_onnx() # pylint: disable=E1120
