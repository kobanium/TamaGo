import torch

from nn.utility import load_network

net = load_network("model/sl-model.bin", False)

board_size = 9
input_tensor = torch.rand((1, 6, board_size, board_size), dtype=torch.float32)

torch.onnx.export(
    net,
    (input_tensor,),
    "sl-model.onnx",
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
