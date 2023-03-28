"""損失関数の実装。
"""
import torch
import torch.nn.functional as F

cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
kld_loss = torch.nn.KLDivLoss(reduction="batchmean")

def calculate_policy_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """_Policyの損失関数値を計算する。

    Args:
        output (torch.Tensor): ニューラルネットワークのPolicyの出力値。
        target (torch.Tensor): Policyのターゲット (分布) 。

    Returns:
        torch.Tensor: Policy loss。
    """
    return torch.sum((-target * (output.float() + 1e-8).log()), dim=1)

def calculate_sl_policy_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """教師あり学習向けのPolicyの損失関数値を計算する。

    Args:
        output (torch.Tensor): ニューラルネットワークのPolicyの出力値。
        target (torch.Tensor): Policyのターゲットクラス。

    Returns:
        torch.Tensor: Policy loss。
    """
    return cross_entropy_loss(output, target)

def calculate_policy_kld_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PolicyのKullback-Leibler divegence損失関数値を計算する。

    Args:
        output (torch.Tensor): ニューラルネットワークのPolicyの出力値。
        target (torch.Tensor): Policyのターゲット（分布）。

    Returns:
        torch.Tensor: PolicyのKullback-Leibler divergence loss。
    """
    return kld_loss(F.log_softmax(output, -1), target)

def calculate_value_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Valueの損失関数値を計算する。

    Args:
        output (torch.Tensor): ニューラルネットワークのValueの出力値。
        target (torch.Tensor): Valueのターゲットクラス。

    Returns:
        torch.Tensor: _description_
    """
    return cross_entropy_loss(output, target)
