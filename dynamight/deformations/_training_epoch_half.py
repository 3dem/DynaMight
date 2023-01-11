import torch


def train_epoch(
    encoder: torch.nn.Module,
    encoder_optimizer: torch.optim.Optimizer,
    decoder: torch.nn.Module,
    decoder_optimizer: torch.optim.Optimizer,
    physical_parameter_optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
):
    # todo: schwab implement and substitute in optimize deformations