from .parameters import get_grouped_llrd_parameters, get_optimizer_params
from torch.optim import AdamW
from torchcontrib.optim import SWA
import torch

def get_optimizer(model, config):
    params = model.parameters()
    no_decay = ["bias", "LayerNorm.weight"]
    differential_layers = config.training.differential_learning_rate_layers
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": config.training.learning_rate,
                "weight_decay": 0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": config.training.differential_learning_rate,
                "weight_decay": config.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": config.training.differential_learning_rate,
                "weight_decay": 0,
            },
        ],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    return optimizer
