from .parameters import get_grouped_llrd_parameters, get_optimizer_params
from torch.optim import AdamW
from torchcontrib.optim import SWA
import torch

# def get_optimizer(model, config):
    
#     if config.optimizer.group_lt_multiplier == 1:
#         optimizer_parameters = get_optimizer_params(model,
#                                                     config.optimizer.encoder_lr,
#                                                     config.optimizer.decoder_lr,
#                                                     weight_decay=config.optimizer.weight_decay)
#     else:
#         optimizer_parameters = get_grouped_llrd_parameters(model,
#                                                            encoder_lr=config.optimizer.encoder_lr,
#                                                            decoder_lr=config.optimizer.decoder_lr,
#                                                            embeddings_lr=config.optimizer.embeddings_lr,
#                                                            lr_mult_factor=config.optimizer.group_lt_multiplier,
#                                                            weight_decay=config.optimizer.weight_decay,
#                                                            n_groups=config.optimizer.n_groups)

#     optimizer = AdamW(optimizer_parameters,
#                       lr=config.optimizer.encoder_lr,
#                       eps=config.optimizer.eps,
#                       betas=config.optimizer.betas)

#     if config.optimizer.use_swa:
#         optimizer = SWA(optimizer,
#                         swa_start=config.optimizer.swa.swa_start,
#                         swa_freq=config.optimizer.swa.swa_freq,
#                         swa_lr=config.optimizer.swa.swa_lr)
#     return optimizer



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



    # if config.optimizer.group_lt_multiplier == 1:
    #     optimizer_parameters = get_optimizer_params(model,
    #                                                 config.optimizer.encoder_lr,
    #                                                 config.optimizer.decoder_lr,
    #                                                 weight_decay=config.optimizer.weight_decay)
    # else:
    #     optimizer_parameters = get_grouped_llrd_parameters(model,
    #                                                        encoder_lr=config.optimizer.encoder_lr,
    #                                                        decoder_lr=config.optimizer.decoder_lr,
    #                                                        embeddings_lr=config.optimizer.embeddings_lr,
    #                                                        lr_mult_factor=config.optimizer.group_lt_multiplier,
    #                                                        weight_decay=config.optimizer.weight_decay,
    #                                                        n_groups=config.optimizer.n_groups)

    # optimizer = AdamW(optimizer_parameters,
    #                   lr=config.optimizer.encoder_lr,
    #                   eps=config.optimizer.eps,
    #                   betas=config.optimizer.betas)

    # return optimizer