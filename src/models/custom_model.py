import torch
import torch.nn as nn
from .pooling_layers import get_pooling_layer
from transformers import AutoModel, AutoConfig
from torch.utils.checkpoint import checkpoint


class CustomModel(nn.Module):
    def __init__(self, cfg, backbone_config):
        super().__init__()
        self.cfg = cfg
        self.backbone_config = backbone_config

        if self.cfg.architecture.pretrained:
            self.backbone = AutoModel.from_pretrained(cfg.architecture.model_name, config=self.backbone_config)
        else:
            self.backbone = AutoModel.from_config(self.backbone_config)

        self.backbone.resize_token_embeddings(len(cfg.tokenizer))

        if hasattr(self.cfg.training, "multi_dropout") and self.cfg.training.multi_dropout:
            self.dropout = nn.Dropout(self.backbone_config.hidden_dropout_prob)
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.2)
            self.dropout3 = nn.Dropout(0.3)
            self.dropout4 = nn.Dropout(0.4)
            self.dropout5 = nn.Dropout(0.5)
        
        
        self.pool = get_pooling_layer(cfg, backbone_config)


        self.fc = nn.Linear(self.pool.output_dim, len(self.cfg.dataset.target_cols))

        if 'bart' in cfg.architecture.model_name:
            self.initializer_range = self.backbone_config.init_std
        else:
            self.initializer_range = self.backbone_config.initializer_range

        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        outputs = self.backbone(**inputs)
        feature = self.pool(inputs, outputs)
        
        if hasattr(self.cfg.training, "multi_dropout") and self.cfg.training.multi_dropout:
            feature = self.dropout(feature)
            logits1 = self.fc(self.dropout1(feature))
            logits2 = self.fc(self.dropout2(feature))
            logits3 = self.fc(self.dropout3(feature))
            logits4 = self.fc(self.dropout4(feature))
            logits5 = self.fc(self.dropout5(feature))
            output = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        
        output = self.fc(feature)
        return output
    



class CustomModel2(nn.Module):
    def __init__(self, cfg, backbone_config):
        super().__init__()
        self.cfg = cfg
        self.backbone_config = backbone_config

        self.backbone_config.update(
            {
                "output_hidden_states": True,
                "add_pooling_layer": False,
            }
        )

        if self.cfg.architecture.pretrained:
            self.backbone = AutoModel.from_pretrained(cfg.architecture.model_name, config=self.backbone_config)
        else:
            self.backbone = AutoModel.from_config(self.backbone_config)

        self.backbone.resize_token_embeddings(len(cfg.tokenizer))

        if hasattr(self.cfg.training, "multi_dropout") and self.cfg.training.multi_dropout:
            self.dropout = nn.Dropout(self.backbone_config.hidden_dropout_prob)
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.2)
            self.dropout3 = nn.Dropout(0.3)
            self.dropout4 = nn.Dropout(0.4)
            self.dropout5 = nn.Dropout(0.5)
        
        
        # self.pool = get_pooling_layer(cfg, backbone_config)
        self.fc = nn.Linear(self.backbone_config.hidden_size, len(self.cfg.dataset.target_cols))

        if 'bart' in cfg.architecture.model_name:
            self.initializer_range = self.backbone_config.init_std
        else:
            self.initializer_range = self.backbone_config.initializer_range

        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        outputs = self.backbone(**inputs).last_hidden_state
        # feature = self.pool(inputs, outputs)
        
        if hasattr(self.cfg.training, "multi_dropout") and self.cfg.training.multi_dropout:
            feature = self.dropout(outputs)
            logits1 = self.fc(self.dropout1(feature))
            logits2 = self.fc(self.dropout2(feature))
            logits3 = self.fc(self.dropout3(feature))
            logits4 = self.fc(self.dropout4(feature))
            logits5 = self.fc(self.dropout5(feature))
            output = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
            return output
        else:
            output = self.fc(outputs)
        return output
    








