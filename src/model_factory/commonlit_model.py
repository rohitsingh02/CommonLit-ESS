import numpy as np
import torch
from torch import nn
from sklearn import metrics
from transformers import (
    AdamW,
    AutoModel,
    AutoConfig,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
    
class RMSELoss(nn.Module):
    """
    Code taken from Y Nakama's notebook (https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train)
    """
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, predictions, targets):
        loss = torch.sqrt(self.mse(predictions, targets) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class CommonlitModel(nn.Module):
    def __init__(self, cfg, model_name, num_train_steps, learning_rate, num_labels, steps_per_epoch, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels  
        self.steps_per_epoch = steps_per_epoch
        self.pool = MeanPooling()     
        hidden_dropout_prob: float = 0.0
        self.config = AutoConfig.from_pretrained(model_name)

        self.config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "add_pooling_layer": True,
                "num_labels": self.num_labels,
            }
        )

#         self.transformer = DebertaV2Model.from_pretrained(model_name, config=config)   
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        self.transformer.resize_token_embeddings(len(tokenizer))        
        # if hasattr(self.cfg.architecture, "gradient_checkpointing") and self.cfg.architecture.gradient_checkpointing:
        #     self.transformer.gradient_checkpointing_enable()
        

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(self.config.hidden_size, self.num_labels)
        self._init_weights(self.output)        
        
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
                
                

    def optimizer_scheduler(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.learning_rate)
        # sch = get_polynomial_decay_schedule_with_warmup(
        #     opt,
        #     num_warmup_steps=int(self.num_train_steps * 0.2),
        #     num_training_steps=self.num_train_steps,
        #     last_epoch=-1,
        # )
        sch = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps= int(0.1 * self.num_train_steps),
            num_training_steps=self.num_train_steps,
            num_cycles=1,
            last_epoch=-1,
        )
        
        return opt, sch

    
    def loss(self, outputs, targets):
        # weights = [1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 1.0, 1.0, 1.3, 1.3, 1.3, 1.3, 0.8]
        # class_weights = torch.FloatTensor(weights).cuda()
        loss_fct = RMSELoss()     
        loss = loss_fct(outputs, targets)
        return loss
    

    def monitor_metrics(self, outputs, targets, attention_mask, num_labels):
        active_loss = (attention_mask.view(-1) == 1).cpu().numpy()
        active_logits = outputs.view(-1, num_labels)
        true_labels = targets.view(-1).cpu().numpy()
        outputs = active_logits.argmax(dim=-1).cpu().numpy()
        idxs = np.where(active_loss == 1)[0]
        f1_score = metrics.f1_score(true_labels[idxs], outputs[idxs], average="macro")
        return {"f1": torch.tensor(f1_score).cuda()}
        
    
    def forward(self, ids, mask, token_type_ids=None, targets=None):

        if token_type_ids is not None:
            transformer_out = self.transformer(
                input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids
            )
        else:
            transformer_out = self.transformer(input_ids=ids, attention_mask=mask)

        sequence_output = transformer_out.last_hidden_state
        if self.cfg.architecture.pool == "Mean":
            sequence_output = self.pool(sequence_output, mask) # new line
        sequence_output = self.dropout(sequence_output)
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
                
        # logits_softmax = torch.softmax(logits_f1, dim=-1)
        
        loss = 0
        if targets is not None:
            loss1 = self.loss(logits1, targets)
            loss2 = self.loss(logits2, targets)
            loss3 = self.loss(logits3, targets)
            loss4 = self.loss(logits4, targets)
            loss5 = self.loss(logits5, targets)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            metric = {"rmse": loss}            
            return logits, loss, metric

        return logits, loss, {}
    
    
    
    

class FeedbackModel2(nn.Module):
    def __init__(self, cfg, model_name, num_train_steps, learning_rate, num_labels, num_labels2, steps_per_epoch, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels  
        self.num_labels2 = num_labels2
        self.steps_per_epoch = steps_per_epoch
        hidden_dropout_prob: float = 0.0
        config = AutoConfig.from_pretrained(model_name)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
                "num_labels2": self.num_labels2,
            }
        )

#         self.transformer = DebertaV2Model.from_pretrained(model_name, config=config)   
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.transformer.resize_token_embeddings(len(tokenizer))        
        if hasattr(self.cfg.architecture, "gradient_checkpointing") and self.cfg.architecture.gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()
        

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output = nn.Linear(config.hidden_size, self.num_labels)
        self.output2 = nn.Linear(config.hidden_size, self.num_labels2)
        
        
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def optimizer_scheduler(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]    
        opt = torch.optim.AdamW(optimizer_parameters, lr=self.learning_rate)
        sch = get_polynomial_decay_schedule_with_warmup(
            opt,
            num_warmup_steps=int(self.num_train_steps * 0.2),
            num_training_steps=self.num_train_steps,
            last_epoch=-1,
        )
        return opt, sch

    
    def loss(self, outputs, targets, attention_mask, num_labels):
        
        
        
        
        loss_fct = nn.CrossEntropyLoss()
        active_loss = attention_mask.view(-1) == 1
        active_logits = outputs.view(-1, num_labels)
        true_labels = targets.view(-1)
        outputs = active_logits.argmax(dim=-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)        
        loss = loss_fct(active_logits, true_labels)
        return loss
    

    

    def monitor_metrics(self, outputs, targets, attention_mask, num_labels):
        active_loss = (attention_mask.view(-1) == 1).cpu().numpy()
        active_logits = outputs.view(-1, num_labels)
        true_labels = targets.view(-1).cpu().numpy()
        outputs = active_logits.argmax(dim=-1).cpu().numpy()
        idxs = np.where(active_loss == 1)[0]
        f1_score = metrics.f1_score(true_labels[idxs], outputs[idxs], average="macro")
        return {"f1": torch.tensor(f1_score).cuda()}
        
    
    def forward(self, ids, mask, token_type_ids=None, targets=None, targets2=None):

        if token_type_ids is not None:
            transformer_out = self.transformer(
                input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids
            )
        else:
            transformer_out = self.transformer(input_ids=ids, attention_mask=mask)

        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits_f1 = self.output(sequence_output)
        logits_softmax = torch.softmax(logits_f1, dim=-1)
        

        logits_f2 = self.output2(sequence_output)
        logits2_softmax = torch.softmax(logits_f2, dim=-1)
        
        
        loss = 0
        if targets is not None:
            loss_f1 = self.loss(logits_f1, targets, attention_mask=mask, num_labels=self.num_labels)            
            loss_f2 = self.loss2(logits_f2, targets2, attention_mask=mask, num_labels=self.num_labels2)
            loss = (loss_f1 + loss_f2) / 2
            f1_score_o1 = self.monitor_metrics(logits_f1, targets, attention_mask=mask, num_labels=self.num_labels)["f1"]                        
            f1_score_o2 = self.monitor_metrics(logits_f2, targets2, attention_mask=mask, num_labels=self.num_labels2)["f1"]

            if hasattr(self.cfg.training, "preferred_f1") and self.cfg.training.preferred_f1 == "fb1":
                f1 = f1_score_o1
            else:
                f1 =  0.5* f1_score_o1 + 0.5*f1_score_o2 # 2*(f1_score_o1*f1_score_o2)/(f1_score_o1+f1_score_o2)
            
            metric = {"f1": f1}
            
            return (logits_softmax, logits2_softmax), loss, metric

        return (logits_softmax, logits2_softmax), loss, {}