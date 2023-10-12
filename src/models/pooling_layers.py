import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


def get_last_hidden_state(backbone_outputs):
    last_hidden_state = backbone_outputs[0]
    return last_hidden_state


def get_all_hidden_states(backbone_outputs):
    all_hidden_states = torch.stack(backbone_outputs[1])
    return all_hidden_states


def get_input_ids(inputs):
    return inputs['input_ids']


def get_attention_mask(inputs):
    return inputs['attention_mask']


class MeanPooling(nn.Module):
    def __init__(self, backbone_config, config):
        super(MeanPooling, self).__init__()
        self.output_dim = backbone_config.hidden_size
        self.config = config

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        last_hidden_state = get_last_hidden_state(backbone_outputs)

        input_ids = inputs['input_ids'].to(last_hidden_state.device)  # Move input_ids to the same device as last_hidden_state
        start_token_index = (input_ids == self.config.text_start_token).nonzero()[:, 1]
        end_token_index = (input_ids == self.config.text_end_token).nonzero()[:, 1]
        # Create an attention mask based on start and end token indices
        batch_size, max_seq_len = attention_mask.shape
        valid_mask = torch.zeros((batch_size, max_seq_len), device=last_hidden_state.device)
        for i in range(batch_size):
            valid_mask[i, start_token_index[i]:end_token_index[i] + 1] = 1.0

        input_mask_expanded = valid_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    

class ConcatPooling(nn.Module):
    def __init__(self, backbone_config, config):
        super(ConcatPooling, self, ).__init__()

        self.config = config
        pooling_config = config.architecture.concat_pooling
        self.n_layers = pooling_config.n_layers
        self.output_dim = backbone_config.hidden_size*pooling_config.n_layers
        print(pooling_config)

    def forward(self, inputs, backbone_outputs):        
        all_hidden_states = get_all_hidden_states(backbone_outputs)
        concatenate_pooling = torch.cat([all_hidden_states[-(i + 1)] for i in range(self.n_layers)], -1)
        concatenate_pooling = concatenate_pooling[:, 0]
        return concatenate_pooling
    


class ConcatMeanPooling(nn.Module):
    def __init__(self, backbone_config, config):
        super(ConcatMeanPooling, self, ).__init__()

        self.config = config
        pooling_config = config.architecture.concat_pooling
        self.n_layers = pooling_config.n_layers
        # self.output_dim = backbone_config.hidden_size*pooling_config.n_layers
        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_outputs): 
        attention_mask = get_attention_mask(inputs)
        all_hidden_states = get_all_hidden_states(backbone_outputs)
        last_hidden_state = torch.mean(all_hidden_states[-(self.n_layers):], dim=0)

        input_ids = inputs['input_ids'].to(last_hidden_state.device)  # Move input_ids to the same device as last_hidden_state
        start_token_index = (input_ids == self.config.text_start_token).nonzero()[:, 1]
        end_token_index = (input_ids == self.config.text_end_token).nonzero()[:, 1]
        # Create an attention mask based on start and end token indices
        batch_size, max_seq_len = attention_mask.shape
        valid_mask = torch.zeros((batch_size, max_seq_len), device=last_hidden_state.device)
        for i in range(batch_size):
            valid_mask[i, start_token_index[i]:end_token_index[i] + 1] = 1.0

        input_mask_expanded = valid_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MeanMaxPooling(nn.Module):
    def __init__(self, backbone_config, config):
        super(MeanMaxPooling, self).__init__()
        self.config = config
        self.feat_mult = 1
        self.output_dim = backbone_config.hidden_size*2

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        last_hidden_state = get_last_hidden_state(backbone_outputs)

        input_ids = get_input_ids(inputs).to(last_hidden_state.device)  # Move input_ids to the same device as last_hidden_state
        start_token_index = (input_ids == self.config.text_start_token).nonzero()[:, 1]
        end_token_index = (input_ids == self.config.text_end_token).nonzero()[:, 1]
        # Create an attention mask based on start and end token indices
        batch_size, max_seq_len = attention_mask.shape
        valid_mask = torch.zeros((batch_size, max_seq_len), device=last_hidden_state.device)
        for i in range(batch_size):
            valid_mask[i, start_token_index[i]:end_token_index[i] + 1] = 1.0

        input_mask_expanded = valid_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        max_embeddings, _ = torch.max(last_hidden_state * input_mask_expanded, dim=1)
        # max_embedding[input_mask_expanded == 0] = -1e4
        mean_max_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)
        return mean_max_embeddings


class LSTMPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config, is_lstm=True):
        super(LSTMPooling, self).__init__()

        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.hidden_size = backbone_config.hidden_size
        self.hidden_lstm_size = pooling_config.hidden_size
        self.dropout_rate = pooling_config.dropout_rate
        self.bidirectional = pooling_config.bidirectional

        self.is_lstm = is_lstm
        self.output_dim = pooling_config.hidden_size*2 if self.bidirectional else pooling_config.hidden_size

        if self.is_lstm:
            self.lstm = nn.LSTM(self.hidden_size,
                                self.hidden_lstm_size,
                                bidirectional=self.bidirectional,
                                batch_first=True)
        else:
            self.lstm = nn.GRU(self.hidden_size,
                               self.hidden_lstm_size,
                               bidirectional=self.bidirectional,
                               batch_first=True)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out


class WeightedLayerPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(WeightedLayerPooling, self).__init__()

        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.layer_start = pooling_config.layer_start
        self.layer_weights = pooling_config.layer_weights if pooling_config.layer_weights is not None else \
            nn.Parameter(torch.tensor([1] * (self.num_hidden_layers + 1 - self.layer_start), dtype=torch.float))

        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average[:, 0]



class AttentionPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.hidden_size = backbone_config.hidden_size
        self.hiddendim_fc = pooling_config.hiddendim_fc
        self.dropout = nn.Dropout(pooling_config.dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().to(self.device)
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().to(self.device)

        self.output_dim = self.hiddendim_fc

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v
    

class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features,attention_mask):
        weights_mask = attention_mask.unsqueeze(-1)
        att = torch.tanh(self.W(features))
        score = self.V(att)
        score[attention_mask==0]=-1e4
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights*weights_mask*features, dim=1)
        return context_vector


class WKPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(WKPooling, self).__init__()

        self.layer_start = pooling_config.layer_start
        self.context_window_size = pooling_config.context_window_size

        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)
        attention_mask = get_attention_mask(inputs)

        ft_all_layers = all_hidden_states
        org_device = ft_all_layers.device
        all_layer_embedding = ft_all_layers.transpose(1, 0)
        all_layer_embedding = all_layer_embedding[:, self.layer_start:, :, :]

        all_layer_embedding = all_layer_embedding.cpu()

        attention_mask = attention_mask.cpu().numpy()
        unmask_num = np.array([sum(mask) for mask in attention_mask]) - 1
        embedding = []

        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index, :, :unmask_num[sent_index], :]
            one_sentence_embedding = []

            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:, token_index, :]
                token_embedding = self.unify_token(token_feature)
                one_sentence_embedding.append(token_embedding)

            one_sentence_embedding = torch.stack(one_sentence_embedding)
            sentence_embedding = self.unify_sentence(sentence_feature, one_sentence_embedding)
            embedding.append(sentence_embedding)

        output_vector = torch.stack(embedding).to(org_device)
        return output_vector




class MaxPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(MaxPooling, self).__init__()
        self.feat_mult = 1
        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        x = get_input_ids(inputs)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        embeddings = x.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MinPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(MinPooling, self).__init__()
        self.feat_mult = 1
        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_outputs):
        attention_mask = get_attention_mask(inputs)
        x = get_input_ids(inputs)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
        embeddings = x.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings



class GeMPooling(nn.Module):
    def __init__(self, backbone_config, config):
        super(GeMPooling, self).__init__()
        self.config = config
        pooling_config = config.architecture.gem_pooling
        self.dim = pooling_config.dim
        self.eps = pooling_config.eps
        self.feat_mult = 1
        self.p = Parameter(torch.ones(1) * pooling_config.p)
        self.output_dim = backbone_config.hidden_size

    def forward(self, inputs, backbone_output):
        last_hidden_state  = get_last_hidden_state(backbone_output)
        attention_mask = get_attention_mask(inputs)

        input_ids = get_input_ids(inputs).to(last_hidden_state.device)  # Move input_ids to the same device as last_hidden_state
        start_token_index = (input_ids == self.config.text_start_token).nonzero()[:, 1]
        end_token_index = (input_ids == self.config.text_end_token).nonzero()[:, 1]
        # # Create an attention mask based on start and end token indices
        batch_size, max_seq_len = attention_mask.shape
        valid_mask = torch.zeros((batch_size, max_seq_len), device=input_ids.device)
        for i in range(batch_size):
            valid_mask[i, start_token_index[i]:end_token_index[i] + 1] = 1.0

        attention_mask_expanded = valid_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # attention_mask_expanded = attention_mask.unsqueeze(-1).expand(input_ids.size()).float()
        input_ids = (last_hidden_state.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = input_ids / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret



# class GeMText(nn.Module):
#     def __init__(self, dim=1, p=3, eps=1e-6):
#         super(GeMText, self).__init__()
#         self.dim = dim
#         self.p = Parameter(torch.ones(1) * p)
#         self.eps = eps

#     def forward(self, x, attention_mask):
#         attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)
#         x = ((x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p)).sum(self.dim)
#         ret = (x/(attention_mask_expanded.sum(self.dim))).clip(min=self.eps)
#         ret = ret.pow(1/self.p)
#         return ret



def get_pooling_layer(config, backbone_config):
    if config.architecture.pooling_type == 'MeanPooling':
        return MeanPooling(backbone_config, config)

    elif config.architecture.pooling_type == 'ConcatMeanPooling':
        return ConcatMeanPooling(backbone_config, config)

    elif config.architecture.pooling_type == 'MeanMaxPooling':
        return MeanMaxPooling(backbone_config, config)


    elif config.architecture.pooling_type == 'MinPooling':
        return MinPooling(backbone_config, config.architecture.gru_pooling)
    
    elif config.architecture.pooling_type == 'MaxPooling':
        return MaxPooling(backbone_config, config.architecture.gru_pooling)
    

    elif config.architecture.pooling_type == 'GeMPooling':
        return GeMPooling(backbone_config, config)
    
    elif config.architecture.pooling_type == 'GRUPooling':
        return LSTMPooling(backbone_config, config.architecture.gru_pooling, is_lstm=False)

    elif config.architecture.pooling_type == 'LSTMPooling':
        return LSTMPooling(backbone_config, config.architecture.lstm_pooling, is_lstm=True)

    elif config.architecture.pooling_type == 'WeightedLayerPooling':
        return WeightedLayerPooling(backbone_config, config.architecture.weighted_pooling)

    elif config.architecture.pooling_type == 'WKPooling':
        return WKPooling(backbone_config, config.architecture.wk_pooling)

    elif config.architecture.pooling_type == 'ConcatPooling':
        return ConcatPooling(backbone_config, config)

    elif config.architecture.pooling_type == 'AttentionPooling':
        return AttentionPooling(backbone_config, config.architecture.attention_pooling)

    else:
        raise ValueError(f'Invalid pooling type: {config.model.pooling_type}')