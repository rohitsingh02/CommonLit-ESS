import torch
import numpy as np

class CommonlitDataset:
    def __init__(self, samples, max_len, tokenizer, mode="valid"):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]["input_ids"]
        input_labels = self.samples[idx]["input_labels"]
        
        input_ids = [self.tokenizer.cls_token_id] + ids
        input_labels = input_labels
        
        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[: self.max_len - 1]
            
            
        # add end token id to the input_ids
        input_ids = input_ids + [self.tokenizer.sep_token_id]
        input_labels = input_labels
        attention_mask = [1] * len(input_ids)

        res = {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "targets": torch.tensor(input_labels, dtype=torch.float),
        }
        
        return res
    
    
    

# class FeedbackDataset2:
#     def __init__(self, samples, max_len, tokenizer, mode="valid"):
#         self.samples = samples
#         self.max_len = max_len
#         self.tokenizer = tokenizer
#         self.length = len(samples)
#         self.mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
#         self.mode = mode

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         ids = self.samples[idx]["input_ids"]
#         input_labels = self.samples[idx]["input_labels"]
#         input_labels2 = self.samples[idx]["input_labels2"]

#         input_labels = [target_id_map[x] for x in input_labels]
#         input_labels2 = [eff_id_map[x] for x in input_labels2]

#         other_label_id = target_id_map["O"]
#         other_label_id2 = eff_id_map["O"]
        
#         input_ids = [self.tokenizer.cls_token_id] + ids
#         input_labels = [other_label_id] + input_labels
#         input_labels2 = [other_label_id2] + input_labels2
        
#         if len(input_ids) > self.max_len - 1:
#             input_ids = input_ids[: self.max_len - 1]
#             input_labels = input_labels[: self.max_len - 1]
#             input_labels2 = input_labels2[: self.max_len - 1]

 
#         # add end token id to the input_ids
#         input_ids = input_ids + [self.tokenizer.sep_token_id]
#         input_labels = input_labels + [other_label_id]
#         input_labels2 = input_labels2 + [other_label_id2]
#         attention_mask = [1] * len(input_ids)
        
#         input_ids = torch.tensor(input_ids, dtype=torch.long)
        
#         if self.mode == "train":
#             ix = torch.rand(size=(len(input_ids),)) < 0.25
#             input_ids[ix] = self.mask_token
        
        
#         res = {
#             # "ids": torch.tensor(input_ids, dtype=torch.long),
#             "ids": input_ids,
#             "mask": torch.tensor(attention_mask, dtype=torch.long),
#             "targets": torch.tensor(input_labels, dtype=torch.long),
#             "targets2": torch.tensor(input_labels2, dtype=torch.long),
#         }
        
#         return res
    