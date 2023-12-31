import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding


class CommonlitDataset(Dataset):
    def __init__(self, cfg, df, mode="train"):
        self.cfg = cfg
        self.df = df
        self.texts = self.df['input_text'].values
        self.mask_token = self.cfg.tokenizer.convert_tokens_to_ids(self.cfg.tokenizer.mask_token)
        self.sep_token = self.cfg.tokenizer.sep_token
        self.mode = mode
        self.labels = None
        if cfg.dataset.target_cols[0] in df.columns and self.mode != "test":
            self.labels = df[cfg.dataset.target_cols].values

        if hasattr(self.cfg.dataset, "use_pseudo_targets") and self.mode == "train":
            self.labels_pseudo = df[['content_pred', 'wording_pred']].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):

        text = self.texts[item]   
        inputs = self.cfg.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.cfg.dataset.max_length,
            padding="max_length",
            truncation=True,
        )
        
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
            
            
        # if  self.cfg.training.mask_ratio > 0 and self.mode == "train":
        #     ix = torch.rand(size=(len(inputs["input_ids"]),)) < self.cfg.training.mask_ratio # 0.25
        #     inputs["input_ids"][ix] = self.mask_token
            
        
        if self.labels is not None:
            if hasattr(self.cfg.dataset, "use_pseudo_targets") and self.mode == "train":
                label_tmp = self.labels_pseudo[item]
                label_original = self.labels_pseudo[item]
                if self.cfg.dataset.use_pseudo_targets.type == "pseudo":
                    label = torch.tensor(label_tmp, dtype=torch.float)
                elif self.cfg.dataset.use_pseudo_targets.type == "mean":
                    label = (label_original + label_tmp) / 2
                    label = torch.tensor(label, dtype=torch.float)
                elif self.cfg.dataset.use_pseudo_targets.type == "mean_diff":
                    label = label_original - ((label_original + label_tmp) / 2)
                    label = torch.tensor(label, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
            else:
                label = torch.tensor(self.labels[item], dtype=torch.float)
            return inputs, label
        return inputs



def get_train_dataloader(cfg, df):
    dataset = CommonlitDataset(cfg, df, mode="train")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.train_batch_size,
        num_workers=cfg.environment.n_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def get_valid_dataloader(cfg, df):
    dataset = CommonlitDataset(cfg, df, mode="val")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.valid_batch_size,
        num_workers=cfg.environment.n_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader



def get_test_dataloader(cfg, df):
    dataset = CommonlitDataset(cfg, df, mode="test")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.valid_batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=cfg.tokenizer, padding='longest'),
        num_workers=cfg.environment.n_workers,
        pin_memory=True,
        drop_last=False
    )
    return dataloader