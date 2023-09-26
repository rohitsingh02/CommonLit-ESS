import glob
import torch
import os
import argparse


def average_checkpoints(input_folder, output_ckpt):
    input_ckpts = sorted(glob.glob(input_folder + '/*.pth'))
    # assert len(input_ckpts) >= 1
    if len(input_ckpts) >= 1:
        data = torch.load(input_ckpts[0], map_location='cpu')['state_dict']
        swa_n = 1
        for ckpt in input_ckpts[1:]:
            if "config" in ckpt:
                continue
            new_data = torch.load(ckpt, map_location='cpu')['state_dict']
            swa_n += 1
            for k, v in new_data.items():
                if v.dtype != torch.float32:
                    print(k)
                else:
                    data[k] += (new_data[k] - data[k]) / swa_n

        torch.save(dict(state_dict=data), output_ckpt)

