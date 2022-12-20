import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import data
import argparse
import os
from tqdm import tqdm
from networks import GMM, UnetGenerator, load_checkpoint
from FASCODE_IMAGE import FASCODE_IMAGE

from torch.utils.tensorboard import SummaryWriter
from visualization import board_add_image, board_add_images, save_images

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
gpus_id = dist.get_rank()

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/DataSet/FASCODE_IMAGE")
    parser.add_argument("--categories", default="top")
    parser.add_argument("--item", default=None)
    
    opt = parser.parse_args()
    return opt

def get_dataloader(opt):
    dataset = FASCODE_IMAGE(opt.root, opt.categories, opt.item)

    return data.DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=data.DistributedSampler(dataset),
        num_workers=6,
        pin_memory=True
        )

if __name__ == "__main__":
    opt = get_opt()

    dataloader = get_dataloader(opt)