#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms
from torchvision.utils import make_grid

import argparse, datetime, os
from PIL import Image
from tqdm import tqdm
from cp_dataset import CPDataset

from networks import GMM, UnetGenerator

from torch.utils.tensorboard import SummaryWriter


dist.init_process_group("nccl")
gpus_id = dist.get_rank()

def get_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--name", default = "TOM")
    parser.add_argument('-j', '--num_workers', type=int, default=6)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    # parser.add_argument("--stage", default = "TOM")
    # parser.add_argument("--data_list", default = "test_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    # parser.add_argument('--checkpoint', type=str, default='checkpoints/TOM/tom_final.pth', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default = 1)
    # parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = get_opt()
    
    # create dataset 
    dataset = CPDataset(opt)

    # create dataloader
    dataloader = DataLoader(dataset = dataset, 
                             batch_size = opt.batch_size, 
                             sampler=DistributedSampler(dataset),
                             num_workers=opt.num_workers,
                             pin_memory=True
                             )
    
    now = datetime.datetime.now()
    now_path = os.path.join(now.strftime("%Y%m%d"), now.strftime("%H:%M:%S"))
    tensorboard_path = os.path.join(opt.tensorboard_dir, now_path)
    result_path = os.path.join(opt.result_dir, now_path)
    checkpoint_GMM_path = "checkpoints/GMM/gmm_final.pth"
    checkpoint_TOM_path = "checkpoints/TOM/tom_final.pth"
    
    os.makedirs(result_path, exist_ok=True)
    
    model_GMM = GMM(opt)
    model_GMM.load_state_dict(torch.load(checkpoint_GMM_path))
    model_GMM = DDP(model_GMM.to(gpus_id), [gpus_id])
    
    model_TOM = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
    model_TOM.load_state_dict(torch.load(checkpoint_TOM_path))
    model_TOM = DDP(model_TOM.to(gpus_id), [gpus_id])
    
    model_GMM.eval()
    model_TOM.eval()
    
    # writer = SummaryWriter(tensorboard_path)
    
    for batch in tqdm(dataloader):
        with torch.no_grad():
            c_names = batch['c_name']
            im_names = batch['im_name']
            im = batch['image'].to(gpus_id)
            im_pose = batch['pose_image'].to(gpus_id)
            im_h = batch['head'].to(gpus_id)
            shape = batch['shape'].to(gpus_id)
            agnostic = batch['agnostic'].to(gpus_id)
            c = batch['cloth'].to(gpus_id)
            cm = batch['cloth_mask'].to(gpus_id)
            im_c =  batch['parse_cloth'].to(gpus_id)
            im_g = batch['grid_image'].to(gpus_id)
            
            grid, _ = model_GMM(agnostic, c)
            
            c = F.grid_sample(c, grid, padding_mode='border', align_corners=True)
            cm = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=True)
            im_g = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=True)
            
            outputs = model_TOM(torch.cat([agnostic, c], 1))
            p_rendered, m_composite = torch.split(outputs, 3,1)
            p_rendered = torch.tanh(p_rendered)
            m_composite = torch.sigmoid(m_composite)
            p_tryon = c * m_composite + p_rendered * (1 - m_composite)
            
            transforms.ToPILImage()(make_grid(p_tryon, normalize=True)).save(os.path.join(result_path, im_names[0].split("_0")[0]+".png"))