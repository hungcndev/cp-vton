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
    
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--grid_size", type=int, default = 5)
    
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

def get_model(opt):
    model = GMM(opt)

    model.load_state_dict(torch.load(opt.checkpoint_path))
    model.cuda()

    return model

def test_gmm(opt, test_loader, model, board):
    model = DDP(model.cuda(), [gpus_id])
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    # save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    save_dir = "data/test"

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')

    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    for step, inputs in enumerate(test_loader.data_loader):
        test_loader.data_loader.sampler.set_epoch(step)
        # iter_start_time = time.time()
        
        c_names = inputs['c_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
            
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=True)
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=True)
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=True)

        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        save_images(warped_cloth, c_names, warp_cloth_dir) 
        save_images(warped_mask*2-1, c_names, warp_mask_dir) 

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            # t = time.time() - iter_start_time
            # print('step: %8d, time: %.3f' % (step+1, t), flush=True)

def test_tom(opt, test_loader, model, board):
    model = DDP(model.cuda(), [gpus_id])
    model.eval()
    
    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    try_on_dir = os.path.join(save_dir, 'try-on')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)

    # print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)

    for step, inputs in enumerate(test_loader.data_loader):
        # iter_start_time = time.time()
        
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, 2*cm-1, m_composite], 
                   [p_rendered, p_tryon, im]]
            
        save_images(p_tryon, im_names, try_on_dir) 
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            # t = time.time() - iter_start_time
            # print('step: %8d, time: %.3f' % (step+1, t), flush=True)

def test(opt):
    dataloader = get_dataloader(opt)
    model = get_model(opt)

    writer = SummaryWriter("tensorboard/FASCODE_DATA/")
    
    with torch.no_grad():
        test_gmm(opt, dataloader, model, writer)

if __name__ == "__main__":
    opt = get_opt()

    test(opt)
