import argparse
import logging
import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from dataset import UNetDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from tqdm import tqdm

from eval import eval_npy
from unet import UNet

from config.base_config import parse_args


parser = argparse.ArgumentParser(description='Train the Auto-shot',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--checkpoint_dir', default=None)
args = parser.parse_args()

parser = argparse.ArgumentParser(description='Train the Auto-shot',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--feature_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--checkpoint_dir', default=None)
args = parser.parse_args()


def train_unet(net, cfg):
    dataset = UNetDataset(args.image_dir, args.mask_dir, cfg.seg_model.num_cls, 
                        cfg.unet_train_config.size,
                        cfg.unet_train_config.augmentation)
    n_val = int(len(dataset) * cfg.unet_train_config.val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=cfg.unet_train_config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=cfg.unet_train_config.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    writer = SummaryWriter(comment=f'Name_{cfg.name}')

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {cfg.unet_train_config.epochs}
        Batch size:      {cfg.unet_train_config.batch_size}
        Learning rate:   {cfg.unet_train_config.lrr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {cfg.device}
        Num. class:      {cfg.seg_model.num_cls}
        Images size:     {cfg.unet_train_config.size}
        Transform:       {cfg.unet_train_config.augmentation}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=cfg.unet_train_config.lr, weight_decay=cfg.unet_train_config.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' , patience=20, min_lr=1e-6)

    for epoch in range(cfg.unet_train_config.epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{cfg.unet_train_config.epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image'].cuda()
                true_masks = batch['mask'].cuda()

                pred_masks = net(imgs)

                loss = F.cross_entropy(pred_masks, true_masks)

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_npy(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                       logging.info('Validation cross entropy: {}'.format(val_score))
                       writer.add_scalar('Loss/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)

        if save_cp and (epoch+1)%50==0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save({'epoch' : epoch,
                        'model_state_dict' : net.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict() },
                        dir_checkpoint + '/'+ f'CP_epoch{epoch + 1}_cont.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')


    writer.close()


def main():
    cfg = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {cfg.device}')
    logging.info(f'Network:\n'
                 f'\t{3} input channels\n'
                 f'\t{cfg.seg_model.num_cls} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    net = UNet(n_channels=3, n_classes=cfg.seg_model.num_cls, bilinear=True)
    if args.checkpoint_dir:
        net.load_state_dict(
            torch.load(args.checkpoint_dir, map_location=cfg.device))
        logging.info(f'Model loaded from {args.checkpoint_dir}')

    net.to(cfg.device)

    train_unet(net, cfg)

if __name__ == '__main__':
    main()

