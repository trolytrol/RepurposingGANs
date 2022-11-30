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


Dir_img = '/home/nontawat/gen_code'
Dir_mask = '/home/nontawat/gen_code'
Dir_checkpoint = '/home2/nontawat/som_8_11/'

parser = argparse.ArgumentParser(description='Train the Auto-shot',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--feature_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--checkpoint_dir', default=None)
args = parser.parse_args()


def train_unet(net,
              device,
              nam,
              num_class,
              size=256,
              trans=True,
              epochs=100,
              start=0,
              batch_size=32,
              lr=0.007,
              val_percent=0.005,
              cut=0,
              save_cp=True,
              opt_dict=None
              ):
    dir_img = os.path.join(Dir_img, nam, 'Image')
    dir_mask = os.path.join(Dir_mask, nam, 'Mask')
    dir_checkpoint = os.path.join(Dir_checkpoint, nam)

    dataset = UNetDataset(dir_img, dir_mask, num_class, size, trans, cut)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    cpp = dir_checkpoint.split('/')[-1]
    writer = SummaryWriter(comment=f'Name_{cfg.name}')

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Num. class:      {num_class}
        Images size:     {size}
        Transform:       {trans}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' , patience=20, min_lr=1e-6)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image'].cuda()
                true_masks = batch['mask'].cuda()

                pred_masks = net(imgs)

                # loss = softmax_cross_entropy_with_softtarget(pred_masks, true_masks, flag_masks)
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


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--name', type=str, required=True)
    parser.add_argument('-c', '--num_class', type=int, required=True, help='Number of class')
    parser.add_argument('-a', '--trans', type=bool, default=True, help='Augmentation')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--size', dest='size', type=int, default=512,
                        help='size of image')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-u', '--cut', dest='cut', type=int, default=0,
                        help='Amount of dataset cut')
    parser.add_argument('-t', '--conttrain', dest='trainload', type=str, default=False,
                        help='Continue training from a .pth file')

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {cfg.device}')

    net = UNet(n_channels=3, n_classes=cfg.seg_model.num_cls, bilinear=True, mse=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    start_ep=0
    if args.checkpoint_dir:
        net.load_state_dict(
            torch.load(args.checkpoint_dir, map_location=cfg.device))
        logging.info(f'Model loaded from {args.checkpoint_dir}')

    net.to(cfg.device)

    train_net(net=net,
            device=device,
            nam=args.name,
            num_class=args.num_class,
            size=args.size,
            trans=args.trans,
            epochs=args.epochs,
            start=start_ep,
            batch_size=args.batchsize,
            lr=args.lr,
            val_percent=args.val / 100,
            cut=args.cut,
            save_cp=True,
            opt_dict=op_cp)

