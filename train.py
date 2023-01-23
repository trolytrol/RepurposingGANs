import torch, os, argparse, logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config.base_config import parse_args

from tqdm import tqdm
from dataset import FewshotDataset
from network import dilated_CNN_61

parser = argparse.ArgumentParser(description='Train the Fewshot segment on features and target masks',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--feature_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--checkpoint_dir', default=None)
args = parser.parse_args()

def train_fewshot(net, cfg, outdir):
    writer = SummaryWriter(comment=f'Name_{cfg.name}')
    trainDataset = FewshotDataset(args.feature_dir, args.mask_dir, 
                        img_size=cfg.seg_model.img_size,
                        mask_size=cfg.seg_model.mask_size,
                        feature_size=cfg.seg_model.feature_size)
    trainLoader = DataLoader(dataset = trainDataset,
                        batch_size=cfg.seg_train_config.batch_size,
                        shuffle=False, num_workers=0, pin_memory=False)

    criterion = torch.nn.NLLLoss().to(cfg.device)
    optimizer = torch.optim.Adam(net.parameters(),
                            lr=cfg.seg_train_config.lr,
                            weight_decay=cfg.seg_train_config.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                gamma=cfg.seg_train_config.lr_decay)
    global_step = 0
    for epoch in range(cfg.seg_train_config.epochs):

        epoch_loss = 0
        with tqdm(total=len(trainDataset), desc=f'Epoch {epoch + 1}/{cfg.seg_train_config.epochs}', unit='img') as pbar:
            for batch in trainLoader:
                data = batch['feature'].to(cfg.device)
                target = batch['label'].to(cfg.device)
                
                prediction = net(data)

                loss = criterion(F.log_softmax(prediction, dim=1), target)

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(data.shape[0])
                global_step += 1
            
            if epoch%50 == 0 and epoch!=0 :
                lr_scheduler.step()
                torch.save(net.state_dict(), os.path.join(outdir, f'{cfg.name}_{epoch}.pt'))
    writer.close()

def main():
    cfg = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {cfg.device}')
    logging.info(f'Network:\n'
                 f'\t{cfg.seg_model.feature_dim} feature channels\n'
                 f'\t{cfg.seg_model.num_cls} output channels (classes)\n')
    
    net = dilated_CNN_61(cfg.seg_model.num_cls, cfg.seg_model.feature_dim)
    if args.checkpoint_dir:
        net.load_state_dict(
            torch.load(args.checkpoint_dir, map_location=cfg.device))
        logging.info(f'Model loaded from {args.checkpoint_dir}')

    net.to(cfg.device)
    train_fewshot(net, cfg)

if __name__ == '__main__':
    main()
    
