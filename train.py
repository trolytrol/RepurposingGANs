import torch, os, argparse
from torch.utils.data import DataLoader
from config.base_config import parse_args

from dataset import FewshotDataset
from network import dilated_CNN_61

parser = argparse.ArgumentParser(description='Train the Fewshot segment on features and target masks',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--feature_dir', required=True)
parser.add_argument('--mask_dir', required=True)
args = parser.parse_args()

def train_fewshot(net, cfg, outdir):
    trainDataset = FewshotDataset(args.feature_dir, args.mask_dir, 
                        img_size=cfg.seg_model.img_size,
                        mask_size=cfg.seg_model.mask_size,
                        feature_size=cfg.seg_model.feature_size)
    trainLoader = DataLoader(dataset = trainDataset,
                        batch_size=cfg.seg_train_config.batch_size,
                        shuffle=False, num_workers=0, pin_memory=False)
    print('Shot :', trainLoader.__len__())

    criterion = torch.nn.NLLLoss().to(cfg.device)
    optimizer = torch.optim.Adam(net.parameters(),
                            lr=cfg.seg_train_config.lr,
                            weight_decay=cfg.seg_train_config.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                gamma=cfg.seg_train_config.lr_decay)

    for step in range(cfg.seg_train_config.epochs):
        for batch in trainLoader:
            data = batch['image'].to(cfg.device)
            target = batch['label'].to(cfg.device)

            total_loss = 0
            optimizer.zero_grad()
            prediction = net(data)
            loss = criterion(prediction, target)
            total_loss+=loss
            loss.backward()
            optimizer.step()
        print('Epoch : {} Loss : {}'.format(step, total_loss.item()/trainLoader.__len__()))
        if step%50 == 0 and step!=0 :
            lr_scheduler.step()
            torch.save(net.state_dict(), os.path.join(outdir, f'{cfg.name}_{step}.pt'))

def main():
    cfg = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {cfg.device}')
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    
    net = dilated_CNN_61(cfg., 4864)
    
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=cfg.device))
        logging.info(f'Model loaded from {args.load}')

    net.to(cfg.device)

    train_fewshot(net, cfg)

if __name__ == '__main__':
    main()
    
