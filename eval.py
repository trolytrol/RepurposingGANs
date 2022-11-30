import torch
import torch.nn.functional as F
from tqdm import tqdm

def eval_unet(net, loader):
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image'].cuda()
            true_masks = batch['mask'].cuda()
            with torch.no_grad():
                mask_pred = net(imgs)
                tot += F.cross_entropy(mask_pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val
