import torch, os, argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from nethook import InstrumentedModel
from stylegan2_pytorch import model
from config.base_config import parse_args

from network import dilated_CNN_61
from utils import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Generate auto-shot dataset')
parser.add_argument('--outdir', required=True)
parser.add_argument('--num_pic', default=500, type=int)
args = parser.parse_args()

def gen_gan(gen_model, seg_model,
            num_pic, img_size,
            mask_size, feat_size,
            img_path, mask_path):
    with torch.no_grad():
        for idx in tqdm(range(num_pic)):
            z_noise = z_sample(seed=idx).unsqueeze(0) # [1, 1, 512]
            stack, rgb_im = get_stack(gen_model, z_noise.to(device), feat_size)

            rgb_im = ((rgb_im[0] + 1) / 2 * 255)
            rgb_im = rgb_im.permute(0, 2, 3, 1).clamp(0, 255).byte().cpu().numpy()


            output = seg_model(stack.to(device)) # Logit: [1, c, h, w] 
            output = output[0].reshape((output.shape[1], -1)).T # [hw, c] for visualize

            _, argmax_out = output.max(dim=1) # [hw, 1]
            mask = viz(argmax_out, feat_size)

            for (prefix, suffix, file, size) in [(mask_path, 'mask.png', mask, mask_size),
                                                (img_path, 'img.jpg', rgb_im[0], img_size)]:
                filename = os.path.join(prefix, '%d-%s' %(idx, suffix))
                Image.fromarray((file).astype(np.uint8)).resize([size, size]).save(filename,
                                                    optimize=True, quality=100)

            logit_file = os.path.join(mask_path, '%d%s' %(idx, '.pt'))
            torch.save(output.cpu(), logit_file)


def main():
    cfg = parse_args()

    gen_model = model.Generator(size=cfg.gen_model.img_size, 
                            style_dim=cfg.gen_model.style_dim, 
                            n_mlp=cfg.gen_model.n_mlp)
    gen_model.load_state_dict(torch.load(cfg.data.gen_model_dir)['g_ema'], strict=False)
    gen_models = InstrumentedModel(gen_model)
    gen_models.eval()
    gen_models.to(device)
    gen_models.retain_layers(cfg.gen_model.stylegan_dict)

    seg_model = dilated_CNN_61(cfg.seg_model.num_cls, cfg.seg_model.feature_dim)
    seg_model.load_state_dict(torch.load(cfg.data.seg_model_dir))
    seg_model.eval()
    seg_model.to(device)

    img_path = os.path.join(args.outdir, 'Image')
    mask_path = os.path.join(args.outdir, 'Mask')
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    gen_gan(gen_models, seg_model,
            args.num_pic, cfg.gen_model.img_size, 
            cfg.seg_model.mask_size, cfg.seg_model.feature_size,
            img_path, mask_path)


if __name__ == "__main__":
    main()
