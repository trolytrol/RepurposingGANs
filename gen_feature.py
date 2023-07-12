import torch, os, argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from nethook import InstrumentedModel
from stylegan2_pytorch import model
from config.base_config import parse_args

from utils import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Generate feature dataset')
parser.add_argument('--outdir', required=True)
parser.add_argument('--num_pic', default=1, type=int)
args = parser.parse_args()

def gen_feature(gen_model, num_pic, 
            img_size, feat_size,
            img_path, feat_path):
    with torch.no_grad():
        for idx in tqdm(range(num_pic)):
            z_noise = utils.z_sample(seed=idx).unsqueeze(0) # [1, 1, 512]
            stack, rgb_im = utils.get_stack(gen_model, z_noise.to(device), feat_size)

            rgb_im = ((rgb_im + 1) / 2 * 255)
            rgb_im = rgb_im.permute(0, 2, 3, 1).clamp(0, 255).byte().cpu().numpy()

            filename = os.path.join(img_path, '%d%s' %(idx, '.png'))
            Image.fromarray((rgb_im[0]).astype(np.uint8)).resize([img_size, img_size]).save(filename,
                                                optimize=True, quality=100)

            feat_file = os.path.join(feat_path, '%d%s' %(idx, '.pt'))
            torch.save(stack.cpu(), feat_file)


def main():
    cfg = parse_args()

    gen_model = model.Generator(size=cfg.gen_model.img_size, 
                            style_dim=cfg.gen_model.style_dim, 
                            n_mlp=cfg.gen_model.n_mlp, 
                            input_is_Wlatent=False)
    gen_model.load_state_dict(torch.load(cfg.data.gen_model_dir)['g_ema'], strict=False)
    gen_models = InstrumentedModel(gen_model)
    gen_models.eval()
    gen_models.to(device)
    gen_models.retain_layers(cfg.gen_model.stylegan_dict)

    img_path = os.path.join(args.outdir, 'images')
    feat_path = os.path.join(args.outdir, 'features')
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(feat_path, exist_ok=True)

    gen_feature(gen_models, args.num_pic, 
            cfg.gen_model.img_size, cfg.seg_model.feature_size,
            img_path, feat_path)


if __name__ == "__main__":
    main()
