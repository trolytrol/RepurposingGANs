from yacs.config import CfgNode as CN
import argparse
import datetime

cfg = CN()

cfg.name = ''
cfg.device = 'cuda'

cfg.gen_model = CN()
cfg.gen_model.img_size = 1024
cfg.gen_model.style_dim = 512
cfg.gen_model.n_mlp = 8
cfg.gen_model.stylegan_dict = [None]

cfg.seg_model = CN()
cfg.seg_model.num_cls = None
cfg.seg_model.mask_size = 512
cfg.seg_model.feature_size = 512
cfg.seg_model.feature_dim = 5056

cfg.seg_train_config = CN()
cfg.seg_train_config.lr = 0.001
cfg.seg_train_config.lr_decay = 0.9
cfg.seg_train_config.batch_size = 1
cfg.seg_train_config.epochs = 1000
cfg.seg_train_config.augmentation = True
cfg.seg_train_config.wd = 1e-3


cfg.unet_train_config = CN()
cfg.unet_train_config.lr = 0.001
cfg.unet_train_config.batch_size = 1
cfg.unet_train_config.epochs = 1000
cfg.unet_train_config.augmentation = True
cfg.unet_train_config.wd = 1e-3

cfg.data = CN()
cfg.data.gen_model_dir = "models/gen_models"
cfg.data.seg_model_dir = "models/seg_models"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args(ipynb={'mode':False, 'cfg':None}):
    '''
    Return dict-like cfg, accesible with cfg.<key1>.<key2> or cfg[<key1>][<key2>]
    e.g. <key1> = dataset, <key2> = training_data
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    args, opts = parser.parse_known_args()
    if ipynb['mode']:
        # Using this with ipynb will have some opts defaults from ipynb and we need to filter out.
        opts=[]
        args.cfg = ipynb['cfg']

    print("Merging with : ", args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    # Merge with cmd-line argument(s)

    if opts != []:
        cfg_list = cmd_to_cfg_format(opts)
        cfg.merge_from_list(cfg_list)

    return cfg

def cmd_to_cfg_format(opts):
    """
    Override config from a list
    src-format : ['--dataset.train', '/data/mint/dataset']
    dst-format : ['dataset.train', '/data/mint/dataset']
    for writing a "dataset.train" key
    """
    opts_new = []
    for i, opt in enumerate(opts):
        if (i+1) % 2 != 0:
            opts_new.append(opt[2:])
        else: 
            opts_new.append(opt)
    return opts_new


if __name__ == '__main__':
    print(parse_args())
    cfg = parse_args()
    print(cfg.dataset)
