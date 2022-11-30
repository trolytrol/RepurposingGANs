from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch, os
import random

from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import logging
from PIL import Image

from utils import get_stack, load_mask


class FewshotDataset(Dataset):
    '''
    Get pairs of precomputed features and masks
    
    '''
    def __init__(self, feature_dir, mask_dir,  mask_size, feat_size):
        self.feature_dir = feature_dir
        self.mask_dir = mask_dir

        self.feature_file = listdir(feature_dir)

        self.mask_size = mask_size
        self.feat_size = feat_size

    def __len__(self):
        return len(self.feature_file)

    def __getitem__(self, i):
        file_name = self.feature_file[i]
        mask_name = self.feature_file[i].split('.')[0]

        feature = torch.load(os.path.join(self.feature_dir, file_name)) #[c, h, w]
        mask = load_mask(self.mask_dir, [mask_name], self.mask_size)

        seg_flat = mask[0].reshape(-1,4)
        seg_flat = np.argmax(seg_flat, axis=1)

        return {
            'feature': torch.from_numpy(feature.T),
            'label': seg_flat
        }

class UNetDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, num_class, size=512, trans=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.num_class = num_class
        self.size = size
        self.trans = trans
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def transform_npy(self, img, label):
        ang, trans, scl, shr = transforms.RandomAffine.get_params(degrees = (-10,10),
                                                                translate = (0.5,0.5),
                                                                scale_ranges=(0.5,2.0),
                                                                shears=None,
                                                                img_size=(self.size,self.size))
        img = transforms.functional.affine(img, ang, trans, scl, shr, fillcolor=0)
        label = transforms.functional.affine(label, ang, trans, scl, shr, fillcolor=0)
        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            label = transforms.functional.hflip(label)
        img = transforms.ToTensor()(img)
        return img, label

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        midx = idx.replace('-img', '')
        mask_file = self.masks_dir + '/' +midx + '.pt'
        img_file = self.imgs_dir + '/' +idx + '.jpg'

        mask = torch.load(mask_file) #[C, 512, 512] {0, 255}
        mask = F.softmax(torch.tensor(mask), dim=0) #[512, 512] # In case of color masks

        img = Image.open(img_file)
        img.load()

        if self.trans == True:
            seed = random.randint(0,2**32)
            random.seed(seed)
            img, mask = self.transform_npy(img, mask, self.size, self.num_class)

        return {
            'image': img,
            'mask': mask,
        }