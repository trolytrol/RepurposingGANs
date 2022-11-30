from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch, os
import random

from torch.utils.data import Dataset
import logging
from PIL import Image

from function import get_stack, load_mask


class FewshotDataset(Dataset):
    def __init__(self, imgs_dir, mask_dir, gen_model, img_size,
                mask_size, feat_size=256):
        
        self.imgs_dir = imgs_dir
        self.mask_dir = mask_dir

        self.img_file = listdir(imgs_dir)

        self.gen_model = gen_model
        self.img_size = img_size
        self.mask_size = mask_size
        self.feat_size = feat_size

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, i):
        img_name = self.img_file[i]
        mask_name = self.img_file[i].split('.')[0]

        path = os.path.join(self.imgs_dir, img_name)
        img = Image.open(path).resize((self.img_size, self.img_size))
        img.load()
        img = np.asarray(img)

        if np.max(img) > 1:
            img = img/255.
        img = np.transpose(img, [2,0,1]).astype(np.float32)

        img = torch.from_numpy(np.expand_dims(img, 0)).cuda()
        mask = load_mask(self.mask_dir, [mask_name], self.mask_size)

        feature = get_stack(self.model, img, self.feat_size)

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

    def transform_npy(self, img, label, size):
        # flag_mask = torch.ones([3, size, size], dtype=torch.int32) # Mask use for Loss 

        ang, trans, scl, shr = transforms.RandomAffine.get_params(degrees = (-10,10),
                                                                translate = (0.4,0.4),
                                                                scale_ranges=(0.5,2.0),
                                                                shears=None,
                                                                img_size=(size,size))
        img = transforms.functional.affine(img, ang, trans, scl, shr, fillcolor=0)
        label = transforms.functional.affine(label, ang, trans, scl, shr, fillcolor=0)
        # flag_mask = transforms.functional.affine(flag_mask, ang, trans, scl, shr, fillcolor=0)

        if random.random() > 0.5:
            img = transforms.functional.hflip(img)
            label = transforms.functional.hflip(label)
            # flag_mask = transforms.functional.hflip(flag_mask)

        img = transforms.ToTensor()(img)
        return img, label #, flag_mask

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
            #'flag': flag
        }