# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp

from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, debug=False):
        self.dataset = dataset
        self.transform = transform
        self.debug = debug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
            if self.debug:
                import numpy as np
                import cv2

                mean, std = self.transform.transforms[-2].mean, self.transform.transforms[-2].std
                mean, std = np.array(mean, dtype=np.float32)[:, None, None], np.array(std, dtype=np.float32)[:, None,
                                                                             None]
                img = ((img.cpu().numpy() * std + mean) * 255).astype(np.uint8)
                img = np.transpose(img, (1, 2, 0))[..., ::-1]
                cv2.imshow(f'image_{pid}', img)
                cv2.waitKey(0)

        return img, pid, camid, img_path
