# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import numpy as np
import torchvision.transforms as T
from PIL import Image

from .transforms import RandomErasing


def build_aug(p=.8):
    import cv2
    from albumentations import (
        CLAHE, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, IAAAdditiveGaussianNoise,
        GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
        IAASharpen, Flip, OneOf, Compose, RGBShift, RandomGamma, ElasticTransform, ImageCompression
    )

    composer = Compose([
        Flip(),
        Transpose(),
        OneOf([
            ImageCompression(quality_lower=40),
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.3, scale_limit=0.5, p=0.2, border_mode=cv2.BORDER_CONSTANT,
                         value=(128, 128, 128)),
        OneOf([
            OpticalDistortion(shift_limit=0, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128)),
            GridDistortion(p=.1, border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128)),
            IAAPiecewiseAffine(p=0.3),
            ElasticTransform(p=0.3, border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128)),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            RandomBrightnessContrast(),
        ], p=0.3),
        OneOf([
            RGBShift(r_shift_limit=(40, 60), g_shift_limit=(40, 60), b_shift_limit=0),
            RandomGamma(gamma_limit=(60, 200)),
            RandomBrightnessContrast(),
        ], p=0.3),
    ], p=p)

    def _wrapper(img):
        img = np.array(img)
        augmented = composer(image=img)['image']
        return Image.fromarray(augmented)

    return _wrapper


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            build_aug(),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
