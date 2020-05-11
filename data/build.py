# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from copy import deepcopy

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .datasets.multidataset import MultiDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        datasets = []
        offsets = dict(num_train_pids=0, num_train_cams=0, num_query_pids=0,
                       num_query_cams=0, num_gallery_pids=0, num_gallery_cams=0)
        for name in cfg.DATASETS.NAMES:
            dataset = init_dataset(name, root=cfg.DATASETS.ROOT_DIR, offsets=deepcopy(offsets))
            offsets['num_train_pids'] += dataset.num_train_pids
            offsets['num_train_cams'] += dataset.num_train_cams
            offsets['num_query_pids'] += dataset.num_query_pids
            offsets['num_query_cams'] += dataset.num_query_cams
            offsets['num_gallery_pids'] += dataset.num_gallery_pids
            offsets['num_gallery_cams'] += dataset.num_gallery_cams
            datasets.append(dataset)
        dataset = MultiDataset(datasets)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
        )

    val_set = ImageDataset(datasets[0].query + datasets[0].gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes
