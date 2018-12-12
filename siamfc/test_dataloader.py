import os
import numpy as np
import cv2

from torch.utils.data import DataLoader

from dataset import VID
from transforms import *
from config import config

if __name__ == '__main__':

    data_dir = "data/ILSVRC_VID_CURATION"
    train_imdb = "data/imdb_video_train.json"
    val_imdb = "data/imdb_video_val.json"

    center_crop_size = config.instance_size - config.total_stride
    random_crop_size = config.instance_size - 2 * config.total_stride

    train_z_transforms = Compose([
        RandomStretch(),
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    train_x_transforms = Compose([
        RandomStretch(),
        CenterCrop((center_crop_size, center_crop_size)),
        RandomCrop((random_crop_size, random_crop_size)),
        ToTensor()
    ])
    val_z_transforms = Compose([
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor(),
    ])
    val_x_transforms = Compose([
        ToTensor()
    ])

    train_dataset = VID(train_imdb, data_dir, train_z_transforms, train_x_transforms, 'train')
    val_dataset = VID(val_imdb, data_dir, val_z_transforms, val_x_transforms, 'val')

    train_loader = DataLoader(train_dataset, batch_size=1, # config.batch_size,
                             shuffle=True, num_workers=config.train_num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.val_num_workers, drop_last=True)

    for idx, (z_crops, x_crops) in enumerate(train_loader):
        print(x_crops.size())
