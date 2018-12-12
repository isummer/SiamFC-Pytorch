import os
import cv2
import numpy as np
import json
from torch.utils.data.dataset import Dataset
from config import config


class VID(Dataset):

    def __init__(self, imdb, data_dir, z_transforms, x_transforms, phase='train'):
        imdb_video      = json.load(open(imdb, 'r'))
        self.videos     = imdb_video['videos']
        self.data_dir   = data_dir
        self.num_videos = int(imdb_video['num_videos'])

        self._imgpath = os.path.join(self.data_dir, "%s", "%06d.%02d.x.jpg")

        self.z_transforms = z_transforms
        self.x_transforms = x_transforms

        assert(phase in ['train', 'val'])
        if phase == 'train':
            self.itemCnt = config.num_pairs
        else:
            self.itemCnt = self.num_videos

    def __getitem__(self, idx):
        idx = idx % self.num_videos
        video = self.videos[idx]
        trajs = video['trajs']
        # sample one trajs
        trackid = np.random.choice(list(trajs.keys()))
        traj = trajs[trackid]

        rand_z = np.random.choice(range(len(traj)))
        possible_x_pos = list(range(len(traj)))
        rand_x = np.random.choice(possible_x_pos[max(rand_z - config.from_range, 0):rand_z] + possible_x_pos[(rand_z + 1):min(rand_z + config.from_range, len(traj))])

        z = traj[rand_z].copy()
        x = traj[rand_x].copy()

        # read z and x
        img_z = cv2.imread(self._imgpath % (video['name'], int(z['fno']), int(trackid)))
        img_z = cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB)

        img_x = cv2.imread(self._imgpath % (video['name'], int(x['fno']), int(trackid)))
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)

        # augmentation
        img_z = self.z_transforms(img_z)
        img_x = self.x_transforms(img_x)

        return img_z, img_x

    def __len__(self):
        return self.itemCnt
