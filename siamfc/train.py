import os
from tqdm import tqdm

import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from net import SiamFC
from dataset import VID
from transforms import *
from config import config

# fix random seed
np.random.seed(1357)
torch.manual_seed(1234)


def create_label(shape, batch_size, use_gpu=True):
    # same for all pairs
    h, w = shape
    y = np.arange(h, dtype=np.float32) - (h-1) / 2.
    x = np.arange(w, dtype=np.float32) - (w-1) / 2.
    y, x = np.meshgrid(y, x)
    dist = np.sqrt(x**2 + y**2)
    mask = np.zeros((h, w))
    mask[dist <= config.radius / config.total_stride] = 1
    mask = mask[np.newaxis, :, :]
    weights = np.ones_like(mask)
    weights[mask == 1] = 0.5 / np.sum(mask == 1)
    weights[mask == 0] = 0.5 / np.sum(mask == 0)
    mask = np.repeat(mask, batch_size, axis=0)[:, np.newaxis, :, :]

    label = mask.astype(np.float32)
    weights = weights.astype(np.float32)
    if use_gpu:
        label = torch.from_numpy(label).cuda()
        weights = torch.from_numpy(weights).cuda()
    return label, weights


def weighted_loss(pred, gt, weight, batch_size):
    return F.binary_cross_entropy_with_logits(pred, gt, weight, reduction='sum') / batch_size


def train(data_dir, train_imdb, val_imdb, model_save_path="./model/", use_gpu=True):

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
    valid_z_transforms = Compose([
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor(),
    ])
    valid_x_transforms = Compose([
        ToTensor()
    ])

    train_dataset = VID(train_imdb, data_dir, train_z_transforms, train_x_transforms, 'train')
    val_dataset = VID(val_imdb, data_dir, valid_z_transforms, valid_x_transforms, 'val')

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                             shuffle=True, num_workers=config.train_num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size,
                           shuffle=True, num_workers=config.val_num_workers, drop_last=True)

    with torch.no_grad():
        train_gt, train_weight = create_label((config.train_response_sz, config.train_response_sz), config.train_batch_size)
        val_gt, val_weight = create_label((config.response_sz, config.response_sz), config.val_batch_size)

    net = SiamFC()
    if use_gpu:
        net.cuda()
        net = torch.nn.DataParallel(net)

    optimizer = torch.optim.SGD(net.parameters(), config.lr, config.momentum, config.weight_decay)
    scheduler = StepLR(optimizer, config.step_size, config.gamma)

    for epoch in range(config.num_epochs):

        scheduler.step()

        train_loss = []
        net.train()
        for i, data in enumerate(tqdm(train_loader)):
            exemplar_imgs, instance_imgs = data
            if use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            optimizer.zero_grad()
            outputs = net(exemplar_imgs, instance_imgs)
            loss = weighted_loss(outputs, train_gt, train_weight, config.train_batch_size)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.data)
        train_loss = np.mean(train_loss)

        val_loss = []
        net.eval()
        for i, data in enumerate(tqdm(val_loader)):
            exemplar_imgs, instance_imgs = data
            if use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            outputs = net(exemplar_imgs, instance_imgs)
            loss = weighted_loss(outputs, val_gt, val_weight, config.val_batch_size)
            val_loss.append(loss.data)
        val_loss = np.mean(val_loss)
        print("EPOCH %d train_loss: %.4f, val_loss: %.4f" %
             (epoch, train_loss, val_loss))
        torch.save(net.state_dict(),
                   "./models/siamfc_{}.pth".format(epoch+1))


if __name__ == "__main__":

    data_dir = "data/ILSVRC2015_VID_CURATION"
    train_imdb = "data/imdb_video_train.json"
    val_imdb = "data/imdb_video_val.json"

    train(data_dir, train_imdb, val_imdb)
